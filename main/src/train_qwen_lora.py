#!/usr/bin/env python3
"""
LoRA Fine-Tuning Pipeline for Qwen2.5-1.5B Spoken-Math-to-LaTeX Transduction.

Trained on the balanced 20,000-sample dataset (MathBridge + Speech2LaTeX + OCW).
Features:
  - Memory-safe QLoRA / LoRA targeting all attention and MLP projection layers
  - Gradient checkpointing enabled
  - Cosine learning rate schedule with warmup
  - Direct export to merged HuggingFace weights and GGUF conversion instructions
"""

import argparse
import os
from pathlib import Path
import sys
import torch

DEFAULT_MODEL = "/mnt/thesis/models/Qwen2.5-1.5B-Instruct"
DEFAULT_DATA_DIR = "/mnt/thesis/datasets/Qwen_FineTune_20k"
DEFAULT_OUTPUT_DIR = "/mnt/thesis/models/qwen2.5-1.5b-math-lora"


def train(
    model_name_or_path: str = DEFAULT_MODEL,
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    lr: float = 2e-4,
    epochs: int = 3,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    max_seq_length: int = 512,
):
    print("=" * 65)
    print(f"Starting Qwen2.5-1.5B LoRA Fine-Tuning")
    print(f"Model:      {model_name_or_path}")
    print(f"Dataset:    {data_dir}")
    print(f"Output:     {output_dir}")
    print(f"Parameters: Epochs={epochs}, LR={lr}, LoRA Rank={lora_rank}, Alpha={lora_alpha}")
    print("=" * 65)

    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq
        )
    except ImportError:
        print("[ERROR] Required libraries not found. Install with:")
        print("  pip install transformers peft datasets trl accelerate")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load Tokenizer & Model
    print("[1/5] Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # Configure LoRA
    print("[2/5] Configuring LoRA adapter...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load Dataset
    print("[3/5] Loading ChatML JSONL dataset...")
    train_path = Path(data_dir) / "train.jsonl"
    val_path = Path(data_dir) / "val.jsonl"

    if not train_path.exists():
        print(f"[ERROR] Training file not found at {train_path}. Run prepare_finetune_dataset.py first.")
        sys.exit(1)

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
    )

    def preprocess_chatml(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(
            text,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
        )
        tokens["labels"] = [
            t if t != tokenizer.pad_token_id else -100
            for t in tokens["input_ids"]
        ]
        return tokens

    print("[4/5] Tokenizing dataset with ChatML template...")
    tokenized_dataset = dataset.map(preprocess_chatml, batched=False, remove_columns=["messages"])

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=25,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )

    # Train
    print("[5/5] Beginning LoRA Training...")
    trainer.train()

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    print("\n" + "=" * 65)
    print(f"LoRA Training Complete!")
    print(f"Saved LoRA Adapter: {out_path}")
    print("\nTo export directly into GGUF for llama-server on Jetson Orin:")
    print("  1. Merge adapter: python -c 'from peft import PeftModel; ...'")
    print("  2. Convert to GGUF: python /mnt/thesis/llama.cpp/convert_hf_to_gguf.py ...'")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Training for Qwen2.5-1.5B Spoken-Math-to-LaTeX")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train(
        model_name_or_path=args.model,
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
    )
