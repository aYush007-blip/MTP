#!/usr/bin/env python3
"""
Standalone Fine-Tuning Pipeline for Qwen2.5-1.5B on Signals & Systems Transduction.
Compatible with Windows (NVIDIA GPU), Linux, and Google Colab.

Features:
  - Memory-safe QLoRA 4-bit (bitsandbytes) by default (runs on 6GB-8GB GPUs)
  - Full FP16/BF16 LoRA option for 16GB+ GPUs
  - Auto-decompresses train.jsonl.gz if train.jsonl is not present
  - Targets all attention and MLP projection layers
  - Merges trained LoRA adapter directly into base weights for 1-click GGUF conversion
"""

import argparse
import gzip
import os
from pathlib import Path
import shutil
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-1.5B on Signals & Systems dataset.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model ID or local directory")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).parent / "data"),
                        help="Directory containing train.jsonl (or train.jsonl.gz)")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).parent / "output_qwen_signals"),
                        help="Directory to save checkpoints and merged model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (effective batch = batch * accum)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence token length")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling parameter")
    parser.add_argument("--qlora", action="store_true", default=True,
                        help="Enable 4-bit NF4 QLoRA quantization (saves VRAM; recommended for <=12GB GPUs)")
    parser.add_argument("--no-qlora", dest="qlora", action="store_false",
                        help="Disable 4-bit QLoRA and use 16-bit LoRA (requires >=16GB VRAM)")
    parser.add_argument("--merge-and-export", action="store_true", default=True,
                        help="Automatically merge LoRA adapter into base model after training")
    return parser.parse_args()


def ensure_decompressed(data_dir: Path) -> tuple[Path, Path]:
    """Ensures train.jsonl and val.jsonl are uncompressed and ready to load."""
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    train_gz = data_dir / "train.jsonl.gz"
    val_gz = data_dir / "val.jsonl.gz"

    if not train_file.exists() and train_gz.exists():
        print(f"[Dataset] Decompressing {train_gz} -> {train_file}...")
        with gzip.open(train_gz, "rb") as f_in, open(train_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    if not val_file.exists() and val_gz.exists():
        print(f"[Dataset] Decompressing {val_gz} -> {val_file}...")
        with gzip.open(val_gz, "rb") as f_in, open(val_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    if not train_file.exists():
        raise FileNotFoundError(f"Training dataset not found in {data_dir}. Expected train.jsonl or train.jsonl.gz")

    return train_file, val_file


def main():
    args = parse_args()

    print("=" * 70)
    print("  QWEN2.5-1.5B SIGNALS & SYSTEMS FINE-TUNING PIPELINE")
    print("=" * 70)
    print(f"Base Model:       {args.model_id}")
    print(f"Data Directory:   {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size:       {args.batch_size} (effective batch = {args.batch_size * args.grad_accum})")
    print(f"Learning Rate:    {args.lr}")
    print(f"Quantization:     {'4-bit QLoRA (NF4)' if args.qlora else '16-bit LoRA'}")
    print(f"LoRA Rank/Alpha:  r={args.lora_rank}, alpha={args.lora_alpha}")
    print("=" * 70)

    # 1. Imports check
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq
        )
    except ImportError as e:
        print(f"\n[ERROR] Missing required packages: {e}")
        print("Please install requirements using:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("\n[WARNING] CUDA is not available! Training will run on CPU, which is extremely slow.")
        print("Ensure NVIDIA drivers, CUDA, and PyTorch with CUDA support are installed.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Device] Using GPU: {gpu_name} ({gpu_mem:.2f} GB VRAM)")

    # 2. Dataset preparation
    data_dir_path = Path(args.data_dir)
    train_path, val_path = ensure_decompressed(data_dir_path)
    print(f"[Dataset] Train file: {train_path} ({train_path.stat().st_size / (1024*1024):.2f} MB)")
    if val_path.exists():
        print(f"[Dataset] Val file:   {val_path} ({val_path.stat().st_size / (1024*1024):.2f} MB)")

    # 3. Load Tokenizer & Model
    print(f"\n[1/5] Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2/5] Loading base model weights...")
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if device == "cuda" else None,
    }

    if args.qlora and device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (torch.float16 if torch.cuda.is_available() else torch.float32)
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    if args.qlora and device == "cuda":
        model = prepare_model_for_kbit_training(model)

    # 4. Configure LoRA
    print(f"[3/5] Configuring LoRA adapter (r={args.lora_rank}, alpha={args.lora_alpha})...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Tokenize Dataset
    print(f"\n[4/5] Tokenizing dataset using ChatML template...")
    data_files = {"train": str(train_path)}
    if val_path.exists():
        data_files["validation"] = str(val_path)

    raw_dataset = load_dataset("json", data_files=data_files)

    def preprocess_chatml(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(
            text,
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = raw_dataset.map(preprocess_chatml, batched=False, remove_columns=["messages"])
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt")

    # 6. Training Arguments & Trainer
    output_path = Path(args.output_dir)
    adapter_path = output_path / "lora_adapter"
    merged_path = output_path / "merged_model"
    output_path.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir=str(adapter_path),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=20,
        eval_strategy="epoch" if val_path.exists() else "no",
        save_strategy="epoch",
        fp16=use_fp16,
        bf16=use_bf16,
        optim="adamw_torch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        save_total_limit=1,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation", None),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 7. Execute Training
    print(f"\n[5/5] Commencing Fine-Tuning across {args.epochs} Epochs...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    print(f"\n[Training Complete] Finished in {elapsed/60:.2f} minutes.")

    # Save trained adapter
    print(f"[Save] Saving LoRA adapter to {adapter_path}...")
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    # 8. Merge LoRA into base weights if requested
    if args.merge_and_export:
        print(f"\n[Merge] Merging LoRA weights into base model for GGUF conversion...")
        try:
            from peft import PeftModel
            base_model_reload = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
            )
            merged_model = PeftModel.from_pretrained(base_model_reload, str(adapter_path))
            merged_model = merged_model.merge_and_unload()

            merged_path.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(str(merged_path))
            tokenizer.save_pretrained(str(merged_path))
            print(f"[Merge Complete] Merged HuggingFace model saved at:\n  {merged_path}")
            print(f"\nNext: Run GGUF conversion to create the Jetson model!")
        except Exception as e:
            print(f"[Warning] Automatic merge failed: {e}")
            print("You can merge manually or convert using llama.cpp's convert_lora_to_gguf.py.")

    print("\n" + "=" * 70)
    print("  ALL TRAINING STAGES FINISHED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
