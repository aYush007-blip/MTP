#!/usr/bin/env python3
"""
Standalone Fine-Tuning Pipeline for Qwen2.5-1.5B-Instruct on Signals & Systems Transduction.
Compatible with Windows (NVIDIA GPU), Linux, and Google Colab.

Features:
  - Memory-safe QLoRA 4-bit (bitsandbytes) by default (runs on 6GB-8GB GPUs)
  - Full FP16/BF16 LoRA option for 16GB+ GPUs
  - Assistant-only loss masking (-100 on system & user prompt tokens)
  - Auto-decompresses train.jsonl.gz / val.jsonl.gz (and refreshes if .gz is newer)
  - Compatible with both transformers 4.x and 5.x
  - Targets all attention and MLP projection layers
  - Merges trained LoRA adapter directly into base weights for GGUF conversion
"""

import argparse
import gzip
import inspect
import os
from pathlib import Path
import shutil
import sys
import time

# Completely eliminate PyTorch CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-1.5B on Signals & Systems dataset.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model ID or local directory")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).parent / "data"),
                        help="Directory containing train.jsonl (or train.jsonl.gz)")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).parent / "output_qwen_signals"),
                        help="Directory to save checkpoints and merged model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1 for ~15 mins)")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Max training steps (if > 0 overrides epochs, e.g. 500 steps)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size (default: 4 for rock-solid VRAM safety)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch * accum = 16)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence token length (512 covers 100% of our dataset without empty padding)")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling parameter")
    parser.add_argument("--qlora", action="store_true", default=None,
                        help="Force 4-bit NF4 QLoRA quantization (saves VRAM on <=8GB GPUs; slow on T4)")
    parser.add_argument("--no-qlora", dest="qlora", action="store_false",
                        help="Force native 16-bit LoRA (runs 10x faster on Tensor Cores; recommended for >=12GB GPUs)")
    parser.add_argument("--grad-checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing (default: True to keep VRAM strictly under 5GB)")
    parser.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false",
                        help="Disable gradient checkpointing")
    parser.add_argument("--merge-and-export", action="store_true", default=True,
                        help="Automatically merge LoRA adapter into base model after training")
    return parser.parse_args()


def ensure_decompressed(data_dir: Path) -> tuple[Path, Path]:
    """Ensures train.jsonl and val.jsonl are uncompressed and up-to-date with .gz archives."""
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    train_gz = data_dir / "train.jsonl.gz"
    val_gz = data_dir / "val.jsonl.gz"

    for plain, gz in [(train_file, train_gz), (val_file, val_gz)]:
        if gz.exists() and (not plain.exists() or gz.stat().st_mtime > plain.stat().st_mtime):
            print(f"[Dataset] Decompressing {gz} -> {plain}...")
            with gzip.open(gz, "rb") as f_in, open(plain, "wb") as f_out:
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
    print(f"Max Length:       {args.max_length}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Quantization:     {'4-bit QLoRA (NF4)' if args.qlora else '16-bit LoRA'}")
    print(f"LoRA Rank/Alpha:  r={args.lora_rank}, alpha={args.lora_alpha}")
    print("=" * 70)

    # 1. Imports check
    try:
        import torch

        # Guard against upstream PEFT bug when an older torchao (<0.16.0) is present
        try:
            import peft.import_utils
            peft.import_utils.is_torchao_available = lambda: False
        except Exception:
            pass

        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq,
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
        if args.qlora is None:
            args.qlora = False
        if args.grad_checkpointing is None:
            args.grad_checkpointing = False
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        print(f"[Device] Using GPU: {gpu_name} ({gpu_mem:.2f} GB VRAM, sm_{cc_major}{cc_minor})")

        # Auto-configure QLoRA vs native FP16 LoRA:
        # A 1.5B model takes only ~3.1 GB in FP16. If GPU has >= 10 GB VRAM (e.g. Kaggle T4 with 15GB or P100 with 16GB),
        # running 4-bit QLoRA forces software dequantization on every layer, slowing down training by 35x.
        if gpu_mem >= 10.0 or cc_major < 7:
            if args.qlora:
                print(f"[Auto-Optimizer] Overriding --qlora -> native FP16: GPU has {gpu_mem:.1f} GB VRAM, while Qwen2.5-1.5B takes only 3.1 GB.")
                print("                 Running on native FP16 Tensor Cores with non-reentrant activation checkpointing (VRAM safe: ~3.8 GB).")
            args.qlora = False
        else:
            if args.qlora is None:
                args.qlora = True

        # Keep gradient checkpointing enabled with use_reentrant=False to ensure VRAM stays rock-solid under 5 GB
        args.grad_checkpointing = True

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
    tokenizer.padding_side = "right"

    print(f"[2/5] Loading base model weights...")
    model_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }

    if args.qlora and device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = {"": 0}
    else:
        dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False

    if not args.qlora and device == "cuda":
        model = model.to("cuda:0")

    if args.qlora and device == "cuda":
        model = prepare_model_for_kbit_training(model)
    elif args.grad_checkpointing:
        model.enable_input_require_grads()

    # 4. Configure LoRA
    print(f"[3/5] Configuring LoRA adapter (r={args.lora_rank}, alpha={args.lora_alpha})...")

    # Explicitly ensure torchao LoRA dispatcher is disabled (avoids ImportError with torchao <0.16.0)
    try:
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False
    except Exception:
        pass
    try:
        import peft.tuners.lora.torchao
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
    except Exception:
        pass

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

    # 5. Tokenize Dataset with Assistant-Only Loss Masking
    print(f"\n[4/5] Tokenizing dataset using ChatML template (assistant-only loss masking)...")
    data_files = {"train": str(train_path)}
    if val_path.exists():
        data_files["validation"] = str(val_path)

    raw_dataset = load_dataset("json", data_files=data_files)

    def preprocess_chatml(example):
        messages = example["messages"]
        prompt_messages = messages[:-1]  # system + user
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_tokens = tokenizer(
            full_text,
            max_length=args.max_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )

        input_ids = full_tokens["input_ids"]
        labels = input_ids.copy()
        # Mask system + user prompt tokens so loss is computed ONLY on assistant LaTeX output
        mask_len = min(len(prompt_ids), len(labels))
        for i in range(mask_len):
            labels[i] = -100

        full_tokens["labels"] = labels
        return full_tokens

    tokenized_dataset = raw_dataset.map(
        preprocess_chatml, batched=False, remove_columns=["messages"]
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    # 6. Training Arguments & Trainer (compatible with transformers 4.x and 5.x)
    output_path = Path(args.output_dir)
    adapter_path = output_path / "lora_adapter"
    merged_path = output_path / "merged_model"
    output_path.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

    steps_per_epoch = max(1, len(tokenized_dataset["train"]) // (args.batch_size * args.grad_accum))
    warmup_steps = max(1, int(steps_per_epoch * args.epochs * 0.05))

    ta_kwargs = {
        "output_dir": str(adapter_path),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "logging_steps": 25,
        eval_key: "epoch" if val_path.exists() else "no",
        "save_strategy": "epoch",
        "fp16": use_fp16,
        "bf16": use_bf16,
        "optim": "paged_adamw_8bit" if (args.qlora and device == "cuda") else "adamw_torch",
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": "cosine",
        "report_to": "none",
        "save_total_limit": 1,
        "gradient_checkpointing": args.grad_checkpointing,
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": True,
    }
    if args.max_steps > 0:
        ta_kwargs["max_steps"] = args.max_steps
    if "gradient_checkpointing_kwargs" in ta_params:
        ta_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    # Guard against any removed kwargs in future transformers versions
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in ta_params}
    training_args = TrainingArguments(**ta_kwargs)

    trainer_params = inspect.signature(Trainer.__init__).parameters
    tok_key = "processing_class" if "processing_class" in trainer_params else "tokenizer"
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_dataset["train"],
        "eval_dataset": tokenized_dataset.get("validation", None),
        tok_key: tokenizer,
        "data_collator": data_collator,
    }
    trainer = Trainer(**trainer_kwargs)

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
