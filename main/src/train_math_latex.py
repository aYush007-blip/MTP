"""
Fine-tuning Google T5-Small for Spoken-Math-to-LaTeX Conversion (v2)
Trained on unified MathSpeech + Realistic Synthetic Dataset (9,350 training rows).
Optimized for NVIDIA Jetson Orin Nano (CUDA 12.6, PyTorch 2.8.0, SM87).
Expanded tokenizer vocabulary with LaTeX syntax and math commands.
"""

import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import transformers.models.t5.modeling_t5 as modeling_t5
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

# Safe LayerNorm on Jetson Orin (avoids integer pow kernel issues)
def _safe_t5_layernorm_forward(self, hidden_states):
    variance = hidden_states.to(torch.float32)
    variance = (variance * variance).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(self.weight.dtype)

modeling_t5.T5LayerNorm.forward = _safe_t5_layernorm_forward

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_DIR = Path("/mnt/thesis/datasets/MathSpeech/splits_v2")
MODEL_DIR = "/mnt/thesis/models/t5-small-baseline"
OUT_DIR = Path("/mnt/thesis/models/math-to-latex-v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path("/mnt/thesis/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "train_t5_v2.log"

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2  # Effective batch size = 8, memory-safe for 8GB Jetson Orin
EPOCHS = 5
LR = 2e-4
MAX_LEN = 128
PREFIX = "translate spoken math to LaTeX: "

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(
    MODEL_DIR, local_files_only=True
).to(device)

# Expand tokenizer vocabulary with essential LaTeX syntax and math symbols
LATEX_TOKENS = [
    "{", "}", "\\", "^", "_", "<", ">", "~",
    "\\frac", "\\sqrt", "\\sin", "\\cos", "\\tan", "\\ln", "\\log",
    "\\pi", "\\theta", "\\alpha", "\\beta", "\\gamma", "\\delta",
    "\\lambda", "\\sigma", "\\mu", "\\omega", "\\phi", "\\int",
    "\\iint", "\\oint", "\\sum", "\\lim", "\\times", "\\cdot",
    "\\pm", "\\div", "\\partial", "\\infty", "\\neq", "\\leq",
    "\\geq", "\\approx", "\\left", "\\right", "\\vec", "\\dot",
    "\\ddot", "\\hat", "\\prime", "\\to", "\\rightarrow",
    "\\ldots", "\\cdots", "\\mathrm", "\\mathbf", "\\displaystyle"
]
num_added = tokenizer.add_tokens(LATEX_TOKENS)
if num_added > 0:
    model.resize_token_embeddings(len(tokenizer))
    print(f"Added {num_added} LaTeX tokens. Vocab size: {len(tokenizer)}")


class MathDataset(Dataset):
    def __init__(self, filename):
        with open(filename, encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        source = tokenizer(
            PREFIX + row["input"],
            max_length=MAX_LEN,
            truncation=True,
            return_tensors="pt",
        )
        target = tokenizer(
            row["target"],
            max_length=MAX_LEN,
            truncation=True,
            return_tensors="pt",
        )
        labels = target.input_ids.squeeze(0)
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": source.input_ids.squeeze(0),
            "attention_mask": source.attention_mask.squeeze(0),
            "labels": labels,
        }


def collate(batch):
    inputs = tokenizer.pad(
        [
            {
                "input_ids": x["input_ids"],
                "attention_mask": x["attention_mask"],
            }
            for x in batch
        ],
        padding=True,
        return_tensors="pt",
    )
    max_target = max(len(x["labels"]) for x in batch)
    labels = torch.full((len(batch), max_target), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        labels[i, : len(x["labels"])] = x["labels"]
    inputs["labels"] = labels
    return {k: v.to(device) for k, v in inputs.items()}


train_ds = MathDataset(DATA_DIR / "train.jsonl")
val_ds = MathDataset(DATA_DIR / "validation.jsonl")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
)

effective_steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
total_optimization_steps = effective_steps_per_epoch * EPOCHS
warmup_steps = int(0.05 * total_optimization_steps)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, foreach=False)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optimization_steps
)

print(f"Training dataset:        {len(train_ds)} examples")
print(f"Validation dataset:      {len(val_ds)} examples")
print(f"Batch size:              {BATCH_SIZE} (Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
print(f"Total optimization steps:{total_optimization_steps}")
print(f"Output directory:        {OUT_DIR}")
print(f"Log file:                {LOG_FILE}\n")


def log_msg(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


log_msg(f"--- Training started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    model.train()
    train_total = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader, 1):
        loss = model(**batch).loss / GRAD_ACCUM_STEPS
        loss.backward()
        train_total += loss.item() * GRAD_ACCUM_STEPS

        if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step % (100 * GRAD_ACCUM_STEPS) == 0 or step == len(train_loader):
            current_lr = scheduler.get_last_lr()[0]
            current_loss = loss.item() * GRAD_ACCUM_STEPS
            log_msg(
                f"Epoch {epoch}/{EPOCHS} | Step {step:4d}/{len(train_loader)} | "
                f"Loss: {current_loss:.4f} | LR: {current_lr:.2e}"
            )

    # Validation
    model.eval()
    val_total = 0.0
    with torch.no_grad():
        for batch in val_loader:
            val_total += model(**batch).loss.item()

    train_loss = train_total / len(train_loader)
    val_loss = val_total / len(val_loader)
    epoch_duration = time.time() - epoch_start

    log_msg(
        f"\n>>> Epoch {epoch} Complete ({epoch_duration:.1f}s): "
        f"Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}"
    )

    # Save checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        log_msg(f"*** New best validation loss! Saving model to {OUT_DIR} ***\n")
        model.save_pretrained(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)
    else:
        log_msg(f"Saving latest checkpoint to {OUT_DIR}\n")
        model.save_pretrained(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)

log_msg(f"--- Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
log_msg(f"Best Validation Loss: {best_val_loss:.4f}")
