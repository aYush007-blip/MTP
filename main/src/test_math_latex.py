"""
Evaluation Suite for T5 Spoken-Math-to-LaTeX Models.
Evaluates accuracy on both MathSpeech real lectures and Synthetic test sets.
Supports comparing baseline vs. v2 models.
"""

import argparse
import json
import re
from pathlib import Path

import torch
import transformers.models.t5.modeling_t5 as modeling_t5
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Safe LayerNorm on Jetson Orin (avoids integer pow kernel issues)
def _safe_t5_layernorm_forward(self, hidden_states):
    variance = hidden_states.to(torch.float32)
    variance = (variance * variance).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(self.weight.dtype)

modeling_t5.T5LayerNorm.forward = _safe_t5_layernorm_forward


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate T5 Math-to-LaTeX model.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/mnt/thesis/models/math-to-latex-v2",
        help="Path to fine-tuned model checkpoint.",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="/mnt/thesis/datasets/MathSpeech/splits_v2/test_mathspeech.jsonl",
        help="Path to test set JSONL.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for evaluation (cpu or cuda).",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for generation (default 1 for fast greedy decoding).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples evaluated.",
    )
    return parser.parse_args()


def normalize_latex(text):
    text = text.strip()
    text = re.sub(r"^\$|\$$", "", text).strip()
    # Normalize backslashes, whitespace
    text = re.sub(r"\s+", "", text)
    return text


def evaluate():
    args = parse_args()
    model_dir = Path(args.model_dir)
    test_path = Path(args.test_path)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test set file does not exist: {test_path}")

    device = torch.device(args.device)
    print(f"Loading model from: {model_dir}")
    print(f"Loading test data:  {test_path}")
    print(f"Evaluation device:  {device}")

    tokenizer = T5Tokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True
    ).to(device)
    model.eval()

    with open(test_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    if args.max_samples:
        rows = rows[: args.max_samples]

    prefix = "translate spoken math to LaTeX: "
    exact_matches = 0
    norm_matches = 0

    print(f"\nEvaluating {len(rows)} test examples...\n" + "=" * 60)

    for i, row in enumerate(rows):
        inputs = tokenizer(
            prefix + row["input"],
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=args.num_beams,
            )

        prediction = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        target = row["target"].strip()

        exact = prediction == target
        norm = normalize_latex(prediction) == normalize_latex(target)

        exact_matches += int(exact)
        norm_matches += int(norm)

        if i < 15 or not norm:
            status = "MATCH" if norm else "MISMATCH"
            print(f"[{i+1}/{len(rows)}] [{status}]")
            print(f"  Input:     {row['input']}")
            print(f"  Expected:  {target}")
            print(f"  Predicted: {prediction}")
            print("-" * 60)

    print("\n" + "=" * 60)
    print(f"Results for Model: {model_dir.name} on {test_path.name}")
    print(f"Total Test Examples:       {len(rows)}")
    print(f"Strict Exact Matches:      {exact_matches}/{len(rows)} ({100*exact_matches/len(rows):.2f}%)")
    print(f"Normalized Exact Matches:  {norm_matches}/{len(rows)} ({100*norm_matches/len(rows):.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
