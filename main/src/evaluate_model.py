#!/usr/bin/env python3
"""
Comprehensive Evaluation & Benchmarking Suite for Spoken-Math-to-LaTeX Models.

Evaluates models on downloaded corpora:
  - MathSpeech real lecture test set (test_mathspeech.jsonl)
  - Synthetic mathematical benchmark (test_synthetic.jsonl)
  - Technical engineering lectures manifest (manifest.csv)
  - Or any user-specified .jsonl / .csv file

Metrics Calculated:
  1. Strict Exact Match (Strict-EM %): Identical raw string equality.
  2. Normalized Exact Match (Norm-EM %): Mathematical notation equivalence
     (normalizes whitespace, brackets, \displaystyle, implicit multiplication).
  3. Token BLEU-4 Score: N-gram precision on LaTeX syntax tokens.
  4. Character Error Rate (CER %): Levenshtein distance on characters.
  5. Token Error Rate (TER %): Levenshtein distance on LaTeX tokens.
  6. Latency Profile: Mean, median, min, max, and P95 latency in ms per sample.

Outputs Saved:
  - Detailed per-sample JSON: <output_dir>/eval_<dataset>_<mode>_<timestamp>.json
  - Spreadsheet CSV: <output_dir>/eval_<dataset>_<mode>_<timestamp>.csv
  - Human-readable Markdown summary: <output_dir>/summary_<dataset>_<mode>_<timestamp>.md

Usage:
  python src/evaluate_model.py --dataset mathspeech
  python src/evaluate_model.py --dataset synthetic --max_samples 100
  python src/evaluate_model.py --dataset manifest --max_samples 50
  python src/evaluate_model.py --dataset /path/to/custom_data.jsonl
  python src/evaluate_model.py --mode model_only
"""

import argparse
import csv
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
import time

# Lazy import handling for T5 and PyTorch
_t5_patched = False

def _ensure_t5_patched():
    global _t5_patched
    if not _t5_patched:
        import torch
        import transformers.models.t5.modeling_t5 as modeling_t5
        def _safe_t5_layernorm_forward(self, hidden_states):
            variance = hidden_states.to(torch.float32)
            variance = (variance * variance).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(self.weight.dtype)
        modeling_t5.T5LayerNorm.forward = _safe_t5_layernorm_forward
        _t5_patched = True


# ===========================================================================
# Metric Utilities
# ===========================================================================

def tokenize_latex(s: str) -> list[str]:
    """Tokenize LaTeX into syntax symbols, commands, numbers, and braces."""
    pattern = r"\\[a-zA-Z]+|[a-zA-Z]|[0-9]+|\^|_|\{|\}|\(|\)|\[|\]|\+|-|=|<|>|/|\*|\.|,"
    return re.findall(pattern, s)


def normalize_latex(text: str) -> str:
    """Normalize LaTeX notation to canonical form for fair equivalence testing."""
    t = text.strip().strip("$").strip()
    # Strip styling commands
    t = re.sub(r"\\displaystyle\s*", "", t)
    t = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", t)
    t = re.sub(r"\\left\s*", "", t)
    t = re.sub(r"\\right\s*", "", t)
    # Strip thin / alignment spaces
    t = re.sub(r"\\[,;!]|\\quad|\\qquad", " ", t)
    # Standardize multiplication
    t = re.sub(r"\\times|\\cdot", "*", t)
    # Remove all whitespace for structural comparison
    t = re.sub(r"\s+", "", t)
    # Simplify redundant single-char braces: e.g. {x} -> x
    t = re.sub(r"\{([a-zA-Z0-9])\}", r"\1", t)
    # Normalize implicit multiplication: e.g. 3*(x-1) -> 3(x-1)
    t = re.sub(r"([0-9a-zA-Z])\*([a-zA-Z\(])", r"\1\2", t)
    t = re.sub(r"(\))\*([a-zA-Z0-9\(])", r"\1\2", t)
    return t


def levenshtein_distance(seq1, seq2) -> int:
    """Compute standard dynamic programming Levenshtein distance."""
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0] * size_y for _ in range(size_x)]
    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x][y] = matrix[x - 1][y - 1]
            else:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,
                    matrix[x][y - 1] + 1,
                    matrix[x - 1][y - 1] + 1,
                )
    return matrix[size_x - 1][size_y - 1]


def calculate_cer(reference: str, candidate: str) -> float:
    """Character Error Rate (CER %)."""
    ref_chars = list(reference)
    cand_chars = list(candidate)
    if not ref_chars:
        return 0.0 if not cand_chars else 100.0
    dist = levenshtein_distance(ref_chars, cand_chars)
    return min(100.0, (dist / len(ref_chars)) * 100.0)


def calculate_ter(reference_tokens: list[str], candidate_tokens: list[str]) -> float:
    """Token Error Rate (TER %) on LaTeX tokens."""
    if not reference_tokens:
        return 0.0 if not candidate_tokens else 100.0
    dist = levenshtein_distance(reference_tokens, candidate_tokens)
    return min(100.0, (dist / len(reference_tokens)) * 100.0)


def calculate_bleu(reference_tokens: list[str], candidate_tokens: list[str], max_n=4) -> float:
    """Sentence-level smoothed BLEU-4 on LaTeX tokens."""
    if not candidate_tokens or not reference_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        if len(candidate_tokens) < n or len(reference_tokens) < n:
            precisions.append(0.01)
            continue
        cand_ngrams = Counter(tuple(candidate_tokens[i : i + n]) for i in range(len(candidate_tokens) - n + 1))
        ref_ngrams = Counter(tuple(reference_tokens[i : i + n]) for i in range(len(reference_tokens) - n + 1))
        overlap = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in cand_ngrams.items())
        total = max(1, len(candidate_tokens) - n + 1)
        precisions.append((overlap + 0.1) / (total + 0.1))

    c, r = len(candidate_tokens), len(reference_tokens)
    bp = 1.0 if c > r else math.exp(1 - r / max(1, c))
    log_sum = sum(math.log(p) for p in precisions) / max_n
    return round(bp * math.exp(log_sum) * 100.0, 2)


# ===========================================================================
# Dataset Loader
# ===========================================================================

DATASET_SHORTCUTS = {
    "mathspeech": "/mnt/thesis/datasets/MathSpeech/splits_v2/test_mathspeech.jsonl",
    "synthetic": "/mnt/thesis/datasets/MathSpeech/splits_v2/test_synthetic.jsonl",
    "validation": "/mnt/thesis/datasets/MathSpeech/splits_v2/validation.jsonl",
    "manifest": "/mnt/thesis/datasets/manifest.csv",
}


def load_dataset(dataset_arg: str, max_samples: int | None = None) -> tuple[list[dict], str]:
    """Load test samples from a known shortcut or custom file path."""
    resolved_path = DATASET_SHORTCUTS.get(dataset_arg.lower(), dataset_arg)
    p = Path(resolved_path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    samples = []
    if p.suffix.lower() == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                inp = data.get("input", "") or data.get("transcript", "")
                tgt = data.get("target", "") or data.get("latex", "")
                if inp and tgt:
                    samples.append({
                        "id": str(line_idx + 1),
                        "input": inp.strip(),
                        "target": tgt.strip(),
                        "domain": data.get("domain", "Math"),
                    })
    elif p.suffix.lower() == ".csv":
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                inp = row.get("transcript", "") or row.get("input", "")
                tgt = row.get("latex", "") or row.get("target", "")
                if inp and tgt:
                    samples.append({
                        "id": row.get("example_id", str(row_idx + 1)),
                        "input": inp.strip(),
                        "target": tgt.strip(),
                        "domain": row.get("domain", "Engineering"),
                    })
    else:
        raise ValueError(f"Unsupported dataset format (expected .jsonl or .csv): {p.name}")

    if max_samples and max_samples > 0:
        samples = samples[:max_samples]

    return samples, p.name


# ===========================================================================
# Main Evaluation Engine
# ===========================================================================

def run_evaluation():
    parser = argparse.ArgumentParser(
        description="Comprehensive Benchmark & Evaluation Suite for Spoken-Math-to-LaTeX."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mathspeech",
        help="Dataset name ('mathspeech', 'synthetic', 'manifest', 'validation') or path to .jsonl/.csv",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/mnt/thesis/models/math-to-latex-v2",
        help="Path to T5 model directory.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pipeline", "model_only", "slm"],
        default="pipeline",
        help="'pipeline' uses MathRouter (canonical rules + T5 + cleaner). 'model_only' evaluates raw T5. 'slm' evaluates Qwen2.5 SLM.",
    )
    parser.add_argument(
        "--slm_model_path",
        type=str,
        default=None,
        help="Path to Qwen2.5 SLM GGUF file or HuggingFace directory (defaults to auto-detected GGUF).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "server", "llama_cli", "transformers"],
        default="auto",
        help="SLM backend: 'auto' (checks server -> llama_cli -> transformers), 'server', 'llama_cli', or 'transformers'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Inference device (cuda for GPU offloading).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit the number of evaluated samples (e.g. 50). Default is all.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/thesis/evaluation_results",
        help="Directory to save JSON, CSV, and Markdown evaluation reports.",
    )
    args = parser.parse_args()

    # 1. Load Data
    print("\n" + "=" * 70)
    print("🚀 SPOKEN-MATH-TO-LATEX BENCHMARK & EVALUATION SUITE")
    print("=" * 70)
    samples, dataset_name = load_dataset(args.dataset, args.max_samples)
    print(f"📁 Dataset:           {dataset_name} ({len(samples)} samples)")
    print(f"🧠 Model:             {Path(args.model_dir).name if args.mode != 'slm' else 'Qwen2.5-1.5B'}")
    print(f"⚙️  Evaluation Mode:   {args.mode.upper()}")
    print(f"💻 Device:            {args.device.upper()}")
    print("=" * 70 + "\n")

    if not samples:
        print("❌ Error: No valid samples found with input & target.")
        sys.exit(1)

    # 2. Initialize Model / Pipeline
    t_load_start = time.monotonic()
    if args.mode == "slm":
        from slm_translator import QwenMathTranslator
        slm_translator = QwenMathTranslator(
            model_path=args.slm_model_path,
            backend=args.backend,
            device=args.device,
        )
        translator = None
    else:
        _ensure_t5_patched()
        from router import MathRouter, T5MathTranslator
        translator = T5MathTranslator(model_dir=args.model_dir, device=args.device)
        router = MathRouter(translator=translator) if args.mode == "pipeline" else None
        slm_translator = None
    print(f"✅ Model ready in {round((time.monotonic() - t_load_start)*1000, 1)} ms.\n")

    # 3. Evaluation Loop
    results = []
    strict_matches = 0
    norm_matches = 0
    bleu_scores = []
    cer_scores = []
    ter_scores = []
    latencies = []

    print(f"Running evaluation on {len(samples)} samples...\n")
    start_eval_time = time.monotonic()

    for idx, sample in enumerate(samples, 1):
        inp_text = sample["input"]
        target_latex = sample["target"]

        # Run Prediction
        t_sample_start = time.monotonic()
        if args.mode == "pipeline":
            # Evaluates full router pipeline
            proc_res = router.process(inp_text)
            pred_latex = proc_res.processed_text
            # If input was pure math or span extracted, use the extracted latex
            if proc_res.spans:
                pred_latex = " ".join(s.latex_text for s in proc_res.spans)
            elif proc_res.processed_text.startswith("$") and proc_res.processed_text.endswith("$"):
                pred_latex = proc_res.processed_text.strip("$").strip()
            else:
                # Direct translation fallback if not formatted as inline
                pred_latex = translator.translate(inp_text)
        elif args.mode == "slm":
            pred_latex = slm_translator.translate(inp_text)
        else:
            # Evaluates raw T5 model directly
            pred_latex = translator.translate(inp_text)

        latency_ms = (time.monotonic() - t_sample_start) * 1000.0

        # Strip surrounding $ signs for uniform comparison
        clean_target = target_latex.strip().strip("$").strip()
        clean_pred = pred_latex.strip().strip("$").strip()

        # Compute Metrics
        is_strict = clean_pred == clean_target
        is_norm = normalize_latex(clean_pred) == normalize_latex(clean_target)

        target_tokens = tokenize_latex(clean_target)
        pred_tokens = tokenize_latex(clean_pred)

        bleu = calculate_bleu(target_tokens, pred_tokens)
        cer = calculate_cer(clean_target, clean_pred)
        ter = calculate_ter(target_tokens, pred_tokens)

        strict_matches += int(is_strict)
        norm_matches += int(is_norm)
        bleu_scores.append(bleu)
        cer_scores.append(cer)
        ter_scores.append(ter)
        latencies.append(latency_ms)

        results.append({
            "index": idx,
            "id": sample["id"],
            "domain": sample["domain"],
            "input": inp_text,
            "target": clean_target,
            "prediction": clean_pred,
            "strict_match": is_strict,
            "norm_match": is_norm,
            "bleu_4": bleu,
            "cer_pct": round(cer, 2),
            "ter_pct": round(ter, 2),
            "latency_ms": round(latency_ms, 1),
        })

        # Progress indicator every 10 samples
        if idx % 10 == 0 or idx == len(samples):
            current_norm_acc = (norm_matches / idx) * 100.0
            avg_lat = sum(latencies) / len(latencies)
            print(f"[{idx:4d}/{len(samples):4d}] Norm-EM: {current_norm_acc:5.1f}% | Avg Latency: {avg_lat:5.1f}ms")

    total_eval_sec = time.monotonic() - start_eval_time

    # 4. Summary Statistics
    total_n = len(samples)
    strict_acc = (strict_matches / total_n) * 100.0
    norm_acc = (norm_matches / total_n) * 100.0
    avg_bleu = sum(bleu_scores) / total_n
    avg_cer = sum(cer_scores) / total_n
    avg_ter = sum(ter_scores) / total_n

    latencies_sorted = sorted(latencies)
    mean_lat = sum(latencies) / total_n
    med_lat = latencies_sorted[total_n // 2]
    p95_lat = latencies_sorted[int(total_n * 0.95)]
    throughput = total_n / max(0.001, total_eval_sec)

    # 5. Print Results Table
    print("\n" + "=" * 70)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"Dataset:                   {dataset_name}")
    print(f"Total Evaluated Samples:   {total_n}")
    print(f"Evaluation Mode:           {args.mode.upper()}")
    print(f"Total Execution Time:      {total_eval_sec:.2f} s")
    print(f"Throughput:                {throughput:.1f} samples/sec")
    print("-" * 70)
    print(f"Strict Exact Match (EM):   {strict_matches}/{total_n} ({strict_acc:.2f}%)")
    print(f"Normalized EM (Norm-EM):   {norm_matches}/{total_n} ({norm_acc:.2f}%)")
    print(f"Token BLEU-4 Score:        {avg_bleu:.2f} / 100")
    print(f"Character Error Rate (CER):{avg_cer:.2f}%")
    print(f"Token Error Rate (TER):    {avg_ter:.2f}%")
    print("-" * 70)
    print(f"Latency (Mean):            {mean_lat:.1f} ms")
    print(f"Latency (Median):          {med_lat:.1f} ms")
    print(f"Latency (P95):             {p95_lat:.1f} ms")
    print("=" * 70 + "\n")

    # 6. Save Reports
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"eval_{Path(dataset_name).stem}_{args.mode}_{timestamp}"

    # JSON Report
    json_path = out_dir / f"{base_name}.json"
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "model": str(Path(args.model_dir).name),
        "mode": args.mode,
        "device": args.device,
        "total_samples": total_n,
        "metrics": {
            "strict_exact_match_pct": round(strict_acc, 2),
            "normalized_exact_match_pct": round(norm_acc, 2),
            "token_bleu_4": round(avg_bleu, 2),
            "character_error_rate_pct": round(avg_cer, 2),
            "token_error_rate_pct": round(avg_ter, 2),
            "latency_ms": {
                "mean": round(mean_lat, 1),
                "median": round(med_lat, 1),
                "p95": round(p95_lat, 1),
                "min": round(min(latencies), 1),
                "max": round(max(latencies), 1),
            },
            "throughput_samples_per_sec": round(throughput, 2),
            "total_elapsed_sec": round(total_eval_sec, 2),
        },
        "samples": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # CSV Report
    csv_path = out_dir / f"{base_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index", "id", "domain", "input", "target", "prediction",
            "strict_match", "norm_match", "bleu_4", "cer_pct", "ter_pct", "latency_ms"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Markdown Summary Report
    md_path = out_dir / f"summary_{Path(dataset_name).stem}_{args.mode}_{timestamp}.md"
    mismatches = [r for r in results if not r["norm_match"]][:10]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark Report: {dataset_name}\n\n")
        f.write(f"- **Model:** `{Path(args.model_dir).name}`\n")
        f.write(f"- **Mode:** `{args.mode.upper()}`\n")
        f.write(f"- **Evaluated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Overall Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| **Total Samples** | {total_n} |\n")
        f.write(f"| **Normalized Exact Match (Norm-EM)** | **{norm_acc:.2f}%** ({norm_matches}/{total_n}) |\n")
        f.write(f"| **Strict Exact Match (Strict-EM)** | **{strict_acc:.2f}%** ({strict_matches}/{total_n}) |\n")
        f.write(f"| **Token BLEU-4** | **{avg_bleu:.2f} / 100** |\n")
        f.write(f"| **Character Error Rate (CER)** | **{avg_cer:.2f}%** |\n")
        f.write(f"| **Token Error Rate (TER)** | **{avg_ter:.2f}%** |\n")
        f.write(f"| **Mean Latency** | **{mean_lat:.1f} ms** |\n")
        f.write(f"| **P95 Latency** | **{p95_lat:.1f} ms** |\n")
        f.write(f"| **Throughput** | **{throughput:.1f} samples/sec** |\n\n")
        
        if mismatches:
            f.write("## Sample Discrepancies (Top 10)\n\n")
            f.write("| Input Spoken Math | Ground Truth Target | Model Prediction |\n")
            f.write("| :--- | :--- | :--- |\n")
            for m in mismatches:
                f.write(f"| `{m['input']}` | `${m['target']}$` | `${m['prediction']}$` |\n")

    print(f"💾 Detailed JSON report: {json_path}")
    print(f"💾 Spreadsheet CSV:      {csv_path}")
    print(f"📄 Markdown summary:     {md_path}\n")


if __name__ == "__main__":
    run_evaluation()
