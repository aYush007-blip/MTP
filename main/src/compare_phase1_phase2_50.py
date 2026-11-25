#!/usr/bin/env python3
"""
Statistically Comparable Phase-1 vs Phase-2 Evaluation on Exact 50-Sample Manifest Subset.

Ensures Phase 1 (MathRouter + T5-Small) and Phase 2 (Qwen2.5-1.5B-Instruct Q4_K_M SLM)
are compared on the exact same 50 samples from /mnt/thesis/datasets/manifest_mathspeech_50.jsonl
with identical input text and consistent metric implementations from evaluate_model.py.

Preserves all original evaluation files in /mnt/thesis/evaluation_results/.
"""

import csv
from datetime import datetime
import json
from pathlib import Path
import sys

EVAL_DIR = Path("/mnt/thesis/evaluation_results")
DATASET_50_PATH = Path("/mnt/thesis/datasets/manifest_mathspeech_50.jsonl")
PHASE1_FULL_JSON = EVAL_DIR / "eval_manifest_pipeline_20261015_145227.json"
PHASE2_50_JSON = EVAL_DIR / "eval_manifest_slm_20261024_113840.json"

MAIN_SRC = Path("/mnt/thesis/main/src")
if str(MAIN_SRC) not in sys.path:
    sys.path.insert(0, str(MAIN_SRC))

from evaluate_model import (
    calculate_bleu,
    calculate_cer,
    calculate_ter,
    normalize_latex,
    tokenize_latex,
)


def compute_metrics_from_predictions(samples: list[dict], source_label: str) -> dict:
    """Recompute all metrics consistently using evaluate_model.py metric functions."""
    total_n = len(samples)
    assert total_n > 0, f"Empty sample list for {source_label}"

    rows = []
    strict_matches = 0
    norm_matches = 0
    bleu_scores = []
    cer_scores = []
    ter_scores = []
    latencies = []

    for idx, s in enumerate(samples, 1):
        clean_target = s["target"].strip().strip("$").strip()
        clean_pred = s["prediction"].strip().strip("$").strip()
        lat_ms = float(s["latency_ms"])

        is_strict = clean_pred == clean_target
        is_norm = normalize_latex(clean_pred) == normalize_latex(clean_target)
        t_tok = tokenize_latex(clean_target)
        p_tok = tokenize_latex(clean_pred)

        bleu = calculate_bleu(t_tok, p_tok)
        cer = calculate_cer(clean_target, clean_pred)
        ter = calculate_ter(t_tok, p_tok)

        strict_matches += int(is_strict)
        norm_matches += int(is_norm)
        bleu_scores.append(bleu)
        cer_scores.append(cer)
        ter_scores.append(ter)
        latencies.append(lat_ms)

        rows.append({
            "index": idx,
            "id": str(s["id"]),
            "domain": s.get("domain", "mathematics"),
            "input": s["input"],
            "target": clean_target,
            "prediction": clean_pred,
            "strict_match": is_strict,
            "norm_match": is_norm,
            "bleu_4": bleu,
            "cer_pct": round(cer, 2),
            "ter_pct": round(ter, 2),
            "latency_ms": round(lat_ms, 1),
        })

    lat_sorted = sorted(latencies)
    mean_lat = sum(latencies) / total_n
    med_lat = lat_sorted[total_n // 2]
    p95_lat = lat_sorted[int(total_n * 0.95)]
    total_sec = sum(latencies) / 1000.0

    summary = {
        "source": source_label,
        "total_samples": total_n,
        "strict_exact_match_count": strict_matches,
        "strict_exact_match_pct": round((strict_matches / total_n) * 100.0, 2),
        "normalized_exact_match_count": norm_matches,
        "normalized_exact_match_pct": round((norm_matches / total_n) * 100.0, 2),
        "token_bleu_4": round(sum(bleu_scores) / total_n, 2),
        "character_error_rate_pct": round(sum(cer_scores) / total_n, 2),
        "token_error_rate_pct": round(sum(ter_scores) / total_n, 2),
        "latency_ms": {
            "mean": round(mean_lat, 1),
            "median": round(med_lat, 1),
            "p95": round(p95_lat, 1),
            "min": round(min(latencies), 1),
            "max": round(max(latencies), 1),
        },
        "throughput_samples_per_sec": round(total_n / max(0.001, total_sec), 2),
    }
    return {"metrics": summary, "samples": rows}


def main():
    with open(DATASET_50_PATH, "r", encoding="utf-8") as f:
        ref_50 = [json.loads(line) for line in f if line.strip()]
    with open(PHASE1_FULL_JSON, "r", encoding="utf-8") as f:
        p1_full = json.load(f)
    with open(PHASE2_50_JSON, "r", encoding="utf-8") as f:
        p2_50 = json.load(f)

    p1_slice = p1_full["samples"][:50]
    p2_slice = p2_50["samples"]

    assert len(ref_50) == 50
    assert len(p1_slice) == 50
    assert len(p2_slice) == 50

    for i in range(50):
        ref_inp = ref_50[i]["input"].strip()
        ref_tgt = ref_50[i]["target"].strip().strip("$").strip()
        assert p1_slice[i]["input"].strip() == ref_inp, f"Phase 1 input mismatch at idx {i}"
        assert p2_slice[i]["input"].strip() == ref_inp, f"Phase 2 input mismatch at idx {i}"
        assert p1_slice[i]["target"].strip().strip("$").strip() == ref_tgt, f"Phase 1 target mismatch at idx {i}"
        assert p2_slice[i]["target"].strip().strip("$").strip() == ref_tgt, f"Phase 2 target mismatch at idx {i}"
        assert str(p1_slice[i]["id"]) == str(ref_50[i]["id"]) == str(p2_slice[i]["id"]), f"ID mismatch at idx {i}"

    p1_eval = compute_metrics_from_predictions(p1_slice, str(PHASE1_FULL_JSON.name))
    p2_eval = compute_metrics_from_predictions(p2_slice, str(PHASE2_50_JSON.name))

    out_json = EVAL_DIR / "eval_manifest_50_comparable.json"
    out_p1_csv = EVAL_DIR / "eval_manifest_50_phase1_pipeline.csv"
    out_p2_csv = EVAL_DIR / "eval_manifest_50_phase2_slm.csv"
    out_md = EVAL_DIR / "summary_manifest_50_comparable.md"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "dataset": str(DATASET_50_PATH),
        "sample_count": 50,
        "identical_inputs_and_targets_verified": True,
        "phase1_source": str(PHASE1_FULL_JSON),
        "phase2_source": str(PHASE2_50_JSON),
        "phase1": p1_eval,
        "phase2": p2_eval,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = [
        "index", "id", "domain", "input", "target", "prediction",
        "strict_match", "norm_match", "bleu_4", "cer_pct", "ter_pct", "latency_ms"
    ]
    for csv_path, rows in ((out_p1_csv, p1_eval["samples"]), (out_p2_csv, p2_eval["samples"])):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    m1 = p1_eval["metrics"]
    m2 = p2_eval["metrics"]
    md_lines = [
        "# Matched 50-Sample Phase-1 vs Phase-2 Benchmark Comparison",
        "",
        f"- **Dataset Subset:** `{DATASET_50_PATH}` (`example_id` 0..49, 50 identical samples)",
        f"- **Phase 1 Source:** `{PHASE1_FULL_JSON.name}` (first 50 samples of original 1,101-sample run; live PyTorch re-run blocked by corrupted `libtorch_python.so`)",
        f"- **Phase 2 Source:** `{PHASE2_50_JSON.name}` (original 50-sample run)",
        f"- **Metric Evaluator:** `main/src/evaluate_model.py` (`normalize_latex`, `tokenize_latex`, `calculate_bleu`, `calculate_cer`, `calculate_ter`)",
        "",
        "| Metric | Phase 1 (T5-Small + Router, n=50) | Phase 2 (Qwen2.5-1.5B SLM, n=50) |",
        "| :--- | :--- | :--- |",
        f"| **Strict Exact Match (Strict-EM)** | {m1['strict_exact_match_pct']:.2f}% ({m1['strict_exact_match_count']}/50) | {m2['strict_exact_match_pct']:.2f}% ({m2['strict_exact_match_count']}/50) |",
        f"| **Normalized Exact Match (Norm-EM)** | {m1['normalized_exact_match_pct']:.2f}% ({m1['normalized_exact_match_count']}/50) | {m2['normalized_exact_match_pct']:.2f}% ({m2['normalized_exact_match_count']}/50) |",
        f"| **Token BLEU-4** | {m1['token_bleu_4']:.2f} / 100 | {m2['token_bleu_4']:.2f} / 100 |",
        f"| **Character Error Rate (CER)** | {m1['character_error_rate_pct']:.2f}% | {m2['character_error_rate_pct']:.2f}% |",
        f"| **Token Error Rate (TER)** | {m1['token_error_rate_pct']:.2f}% | {m2['token_error_rate_pct']:.2f}% |",
        f"| **Mean Latency** | {m1['latency_ms']['mean']:.1f} ms | {m2['latency_ms']['mean']:.1f} ms |",
        f"| **P95 Latency** | {m1['latency_ms']['p95']:.1f} ms | {m2['latency_ms']['p95']:.1f} ms |",
        "",
    ]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print("Phase 1 (n=50):", json.dumps(m1))
    print("Phase 2 (n=50):", json.dumps(m2))
    print("Saved:", out_json, out_p1_csv, out_p2_csv, out_md)


if __name__ == "__main__":
    main()

