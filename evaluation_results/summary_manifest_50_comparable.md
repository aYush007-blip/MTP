# Matched 50-Sample Phase-1 vs Phase-2 Benchmark Comparison

- **Dataset Subset:** `/mnt/thesis/datasets/manifest_mathspeech_50.jsonl` (`example_id` 0..49, 50 identical samples)
- **Phase 1 Source:** `eval_manifest_pipeline_20261015_145227.json` (first 50 samples of original 1,101-sample run; live PyTorch re-run blocked by corrupted `libtorch_python.so`)
- **Phase 2 Source:** `eval_manifest_slm_20261024_113840.json` (original 50-sample run)
- **Metric Evaluator:** `main/src/evaluate_model.py` (`normalize_latex`, `tokenize_latex`, `calculate_bleu`, `calculate_cer`, `calculate_ter`)

| Metric | Phase 1 (T5-Small + Router, n=50) | Phase 2 (Qwen2.5-1.5B SLM, n=50) |
| :--- | :--- | :--- |
| **Strict Exact Match (Strict-EM)** | 8.00% (4/50) | 28.00% (14/50) |
| **Normalized Exact Match (Norm-EM)** | 8.00% (4/50) | 48.00% (24/50) |
| **Token BLEU-4** | 22.56 / 100 | 60.29 / 100 |
| **Character Error Rate (CER)** | 66.90% | 29.62% |
| **Token Error Rate (TER)** | 64.42% | 27.08% |
| **Mean Latency** | 721.0 ms | 566.2 ms |
| **P95 Latency** | 1712.2 ms | 1023.9 ms |
