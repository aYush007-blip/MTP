# Edge-Native Mathematical Speech-to-LaTeX System

An end-to-end, 100% offline edge-native speech-to-LaTeX transcription system optimized for **NVIDIA Jetson Orin Nano (Ampere SM87)**. Converts spoken mathematical and technical lectures into formatted LaTeX notation in real time (<500 ms latency).

## 🚀 Key Features

- **Dual-Engine Architecture:**
  - **Phase 2 SLM (Qwen2.5-1.5B-Instruct Q4_K_M):** High-precision conversational prose and complex mathematical formula extraction via edge C++ `llama.cpp` CUDA runtime (~260 ms inference).
  - **Phase 1 Pipeline (T5-Small v2 + Regex Router):** Ultra-low-latency formula translation (~11 ms inference).
- **Edge Acoustic Pipeline:**
  - Pure NumPy vectorized 3:1 integer decimation (48 kHz stereo hardware mic $\to$ 16 kHz mono PCM) with zero PyTorch/SciPy ingestion overhead.
  - 500 ms rolling circular audio buffer with Silero Neural VAD for natural utterance segmentation.
  - C++ `whisper.cpp` (`base.en`) for real-time offline ASR.
- **Live Showcase Web GUI:**
  - Built with FastAPI / Aiohttp backend and a modern responsive dark-mode frontend.
  - Live KaTeX auto-rendering over WebSockets.
  - Interactive multi-engine pill toggle (`SLM` vs `T5-Small`).

---

## 📊 Empirical Benchmark Results (MathSpeech Corpus)

Evaluated on 107 real-world university mathematics and engineering lecture utterances:

| Metric | Phase 1 Baseline (T5-Small + Heuristic Router) | Phase 2 SLM (Qwen2.5-1.5B Q4_K_M) | Improvement |
| :--- | :--- | :--- | :--- |
| **Normalized Exact Match (Norm-EM)** | **8.41%** (9/107) | **40.19%** (43/107) | **+377.9% (4.78x higher!)** |
| **Strict Exact Match (Strict-EM)** | **0.93%** (1/107) | **4.67%** (5/107) | **+402.2% (5.02x higher!)** |
| **Token BLEU-4 Score** | **27.06 / 100** | **47.79 / 100** | **+76.6% relative gain** |
| **Token Error Rate (TER)** | **61.15%** | **30.75%** | **-49.7% error reduction** |
| **Character Error Rate (CER)** | **72.74%** | **47.01%** | **-35.4% error reduction** |
| **Mean Inference Latency** | **787.4 ms** | **519.8 ms** | **34.0% faster on Orin GPU** |
| **P95 Tail Latency** | **2125.1 ms** | **969.2 ms** | **2.19x faster tail latency** |

---

## 🏗️ Repository Architecture

```
.
├── SYSTEM_ARCHITECTURE_AND_DECISIONS.md # Detailed architectural decisions & sequence flows
├── CURRENT_MODEL_DOCUMENTATION.md       # Phase 1 model documentation & failure mode analysis
├── README.md                            # Project overview & quickstart
├── evaluation_results/                  # Quantitative benchmark reports (JSON, CSV, MD)
└── main/
    ├── gui/                             # Web showcase application
    │   ├── server.py                    # Aiohttp server daemon (port 8000)
    │   ├── index.html                   # Responsive glassmorphism interface
    │   └── static/                      # CSS, app logic, and local KaTeX bundle
    └── src/                             # Core algorithmic pipeline
        ├── audio.py                     # Vectorized audio decimation (pure NumPy)
        ├── slm_translator.py            # Qwen2.5 SLM edge translator & ChatML wrapper
        ├── router.py                    # Regex heuristic router & T5-small translator
        ├── live_vad.py                  # Real-time microphone capture & Neural VAD
        ├── evaluate_model.py            # Quantitative evaluation & benchmarking suite
        ├── run_live_test.py             # Live microphone acoustic test runner
        └── download_slm.py              # Verified Hugging Face Hub downloader
```

---

## ⚡ Quickstart

### 1. Launch the SLM Inference Daemon
```bash
/mnt/thesis/llama.cpp/build/bin/llama-server \
  -m /mnt/thesis/models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -c 2048 -ngl 99 --port 8080 --host 127.0.0.1
```

### 2. Launch the Web Showcase GUI
```bash
cd main
source .venv/bin/activate
python gui/server.py
```
Open `http://localhost:8000` in your browser.

### 3. Run Benchmark Suite
```bash
python main/src/evaluate_model.py --mode slm --dataset mathspeech
```

---

## 📜 Complete Architecture & Decision Log

For the full log of all 19 system design decisions, edge failure remediations, and Mermaid runtime sequence diagrams, refer to [SYSTEM_ARCHITECTURE_AND_DECISIONS.md](SYSTEM_ARCHITECTURE_AND_DECISIONS.md).
