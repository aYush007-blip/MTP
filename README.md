# Edge-Native Mathematical Speech-to-LaTeX System

An end-to-end, 100% offline edge-native speech-to-LaTeX transcription and typesetting system optimized for the **NVIDIA Jetson Orin Nano (8GB Unified Memory, Ampere SM87)**. Converts continuous engineering and mathematical lectures into publication-ready LaTeX documents and synchronized subtitles in real time.

---

## 🚀 Key Features

* **Dual-Engine Edge Pipeline:**
  * **Acoustic ASR Engine:** `whisper.cpp` (`small.en`) with **CUDA Flash Attention (`-fa`)**, multi-threaded decoding, and adaptive Voice Activity Detection (VAD) (~540 ms latency).
  * **Spoken-Math Typesetting SLM:** **Qwen2.5-1.5B-Instruct (Q4_K_M GGUF)** offloaded to Ampere GPU SM87 via `llama.cpp` server runtime (~290 ms inference).
* **Continuous Long-Form Streaming:**
  * Tested and validated on full **32-minute** and **1-hour** university lectures.
  * Achieves **`0.27x – 0.34x` Real-Time Factor (RTF)** (**2.9x to 3.7x faster than real time**), providing $>65\%$ compute headroom for zero-latency live streaming.
* **ASR Fidelity & Zero-Loss Acoustic Pipeline:**
  * **Acoustic Padding ($\pm 250\text{ms}$):** Pre-roll and post-roll safety margins eliminate boundary word clipping.
  * **Dynamic Prompt Chaining:** Whisper's autoregressive decoder is continuously conditioned on preceding transcript history, ensuring seamless sentence flow across chunks.
  * **Minimum Chunk Safeguard ($\ge 2.5\text{s}$):** Merges ultra-short fragments to prevent Whisper attention collapse.
  * **Lowered No-Speech Threshold (`0.40`) & Temperature Fallback (`0.0 \to 0.2`):** Captures soft, accented, and trailing academic speech.
* **Live Showcase Web GUI & Headless Capabilities:**
  * Modern glassmorphism UI with live KaTeX rendering over WebSockets.
  * Full export suite: Markdown notes, standalone `.tex` document, SRT/VTT subtitles, per-chunk metrics CSV, and instant ZIP download.

---

## 📊 Long-Form Lecture Benchmark Results

| Lecture Recording | Duration | Wall-Clock Time | Real-Time Factor (RTF) | Utterance Chunks | Formulas Detected | LaTeX Syntax Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lecture 12 Z - Transform.wav** | 32.37 min (1,942 s) | 10.99 min (659 s) | **`0.3395x`** (2.95x real-time) | 560 chunks | 243 formulas | **99.6%** |
| **Modeling of Synchronous Machines.wav** | 59.57 min (3,574 s) | 16.04 min (962 s) | **`0.2693x`** (3.71x real-time) | 977 chunks | 80 formulas | **100.0%** |

### Latency Percentiles (1-Hour Continuous Lecture):
* **ASR Latency (Flash Attention):** Mean = 685.9 ms | P50 = 540.6 ms | P90 = 898.9 ms
* **SLM Latency (CUDA SM87):** Mean = 298.2 ms | P50 = 0.1 ms (selective pass-through) | P90 = 1,055.9 ms
* **End-to-End Pipeline Latency:** Mean = 984.9 ms | P50 = 577.3 ms | P90 = 1,859.1 ms

---

## ⚡ Quickstart

### 1. Launch the Web Showcase GUI
```bash
cd /mnt/thesis
./run_gui.sh
```
Open `http://localhost:8000` in your browser.

### 2. Run Terminal-Based Live Streaming
```bash
/mnt/thesis/main/.venv/bin/python /mnt/thesis/main/src/live_streaming_terminal.py
```

### 3. Run Benchmark on Any Lecture WAV
```bash
/mnt/thesis/main/.venv/bin/python /mnt/thesis/main/src/benchmark_lecture.py \
  --audio "/mnt/thesis/Lecture 12 Z - Transform.wav" \
  --name "z-transform_lecture_test"
```

For complete setup instructions, flags, and deployment options, see [RUNNING_GUIDE.md](RUNNING_GUIDE.md).

---

## 🛡️ Model Integrity & Backup Recovery

All models are backed up in read-only storage on the internal eMMC (`/home/ayush02/models/backup_working_models/`).

To verify or restore the exact models:
```bash
/mnt/thesis/scripts/setup_models.sh
```
For cryptographic SHA-256 hashes and storage architecture, refer to [MODELS_MANIFEST.md](MODELS_MANIFEST.md).

---

## 🏗️ Repository Architecture

```
.
├── RUNNING_GUIDE.md                     # Comprehensive operation & benchmarking manual
├── MODELS_MANIFEST.md                   # Exact model registry, SHA-256 hashes & backup topology
├── SYSTEM_ARCHITECTURE_AND_DECISIONS.md # Detailed architectural log & design decisions
├── README.md                            # Project overview & benchmark summary
├── run_gui.sh                           # Self-bootstrapping Web GUI launcher
├── speech2latex.service                 # systemd user service for background deployment
├── Speech2LaTeX.desktop                 # Desktop application shortcut
├── evaluation_results/                  # Benchmark deliverables (Z-transform, Synchronous Machines)
├── scripts/
│   └── setup_models.sh                  # One-command model verification & recovery tool
└── main/
    ├── gui/                             # Web showcase application
    │   ├── server.py                    # Aiohttp server daemon (port 8000)
    │   ├── index.html                   # Responsive glassmorphism interface
    │   └── static/                      # CSS, app logic, and local KaTeX bundle
    └── src/                             # Core algorithmic pipeline
        ├── benchmark_lecture.py         # Long-form academic lecture benchmark engine
        ├── live_streaming_terminal.py   # Console-based real-time streaming runner
        ├── slm_translator.py            # Qwen2.5 SLM edge translator & ChatML wrapper
        ├── router.py                    # Fast heuristic math classifier & fallback router
        ├── audio.py                     # Vectorized audio decimation (pure NumPy)
        └── live_vad.py                  # Real-time microphone capture & Neural VAD
```
