# Edge-Native Mathematical Speech-to-LaTeX System

An end-to-end, 100% offline edge-native speech-to-LaTeX transcription and typesetting system optimized for the **NVIDIA Jetson Orin Nano (8GB Unified Memory, Ampere SM87)**. Converts continuous engineering and mathematical lectures into publication-ready LaTeX monographs, standalone PDFs with vector graphics, and synchronized subtitles in real time.

---

## 🚀 Key Features

* **Dual-Engine Edge Pipeline:**
  * **Acoustic ASR Engine:** `whisper.cpp` (`small.en`) with **CUDA Flash Attention (`-fa`)**, multi-threaded decoding, and adaptive Voice Activity Detection (VAD) (~540 ms latency).
  * **Spoken-Math Typesetting SLM:** **Qwen2.5-1.5B-Instruct (Q4_K_M GGUF)** offloaded to Ampere GPU SM87 via `llama.cpp` server runtime (~290 ms inference).
* **Headless Voice-Activated Edge Agent (Zero-Touch Monitorless Boot):**
  * **Voice Command Navigation:** Controlled entirely by speech without an external monitor, keyboard, or mouse:
    * Activation: *"Hey Jetson, start recording"* $\to$ spoken Piper neural countdown *"Ok, starting recording in 5... 4... 3... 2... 1... Recording started."*
    * Termination: *"Hey Jetson, stop recording"* $\to$ spoken notice *"Recording stopped. Processing lecture, please wait."*
  * **Automated USB Storage (`AYUSH`):** Operates 100% offline with zero GitHub uploads during recording. Automatically exports `.pdf`, `.tex`, `.md`, `.srt`, `.vtt`, `.wav`, and `.zip` directly to the inserted USB drive labeled **`AYUSH`** (`/media/ayush02/AYUSH/Lectures/<timestamp>/`) and flushes storage buffers via `sync`.
  * **Safe-Shutdown Spoken Signal:** Announces *"Transcription successfully saved. You can turn off the system."* when all deliverables are written to flash.
* **Textbook-Grade LaTeX Monograph & TikZ Vector Graphics:**
  * **Timestamp-Free Academic Prose:** Eliminates all chunk timestamps from the paper, assembling spoken clauses into continuous academic paragraphs with hierarchical sectioning.
  * **Numbered Display Equations:** Automatically promotes core definitions and laws into numbered `\begin{equation} ... \label{eq:...} \end{equation}` blocks.
  * **Automated TikZ / PGFPlots Vector Illustrations:** Detects geometric coordinate discussions (e.g. real/imaginary $z$-plane, Region of Convergence (ROC), unit circle, synchronous machine $d-q$ rotor frames, phasors, Bode plots) and generates native TikZ vector graphics inside figures.
  * **Multi-Pass Local AI Review:** Local Qwen2.5-1.5B SLM harmonizes cross-chunk mathematical notation and checks syntax.
  * **Standalone PDF Compilation:** Automatically compiles the document into a high-fidelity `.pdf`.
* **Continuous Long-Form Streaming:**
  * Tested and validated on full **32-minute** and **1-hour** university lectures.
  * Achieves **`0.27x – 0.34x` Real-Time Factor (RTF)** (**2.9x to 3.7x faster than real time**), providing $>65\%$ compute headroom for zero-latency live streaming.

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

### 1. Run Headless Voice-Activated Agent
```bash
cd /mnt/thesis
PYTHONPATH=/mnt/thesis/main/src /mnt/thesis/main/.venv/bin/python /mnt/thesis/main/src/headless_voice_agent.py
```
*(Say "Hey Jetson start recording" to begin; say "Hey Jetson stop recording" when done).*

### 2. Launch the Web Showcase GUI
```bash
cd /mnt/thesis
./run_gui.sh
```
Open `http://localhost:8000` in your browser.

### 3. Run Benchmark on Any Lecture WAV
```bash
/mnt/thesis/main/.venv/bin/python /mnt/thesis/main/src/benchmark_lecture.py \
  --audio "/mnt/thesis/Lecture 12 Z - Transform.wav" \
  --name "z-transform_lecture_test"
```

For complete setup instructions, flags, and deployment options, see [RUNNING_GUIDE.md](RUNNING_GUIDE.md).

---

## 🛡️ Model Integrity & Backup Recovery

All models are backed up in read-only storage on internal eMMC (`/home/ayush02/models/backup_working_models/`) and USB drive `AYUSH` (`/media/ayush02/AYUSH/Speech2LaTeX_Backup/`).

To verify or restore the exact models:
```bash
/mnt/thesis/scripts/setup_models.sh
```
For cryptographic SHA-256 hashes and storage architecture, refer to [MODELS_MANIFEST.md](MODELS_MANIFEST.md).

---

## 🏗️ Repository Architecture

```
.
├── RUNNING_GUIDE.md                     # Comprehensive operation manual (Headless, GUI, Benchmark)
├── MODELS_MANIFEST.md                   # Exact model registry, SHA-256 hashes & backup topology
├── SYSTEM_ARCHITECTURE_AND_DECISIONS.md # Detailed architectural log & design decisions
├── README.md                            # Project overview & benchmark summary
├── headless_voice_agent_plan.md         # Full specification of offline headless edge agent
├── speech2latex-headless.service        # systemd user service for auto-boot headless operation
├── run_gui.sh                           # Self-bootstrapping Web GUI launcher
├── speech2latex.service                 # systemd service for GUI showcase
├── evaluation_results/                  # Benchmark deliverables (Z-transform, Synchronous Machines)
├── scripts/
│   └── setup_models.sh                  # One-command model verification & recovery tool
└── main/
    ├── assets/voice_prompts/            # Pre-rendered zero-latency TTS prompts
    ├── gui/                             # Web showcase application
    └── src/                             # Core algorithmic pipeline
        ├── headless_voice_agent.py      # Master autonomous edge voice daemon
        ├── edge_tts.py                  # Offline Piper neural TTS engine
        ├── lecture_to_book.py           # Textbook monograph & TikZ vector graph synthesizer
        ├── postprocess_lecture_ai.py    # Global AI review pass & standalone PDF compiler
        ├── usb_exporter.py              # Automated USB AYUSH storage & sync manager
        ├── benchmark_lecture.py         # Long-form academic lecture benchmark engine
        ├── live_streaming_terminal.py   # Console-based real-time streaming runner
        ├── slm_translator.py            # Qwen2.5 SLM edge translator & ChatML wrapper
        ├── router.py                    # Fast heuristic math classifier & fallback router
        ├── audio.py                     # Vectorized audio decimation (pure NumPy)
        └── energy_vad.py                # Real-time microphone capture & Adaptive VAD
```
