# Edge-Native Mathematical Speech-to-LaTeX System

An end-to-end, 100% offline edge-native speech-to-LaTeX transcription and typesetting architecture optimized for the **NVIDIA Jetson Orin Nano (8GB Unified Memory, Ampere SM87)**. 

The repository is organized into two distinct, self-contained versions:
1. **`headless_agent/` (New Headless Version):** Fully autonomous, monitorless edge recording agent with offline Piper neural TTS audio feedback, voice-command state machine (*"Hey Jetson, start recording"* $\to$ 5s countdown $\to$ *"Hey Jetson, stop recording"*), textbook monograph paragraph flow, numbered display equations (`\begin{equation}`), automated native TikZ vector graphics ($z$-plane, ROC, $d-q$ axes, phasors), and direct automated export to USB flash drive **`AYUSH`**.
2. **`main/` (Previous Version):** Interactive Web Showcase GUI (glassmorphism UI with live KaTeX rendering and WebSocket streaming), console-based streaming runner (`live_streaming_terminal.py`), and long-form university lecture benchmarking suite (`benchmark_lecture.py`).

---

## 🚀 Key Features by Version

### 1. Headless Version (`headless_agent/`)
* **Zero-Touch Monitorless Boot:** Runs as an auto-booting background daemon (`speech2latex-headless.service`) with user lingering enabled. No external monitor, keyboard, or mouse required.
* **Offline Spoken Voice Interaction:**
  * Powered by `piper-tts` with high-clarity voice (`en_US-lessac-medium.onnx`), running locally in ~50ms on Jetson CPU.
  * Controlled entirely by voice:
    * Activation: *"Hey Jetson, start recording"* $\to$ spoken countdown *"Ok, starting recording in 5... 4... 3... 2... 1... Recording started."*
    * Termination: *"Hey Jetson, stop recording"* $\to$ spoken notice *"Recording stopped. Processing lecture, please wait."*
    * Safe Shutdown: *"Transcription successfully saved. You can turn off the system."*
* **Textbook-Grade LaTeX Monograph & TikZ Graphics:**
  * Spoken timestamps are removed from the LaTeX document (retained solely in `.srt` and `.vtt` subtitle files).
  * Clauses are fused into coherent academic paragraphs with logical section headings.
  * Formulas are promoted to numbered display equations (`\begin{equation} ... \label{eq:...} \end{equation}`).
  * Synthesizes native TikZ / PGFPlots vector graphics for complex $z$-plane, ROC circle, $d-q$ rotor axes, phasors, and Bode plots.
  * Local Qwen2.5-1.5B SLM review pass harmonizes mathematical notation across the lecture.
  * Compiles standalone `.pdf` document.
* **Automated USB Storage (`AYUSH`):**
  * Operates 100% offline with zero GitHub uploads during recording.
  * Automatically exports `.pdf`, `.tex`, `.md`, `.srt`, `.vtt`, `.wav`, and `.zip` directly to `/media/ayush02/AYUSH/Lectures/<timestamp>/` and flushes storage buffers via `sync`.

### 2. Interactive & Benchmark Version (`main/`)
* **Web Showcase GUI:** Glassmorphism dashboard on port `8000` with live waveform visualizer, KaTeX auto-rendering, and one-click ZIP download.
* **Terminal Live Stream:** Real-time console output with VU meter and Schmitt trigger VAD.
* **Lecture Benchmark Engine:** Evaluates full 30-minute and 1-hour recordings, achieving `0.27x – 0.34x` RTF (2.9x to 3.7x faster than real time).

---

## 📁 Repository Structure

```
.
├── headless_agent/                      # [NEW VERSION] 100% Offline Headless Voice Agent
│   ├── src/
│   │   ├── headless_voice_agent.py      # Master autonomous voice daemon & state machine
│   │   ├── edge_tts.py                  # Offline Piper neural TTS engine & audio player
│   │   ├── lecture_to_book.py           # Textbook monograph & TikZ vector graphics
│   │   ├── postprocess_lecture_ai.py    # Post-lecture AI review pass & standalone PDF compiler
│   │   ├── usb_exporter.py              # USB drive AYUSH auto-export & sync manager
│   │   ├── audio.py                     # 48k stereo -> 16k mono decimation
│   │   ├── energy_vad.py                # Adaptive energy VAD
│   │   └── slm_translator.py            # Qwen2.5 SLM mathematical formatting
│   ├── assets/voice_prompts/            # Pre-rendered zero-latency WAV prompt audio
│   ├── config/
│   │   └── speech2latex-headless.service # systemd auto-boot service
│   ├── run_headless.sh                  # Local executable launcher
│   └── README.md                        # Headless version manual
│
├── main/                                # [PREVIOUS VERSION] Interactive GUI & Benchmarks
│   ├── gui/                             # Web showcase application (port 8000)
│   │   ├── server.py                    # Aiohttp server daemon
│   │   ├── index.html                   # Glassmorphism interface
│   │   └── static/                      # CSS, app logic, local KaTeX bundle
│   └── src/                             # Algorithmic streaming pipeline
│       ├── benchmark_lecture.py         # Long-form academic lecture benchmark engine
│       ├── live_streaming_terminal.py   # Console-based real-time streaming runner
│       ├── slm_translator.py            # Qwen2.5 SLM edge translator & ChatML wrapper
│       ├── router.py                    # Fast heuristic math classifier
│       ├── audio.py                     # Vectorized audio decimation (pure NumPy)
│       └── energy_vad.py                # Real-time microphone capture & Adaptive VAD
│
├── run_headless.sh                      # Root shortcut launcher for Headless Version
├── run_gui.sh                           # Root shortcut launcher for Web GUI Version
├── speech2latex-headless.service        # systemd user service for Headless Version
├── speech2latex.service                 # systemd user service for GUI Version
├── RUNNING_GUIDE.md                     # Comprehensive operation manual for both versions
├── MODELS_MANIFEST.md                   # Model registry, SHA-256 hashes & backup topology
├── scripts/
│   └── setup_models.sh                  # Model verification & recovery tool
└── evaluation_results/                  # Benchmark deliverables (Z-transform, Synchronous Machines)
```

---

## ⚡ Quick Start

### Running the New Headless Version:
```bash
# Interactive microphone voice-controlled mode:
./run_headless.sh

# Testing simulation on a pre-recorded WAV file:
./run_headless.sh --sim-file "/mnt/thesis/Lecture 12 Z - Transform.wav"
```
*(Say "Hey Jetson start recording" to begin; say "Hey Jetson stop recording" when done).*

### Running the Previous GUI Version:
```bash
./run_gui.sh
```
*(Open `http://localhost:8000` in your browser).*

For complete configuration and deployment instructions, refer to [RUNNING_GUIDE.md](RUNNING_GUIDE.md).
