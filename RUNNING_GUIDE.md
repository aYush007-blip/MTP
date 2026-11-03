# Jetson Orin Speech-to-LaTeX: Complete Running Guide

This guide provides end-to-end instructions for deploying, running, and benchmarking the **Edge Speech-to-LaTeX** transcription system on the **NVIDIA Jetson Orin Nano (8GB Unified Memory)**.

---

## 📋 System Requirements

* **Hardware:** NVIDIA Jetson Orin Nano Developer Kit (8GB Unified Memory, Ampere GPU SM87).
* **Operating System:** Linux (Ubuntu 20.04 / 22.04 LTS, JetPack 5.1.x / 6.x).
* **CUDA:** Version 12.x with cuBLAS and Flash Attention.
* **Audio Input:** USB Headset / Earphones Microphone (e.g. `AB13X USB Audio` on `hw:2,0` or generic USB PnP audio).
* **Storage:** $\ge 15\text{ GB}$ free disk space.
* **Network:** **100% Offline** (Zero external internet access required during execution).

---

## ⚡ Quick Start: 3 Ways to Run

```
                         ┌─────────────────────────────────────────┐
                         │           Speech2LaTeX System           │
                         └────────────────────┬────────────────────┘
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         ▼                                    ▼                                    ▼
┌──────────────────┐               ┌───────────────────────┐            ┌──────────────────────┐
│  Mode 1: Web GUI │               │ Mode 2: Terminal Live │            │ Mode 3: Benchmark    │
│  ./run_gui.sh    │               │ live_streaming_       │            │ benchmark_lecture.py │
│  (Port 8000)     │               │ terminal.py           │            │ (Full 1-Hr WAVs)     │
└──────────────────┘               └───────────────────────┘            └──────────────────────┘
```

---

### Mode 1: Web Showcase GUI (Recommended)

The Web GUI provides a glassmorphism interface with live continuous lecture streaming, KaTeX auto-rendering, live audio waveform visualizer, and one-click ZIP export.

#### Launch Command:
```bash
cd /mnt/thesis
./run_gui.sh
```

*(Alternatively, via Python virtual environment:)*
```bash
/mnt/thesis/main/.venv/bin/python /mnt/thesis/main/gui.py
```

* **Access the UI:** Open `http://localhost:8000` (or `http://<jetson-ip>:8000` from any device on the local network).
* **What Happens Automatically:**
  1. The launcher checks and auto-starts `whisper-server` on port `8081`.
  2. The launcher checks and auto-starts `llama-server` on port `8080`.
  3. The web server listens on port `8000` and automatically opens your default browser.
  4. Audio devices are scanned and negotiated (defaulting to `AB13X USB Audio`).

#### Using Continuous Lecture Mode:
1. In the Web UI, select your microphone from the top dropdown.
2. Click **"Start Continuous Lecture"**.
3. Speak naturally. The system segments speech using adaptive VAD, applies a 120Hz high-pass filter, transcribes via Whisper Flash Attention, and formats formulas into LaTeX with Qwen2.5-1.5B in real time.
4. When done, click **"Stop & Export"**.
5. Download the complete bundle (`.tex`, `.md`, `.srt`, `.vtt`, `.wav`, and `.zip`).

---

### Mode 2: Terminal-Based Live Streaming

For headless or SSH-only sessions where no web browser is available:

```bash
cd /mnt/thesis
/mnt/thesis/main/.venv/bin/python /mnt/thesis/main/src/live_streaming_terminal.py
```

#### Terminal Features:
* Real-time audio VU meter in console.
* Ambient noise floor calibration on startup.
* Instantaneous terminal printing of spoken words followed by typeset inline LaTeX.
* Press `Ctrl+C` to stop and save the session.

---

### Mode 3: Long-Form Lecture Benchmark Engine

To evaluate the system on pre-recorded audio lectures (e.g. 30-minute or 1-hour `.wav` files):

```bash
cd /mnt/thesis
/mnt/thesis/main/.venv/bin/python /mnt/thesis/main/src/benchmark_lecture.py \
  --audio "/mnt/thesis/Mod-01 Lec-11 Modeling of Synchronous Machines.wav" \
  --name "synchronous_machines_lecture_test" \
  --output_dir "/mnt/thesis/evaluation_results"
```

#### Command-Line Options:
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--audio` | `/mnt/thesis/Lecture 12 Z - Transform.wav` | Absolute path to the source audio `.wav` file |
| `--name` | `z-transform_lecture_test` | Benchmark session name (used for output directory) |
| `--duration_sec` | `0.0` (Full recording) | Slice duration in seconds (`0` processes the entire lecture) |
| `--start_sec` | `0.0` | Start offset in seconds |
| `--output_dir` | `/mnt/thesis/evaluation_results` | Directory where all 8 deliverable files are saved |

---

## 🏛️ Edge Architecture & Port Mapping

| Service / Daemon | Port | Binary / Command | Role & Configuration |
| :--- | :---: | :--- | :--- |
| **Whisper Server** | `8081` | `whisper-server` | CUDA Flash Attention (`-fa`), ggml-small.en, `-t 4`, `-nth 0.40` |
| **LLaMA Server** | `8080` | `llama-server` | Qwen2.5-1.5B Q4_K_M GGUF, CUDA SM87 (`-ngl 28`), context (`-c 4096`) |
| **Web Showcase GUI** | `8000` | `gui.py` / `server.py` | Aiohttp async webserver, WebSockets, REST APIs |

### Self-Healing & Auto-Revival
Both `main/gui/server.py` and `main/src/slm_translator.py` feature **automatic daemon health monitoring**. If a daemon crashes or encounters an out-of-memory event during extended 1-hour streaming, the pipeline automatically intercepts the failure, auto-revives the process, waits for the `/health` endpoint, and retries the request without dropping any speech chunks.

---

## 📁 Export Deliverables Suite

Every continuous session or benchmark generates the following 8 deliverables in `evaluation_results/<session_name>/`:

1. **`*_report.md`**: Executive evaluation report with latency percentiles (P50, Mean, P90, P95), RTF, and formula counts.
2. **`*_transcript.md`**: Clean Markdown lecture notes formatted with LaTeX mathematical formulas.
3. **`*_paper.tex`**: Compilable standalone LaTeX article document with `\usepackage{amsmath,amssymb}` ready for `pdflatex`.
4. **`*_transcript.txt`**: Plain text transcript with timestamped utterance records.
5. **`*_subtitles.srt`**: SubRip subtitles with microsecond timestamps and LaTeX formulas for video synchronization.
6. **`*_subtitles.vtt`**: WebVTT subtitles for HTML5 video players.
7. **`*_metrics.csv`**: Comprehensive per-chunk analytics (ASR ms, SLM ms, total ms, duration, RTF, math flag).
8. **`*_manifest.json`**: Structured machine-readable JSON containing full metadata and chunk history.

---

## 🛡️ Model Verification & Backup Recovery

All models are backed up in read-only storage on the internal eMMC (`/home/ayush02/models/backup_working_models/`).

To verify or restore models at any time, run:
```bash
/mnt/thesis/scripts/setup_models.sh
```

For cryptographic SHA-256 checksums and model architecture details, refer to [MODELS_MANIFEST.md](MODELS_MANIFEST.md).

---

## 🔧 Systemd Service Setup (Optional Auto-Start on Boot)

To run the Web Showcase automatically as a background user service:

```bash
# 1. Copy service file
mkdir -p ~/.config/systemd/user
cp /mnt/thesis/speech2latex.service ~/.config/systemd/user/

# 2. Reload daemon and enable service
systemctl --user daemon-reload
systemctl --user enable speech2latex.service
systemctl --user start speech2latex.service

# 3. Check service status
systemctl --user status speech2latex.service
```
