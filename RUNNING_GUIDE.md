# Jetson Orin Speech-to-LaTeX: Complete Running Guide

This guide provides end-to-end instructions for deploying, running, and benchmarking the **Edge Speech-to-LaTeX** transcription system on the **NVIDIA Jetson Orin Nano (8GB Unified Memory)**.

---

## 📋 System Requirements

* **Hardware:** NVIDIA Jetson Orin Nano Developer Kit (8GB Unified Memory, Ampere GPU SM87).
* **Operating System:** Linux (Ubuntu 20.04 / 22.04 LTS, JetPack 5.1.x / 6.x).
* **CUDA:** Version 12.x with cuBLAS and Flash Attention.
* **Audio Input:** USB Headset / Earphones Microphone (e.g. `AB13X USB Audio` on `hw:2,0` or generic USB PnP audio).
* **Storage:** $\ge 15\text{ GB}$ free disk space; USB flash drive labeled `AYUSH` formatted FAT32/ext4.
* **Network:** **100% Offline** (Zero internet connection or external APIs required during execution).

---

## ⚡ Modes of Operation

```
                         ┌─────────────────────────────────────────────────────────┐
                         │               Speech2LaTeX Edge System                  │
                         └────────────────────────────┬────────────────────────────┘
                                                      │
         ┌────────────────────────────┬───────────────┴───────────────┬────────────────────────────┐
         ▼                            ▼                               ▼                            ▼
┌───────────────────┐        ┌──────────────────┐            ┌───────────────────────┐    ┌──────────────────────┐
│ Mode 0: Headless  │        │ Mode 1: Web GUI  │            │ Mode 2: Terminal Live │    │ Mode 3: Benchmark    │
│ Voice Agent       │        │ ./run_gui.sh     │            │ live_streaming_       │    │ benchmark_lecture.py │
│ (Auto-boot, Piper │        │ (Port 8000)      │            │ terminal.py           │    │ (Full 1-Hr WAVs)     │
│ TTS, USB AYUSH)   │        │                  │            │                       │    │                      │
└───────────────────┘        └──────────────────┘            └───────────────────────┘    └──────────────────────┘
```

---

### Mode 0: Autonomous Headless Edge Agent (Primary Production Mode)

This mode runs **completely monitorless and zero-touch** for real classroom lectures. When the Jetson powers on, the background daemon auto-configures the audio hardware, starts the edge GPU daemons, gives spoken voice feedback, transcribes the lecture, compiles a textbook PDF with TikZ diagrams, and automatically exports all deliverables to the inserted USB drive labeled **`AYUSH`**.

#### 1. Zero-Touch Voice Interaction Flow:
1. **Cold Boot / Power On:** Jetson boots up silently.
2. **Audio Greeting:** Jetson speaks through connected headphones/speakers:
   > *"Jetson speech agent is ready. Say hey Jetson start recording to begin."*
3. **Voice Activation Trigger:** The lecturer speaks:
   > *"Hey Jetson, start recording"* (or *"Jetson, start lecture"*).
4. **Spoken Countdown:** Jetson speaks:
   > *"Ok, starting recording in 5... 4... 3... 2... 1... Recording started."*
5. **Continuous Lecture Capture:** Captures high-fidelity speech, applies 120Hz mains hum filter, segments via adaptive VAD, and translates math into LaTeX.
6. **Voice Termination Trigger:** When the lecture finishes, the lecturer speaks:
   > *"Hey Jetson, stop recording"* (or *"Jetson, stop lecture"*).
7. **Spoken Acknowledgment:** Jetson speaks:
   > *"Recording stopped. Processing lecture, please wait."*
8. **Typesetting & AI Review:**
   - Assembles speech clauses into textbook-style academic paragraphs (zero timestamps in `.tex`).
   - Promotes formulas into numbered display equations (`\begin{equation} ... \label{eq:...} \end{equation}`).
   - Synthesizes vector **TikZ / PGFPlots diagrams** (complex $z$-plane, ROC, $d-q$ rotor frame, phasors, Bode plots).
   - Runs Qwen2.5-1.5B global review pass to harmonize mathematical notation.
   - Compiles a publication-grade standalone `.pdf`.
9. **Automated USB Export (`AYUSH`):**
   - Copies `.pdf`, `.tex`, `.md`, `.srt`, `.vtt`, `.wav`, `.zip`, and `SHA256SUMS.txt` directly to `/media/ayush02/AYUSH/Lectures/<timestamp>_<title>/`.
   - Flushes dirty filesystem buffers (`sync`).
10. **Spoken Safe-Shutdown Announcement:**
    > *"Transcription successfully saved. You can turn off the system."*

#### 2. Manual Terminal Launch:
```bash
cd /mnt/thesis
PYTHONPATH=/mnt/thesis/main/src /mnt/thesis/main/.venv/bin/python /mnt/thesis/main/src/headless_voice_agent.py
```

#### 3. Verification Simulation on WAV File:
To test the complete headless pipeline on an existing lecture audio recording:
```bash
cd /mnt/thesis
PYTHONPATH=/mnt/thesis/main/src /mnt/thesis/main/.venv/bin/python /mnt/thesis/main/src/headless_voice_agent.py \
  --sim-file "/mnt/thesis/Lecture 12 Z - Transform.wav"
```

#### 4. Auto-Boot Systemd Service Configuration:
The service is pre-installed and enabled under user systemd with user lingering active:
```bash
# Check service status
systemctl --user status speech2latex-headless.service

# Start or restart service manually
systemctl --user restart speech2latex-headless.service

# View live background logs
journalctl --user -u speech2latex-headless.service -f
```

---

### Mode 1: Web Showcase GUI

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
* **Using Continuous Lecture Mode:**
  1. In the Web UI, select your microphone from the top dropdown.
  2. Click **"Start Continuous Lecture"**.
  3. Speak naturally.
  4. When done, click **"Stop & Export"**.

---

### Mode 2: Terminal-Based Live Streaming

For interactive terminal sessions where no web browser is needed:

```bash
cd /mnt/thesis
PYTHONPATH=/mnt/thesis/main/src /mnt/thesis/main/.venv/bin/python /mnt/thesis/main/src/live_streaming_terminal.py
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
PYTHONPATH=/mnt/thesis/main/src /mnt/thesis/main/.venv/bin/python /mnt/thesis/main/src/benchmark_lecture.py \
  --audio "/mnt/thesis/Mod-01 Lec-11 Modeling of Synchronous Machines.wav" \
  --name "synchronous_machines_lecture_test" \
  --output_dir "/mnt/thesis/evaluation_results"
```

---

## 🏛️ Edge Architecture & Port Mapping

| Service / Daemon | Port | Binary / Command | Role & Configuration |
| :--- | :---: | :--- | :--- |
| **Whisper Server** | `8081` | `whisper-server` | CUDA Flash Attention (`-fa`), `ggml-small.en.bin`, `-t 4`, `-nth 0.40` |
| **LLaMA Server** | `8080` | `llama-server` | Qwen2.5-1.5B Q4_K_M GGUF, CUDA SM87 (`-ngl 28`), context (`-c 2048`) |
| **Piper Neural TTS** | — | `piper-tts` (ONNX) | Instant offline speech synthesis (~50ms latency), `en_US-lessac-medium` |
| **Web Showcase GUI** | `8000` | `gui.py` / `server.py` | Aiohttp async webserver, WebSockets, REST APIs |

### Self-Healing & Auto-Revival
Both `headless_voice_agent.py` and `slm_translator.py` feature **automatic daemon health monitoring**. If a daemon crashes or encounters a memory reset during extended 1-hour streaming, the pipeline automatically intercepts the failure, auto-revives the process, waits for the `/health` endpoint, and retries the request without dropping any speech chunks.

---

## 📁 Export Deliverables Suite & USB Sync

All lectures are saved locally in `/mnt/thesis/live_tests/<timestamp>_<title>/` and automatically mirrored to `/media/ayush02/AYUSH/Lectures/<timestamp>_<title>/`:

1. **`<title>.pdf`**: Compiled publication-grade textbook monograph with abstract, numbered sections, equations, and vector graphics.
2. **`<title>.tex`**: Full LaTeX source with native TikZ/PGFPlots code and numbered `\begin{equation}` environments.
3. **`<title>.md`**: Academic Markdown notes formatted with math blocks.
4. **`<title>.srt`**: SubRip subtitle file with microsecond timestamps and math formulas.
5. **`<title>.vtt`**: WebVTT subtitles for multimedia synchronization.
6. **`<title>_raw.wav`**: High-fidelity captured audio recording of the lecture.
7. **`<title>_bundle.zip`**: Complete self-contained distribution archive.
8. **`SHA256SUMS.txt`**: Cryptographic checksum manifest.

### Offline USB Sync Command:
If the USB drive `AYUSH` was inserted after a lecture completed, sync all past lectures with a single command:
```bash
PYTHONPATH=/mnt/thesis/main/src /mnt/thesis/main/.venv/bin/python /mnt/thesis/main/src/usb_exporter.py --sync-all
```

---

## 🛡️ Model Verification & Backup Recovery

All models are backed up in read-only storage on internal eMMC (`/home/ayush02/models/backup_working_models/`) and USB drive `AYUSH` (`/media/ayush02/AYUSH/Speech2LaTeX_Backup/`).

To verify or restore models at any time, run:
```bash
/mnt/thesis/scripts/setup_models.sh
```
