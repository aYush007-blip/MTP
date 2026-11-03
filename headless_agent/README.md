# Speech2LaTeX Headless Voice-Activated Edge Agent

An autonomous, monitorless academic recording agent designed for the **NVIDIA Jetson Orin Nano (8GB Unified Memory)**. Operates 100% offline with zero internet dependency, auto-configuring hardware on power-up, responding to voice commands with Piper neural speech synthesis, typeset in textbook-quality LaTeX with TikZ vector graphs and numbered equations, and automatically exporting all deliverables to an inserted USB drive labeled **`AYUSH`**.

---

## 🚀 Key Features

* **Zero-Touch Monitorless Boot:**
  * Runs as a persistent background daemon (`speech2latex-headless.service`) on boot.
  * Auto-binds to `AB13X USB Audio` (`hw:2,0` or active ALSA soundcard) and calibrates hardware gain.
  * No monitor, keyboard, or mouse required.
* **Offline Spoken Voice Interaction:**
  * Powered by `piper-tts` with high-clarity natural voice model (`en_US-lessac-medium.onnx`).
  * **Wake Command:** *"Hey Jetson, start recording"* $\to$ spoken countdown: *"Ok, starting recording in 5... 4... 3... 2... 1... Recording started."*
  * **Stop Command:** *"Hey Jetson, stop recording"* $\to$ spoken notice: *"Recording stopped. Processing lecture, please wait."*
  * **Shutdown Notice:** *"Transcription successfully saved. You can turn off the system."*
* **Textbook Monograph Formatting:**
  * Spoken timestamps are removed from the LaTeX document (preserved in `.srt` and `.vtt` subtitles).
  * Clauses are fused into coherent academic paragraphs with logical section headings.
  * Key formulas are promoted to numbered display environments:
    $$\begin{equation}
      \label{eq:1}
      X(z) = \sum_{n=-\infty}^{\infty} x[n] z^{-n}
    \end{equation}$$
* **Automated TikZ / PGFPlots Vector Graphics:**
  * Synthesizes native TikZ illustrations when coordinate systems are spoken:
    * Complex $z$-plane, unit circle $|z|=1$, and Region of Convergence (ROC) $|z| > r$.
    * Direct & quadrature ($d-q$) rotor axes rotated by $\theta_r$.
    * Voltage/current phasor diagrams with angle $\phi$.
    * Logarithmic Bode frequency response plots.
* **Local AI Review Pass:**
  * Overlooks the lecture transcript via local Qwen2.5-1.5B (port 8080) to harmonize mathematical notation across chunks.
  * Automatically compiles standalone `.pdf` document.
* **Automated USB Storage (`AYUSH`):**
  * Automatically exports to `/media/ayush02/AYUSH/Lectures/<timestamp>_<title>/` (`.pdf`, `.tex`, `.md`, `.srt`, `.vtt`, `.wav`, `.zip`, `SHA256SUMS.txt`).
  * Flushes storage cache via `sync` to prevent FAT32 data loss upon unplugging or system power-off.

---

## 📁 Directory Structure

```
headless_agent/
├── src/
│   ├── headless_voice_agent.py   # Master autonomous voice daemon & state machine
│   ├── edge_tts.py               # Offline Piper neural TTS engine & audio player
│   ├── lecture_to_book.py        # Textbook paragraph assembler & TikZ vector graphics
│   ├── postprocess_lecture_ai.py # Post-lecture AI review pass & standalone PDF compiler
│   ├── usb_exporter.py           # USB drive AYUSH auto-export & sync manager
│   ├── audio.py                  # 48 kHz stereo to 16 kHz mono decimation
│   ├── energy_vad.py             # Adaptive dual-threshold Schmitt trigger VAD
│   ├── slm_translator.py         # Qwen2.5 SLM mathematical formatting client
│   └── router.py                 # Fast heuristic math classifier
├── assets/
│   └── voice_prompts/            # Pre-rendered zero-latency WAV prompt audio
│       ├── countdown_start.wav
│       ├── recording_stopped.wav
│       ├── transcription_saved.wav
│       ├── system_ready.wav
│       └── usb_warning.wav
├── config/
│   └── speech2latex-headless.service  # systemd service template
├── run_headless.sh               # Local executable launcher
└── README.md                     # This manual
```

---

## ⚡ Quick Start

### 1. Interactive Live Microphone Mode:
```bash
/mnt/thesis/headless_agent/run_headless.sh
```
*(Or from thesis root: `/mnt/thesis/run_headless.sh`)*

### 2. Test Simulation on a Pre-Recorded WAV File:
```bash
/mnt/thesis/headless_agent/run_headless.sh --sim-file "/mnt/thesis/Lecture 12 Z - Transform.wav"
```

### 3. Sync Past Un-synced Lectures to USB Drive:
If the USB drive `AYUSH` was inserted after a lecture finished:
```bash
PYTHONPATH=/mnt/thesis/headless_agent/src /mnt/thesis/headless_agent/.venv/bin/python \
  /mnt/thesis/headless_agent/src/usb_exporter.py --sync-all
```

### 4. Background Service Management:
```bash
# Check service status
systemctl --user status speech2latex-headless.service

# Restart service
systemctl --user restart speech2latex-headless.service

# View live systemd logs
journalctl --user -u speech2latex-headless.service -f
```

