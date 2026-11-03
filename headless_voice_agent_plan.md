# Headless Voice-Activated Edge Speech-to-LaTeX: 100% Offline Architecture & Implementation Plan

This document outlines the complete architectural design and step-by-step implementation for transforming the NVIDIA Jetson Orin Nano Speech-to-LaTeX system into a **fully autonomous, monitorless, 100% offline edge academic recording agent**.

The system operates with zero internet dependency: upon boot, it auto-configures the audio hardware, listens for voice activation, gives spoken audio feedback, transcribes and structures the lecture into a textbook-quality academic document with TikZ vector graphs and numbered equations, compiles a PDF, and automatically exports all final deliverables to the inserted USB drive labeled **`AYUSH`** (`/media/ayush02/AYUSH`).

---

## 1. End-to-End System Architecture

```mermaid
graph TD
    A[Cold Boot / Power On Jetson] --> B[systemd: speech2latex-headless.service]
    B --> C[Auto-Detect USB Mic/Earphones: ALSA hw:2,0]
    B --> D[Auto-Spawn GPU Daemons: Whisper & Qwen2.5-1.5B]
    C & D --> E[Offline Wake-Word Listener: 'Hey Jetson...']
    
    E -- 'Hey Jetson start recording' --> F[Offline Piper TTS: 'Ok starting recording in 5 4 3 2 1 seconds, recording started']
    F --> G[Continuous Lecture Capture & Low-Latency Stream Processing]
    
    G -- 'Hey Jetson stop recording' --> H[Audio Halts & Piper TTS: 'Recording stopped. Processing lecture, please wait.']
    H --> I[Post-Lecture Global AI Review & Formatting Engine]
    
    subgraph Offline Post-Processing & Typesetting Engine
        I --> I1[Textbook Flow: Remove timestamps, synthesize cohesive paragraphs]
        I --> I2[Equation Environment: Numbered \\begin{equation} with labels]
        I --> I3[Automated TikZ/PGFPlots Vector Graph Synthesis]
        I --> I4[Global Notation Harmonization & Syntax Self-Correction Pass]
        I --> I5[Standalone PDF Compilation via TeX Engine]
    end
    
    I5 --> J[Automated USB Export: /media/ayush02/AYUSH/Lectures/<timestamp>/]
    J --> K[Piper Neural TTS Spoken Announcement: 'Transcription successfully saved. You can turn off the system.']
```

---

## 2. Key Features & Design Specifications

### 2.1 Feature 1: Textbook-Grade Formatting & Automated TikZ Vector Graphics
* **Zero Timestamps in Document:** Spoken timestamps are removed from the LaTeX document (`.tex`) and PDF (retained solely in `.srt` and `.vtt` sidecars for multimedia alignment).
* **Academic Monograph Paragraph Flow:** Consecutive speech bursts are fused into cohesive, grammatically sound prose with logical section headings (`\section{...}`, `\subsection{...}`).
* **Numbered Display Equations (`\begin{equation}`):**
  * Spoken mathematical relations (e.g. Park transform, $Z$-transform definition, Fourier integral, flux linkages) are automatically promoted to centered, numbered display environments:
    $$\begin{equation}
      \label{eq:z_transform}
      X(z) = \sum_{n=-\infty}^{\infty} x[n] z^{-n}
    \end{equation}$$
  * Subsequent spoken references are cross-referenced via `\eqref{eq:...}`.
* **Automated TikZ / PGFPlots Vector Graphics:**
  * Detects when the lecturer discusses geometric planes, axes, or graphs (e.g., *"real and imaginary axes"*, *"unit circle"*, *"region of convergence"*, *"phasor diagram"*, *"direct and quadrature axes"*, *"Bode plot"*).
  * Automatically generates self-contained native TikZ vector graphics embedded directly inside `figure` environments.

### 2.2 Feature 2: 100% Headless Standalone Edge Agent with Spoken Voice Control
* **Monitorless Boot:** Runs automatically in the background via systemd upon Jetson power-on without requiring an external monitor, keyboard, or mouse.
* **Auto-Configuring Mic & Sound Card:** Auto-binds to `AB13X USB Audio` (`hw:2,0` or active ALSA audio device), calibrates gain, and prepares ALSA playback sinks.
* **Offline Spoken Voice Activation & Stopping:**
  * **Start Trigger:** Spoken phrase `"Hey Jetson, start recording"` or `"Jetson, start lecture"`.
  * **Spoken Countdown:** Jetson speaks through connected headphones/speakers:
    > *"Ok, starting recording in 5... 4... 3... 2... 1... Recording started."*
  * **Stop Trigger:** Spoken phrase `"Hey Jetson, stop recording"` or `"Jetson, stop lecture"`.
  * **Spoken Acknowledgment:** Jetson speaks:
    > *"Recording stopped. Processing lecture, please wait."*
* **Offline Neural TTS (Piper):** Fast, fully offline, ONNX-accelerated speech synthesis (~50ms latency on Jetson CPU) with natural voice quality (`en_US-lessac-medium`).

### 2.3 Feature 3: Post-Lecture Global AI Review & Error Minimization
* **Post-Lecture Multi-Pass Review:**
  * **Pass 1 (Mathematical Unification):** Reconciles notation across the entire hour-long recording, fixing acoustic misnomers and enforcing symbol consistency.
  * **Pass 2 (Bookish Structuring):** Cleans conversational fillers, formats sections, abstracts, and numbered equations.
  * **Pass 3 (TikZ Graphic Injection):** Synthesizes diagrams for geometric concepts.
  * **Pass 4 (Syntax Guard & PDF Build):** Validates LaTeX syntax, compiles standalone `.pdf`, and automatically fixes compile anomalies.

### 2.4 Feature 4: 100% Offline Local & USB Drive Storage (`AYUSH`)
* **Strict Offline Operation:** Zero dependence on GitHub or internet services during live recording and post-processing.
* **Automated USB Export:**
  * Detects the mounted USB drive at `/media/ayush02/AYUSH`.
  * Creates an organized timestamped folder: `/media/ayush02/AYUSH/Lectures/<timestamp>_<lecture_title>/`.
  * Copies:
    1. `<lecture_title>.pdf` (Compiled textbook document)
    2. `<lecture_title>.tex` (Full LaTeX source with TikZ figures)
    3. `<lecture_title>.md` (Clean Markdown book)
    4. `<lecture_title>.srt` & `<lecture_title>.vtt` (Synchronized subtitle files)
    5. `<lecture_title>_raw.wav` (Original captured audio)
    6. `<lecture_title>_bundle.zip` (Complete archive for portable distribution)
  * Runs system `sync` to ensure safe USB write completion.
* **Spoken Safe-Shutdown Notice:** Once the write is verified, the Jetson speaks:
  > *"Transcription successfully saved. You can turn off the system."*

---

## 3. Implementation Plan & Execution Milestones

| Milestone | Component | Target Files | Key Objectives |
| :---: | :--- | :--- | :--- |
| **Step 1** | **Offline Neural TTS Engine** | `main/src/edge_tts.py` | Standalone Piper TTS with ALSA playback (`hw:2,0` / default sink); pre-render or synthesize prompt audio with zero network dependency. |
| **Step 2** | **Bookish LaTeX & TikZ Synthesis** | `main/src/lecture_to_book.py` | Convert chunk transcripts to textbook chapters; numbered equations (`\begin{equation}`); automated TikZ generation for $z$-plane, ROC, $d-q$ axes, phasors. |
| **Step 3** | **Global AI Review & PDF Compiler** | `main/src/postprocess_lecture_ai.py` | Multi-pass LLM review pass for symbol consistency; standalone compilation via TeX engine; error self-healing. |
| **Step 4** | **USB Auto-Export Manager** | `main/src/usb_exporter.py` | Detect `/media/ayush02/AYUSH`; export PDF, TeX, MD, audio, and ZIP bundle; verify checksums and flush disk cache (`sync`). |
| **Step 5** | **Headless Voice Agent Daemon** | `main/src/headless_voice_agent.py` | Hardware mic auto-negotiation; continuous voice command detection; countdown sequence; orchestrate recording $\to$ post-process $\to$ USB export $\to$ audio shutdown signal. |
| **Step 6** | **Systemd Service Setup & Verification** | `speech2latex-headless.service` | Configure auto-boot on Jetson; test full end-to-end flow with USB drive `AYUSH` and live voice control. |
