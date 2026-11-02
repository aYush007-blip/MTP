# Headless Voice-Activated Edge Speech-to-LaTeX: Architecture & Implementation Plan

This document outlines the end-to-end design for transforming the NVIDIA Jetson Orin Nano Speech-to-LaTeX system into a **fully autonomous, monitorless academic recording agent** that generates publication-grade, textbook-formatted LaTeX papers with vector graphics.

---

## System Architecture Overview

```mermaid
graph TD
    A[Jetson Cold Boot / Power On] --> B[systemd: speech2latex-headless.service]
    B --> C[Auto-Detect USB Mic/Earphones: ALSA hw:2,0]
    B --> D[Auto-Spawn GPU Daemons: Whisper & Qwen2.5 SLM]
    C & D --> E[Passive Wake-Word Listener: 'Hey Jetson...']
    
    E -- 'Hey Jetson start recording' --> F[Offline Neural TTS: 'Starting in 5..4..3..2..1']
    F --> G[Continuous Lecture Capture & Real-Time ASR/SLM]
    
    G -- 'Hey Jetson stop recording' --> H[Audio Halts & TTS: 'Processing lecture...']
    H --> I[Post-Lecture Global AI Review & Formatting Engine]
    
    subgraph Post-Processing Engine
        I --> I1[Textbook Flow: Remove timestamps, format paragraphs]
        I --> I2[Equation Environment: \\begin{equation} with numbers]
        I --> I3[Automated TikZ/PGFPlots Visual Graph Generation]
        I --> I4[Global Notation Harmonization & LaTeX Syntax Validation]
    end
    
    I --> J[Compile PDF via pdflatex & Export Deliverables]
    J --> K[Offline TTS: 'Transcription successfully saved. You can turn off the system.']
    J --> L[Auto-Push to GitHub & Backup to USB Drive]
```

---

## 1. Feature 1: Textbook-Style LaTeX & Automated TikZ Vector Graphics

### 1.1 From Chunk Subtitles to Academic Book Chapters
Currently, exports retain per-utterance chunks with timestamps. The new typesetting engine converts this into an academic monograph:
* **Timestamp Elimination:** Timestamps are completely omitted from the `.tex` document (retained exclusively in `.srt`/`.vtt` subtitle files for video playback).
* **Semantic Paragraph Assembly:** Successive spoken sentences are grouped into cohesive paragraphs using punctuation and topical transitions rather than VAD cut points.
* **Hierarchical Sectioning:** The engine automatically structures the lecture into logical divisions:
  ```latex
  \section{Introduction and Physical Foundations}
  ...
  \subsection{Direct and Quadrature Axis Transformation}
  ...
  ```

### 1.2 Display Equations with Numbering (`\begin{equation}`)
* **Inline vs. Display Math Classification:**
  * Quick variables, indices, and units stay inline: `$x[n]$`, `$z$`, `$\omega$`, `$5\text{ V}$`.
  * Major definitions, laws, and transfer functions are automatically promoted to numbered display environments:
    ```latex
    \begin{equation}
      \label{eq:park_transform}
      \begin{bmatrix} v_d \\ v_q \\ v_0 \end{bmatrix} = \mathbf{P}(\theta) \begin{bmatrix} v_a \\ v_b \\ v_c \end{bmatrix}
    \end{equation}
    ```
* **Equation Referencing:** If the speaker later says *"as shown in the earlier equation"* or refers back to the flux equation, the AI links it via `\eqref{eq:...}`.

### 1.3 Automated TikZ Vector Graphs & Geometric Illustrations
When the lecturer describes geometric planes, graphs, or circuit topologies, the AI injects native **TikZ / PGFPlots** vector code directly into the LaTeX source:
* **Complex $z$-Plane & ROC:**
  ```latex
  \begin{figure}[htbp]
  \centering
  \begin{tikzpicture}[scale=1.2]
    \draw[->, thick] (-2.2, 0) -- (2.2, 0) node[right] {$\text{Re}(z)$};
    \draw[->, thick] (0, -2.2) -- (0, 2.2) node[above] {$\text{Im}(z)$};
    \draw[dashed, blue, thick] (0,0) circle (1.0);
    \node[below right] at (1.0, 0) {$1$};
    % ROC Shading outside radius r
    \fill[red!15, even odd rule] (0,0) circle (2.0) (0,0) circle (1.4);
    \draw[red, thick] (0,0) circle (1.4);
    \node[above left] at (1.4, 0) {$r$};
    \node[red] at (1.1, 1.1) {$\text{ROC}: |z| > r$};
  \end{tikzpicture}
  \caption{Region of Convergence (ROC) in the complex $z$-plane.}
  \label{fig:z_plane_roc}
  \end{figure}
  ```
* **Bode Plots, State Trajectories & Machine Phasors:** Generated using `\usepackage{pgfplots}` without requiring any external graphical dependencies.

---

## 2. Feature 2: 100% Headless Standalone Operation with Voice Commands & Audio Feedback

### 2.1 Monitorless Auto-Boot (`speech2latex-headless.service`)
* Configured as a systemd system service triggered at boot:
  ```ini
  [Unit]
  Description=Speech2LaTeX Autonomous Headless Edge Agent
  After=sound.target network.target

  [Service]
  Type=simple
  User=ayush02
  WorkingDirectory=/mnt/thesis/main
  ExecStart=/mnt/thesis/main/.venv/bin/python -u src/headless_voice_agent.py
  Restart=always
  RestartSec=3

  [Install]
  WantedBy=multi-user.target
  ```
* **Hardware Mic Negotiation:** On boot, the daemon executes our `get_best_input_stream_config` routine, binding to `AB13X USB Audio` (`hw:2,0`) or fallback ALSA card, and initializes audio output.

### 2.2 Offline Wake-Word Detection
* Uses a lightweight continuous listening loop operating on 1.5-second audio buffers:
  * **Trigger Phrases:**
    * Start: `"Hey Jetson, start recording"` / `"Jetson, start lecture"`
    * Stop: `"Hey Jetson, stop recording"` / `"Jetson, stop lecture"`
  * **Low-Power KWS:** Runs Whisper with a 1.5s slice or acoustic keyword detector. Compute utilization during standby is $< 5\%$.

### 2.3 Offline Neural Text-to-Speech (Piper TTS)
* **Engine:** `piper-tts` (fast, offline, ONNX-accelerated neural voice engine).
  * Runs on Jetson in $\sim 50\text{ ms}$ with zero network connection.
  * Voice model: High-clarity natural English (e.g. `en_US-lessac-medium.onnx`).
* **Audible Interaction Flow:**
  1. **On Wake Command:** Jetson outputs through earphones/speaker:
     > *"Okay. Starting recording in 5... 4... 3... 2... 1... Recording started."*
     *(The 5-second countdown allows the lecturer to position themselves and clear audio buffers).*
  2. **On Stop Command:** Jetson outputs:
     > *"Recording stopped. Compiling and typesetting lecture paper. Please wait."*
  3. **On Completion:**
     > *"Transcription and paper successfully saved. You can now safely turn off the system."*

---

## 3. Feature 3: Post-Lecture Global AI Review & Formatting Pass

### 3.1 The Global Synthesis Engine (`postprocess_lecture_ai.py`)
In live mode, processing occurs in 3–9 second chunks. Once the recording stops, the post-processor ingests the entire lecture transcript and executes a structured 4-step AI pass:

```mermaid
graph LR
    R1[Raw Chunks Transcript] --> P1[Pass 1: Global Math Unification]
    P1 --> P2[Pass 2: Bookish Flow & Sectioning]
    P2 --> P3[Pass 3: TikZ Diagram Generation]
    P3 --> P4[Pass 4: LaTeX Compilation & Syntax Guard]
    P4 --> OUT[Final Standalone PDF & Clean TeX]
```

1. **Pass 1 — Global Mathematical Consistency:**
   * Reconciles symbols across the entire hour. If the professor introduces stator flux as $\psi_s$ in minute 5, ensures it is consistently formatted throughout, eliminating transient acoustic misnomers.
2. **Pass 2 — Bookish Flow & Display Equations:**
   * Removes conversational fillers ("okay", "so let me see", "um").
   * Identifies core mathematical relations and wraps them in numbered `\begin{equation} ... \end{equation}` blocks.
   * Generates descriptive abstract and introduction.
3. **Pass 3 — Contextual TikZ Visual Graph Insertion:**
   * Detects spoken prompts like *"let us draw the axes"*, *"here is the phasor diagram"*, *"on the complex plane"*.
   * Synthesizes precise TikZ code visualizing the discussed concepts.
4. **Pass 4 — Automated pdflatex Compilation & Syntax Guard:**
   * Compiles the generated `.tex` to `.pdf` via `pdflatex -interaction=nonstopmode`.
   * Automatically captures any compile errors and runs a self-correction loop to guarantee a 100% valid document.

---

## 4. Implementation Steps (Scheduled for Tomorrow)

| Step | Component | Target Files | Key Objectives |
| :---: | :--- | :--- | :--- |
| **1** | **Piper Neural TTS Setup** | `main/src/edge_tts.py` | Install standalone `piper` binary & model; test playback through `AB13X USB Audio`. |
| **2** | **Headless Voice Agent Daemon** | `main/src/headless_voice_agent.py` | Wake-word listener, auto-mic configuration, countdown, recording state-machine. |
| **3** | **Bookish LaTeX & TikZ Engine** | `main/src/lecture_to_book.py` | Transform transcripts into textbook format with numbered display math and TikZ graphs. |
| **4** | **Global AI Review & PDF Compiler** | `main/src/postprocess_lecture_ai.py` | Full-lecture proofreading pass, notation consistency, `pdflatex` compilation. |
| **5** | **Systemd Service Setup** | `/etc/systemd/system/speech2latex-headless.service` | Auto-start on boot, restart protection, headless validation. |
| **6** | **End-to-End Validation** | Jetson Orin Nano Standalone | Test complete boot-to-export voice loop on live microphone and audio recordings. |

---

## 5. Deliverables Expected Upon Execution

1. `speech2latex-headless.service`: Zero-touch background daemon for monitorless Jetson deployment.
2. `edge_tts.py`: Instantaneous offline voice feedback module.
3. `lecture_to_book.py`: Publication-grade LaTeX formatter with native TikZ graph synthesis.
4. `postprocess_lecture_ai.py`: Multi-pass AI editor delivering compilable, numbered academic papers.
