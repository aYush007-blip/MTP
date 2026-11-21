# Master's Thesis Defense: Presentation Slide Deck (15 Slides)
**Title:** Edge-Native Autonomous Mathematical Speech-to-LaTeX Transduction & Bilingual Monograph Synthesis  
**Presenter:** Ayush Kumar (Roll No: 2024-MTP-ECE-07)  
**Department:** Department of Electrical & Computer Engineering, Indian Institute of Technology  
**Target Platform:** NVIDIA Jetson Orin Nano (8GB Unified LPDDR5, Ampere SM87, 15W TDP)  
**Date:** November 2026  

---

## Slide 1: Title & Author Information
* **Title:** Edge-Native Autonomous Mathematical Speech-to-LaTeX Transduction and Bilingual STEM Monograph Synthesis
* **Candidate:** Ayush Kumar (M.Tech, Department of Electrical & Computer Engineering)
* **Theme:** 100% Offline Edge Appliance for NEP 2020 Bilingual Classroom Digitization on NVIDIA Jetson Orin Nano.
> *Speaker Note:* Good morning esteemed members of the faculty evaluation committee. Today, I present my Master's thesis on building an autonomous edge appliance capable of capturing live spoken mathematics in technical lectures and converting it into publication-grade bilingual monographs.

---

## Slide 2: Context & Motivation: The Triad Problem
* **1. Phonetic ASR Degradation:** Traditional ASR models transcribe spoken math phonetically into nonsensical strings (*"s times x of s"* -> `s xfx`).
* **2. Cloud Vulnerability & Latency:** Inability of cloud APIs to operate reliably in intermittent university Wi-Fi; high recurring token costs; data privacy risks.
* **3. Vernacular Barrier (NEP 2020):** Cognitive overload for rural/regional students assimilating complex math while decoding rapid spoken English.
> *Speaker Note:* In STEM lectures, spoken discourse is fundamentally non-linear and symbolic. Standard speech recognition engines fail catastrophically because they transcribe phonetically. Cloud APIs introduce latency jitter, high recurring costs, and privacy leaks. Concurrently, under NEP 2020, we have an urgent imperative to make technical education accessible in regional Indian languages.

---

## Slide 3: Research Objectives and System Scope
* **100% Offline Edge Autonomy:** Zero cloud reliance under a 15W power envelope.
* **Domain-Agnostic Math Transduction:** Universal coverage across Linear Algebra, Calculus, Signal Processing, and Physics.
* **SLM Edge Acceleration:** Qwen2.5-1.5B Q4_K_M with persistent KV-cache session reuse.
* **Mathematical Invariant Preservation:** Synthesis of fluent Hindi Devanagari notes with 100% verbatim formula preservation.
* **Monitorless Appliance Daemon:** Voice-triggered Linux daemon with neural TTS auditory feedback and cryptographic USB auto-sync.
> *Speaker Note:* Our objective was not to build another cloud wrapper, but a sovereign physical appliance that an instructor can place on a lectern, activate by voice, and walk away with a complete publication monograph on their USB drive.

---

## Slide 4: End-to-End Edge Architecture & Hardware Partitioning
* **GPU Subsystem (1024 CUDA Cores):** Dedicated to Whisper ASR with FlashAttention FP16 (18.2x real-time speed, 1,180 MB VRAM).
* **CPU Subsystem (6-Core ARM Cortex-A78AE):** Pinned to Qwen2.5-1.5B GGUF Q4_K_M to prevent unified memory contention.
* **Unified Memory Footprint:** Total active peak memory of 6.1 GB / 7.6 GB (zero swap thrashing).
* **Decoupled Pipeline Stages:** Ingestion -> GPU ASR -> CPU SLM -> Bilingual Translation -> Tectonic Publishing.
> *Speaker Note:* A key architectural contribution is compute partitioning. Because Jetson shares an 8GB unified memory pool, running both Whisper and an SLM on the GPU causes immediate OOM crashes. By pinning Whisper to the GPU and the quantized SLM to the 6 ARM cores, we achieve concurrent execution with zero swap thrashing.

---

## Slide 5: Stage 1 - Decoupled Acoustic Ingestion & Adaptive VAD
* **Non-Blocking Capture:** PortAudio ingestion thread captures 16-bit PCM at 16 kHz without blocking downstream processing.
* **Vectorized Decimation:** 48 kHz stereo to 16 kHz mono decimation in NumPy.
* **4th-Order Butterworth HPF (fc = 65 Hz):** Eliminates AC rumble, projector noise, and desk vibration.
* **Adaptive Dual-Rate Energy VAD:** Dynamic noise floor tracking over 30ms windows.
* **500ms Pre-Speech Ring Buffer:** Guarantees initial unvoiced mathematical plosives (/p/, /t/, /k/) are never clipped.
> *Speaker Note:* Acoustic reliability starts at ingestion. Our digital Butterworth filter strips out air conditioning rumble, while an adaptive ring buffer ensures mathematical plosives like 't' or 'pi' are never clipped by the voice detector.

---

## Slide 6: Stage 2 - GPU-Accelerated ASR & Universal Prompting
* **Whisper CUDA Engine:** whisper.cpp compiled for Jetson Ampere with FlashAttention FP16.
* **Universal Domain-Agnostic Conditioning:**
  * Replaced narrow domain priors with universal STEM conditioning prompt.
  * Primes beam-search decoder to bias phonemes toward mathematical variables (x, y, z, t, s, k, n).
* **Throughput:** 18.2x real-time factor (31.4 min lecture transcribed in 103s).
> *Speaker Note:* To ensure the system works across all university disciplines, we developed a universal conditioning prompt that primes the Whisper decoder for mathematical syntax without biasing it toward any single engineering subfield.

---

## Slide 7: Stage 3 - SLM Mathematical Transduction Engine
* **Quantized Model:** Qwen2.5-1.5B-Instruct in Q4_K_M GGUF format (1.11 GB).
* **KV-Cache Reuse:** System prompt evaluated once at boot; per-query TTFT drops from 1,200ms to 42ms.
* **Phonetic-to-LaTeX Transformation:** Spoken clauses mapped into clean inline LaTeX ($...$).
* **Display Equation Promotion:** Primary theoretical expressions promoted to numbered display environments.
> *Speaker Note:* Our Small Language Model translates conversational spoken math into precise LaTeX syntax. With prompt caching, query evaluation takes only 42 milliseconds on the edge CPU.

---

## Slide 8: Stage 4 - Cross-Lingual Pedagogical Translation Engine
* **Didactic Devanagari Prose:** Explanatory text rendered in formal academic Hindi.
* **Dual-Competency Architecture:** English technical terminology preserved parenthetically (e.g. *सतत-समय LTI प्रणालियाँ (continuous-time LTI systems)*).
* **Parallel Batch Subtitles:** Subtitle lines batched in blocks of 4, slashing translation time by 70%.
* **1:1 Structural Parity:** Section titles, theorems, and equations correspond exactly between English and Hindi.
> *Speaker Note:* In line with NEP 2020, we translate explanations into natural Hindi while preserving English engineering terms in parentheses, enabling students to bridge vernacular understanding with global literature.

---

## Slide 9: The Mathematical Invariant Preservation Principle
* **Formal Principle:** All mathematical tokens M are 100% invariant under translation mapping F: F(m_j) = m_j.
* **Why Standard MT Fails:** Sub-word tokenizers split LaTeX syntax, corrupting formulas and hallucinating symbols.
* **Syntactic Masking Solution:** Display equations and vector graphics are extracted and masked with invariant tokens, prose is translated, and exact equations are re-injected.
> *Speaker Note:* Mathematical truth is universal. By formally isolating equations before translation and re-injecting them verbatim, we achieve a 100% mathematical preservation rate across languages.

---

## Slide 10: Stage 5 - Autonomous Typesetting & Vector Graphics
* **Embedded Tectonic XeTeX:** Modern Rust/C++ XeTeX engine compiling native vector PDFs without TeXLive bloat.
* **Native Devanagari Typography:** Fontspec with Lohit Devanagari TTF for sub-pixel conjunct rendering.
* **Dynamic TikZ Synthesis:** Transcripts scanned for visual triggers; synthesizes vector figures (Laplace ROC complex plane, discrete stem plots).
* **Sub-Second Compilation:** Full multi-page monographs compiled in under 6 seconds.
> *Speaker Note:* Rather than crude HTML or ReportLab boxes, we embed Tectonic XeTeX, generating textbook-grade vector typography and dynamic TikZ diagrams at 1200 DPI print quality.

---

## Slide 11: Stage 6 - Headless Edge Daemon & Cryptographic Sync
* **Systemd Appliance Service:** speech2latex-headless.service boots on power-on with zero monitor requirements.
* **Hands-Free Operation:** Acoustic wake-words ('Hey Jetson, start recording' / 'stop recording').
* **Piper Neural TTS:** Voice prompts announce system status over USB speakers.
* **Cryptographic Auto-Sync:** Hotplug USB detection, atomic file write-and-rename, and SHA256SUMS.txt manifest.
> *Speaker Note:* The system operates entirely hands-free. The lecturer says 'Hey Jetson start recording', speaks their lecture, says 'stop recording', and the system compiles and syncs all deliverables to their USB drive with SHA-256 verification.

---

## Slide 12: Experimental Testbed & Benchmark Methodology
* **Edge Hardware Testbed:** NVIDIA Jetson Orin Nano (8GB Unified LPDDR5, Ampere SM87, 15W TDP).
* **Test Corpora:**
  1. Continuous-Time LTI Systems: 31.4 min, 58 MB WAV, 2,727 words.
  2. Energy of a Discrete Sequence: 5.7 min, 11 MB WAV, 600 words.
* **Measured Dimensions:** Real-Time Factor (RTF), Time-To-First-Token (TTFT), Throughput, Memory footprint, and Formula Preservation.
> *Speaker Note:* We verified the entire system on full-length real-world classroom lectures, including a 31.4-minute lecture on LTI systems and a discrete signal processing lecture.

---

## Slide 13: Quantitative Hardware Profile & Benchmarks
* **Whisper ASR Speed:** 18.2x real-time (RTF = 0.055, 1,180 MB GPU VRAM).
* **SLM Context Processing:** 48.5 tokens/sec (TTFT = 42 ms, 1,111 MB CPU RAM).
* **SLM Token Generation:** 3.35 tokens/sec on 6 ARM cores.
* **Tectonic XeTeX Compilation:** 1.25 sec/page (7 pages compiled in 5.5s).
* **Total Unified System Memory:** 6.1 GB peak (zero swap thrashing within 8GB budget).
> *Speaker Note:* As shown in our benchmark profile, the entire multi-model pipeline operates comfortably within 6.1 GB of RAM, ensuring complete stability under continuous 15W operation.

---

## Slide 14: Monograph Quality & Bilingual Output Verification
* **LTI Systems Lecture (31.4 min):**
  - English Monograph: 7 pages, 6 formal sections, 35 numbered equations, TikZ ROC diagram.
  - Hindi Monograph: 7 pages, 6 formal sections in Devanagari, 35 identical equations.
  - Subtitles: 170 synchronized blocks (.srt / .vtt) in English and Hindi.
* **Energy of a Sequence (5.7 min):**
  - English Monograph: 3 pages, 9 sections, 8 numbered display equations.
  - Hindi Monograph: 3 pages, 9 sections, 8 identical equations, discrete stem plot.
  - Subtitles: 47 synchronized blocks in both languages.
> *Speaker Note:* Here we see the 1:1 structural parity achieved: both English and Hindi monographs have identical page counts, section titles, and mathematical equations.

---

## Slide 15: Conclusion, Socio-Educational Impact & Future Horizons
* **Accomplishment:** An end-to-end, edge-native lecture capture and bilingual publishing appliance operating under 15W TDP on NVIDIA Jetson Orin Nano.
* **Educational Impact:** Empowers vernacular-medium students across Indian universities with synchronized bilingual lecture notes and global mathematical standards.
* **Final Deliverables:**
  - 18-page Master's Thesis Report (thesis_report.pdf).
  - 15-slide Presentation Deck (presentation_deck.pptx & .pdf).
  - Verified lecture bundles with SHA-256 integrity on USB drive AYUSH.
* **Future Work:** Multimodal whiteboard vision fusion, multi-speaker diarization, and LoRA adaptation for Tamil, Telugu, and Bengali.
> *Speaker Note:* In conclusion, EdgeLectureAI proves that high-tier AI-driven pedagogical publishing can be achieved entirely on low-power edge silicon. Thank you, and I look forward to your questions.
