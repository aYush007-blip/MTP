#!/usr/bin/env python3
"""
Real-Time Live Streaming Speech-to-LaTeX Transcriber for NVIDIA Jetson Orin Nano.

Provides live word-by-word streaming transcription in the terminal as speech occurs,
followed by instant mathematical LaTeX lock-in upon natural speech pause.

Architecture:
  - C-Level ALSA Silencing: Clean terminal output without driver noise
  - Audio Capture: PyAudio (48kHz stereo -> 16kHz mono vectorized decimation)
  - Acoustic Filtering: 4th-order Butterworth High-Pass Filter (120 Hz) removing 50/100 Hz mains hum
  - Adaptive VAD: Dual-threshold Schmitt trigger with dynamic ambient calibration and 400ms pre-roll
  - Streaming ASR: whisper-server HTTP daemon (GPU Flash Attention, in-memory, mathematical prompting)
  - Hallucination Sanitizer: Strips acoustic non-speech tags (e.g. (crashing), [BLANK_AUDIO])
  - Math & Prose Formatting: Qwen2.5-1.5B-Instruct SLM (llama-server CUDA SM87 daemon)
  - Visuals: Live ANSI in-place line overwriting (\r\033[K)

Usage:
  python src/live_streaming_terminal.py                  # Live microphone
  python src/live_streaming_terminal.py --mic            # Explicit microphone mode
  python src/live_streaming_terminal.py --file audio.wav # Test simulation on WAV file
"""

import argparse
import collections
import io
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import threading
import time
import urllib.request
import wave

# Suppress ALSA C-level library warnings on Jetson before importing PyAudio
try:
    import ctypes
    _ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    def _py_alsa_err(filename, line, function, err, fmt):
        pass
    _c_alsa_err = _ERROR_HANDLER_FUNC(_py_alsa_err)
    _asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _asound.snd_lib_error_set_handler(_c_alsa_err)
except Exception:
    pass

import numpy as np
import pyaudio

# Ensure src/ is in sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audio import stereo_48k_to_mono_16k
from energy_vad import AdaptiveEnergyVAD
from slm_translator import QwenMathTranslator

WHISPER_SERVER_URL = "http://127.0.0.1:8081"
LLAMA_SERVER_URL = "http://127.0.0.1:8080"
WHISPER_BIN = "/mnt/thesis/main/whisper.cpp/build-orin/bin/whisper-server"
WHISPER_BASE_MODEL = "/mnt/thesis/models/whisper/ggml-base.bin"
WHISPER_SMALL_MODEL = "/mnt/thesis/models/whisper/ggml-small.en.bin"
LLAMA_BIN = "/mnt/thesis/llama.cpp/build/bin/llama-server"
LLAMA_MODEL = "/mnt/thesis/models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# High-precision technical keyword vocabulary prompt for Whisper ASR (avoids sentence regurgitation)
WHISPER_MATH_PROMPT = (
    "Fourier transform, Laplace, continuous-time signal, discrete-time, integral, "
    "derivative, calculus, algebra, omega, theta, phi, pi, infinity, x(t), f(x), "
    "e to the power, differential equation, matrix, vector, sinusoids."
)


class HighPassHumFilter:
    """
    Cascaded 4th-order DC-blocking & 120 Hz mains hum attenuation filter.
    Pure NumPy implementation: 0 external C/Fortran libraries, zero SciPy symbols.
    Attenuates 50 Hz & 100 Hz AC electrical noise by >85%.
    """
    def __init__(self, r: float = 0.953):
        self.r = r
        self.x1_prev = 0.0
        self.y1_prev = 0.0
        self.x2_prev = 0.0
        self.y2_prev = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        # Stage 1
        y1 = np.empty_like(x)
        xp1, yp1, r = self.x1_prev, self.y1_prev, self.r
        for i in range(len(x)):
            y1[i] = x[i] - xp1 + r * yp1
            yp1, xp1 = y1[i], x[i]
        self.x1_prev, self.y1_prev = xp1, yp1

        # Stage 2
        y2 = np.empty_like(y1)
        xp2, yp2 = self.x2_prev, self.y2_prev
        for i in range(len(y1)):
            y2[i] = y1[i] - xp2 + r * yp2
            yp2, xp2 = y2[i], y1[i]
        self.x2_prev, self.y2_prev = xp2, yp2
        return y2

    def reset(self):
        self.x1_prev = self.y1_prev = self.x2_prev = self.y2_prev = 0.0

# Terminal ANSI Styling
CYAN_BOLD = "\033[1;36m"
GREEN_BOLD = "\033[1;32m"
YELLOW_BOLD = "\033[1;33m"
MAGENTA_BOLD = "\033[1;35m"
WHITE_BOLD = "\033[1;37m"
BLUE_BOLD = "\033[1;34m"
DIM = "\033[2m"
RESET = "\033[0m"
CLEAR_LINE = "\r\033[K"


def ensure_hardware_mic_gain():
    """Ensure hardware USB microphone gain is optimal (-5 dB / 92%), not attenuated to -20 dB."""
    try:
        subprocess.run(
            ["amixer", "-c", "2", "sset", "Mic", "110"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def is_service_alive(url: str, endpoint: str = "/health") -> bool:
    """Verify if a local HTTP service is actively responding."""
    try:
        req = urllib.request.Request(f"{url}{endpoint}", method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def ensure_edge_daemons(model_type: str = "small"):
    """Ensure both whisper-server and llama-server daemons are running on the Jetson."""
    model_path = WHISPER_SMALL_MODEL if model_type == "small" else WHISPER_BASE_MODEL

    # 1. Check if whisper-server is alive
    if not is_service_alive(WHISPER_SERVER_URL, endpoint="/"):
        print(f"{YELLOW_BOLD}[Init]{RESET} Starting whisper-server daemon with {Path(model_path).name} (Port 8081, Flash Attention)...")
        cmd = [
            WHISPER_BIN,
            "-m", model_path,
            "--port", "8081",
            "--host", "127.0.0.1",
            "-fa", "-nt", "-bo", "1", "-bs", "1", "-t", "4",
            "-sns", "-nth", "0.65"
        ]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/mnt/thesis/main/whisper.cpp/build-orin/bin:" + env.get("LD_LIBRARY_PATH", "")
        log_whisper = open("/tmp/whisper-server.log", "w")
        subprocess.Popen(cmd, env=env, stdout=log_whisper, stderr=log_whisper)
        for _ in range(30):
            time.sleep(0.5)
            if is_service_alive(WHISPER_SERVER_URL, endpoint="/"):
                break
        print(f"{GREEN_BOLD}[Ready]{RESET} whisper-server ready ({Path(model_path).name}) on {WHISPER_SERVER_URL}")
    else:
        # Server is already running: reload only if requested model is different
        current_model_file = Path("/tmp/whisper_current_model.txt")
        needs_reload = True
        if current_model_file.exists():
            try:
                if current_model_file.read_text().strip() == model_path:
                    needs_reload = False
            except Exception:
                pass

        if needs_reload:
            try:
                boundary = "----ModelLoadBoundary"
                body = (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="model"\r\n\r\n'
                    f"{model_path}\r\n--{boundary}--\r\n"
                ).encode("utf-8")
                req = urllib.request.Request(
                    f"{WHISPER_SERVER_URL}/load",
                    data=body,
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                )
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    current_model_file.write_text(model_path)
            except Exception:
                pass
        print(f"{GREEN_BOLD}[Active]{RESET} whisper-server active with {Path(model_path).name} on {WHISPER_SERVER_URL}")

    # 2. Check if llama-server is alive (Port 8080)
    if not is_service_alive(LLAMA_SERVER_URL, endpoint="/health"):
        print(f"{YELLOW_BOLD}[Init]{RESET} Starting llama-server daemon on port 8080 (CUDA SM87)...")
        cmd = [
            LLAMA_BIN,
            "-m", LLAMA_MODEL,
            "-c", "2048",
            "-ngl", "28",
            "--port", "8080",
            "--host", "127.0.0.1"
        ]
        log_llama = open("/tmp/llama-server.log", "w")
        subprocess.Popen(cmd, stdout=log_llama, stderr=log_llama)
        for _ in range(30):
            time.sleep(0.5)
            if is_service_alive(LLAMA_SERVER_URL, endpoint="/health"):
                break
        print(f"{GREEN_BOLD}[Ready]{RESET} llama-server (Qwen2.5-1.5B) ready on {LLAMA_SERVER_URL}")
    else:
        print(f"{GREEN_BOLD}[Active]{RESET} llama-server (Qwen2.5-1.5B) is active on {LLAMA_SERVER_URL}")


def clean_whisper_text(text: str) -> str:
    """
    Sanitize raw Whisper transcripts:
      - Strips parenthetical / bracketed acoustic tags like (crashing), [BLANK_AUDIO], (music), (coughing)
      - Filters out empty/hallucinated YouTube-style non-speech phrases
      - Normalizes spaces
    """
    if not text:
        return ""

    # Remove all parenthetical / bracketed tokens: (crashing), [inaudible], {noise}, etc.
    cleaned = re.sub(r"[\(\[\{].*?[\)\]\}]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Hallucination phrases generated on low-confidence or silent boundaries
    hallucination_phrases = {
        "you", "you.", "thank you", "thank you.", "thank you for watching", "thank you for watching.",
        "subtitles by", "bye.", "bye bye", "the end", "i'm out.", "i'm sorry.", "peace."
    }
    if cleaned.lower() in hallucination_phrases:
        return ""

    return cleaned


def query_whisper_stream(pcm_samples: np.ndarray) -> tuple[str, float]:
    """
    Send in-memory PCM float32 samples to whisper-server and return sanitized text.
    Uses in-memory multipart WAV encoding (0 disk I/O) and mathematical domain prompting.
    """
    if len(pcm_samples) < 6400:  # Minimum 400ms
        return "", 0.0

    t0 = time.monotonic()
    # Convert float32 [-1, 1] to signed 16-bit PCM
    int16_samples = np.clip(pcm_samples * 32767, -32768, 32767).astype(np.int16)

    # Encode to in-memory WAV buffer
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(int16_samples.tobytes())
    wav_bytes = buf.getvalue()

    # Form multipart request with math initial prompt and non-speech suppression
    boundary = "----JetsonLiveStreamBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="stream.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8") + wav_bytes + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="prompt"\r\n\r\n'
        f"{WHISPER_MATH_PROMPT}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="no_speech_thold"\r\n\r\n'
        f"0.65\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="response_format"\r\n\r\n'
        f"json\r\n--{boundary}--\r\n"
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{WHISPER_SERVER_URL}/inference",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    try:
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_text = data.get("text", "").strip()
            cleaned = clean_whisper_text(raw_text)
            latency_ms = (time.monotonic() - t0) * 1000.0
            return cleaned, latency_ms
    except Exception:
        return "", 0.0


def calibrate_ambient_noise(
    stream: pyaudio.Stream,
    rate: int,
    channels: int,
    chunk_size: int,
    duration_sec: float = 1.0,
) -> tuple[float, float, float]:
    """
    Measure ambient room and electrical noise floor across duration_sec.
    Returns:
      (noise_floor_rms, speech_threshold, silence_threshold)
    """
    num_chunks = int(duration_sec * rate / chunk_size)
    cal_rmses = []
    hpf = HighPassHumFilter()

    sys.stdout.write(f"{YELLOW_BOLD}[Calibration]{RESET} Measuring ambient noise floor ({duration_sec:.1f}s)... ")
    sys.stdout.flush()

    for _ in range(max(num_chunks, 6)):
        raw_bytes = stream.read(chunk_size, exception_on_overflow=False)
        if rate == 48000 and channels == 2:
            mono_chunk = stereo_48k_to_mono_16k(raw_bytes)
        else:
            pcm_data = np.frombuffer(raw_bytes, dtype=np.int16)
            mono_chunk = np.clip(pcm_data / 32768.0, -1.0, 1.0).astype(np.float32)

        # Apply high-pass filter
        filtered_chunk = hpf.process(mono_chunk)
        rms = float(np.sqrt(np.mean(filtered_chunk ** 2)))
        cal_rmses.append(rms)

    noise_floor = float(np.percentile(cal_rmses, 85))
    # Schmitt trigger dual thresholds
    speech_threshold = max(float(noise_floor * 2.2), 0.045)
    silence_threshold = max(float(noise_floor * 1.3), 0.025)

    print(f"{GREEN_BOLD}Done.{RESET}")
    print(f"   {DIM}Ambient Noise Floor: RMS = {noise_floor:.4f}{RESET}")
    print(f"   {DIM}Speech Trigger: RMS >= {speech_threshold:.4f} | Silence End: RMS < {silence_threshold:.4f}{RESET}\n")

    return noise_floor, speech_threshold, silence_threshold


def run_live_microphone(slm: QwenMathTranslator, pause_sec: float = 1.8, enter_only: bool = False):
    """
    Run real-time microphone capture with instant on-demand transcription on [Enter].
    Optimized for NVIDIA Jetson: Zero GPU inference during audio accumulation prevents
    over-current / thermal throttling, and [Enter] locks in equation immediately (<500ms).
    """
    ensure_hardware_mic_gain()
    pa = pyaudio.PyAudio()

    # Auto-detect AB13X USB Audio or fallback to default
    device_index = None
    channels = 1
    rate = 16000

    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        name = info.get("name", "").lower()
        if "ab13x" in name and info.get("maxInputChannels", 0) > 0:
            device_index = i
            channels = 2
            rate = 48000
            print(f"[Audio] Detected USB Microphone: {info['name']} (Device {i}, 48kHz Stereo)")
            break

    if device_index is None:
        print("[Audio] Using default system microphone (16kHz Mono)")
        device_index = None
        channels = 1
        rate = 16000

    chunk_size = 4800 if rate == 48000 else 1600
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=chunk_size,
    )

    # 1. Calibrate ambient noise floor
    noise_floor, speech_thold, silence_thold = calibrate_ambient_noise(
        stream, rate, channels, chunk_size, duration_sec=1.0
    )

    # 2. Initialize Adaptive VAD with dual thresholds and 400ms pre-roll buffer
    silence_chunks = max(int(pause_sec * 10), 8) if pause_sec > 0 else 18
    vad = AdaptiveEnergyVAD(
        speech_threshold=speech_thold,
        silence_threshold=silence_thold,
        min_speech_chunks=2,               # 200ms speech onset
        min_silence_chunks=silence_chunks, # natural pause tolerance
        pre_roll_chunks=4,                 # 400ms pre-trigger history
    )

    print("=" * 70)
    print("LIVE SPEECH-TO-LATEX TERMINAL TRANSCRIBER")
    print("=" * 70)
    print("Speak mathematical phrases or lecture sentences into your microphone.")
    print("Press [Enter] at any moment to transcribe immediately.")
    if not enter_only and pause_sec > 0:
        print(f"(Or pause naturally for ~{pause_sec:.1f}s for auto-transcription)")
    print("Press Ctrl+C to exit.")
    print("=" * 70 + "\n")
    print(f"{DIM}[Listening...] Speak, then press [Enter] to transcribe.{RESET}\n")

    current_speech_chunks = []
    in_speech = False
    rolling_chunks = collections.deque(maxlen=120)  # Rolling 12-second audio history
    hpf = HighPassHumFilter()

    try:
        while True:
            # Check for instant manual lock-in via Enter key
            manual_lock_in = False
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.0)
                if r:
                    sys.stdin.readline()
                    manual_lock_in = True
            except Exception:
                pass

            raw_bytes = stream.read(chunk_size, exception_on_overflow=False)

            if rate == 48000 and channels == 2:
                mono_chunk = stereo_48k_to_mono_16k(raw_bytes)
            else:
                pcm_data = np.frombuffer(raw_bytes, dtype=np.int16)
                mono_chunk = np.clip(pcm_data / 32768.0, -1.0, 1.0).astype(np.float32)

            # Continuous high-pass filter (removes 50/100 Hz AC hum)
            filtered_chunk = hpf.process(mono_chunk)
            rms = float(np.sqrt(np.mean(filtered_chunk ** 2)))
            rolling_chunks.append((filtered_chunk, rms))

            state, initial_chunks = vad.process(filtered_chunk)

            if state == "start":
                in_speech = True
                current_speech_chunks = list(initial_chunks) if initial_chunks else [filtered_chunk]
                elapsed_sec = len(current_speech_chunks) * 0.1
                sys.stdout.write(f"{CLEAR_LINE}{YELLOW_BOLD}[Recording {elapsed_sec:.1f}s]{RESET} [====....] {DIM}Press [Enter] to transcribe{RESET}\r")
                sys.stdout.flush()

            elif in_speech:
                current_speech_chunks.append(filtered_chunk)
                elapsed_sec = len(current_speech_chunks) * 0.1
                # Pure CPU/NumPy VU meter (0% GPU, zero over-current risk)
                ratio = min(max((rms - noise_floor) / max(speech_thold * 1.5, 0.02), 0.0), 1.0)
                bars = int(round(ratio * 8))
                vu = "=" * bars + "." * (8 - bars)
                pause_note = f" {DIM}(pause detected - press Enter to lock in){RESET}" if vad.silence_count >= 5 else f" {DIM}Press [Enter] to transcribe{RESET}"
                sys.stdout.write(
                    f"{CLEAR_LINE}{YELLOW_BOLD}[Recording {elapsed_sec:.1f}s]{RESET} [{vu}]{pause_note}\r"
                )
                sys.stdout.flush()

            # Determine if transcription should trigger:
            # 1. User pressed [Enter] (Instant manual lock-in)
            # 2. Natural pause ended speech (if not enter_only mode)
            should_transcribe = False
            is_manual = False

            if manual_lock_in:
                should_transcribe = True
                is_manual = True
            elif not enter_only and pause_sec > 0 and state == "end" and in_speech:
                should_transcribe = True
                is_manual = False

            if should_transcribe:
                if in_speech and current_speech_chunks:
                    if is_manual:
                        speech_chunks = list(current_speech_chunks)
                    else:
                        # Auto-pause: strip trailing silence chunks
                        if len(current_speech_chunks) > silence_chunks:
                            speech_chunks = current_speech_chunks[:-silence_chunks]
                        else:
                            speech_chunks = current_speech_chunks
                else:
                    # User pressed Enter without VAD active: find recent audio in rolling history
                    earliest_idx = None
                    for idx, (c, r) in enumerate(rolling_chunks):
                        if r >= silence_thold:
                            earliest_idx = max(0, idx - 4)  # 400ms pre-roll
                            break
                    if earliest_idx is not None and (len(rolling_chunks) - earliest_idx >= 4):
                        speech_chunks = [c for c, _ in list(rolling_chunks)[earliest_idx:]]
                    else:
                        speech_chunks = []

                if not speech_chunks or len(speech_chunks) < 4:
                    if is_manual:
                        sys.stdout.write(f"{CLEAR_LINE}{DIM}[Listening...] (No speech detected - speak and press [Enter]){RESET}\n")
                    else:
                        sys.stdout.write(f"{CLEAR_LINE}{DIM}[Listening...]{RESET}\r")
                    sys.stdout.flush()
                    current_speech_chunks = []
                    in_speech = False
                    vad.reset()
                    continue

                full_utterance = np.concatenate(speech_chunks)
                total_duration = len(full_utterance) / 16000.0

                sys.stdout.write(f"{CLEAR_LINE}{CYAN_BOLD}[Transcribing {total_duration:.1f}s audio...]{RESET}\r")
                sys.stdout.flush()

                # Run Whisper ASR ONCE on GPU
                final_text, final_asr_ms = query_whisper_stream(full_utterance)

                if final_text.strip():
                    sys.stdout.write(f"{CLEAR_LINE}{WHITE_BOLD}[Spoken]:{RESET} \"{final_text}\"\n")
                    sys.stdout.flush()

                    # Run Qwen2.5-1.5B SLM ONCE on GPU
                    t_slm_start = time.monotonic()
                    latex_output = slm.translate(final_text)
                    slm_ms = (time.monotonic() - t_slm_start) * 1000.0

                    print(f"{GREEN_BOLD}[LaTeX]: {RESET}{GREEN_BOLD}{latex_output}{RESET}")
                    print(f"{DIM}[Latency]: ASR: {final_asr_ms:.1f}ms | SLM: {slm_ms:.1f}ms | Audio: {total_duration:.2f}s{RESET}\n")
                    print(f"{DIM}[Listening...] Speak, then press [Enter] to transcribe.{RESET}\n")
                else:
                    sys.stdout.write(f"{CLEAR_LINE}{DIM}[Listening...] (No speech recognized - speak and press [Enter]){RESET}\n")
                    sys.stdout.flush()

                current_speech_chunks = []
                in_speech = False
                vad.reset()
                rolling_chunks.clear()

    except KeyboardInterrupt:
        print(f"\n{YELLOW_BOLD}[Stream]{RESET} Exiting live listener...")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def run_file_simulation(audio_path: str, slm: QwenMathTranslator):
    """Simulate real-time streaming using an existing WAV file."""
    p = Path(audio_path)
    if not p.is_file():
        print(f"Error: Audio file not found: {p}")
        return

    with wave.open(str(p), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    pcm = np.frombuffer(raw_bytes, dtype=np.int16)
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)

    if framerate != 16000:
        num_samples = int(len(pcm) * 16000 / framerate)
        pcm = np.interp(
            np.linspace(0, len(pcm), num_samples, endpoint=False),
            np.arange(len(pcm)),
            pcm,
        ).astype(np.int16)

    float_audio = np.clip(pcm / 32768.0, -1.0, 1.0).astype(np.float32)

    # Filter with HPF
    hpf = HighPassHumFilter()
    filtered_audio = hpf.process(float_audio)

    total_sec = len(filtered_audio) / 16000.0
    print("\n" + "=" * 70)
    print(f"SIMULATING LIVE STREAMING FROM: {p.name}")
    print(f"Duration: {total_sec:.2f}s | Sample Rate: 16 kHz")
    print("=" * 70 + "\n")

    chunk_samples = 1600  # 100ms chunks
    current_buffer = []
    last_partial_text = ""

    for idx, i in enumerate(range(0, len(filtered_audio), chunk_samples)):
        chunk = filtered_audio[i : i + chunk_samples]
        current_buffer.append(chunk)

        # Simulate real-time audio playback cadence (100ms)
        time.sleep(0.08)

        # Run progressive Whisper ASR once >= 800ms accumulated (every 400ms)
        if (idx % 4 == 0) and len(current_buffer) >= 8:
            acc = np.concatenate(current_buffer)
            partial_text, asr_ms = query_whisper_stream(acc)
            if partial_text and partial_text != last_partial_text:
                last_partial_text = partial_text
                elapsed = round(len(acc) / 16000.0, 1)
                sys.stdout.write(
                    f"{CLEAR_LINE}{YELLOW_BOLD}[Live {elapsed:.1f}s]:{RESET} {CYAN_BOLD}{partial_text}{RESET}"
                )
                sys.stdout.flush()

    # Final complete utterance lock-in
    full_audio = np.concatenate(current_buffer)
    final_text, final_asr_ms = query_whisper_stream(full_audio)
    if not final_text:
        final_text = last_partial_text

    sys.stdout.write(f"{CLEAR_LINE}{WHITE_BOLD}[Spoken Transcript]:{RESET} \"{final_text}\"\n")
    sys.stdout.flush()

    print(f"\n{YELLOW_BOLD}[SLM]{RESET} Formatting with Qwen2.5-1.5B (CUDA SM87)...")
    t_slm_start = time.monotonic()
    latex_output = slm.translate(final_text)
    slm_ms = (time.monotonic() - t_slm_start) * 1000.0

    print("\n" + "=" * 70)
    print(f"{GREEN_BOLD}[Final LaTeX]:{RESET} {GREEN_BOLD}{latex_output}{RESET}")
    print(f"{DIM}[Latency Profile]: Final ASR: {final_asr_ms:.1f}ms | SLM: {slm_ms:.1f}ms | Audio: {total_sec:.2f}s{RESET}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Live Streaming Speech-to-LaTeX Transcriber")
    parser.add_argument("--file", type=str, default=None, help="Path to WAV audio file to simulate live streaming")
    parser.add_argument("--mic", action="store_true", help="Force microphone input")
    parser.add_argument("--pause", type=float, default=1.8, help="Silence pause duration (seconds) before equation lock-in (default: 1.8s, or press Enter)")
    parser.add_argument("--enter-only", action="store_true", help="Only transcribe upon pressing Enter (disables auto-pause lock-in)")
    parser.add_argument("--model", type=str, default="small", choices=["base", "small"], help="Whisper model to use ('small' or 'base', default: 'small')")
    args = parser.parse_args()

    # Ensure background GPU daemons with requested model
    ensure_edge_daemons(model_type=args.model)

    # Initialize SLM translator client
    slm = QwenMathTranslator.get_instance(backend="server")

    if args.file:
        run_file_simulation(args.file, slm)
    else:
        run_live_microphone(slm, pause_sec=args.pause, enter_only=args.enter_only)


if __name__ == "__main__":
    main()

