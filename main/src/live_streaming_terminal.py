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
from scipy import signal

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
WHISPER_MODEL = "/mnt/thesis/models/whisper/ggml-base.bin"
LLAMA_BIN = "/mnt/thesis/llama.cpp/build/bin/llama-server"
LLAMA_MODEL = "/mnt/thesis/models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# 4th-order Butterworth High-Pass Filter (120 Hz) designed for 16 kHz audio
# Eliminates 50 Hz & 100 Hz AC electrical mains hum and low-frequency rumble
SOS_HPF_16K = signal.butter(4, 120, "hp", fs=16000, output="sos")

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


def is_service_alive(url: str, endpoint: str = "/health") -> bool:
    """Verify if a local HTTP service is actively responding."""
    try:
        req = urllib.request.Request(f"{url}{endpoint}", method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def ensure_edge_daemons():
    """Ensure both whisper-server and llama-server daemons are running on the Jetson."""
    # 1. Check whisper-server (Port 8081)
    if not is_service_alive(WHISPER_SERVER_URL, endpoint="/"):
        print(f"{YELLOW_BOLD}[Init]{RESET} Starting whisper-server daemon on port 8081 (Flash Attention, Anti-Hallucination)...")
        cmd = [
            WHISPER_BIN,
            "-m", WHISPER_MODEL,
            "--port", "8081",
            "--host", "127.0.0.1",
            "-fa", "-nt", "-bo", "1", "-bs", "1", "-t", "6",
            "-sns", "-nth", "0.65"
        ]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/mnt/thesis/main/whisper.cpp/build-orin/bin:" + env.get("LD_LIBRARY_PATH", "")
        subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(20):
            time.sleep(0.5)
            if is_service_alive(WHISPER_SERVER_URL, endpoint="/"):
                break
        print(f"{GREEN_BOLD}✅ whisper-server ready on {WHISPER_SERVER_URL}{RESET}")
    else:
        print(f"{GREEN_BOLD}✅ whisper-server is active on {WHISPER_SERVER_URL}{RESET}")

    # 2. Check llama-server (Port 8080)
    if not is_service_alive(LLAMA_SERVER_URL, endpoint="/health"):
        print(f"{YELLOW_BOLD}[Init]{RESET} Starting llama-server daemon on port 8080 (CUDA SM87)...")
        cmd = [
            LLAMA_BIN,
            "-m", LLAMA_MODEL,
            "-c", "2048",
            "-ngl", "99",
            "--port", "8080",
            "--host", "127.0.0.1"
        ]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(20):
            time.sleep(0.5)
            if is_service_alive(LLAMA_SERVER_URL, endpoint="/health"):
                break
        print(f"{GREEN_BOLD}✅ llama-server (Qwen2.5-1.5B) ready on {LLAMA_SERVER_URL}{RESET}")
    else:
        print(f"{GREEN_BOLD}✅ llama-server (Qwen2.5-1.5B) is active on {LLAMA_SERVER_URL}{RESET}")


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
        f"Mathematical equations, formulas, calculus, algebra, variables, sine, cosine, theta, pi, integral, derivative, matrix.\r\n"
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
    except Exception as e:
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

    sys.stdout.write(f"{YELLOW_BOLD}🎙️  [Calibration]{RESET} Measuring ambient noise floor ({duration_sec:.1f}s)... ")
    sys.stdout.flush()

    for _ in range(max(num_chunks, 6)):
        raw_bytes = stream.read(chunk_size, exception_on_overflow=False)
        if rate == 48000 and channels == 2:
            mono_chunk = stereo_48k_to_mono_16k(raw_bytes)
        else:
            pcm_data = np.frombuffer(raw_bytes, dtype=np.int16)
            mono_chunk = np.clip(pcm_data / 32768.0, -1.0, 1.0).astype(np.float32)

        # Apply high-pass filter
        filtered_chunk = signal.sosfilt(SOS_HPF_16K, mono_chunk)
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


def run_live_microphone(slm: QwenMathTranslator):
    """Run real-time microphone capture with live word-by-word streaming."""
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
            print(f"🎙️  Found USB Microphone: {info['name']} (Device {i}, 48kHz Stereo)")
            break

    if device_index is None:
        print("🎙️  Using default system microphone (16kHz Mono)")
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
    vad = AdaptiveEnergyVAD(
        speech_threshold=speech_thold,
        silence_threshold=silence_thold,
        min_speech_chunks=2,       # 200ms speech onset
        min_silence_chunks=7,      # 700ms natural speech pause
        pre_roll_chunks=4,         # 400ms pre-trigger history
    )

    print("=" * 70)
    print("🚀 LIVE SPEECH-TO-LATEX TERMINAL STREAMER READY")
    print("=" * 70)
    print("Speak mathematical phrases or lecture sentences into your microphone.")
    print("Spoken words will appear LIVE in real-time as you talk.")
    print("Natural pauses (~700ms) will lock in the final formatted LaTeX equation.")
    print("Press Ctrl+C to exit.")
    print("=" * 70 + "\n")
    print(f"{DIM}🎧 Listening...{RESET}\n")

    current_speech_chunks = []
    in_speech = False
    last_transcribe_time = 0.0
    last_partial_text = ""

    # Streaming filter initial conditions for seamless chunk continuity
    filter_zi = signal.sosfilt_zi(SOS_HPF_16K)

    try:
        while True:
            raw_bytes = stream.read(chunk_size, exception_on_overflow=False)

            if rate == 48000 and channels == 2:
                mono_chunk = stereo_48k_to_mono_16k(raw_bytes)
            else:
                pcm_data = np.frombuffer(raw_bytes, dtype=np.int16)
                mono_chunk = np.clip(pcm_data / 32768.0, -1.0, 1.0).astype(np.float32)

            # Continuous high-pass filter (removes 50/100 Hz AC hum with zero boundary clicks)
            filtered_chunk, filter_zi = signal.sosfilt(SOS_HPF_16K, mono_chunk, zi=filter_zi)

            state, initial_chunks = vad.process(filtered_chunk)

            if state == "start":
                in_speech = True
                # Prepend the pre-roll history to avoid clipping initial consonants
                current_speech_chunks = list(initial_chunks) if initial_chunks else [filtered_chunk]
                last_partial_text = ""
                last_transcribe_time = time.monotonic()
                sys.stdout.write(f"{CLEAR_LINE}{YELLOW_BOLD}🎙️  [Speaking...]{RESET} ")
                sys.stdout.flush()

            elif in_speech:
                current_speech_chunks.append(filtered_chunk)
                now = time.monotonic()

                # Run live partial ASR every 380ms, but only once >= 800ms has accumulated
                accumulated_samples = sum(len(c) for c in current_speech_chunks)
                if (accumulated_samples >= 12800) and (now - last_transcribe_time >= 0.38):
                    accumulated = np.concatenate(current_speech_chunks)
                    # Limit rolling ASR window to last 5 seconds for bounded edge latency
                    if len(accumulated) > 16000 * 5:
                        accumulated = accumulated[-16000 * 5:]

                    partial_text, asr_ms = query_whisper_stream(accumulated)
                    if partial_text and partial_text != last_partial_text:
                        last_partial_text = partial_text
                        elapsed_sec = round(len(accumulated) / 16000.0, 1)
                        sys.stdout.write(
                            f"{CLEAR_LINE}{YELLOW_BOLD}🎙️  [LIVE {elapsed_sec:.1f}s]:{RESET} {CYAN_BOLD}{partial_text}{RESET}"
                        )
                        sys.stdout.flush()
                    last_transcribe_time = now

            if state == "end" and in_speech:
                in_speech = False
                full_utterance = np.concatenate(current_speech_chunks)
                total_duration = len(full_utterance) / 16000.0

                # Reject short acoustic transients (< 0.60s like coughs or clicks)
                if total_duration < 0.60:
                    sys.stdout.write(f"{CLEAR_LINE}{DIM}🎧 Listening...{RESET}\r")
                    sys.stdout.flush()
                    current_speech_chunks = []
                    last_partial_text = ""
                    continue

                final_text, final_asr_ms = query_whisper_stream(full_utterance)
                if not final_text:
                    final_text = last_partial_text

                if final_text.strip():
                    sys.stdout.write(f"{CLEAR_LINE}{WHITE_BOLD}🗣️  Spoken:{RESET} \"{final_text}\"\n")
                    sys.stdout.flush()

                    # Trigger Qwen2.5-1.5B SLM LaTeX translation
                    t_slm_start = time.monotonic()
                    latex_output = slm.translate(final_text)
                    slm_ms = (time.monotonic() - t_slm_start) * 1000.0

                    print(f"{GREEN_BOLD}✨ LaTeX: {RESET}{GREEN_BOLD}{latex_output}{RESET}")
                    print(f"{DIM}⚡ Latency: ASR: {final_asr_ms:.1f}ms | SLM: {slm_ms:.1f}ms | Total Audio: {total_duration:.2f}s{RESET}\n")
                    print(f"{DIM}🎧 Listening...{RESET}\n")

                current_speech_chunks = []
                last_partial_text = ""

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
    filtered_audio = signal.sosfilt(SOS_HPF_16K, float_audio)

    total_sec = len(filtered_audio) / 16000.0
    print("\n" + "=" * 70)
    print(f"🎬 SIMULATING LIVE MICROPHONE STREAMING FROM: {p.name}")
    print(f"⏱️  Duration: {total_sec:.2f}s | Sample Rate: 16 kHz")
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
                    f"{CLEAR_LINE}{YELLOW_BOLD}🎙️  [LIVE {elapsed:.1f}s]:{RESET} {CYAN_BOLD}{partial_text}{RESET}"
                )
                sys.stdout.flush()

    # Final complete utterance lock-in
    full_audio = np.concatenate(current_buffer)
    final_text, final_asr_ms = query_whisper_stream(full_audio)
    if not final_text:
        final_text = last_partial_text

    sys.stdout.write(f"{CLEAR_LINE}{WHITE_BOLD}🗣️  Final Spoken Transcript:{RESET} \"{final_text}\"\n")
    sys.stdout.flush()

    print(f"\n{YELLOW_BOLD}[SLM]{RESET} Formatting with Qwen2.5-1.5B (CUDA SM87)...")
    t_slm_start = time.monotonic()
    latex_output = slm.translate(final_text)
    slm_ms = (time.monotonic() - t_slm_start) * 1000.0

    print("\n" + "=" * 70)
    print(f"{GREEN_BOLD}✨ FINAL LATEX OUTPUT:{RESET} {GREEN_BOLD}{latex_output}{RESET}")
    print(f"{DIM}⚡ Latency Profile: Final ASR: {final_asr_ms:.1f}ms | SLM: {slm_ms:.1f}ms | Total Audio: {total_sec:.2f}s{RESET}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Live Streaming Speech-to-LaTeX Transcriber")
    parser.add_argument("--file", type=str, default=None, help="Path to WAV audio file to simulate live streaming")
    parser.add_argument("--mic", action="store_true", help="Force microphone input")
    args = parser.parse_args()

    # Ensure background GPU daemons
    ensure_edge_daemons()

    # Initialize SLM translator client
    slm = QwenMathTranslator.get_instance(backend="server")

    if args.file:
        run_file_simulation(args.file, slm)
    else:
        run_live_microphone(slm)


if __name__ == "__main__":
    main()
