#!/usr/bin/env python3
"""
Real-Time Live Streaming Speech-to-LaTeX Transcriber for NVIDIA Jetson Orin Nano.

Provides live word-by-word streaming transcription in the terminal as speech occurs,
followed by instant mathematical LaTeX lock-in upon natural speech pause.

Architecture:
  - Microphone Capture: PyAudio (48kHz stereo -> 16kHz mono vectorized decimation)
  - Streaming ASR: whisper-server HTTP daemon (GPU Flash Attention, in-memory)
  - Math & Prose Formatting: Qwen2.5-1.5B-Instruct SLM (llama-server CUDA SM87 daemon)
  - Visuals: Live ANSI in-place line overwriting (\r\033[K)

Usage:
  python src/live_streaming_terminal.py                  # Live microphone
  python src/live_streaming_terminal.py --mic            # Explicit microphone mode
  python src/live_streaming_terminal.py --file audio.wav # Test simulation on WAV file
"""

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
import wave

import numpy as np
import pyaudio

# Ensure src/ is in sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audio import stereo_48k_to_mono_16k
from energy_vad import EnergyVAD
from slm_translator import QwenMathTranslator

WHISPER_SERVER_URL = "http://127.0.0.1:8081"
LLAMA_SERVER_URL = "http://127.0.0.1:8080"
WHISPER_BIN = "/mnt/thesis/main/whisper.cpp/build-orin/bin/whisper-server"
WHISPER_MODEL = "/mnt/thesis/models/whisper/ggml-base.bin"
LLAMA_BIN = "/mnt/thesis/llama.cpp/build/bin/llama-server"
LLAMA_MODEL = "/mnt/thesis/models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Terminal ANSI Styling
CYAN_BOLD = "\033[1;36m"
GREEN_BOLD = "\033[1;32m"
YELLOW_BOLD = "\033[1;33m"
MAGENTA_BOLD = "\033[1;35m"
WHITE_BOLD = "\033[1;37m"
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
        print(f"{YELLOW_BOLD}[Init]{RESET} Starting whisper-server daemon on port 8081 (Flash Attention)...")
        cmd = [
            WHISPER_BIN,
            "-m", WHISPER_MODEL,
            "--port", "8081",
            "--host", "127.0.0.1",
            "-fa", "-nt", "-bo", "1", "-bs", "1", "-t", "6"
        ]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/mnt/thesis/main/whisper.cpp/build-orin/bin:" + env.get("LD_LIBRARY_PATH", "")
        subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(15):
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


def query_whisper_stream(pcm_samples: np.ndarray) -> tuple[str, float]:
    """Send in-memory PCM float32 samples to whisper-server and return recognized text."""
    if len(pcm_samples) < 3200:  # Minimum 200ms
        return "", 0.0

    t0 = time.monotonic()
    # Convert float32 [-1, 1] to signed 16-bit PCM
    int16_samples = np.clip(pcm_samples * 32767, -32768, 32767).astype(np.int16)

    # Encode to in-memory WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(int16_samples.tobytes())
    wav_bytes = buf.getvalue()

    # Form multipart request
    boundary = "----JetsonLiveStreamBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="stream.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8") + wav_bytes + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="response_format"\r\n\r\n'
        f"json\r\n--{boundary}--\r\n"
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{WHISPER_SERVER_URL}/inference",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data.get("text", "").strip()
            # Filter out non-speech noise tags like [BLANK_AUDIO], [inaudible], (music)
            if text in ("[BLANK_AUDIO]", "[inaudible]", "(crickets chirping)", "[silence]"):
                text = ""
            latency_ms = (time.monotonic() - t0) * 1000.0
            return text, latency_ms
    except Exception:
        return "", 0.0


def run_live_microphone(slm: QwenMathTranslator):
    """Run real-time microphone capture with live word-by-word streaming."""
    pa = pyaudio.PyAudio()

    # Auto-detect AB13X USB Audio or default microphone
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

    vad = EnergyVAD(threshold=0.012, min_speech_chunks=2, min_silence_chunks=8)

    print("\n" + "=" * 70)
    print("🚀 LIVE SPEECH-TO-LATEX TERMINAL STREAMER READY")
    print("=" * 70)
    print("Speak mathematical phrases or lecture sentences into your microphone.")
    print("Spoken words will appear LIVE in real-time as you talk.")
    print("Natural pauses (~600ms) will lock in the final formatted LaTeX equation.")
    print("Press Ctrl+C to exit.")
    print("=" * 70 + "\n")
    print(f"{DIM}🎧 Listening...{RESET}\n")

    current_speech_chunks = []
    in_speech = False
    last_transcribe_time = 0.0
    last_partial_text = ""

    try:
        while True:
            raw_bytes = stream.read(chunk_size, exception_on_overflow=False)

            if rate == 48000 and channels == 2:
                mono_chunk = stereo_48k_to_mono_16k(raw_bytes)
            else:
                pcm_data = np.frombuffer(raw_bytes, dtype=np.int16)
                mono_chunk = np.clip(pcm_data / 32768.0, -1.0, 1.0).astype(np.float32)

            state = vad.process(mono_chunk)

            if state == "start":
                in_speech = True
                current_speech_chunks = [mono_chunk]
                last_partial_text = ""
                last_transcribe_time = time.monotonic()
                sys.stdout.write(f"{CLEAR_LINE}{YELLOW_BOLD}🎙️  [Speaking...]{RESET} ")
                sys.stdout.flush()

            elif in_speech:
                current_speech_chunks.append(mono_chunk)
                now = time.monotonic()

                # Run live partial ASR every 400ms while user is speaking
                if now - last_transcribe_time >= 0.40:
                    accumulated = np.concatenate(current_speech_chunks)
                    # Limit rolling ASR window to last 5 seconds for bounded latency
                    if len(accumulated) > 16000 * 5:
                        accumulated = accumulated[-16000 * 5:]

                    partial_text, asr_ms = query_whisper_stream(accumulated)
                    if partial_text and partial_text != last_partial_text:
                        last_partial_text = partial_text
                        sys.stdout.write(f"{CLEAR_LINE}{YELLOW_BOLD}🎙️  [LIVE]:{RESET} {CYAN_BOLD}{partial_text}{RESET}")
                        sys.stdout.flush()
                    last_transcribe_time = now

            if state == "end" and in_speech:
                in_speech = False
                full_utterance = np.concatenate(current_speech_chunks)
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
                    print(f"{DIM}⚡ Latency: ASR: {final_asr_ms:.1f}ms | SLM: {slm_ms:.1f}ms | Total: {final_asr_ms + slm_ms:.1f}ms{RESET}\n")
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

    total_sec = len(float_audio) / 16000.0
    print("\n" + "=" * 70)
    print(f"🎬 SIMULATING LIVE MICROPHONE STREAMING FROM: {p.name}")
    print(f"⏱️  Duration: {total_sec:.2f}s | Sample Rate: 16 kHz")
    print("=" * 70 + "\n")

    chunk_samples = 1600  # 100ms chunks
    current_buffer = []
    last_print_time = 0.0
    last_partial_text = ""

    t_start = time.monotonic()

    for idx, i in enumerate(range(0, len(float_audio), chunk_samples)):
        chunk = float_audio[i : i + chunk_samples]
        current_buffer.append(chunk)

        # Simulate real-time audio playback cadence (100ms)
        time.sleep(0.08)

        # Every 400ms, run progressive Whisper ASR
        if (idx % 4 == 0) and len(current_buffer) >= 4:
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
