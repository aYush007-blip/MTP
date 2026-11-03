#!/usr/bin/env python3
"""
Headless Autonomous Voice-Activated Edge Agent for NVIDIA Jetson Orin Nano.

Zero-Touch, 100% Offline Workflow:
  1. Boot-up auto-configuration: mic detection (AB13X USB Audio hw:2,0),
     gain calibration, and GPU daemon health validation.
  2. Spoken greeting: "Jetson speech agent is ready. Say hey Jetson start recording to begin."
  3. Passive wake-word listener ("Hey Jetson start recording").
  4. Spoken countdown: "Ok, starting recording in 5 4 3 2 1 seconds, recording started."
  5. Continuous lecture streaming, VAD chunking, Whisper ASR + Qwen SLM formatting.
  6. Spoken stopping trigger ("Hey Jetson stop recording"):
     -> "Recording stopped. Processing lecture, please wait."
  7. Post-lecture global AI review, numbered display equations, and TikZ vector graphs.
  8. Standalone PDF compilation.
  9. Automated USB drive export to /media/ayush02/AYUSH/Lectures/<timestamp>/ + sync.
 10. Spoken shutdown signal: "Transcription successfully saved. You can turn off the system."
"""

import argparse
import collections
from enum import Enum
import io
import json
import os
from pathlib import Path
import re
import select
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple
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
from edge_tts import play_prompt, speak
from energy_vad import AdaptiveEnergyVAD
from lecture_to_book import LectureToBookBuilder, TranscriptChunk, generate_subtitles
from postprocess_lecture_ai import PostLectureProcessor
from slm_translator import QwenMathTranslator
from usb_exporter import LectureExporter, find_usb_mount

WHISPER_SERVER_URL = "http://127.0.0.1:8081"
LLAMA_SERVER_URL = "http://127.0.0.1:8080"
WHISPER_BIN = "/mnt/thesis/main/whisper.cpp/build-orin/bin/whisper-server"
WHISPER_MODEL = "/mnt/thesis/models/whisper/ggml-small.en.bin"
LLAMA_BIN = "/mnt/thesis/llama.cpp/build/bin/llama-server"
LLAMA_MODEL = "/mnt/thesis/models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"

WHISPER_MATH_PROMPT = (
    "Fourier transform, Laplace, Z-transform, region of convergence, continuous-time signal, "
    "discrete-time, integral, derivative, calculus, omega, theta, phi, pi, infinity, x(t), f(x), "
    "e to the power, differential equation, matrix, vector, sinusoids, Park transform, direct axis, quadrature axis."
)

START_TRIGGERS = [
    "hey jetson start recording",
    "jetson start recording",
    "start recording",
    "hey jetson start lecture",
    "jetson start lecture",
    "start lecture",
    "hey jetson start",
    "jetson start",
    "start",
    "begin recording",
    "begin lecture",
]

STOP_TRIGGERS = [
    "hey jetson stop recording",
    "jetson stop recording",
    "stop recording",
    "hey jetson stop lecture",
    "jetson stop lecture",
    "stop lecture",
    "hey jetson stop",
    "jetson stop",
    "stop",
    "finish recording",
    "end recording",
    "turn off recording",
]


class AgentState(Enum):
    STANDBY = "STANDBY"
    COUNTDOWN = "COUNTDOWN"
    RECORDING = "RECORDING"
    PROCESSING = "PROCESSING"


class HighPassHumFilter:
    """Cascaded 4th-order DC-blocking & 120 Hz mains hum attenuation filter."""
    def __init__(self, r: float = 0.953):
        self.r = r
        self.x1_prev = self.y1_prev = self.x2_prev = self.y2_prev = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        y1 = np.empty_like(x)
        xp1, yp1, r = self.x1_prev, self.y1_prev, self.r
        for i in range(len(x)):
            y1[i] = x[i] - xp1 + r * yp1
            yp1, xp1 = y1[i], x[i]
        self.x1_prev, self.y1_prev = xp1, yp1

        y2 = np.empty_like(y1)
        xp2, yp2 = self.x2_prev, self.y2_prev
        for i in range(len(y1)):
            y2[i] = y1[i] - xp2 + r * yp2
            yp2, xp2 = y2[i], y1[i]
        self.x2_prev, self.y2_prev = xp2, yp2
        return y2

    def reset(self):
        self.x1_prev = self.y1_prev = self.x2_prev = self.y2_prev = 0.0


def is_service_alive(url: str, endpoint: str = "/health") -> bool:
    try:
        req = urllib.request.Request(f"{url}{endpoint}", method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def ensure_edge_daemons():
    """Ensure whisper-server and llama-server are running."""
    # 1. whisper-server
    if not is_service_alive(WHISPER_SERVER_URL, endpoint="/"):
        print("[Agent] Starting whisper-server daemon on port 8081...")
        cmd = [
            WHISPER_BIN,
            "-m", WHISPER_MODEL,
            "--port", "8081",
            "--host", "127.0.0.1",
            "-fa", "-nt", "-bo", "1", "-bs", "1", "-t", "4",
            "-sns", "-nth", "0.65"
        ]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/mnt/thesis/main/whisper.cpp/build-orin/bin:" + env.get("LD_LIBRARY_PATH", "")
        log_whisper = open("/tmp/whisper-server.log", "w")
        subprocess.Popen(cmd, env=env, stdout=log_whisper, stderr=log_whisper, start_new_session=True)
        for _ in range(60):
            time.sleep(1.0)
            if is_service_alive(WHISPER_SERVER_URL, endpoint="/"):
                break

    # 2. llama-server
    if not is_service_alive(LLAMA_SERVER_URL, endpoint="/health"):
        print("[Agent] Starting llama-server daemon on port 8080...")
        cmd = [
            LLAMA_BIN,
            "-m", LLAMA_MODEL,
            "-c", "2048",
            "-ngl", "28",
            "--port", "8080",
            "--host", "127.0.0.1"
        ]
        log_llama = open("/tmp/llama-server.log", "w")
        subprocess.Popen(cmd, stdout=log_llama, stderr=log_llama, start_new_session=True)
        for _ in range(60):
            time.sleep(1.0)
            if is_service_alive(LLAMA_SERVER_URL, endpoint="/health"):
                break


def query_whisper_stream(pcm_samples: np.ndarray, no_speech_thold: float = 0.40) -> str:
    """Send PCM float32 samples to whisper-server and return cleaned text."""
    if len(pcm_samples) < 6400:  # < 400ms
        return ""

    int16_samples = np.clip(pcm_samples * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(int16_samples.tobytes())
    wav_bytes = buf.getvalue()

    boundary = "----JetsonHeadlessAgentBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="stream.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode("utf-8")
        + wav_bytes
        + f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="prompt"\r\n\r\n'
        f"{WHISPER_MATH_PROMPT}\r\n".encode("utf-8")
        + f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="no_speech_thold"\r\n\r\n'
        f"{no_speech_thold}\r\n".encode("utf-8")
        + f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="response_format"\r\n\r\n'
        f"json\r\n--{boundary}--\r\n".encode("utf-8")
    )

    req = urllib.request.Request(
        f"{WHISPER_SERVER_URL}/inference",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    try:
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw = data.get("text", "").strip()
            # Clean hallucination brackets and music tags
            cleaned = re.sub(r'[\(\[\{].*?[\)\]\}]', '', raw)
            cleaned = re.sub(r'[♪♫]+', '', cleaned).strip()
            if not any(c.isalnum() for c in cleaned):
                return ""
            return cleaned
    except Exception:
        return ""


def clean_trigger_text(text: str) -> str:
    """Normalize text for voice trigger matching."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


class HeadlessVoiceAgent:
    """
    Main edge controller for zero-touch speech-to-LaTeX monograph generation.
    """
    def __init__(self):
        self.state = AgentState.STANDBY
        self.exporter = LectureExporter()
        self.book_builder = LectureToBookBuilder()
        self.post_processor = PostLectureProcessor()
        self.slm = None
        self.hpf = HighPassHumFilter()

    def initialize_hardware(self) -> Tuple[pyaudio.PyAudio, int, int, int]:
        """Detect and configure USB mic and audio subsystem."""
        # Optimal mic gain
        try:
            subprocess.run(["amixer", "-c", "2", "sset", "Mic", "110"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass

        pa = pyaudio.PyAudio()
        device_index = None
        channels = 1
        rate = 16000

        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
                name = info.get("name", "").lower()
                if "ab13x" in name and info.get("maxInputChannels", 0) > 0:
                    device_index = i
                    channels = 2
                    rate = 48000
                    print(f"[Agent] Bound to USB Microphone: {info['name']} (Index {i})")
                    break
            except Exception:
                pass

        if device_index is None:
            print("[Agent] USB Mic not found, falling back to system default input.")
            try:
                default_info = pa.get_default_input_device_info()
                device_index = default_info.get("index", 0)
                rate = int(default_info.get("defaultSampleRate", 16000))
                channels = int(default_info.get("maxInputChannels", 1))
            except Exception:
                device_index = 0

        return pa, device_index, channels, rate

    def run_continuous_loop(self):
        """
        Master background loop listening for voice commands and recording lectures.
        """
        print("[Agent] Initializing daemons and audio subsystems...")
        ensure_edge_daemons()
        self.slm = QwenMathTranslator.get_instance(backend="server")

        pa, dev_idx, channels, rate = self.initialize_hardware()

        print("[Agent] Announcing system ready via TTS...", flush=True)
        play_prompt("system_ready")

        # Verify USB drive presence
        usb_root = find_usb_mount()
        if not usb_root:
            print("[Agent] Notice: USB drive AYUSH not currently inserted.")
            play_prompt("usb_warning", non_blocking=True)
        else:
            print(f"[Agent] USB Drive detected at: {usb_root}")

        chunk_samples = 4800 if rate == 48000 else 1600
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=dev_idx,
            frames_per_buffer=chunk_samples,
        )

        # Calibrate ambient noise floor
        print("[Agent] Calibrating microphone noise floor...", flush=True)
        time.sleep(0.2)
        noise_samples = []
        for _ in range(8):
            b = stream.read(chunk_samples, exception_on_overflow=False)
            if rate == 48000 and channels == 2:
                mc = stereo_48k_to_mono_16k(b)
            else:
                p_arr = np.frombuffer(b, dtype=np.int16)
                mc = np.clip(p_arr / 32768.0, -1.0, 1.0).astype(np.float32)
            fc = self.hpf.process(mc)
            noise_samples.append(float(np.sqrt(np.mean(fc ** 2))))

        noise_floor = min(max(float(np.median(noise_samples)), 0.003), 0.020)
        speech_thold = min(max(noise_floor * 2.5, 0.020), 0.045)
        silence_thold = min(max(noise_floor * 1.4, 0.012), 0.025)
        print(f"[Agent] Calibrated: Noise floor={noise_floor:.4f} | Speech Trigger={speech_thold:.4f}", flush=True)

        vad = AdaptiveEnergyVAD(speech_threshold=speech_thold, silence_threshold=silence_thold)

        # Buffers
        current_speech_chunks = []
        pre_roll_buffer = collections.deque(maxlen=3)
        lecture_chunks: List[TranscriptChunk] = []
        raw_audio_frames = []

        lecture_start_time = 0.0
        in_speech = False
        silence_chunk_count = 0

        print(f"\n[Agent] Standby mode active. Say 'Hey Jetson start recording' (or press Enter) to begin...", flush=True)

        try:
            while True:
                # Non-blocking terminal keyboard check
                try:
                    if sys.stdin.isatty():
                        r, _, _ = select.select([sys.stdin], [], [], 0.0)
                        if r:
                            sys.stdin.readline()
                            if self.state == AgentState.STANDBY:
                                print("\n[Agent] Keyboard Enter detected! Starting countdown...", flush=True)
                                self.state = AgentState.COUNTDOWN
                                stream.stop_stream()
                                play_prompt("countdown_start")
                                stream.start_stream()
                                current_speech_chunks.clear()
                                pre_roll_buffer.clear()
                                lecture_chunks.clear()
                                raw_audio_frames.clear()
                                self.hpf.reset()
                                vad.reset()
                                lecture_start_time = time.monotonic()
                                self.state = AgentState.RECORDING
                                print("[Agent] Recording started! Capturing live lecture...", flush=True)
                                in_speech = False
                                continue
                            elif self.state == AgentState.RECORDING:
                                print("\n[Agent] Keyboard Enter detected! Stopping recording...", flush=True)
                                self.state = AgentState.PROCESSING
                                stream.stop_stream()
                                play_prompt("recording_stopped")
                                current_speech_chunks.clear()
                                pre_roll_buffer.clear()
                                in_speech = False
                                vad.reset()
                                self.process_lecture(lecture_chunks, raw_audio_frames, lecture_start_time)
                                self.state = AgentState.STANDBY
                                stream.start_stream()
                                print("\n[Agent] Standby mode active. Say 'Hey Jetson start recording' to begin...", flush=True)
                                continue
                except Exception:
                    pass

                raw_bytes = stream.read(chunk_samples, exception_on_overflow=False)
                
                # Decimate to 16 kHz mono float32
                if rate == 48000 and channels == 2:
                    mono_chunk = stereo_48k_to_mono_16k(raw_bytes)
                else:
                    pcm = np.frombuffer(raw_bytes, dtype=np.int16)
                    mono_chunk = np.clip(pcm / 32768.0, -1.0, 1.0).astype(np.float32)

                filtered_chunk = self.hpf.process(mono_chunk)

                # =========================================================
                # 1. STANDBY STATE (Listening for Wake-Word)
                # =========================================================
                if self.state == AgentState.STANDBY:
                    is_speech, energy = vad.process_chunk(filtered_chunk)

                    if is_speech:
                        if not in_speech:
                            in_speech = True
                            current_speech_chunks.extend(pre_roll_buffer)
                            pre_roll_buffer.clear()
                        current_speech_chunks.append(filtered_chunk)
                        silence_chunk_count = 0
                        sys.stdout.write(f"\r[Listening] Voice detected (RMS={energy:.3f})... ")
                        sys.stdout.flush()
                    else:
                        if not in_speech:
                            pre_roll_buffer.append(filtered_chunk)
                        else:
                            silence_chunk_count += 1
                            current_speech_chunks.append(filtered_chunk)

                            # Speech boundary (0.6s silence or max 4.5s phrase)
                            if silence_chunk_count >= 6 or len(current_speech_chunks) >= 45:
                                utterance = np.concatenate(current_speech_chunks)
                                spoken = query_whisper_stream(utterance, no_speech_thold=0.30)
                                norm_spoken = clean_trigger_text(spoken)

                                if spoken:
                                    print(f"\n[Standby Heard]: \"{spoken}\"", flush=True)

                                # Check triggers: exact match, or contains "start", or "jetson" and "record"
                                is_start = any(trig in norm_spoken for trig in START_TRIGGERS) or \
                                           ("start" in norm_spoken and ("recording" in norm_spoken or "lecture" in norm_spoken)) or \
                                           ("jetson" in norm_spoken and "start" in norm_spoken) or \
                                           (norm_spoken.startswith("start") and len(norm_spoken.split()) <= 3)

                                if is_start:
                                    print(f"[Agent] Wake-Word Activated! Starting countdown...", flush=True)
                                    self.state = AgentState.COUNTDOWN
                                    stream.stop_stream()
                                    play_prompt("countdown_start")
                                    stream.start_stream()

                                    current_speech_chunks.clear()
                                    pre_roll_buffer.clear()
                                    lecture_chunks.clear()
                                    raw_audio_frames.clear()
                                    self.hpf.reset()
                                    vad.reset()

                                    lecture_start_time = time.monotonic()
                                    self.state = AgentState.RECORDING
                                    print("[Agent] Recording started! Capturing live lecture...", flush=True)
                                    in_speech = False
                                    continue
                                else:
                                    if spoken:
                                        print("[Standby] (Say 'Hey Jetson start recording' or 'Start recording' to begin)", flush=True)

                                current_speech_chunks.clear()
                                pre_roll_buffer.clear()
                                in_speech = False
                                vad.reset()

                # =========================================================
                # 2. RECORDING STATE (Continuous Lecture Transcription)
                # =========================================================
                elif self.state == AgentState.RECORDING:
                    # Accumulate raw audio for archive
                    int16_pcm = np.clip(filtered_chunk * 32767, -32768, 32767).astype(np.int16)
                    raw_audio_frames.append(int16_pcm.tobytes())

                    is_speech, energy = vad.process_chunk(filtered_chunk)

                    if is_speech:
                        if not in_speech:
                            in_speech = True
                            current_speech_chunks.extend(pre_roll_buffer)
                            pre_roll_buffer.clear()
                        current_speech_chunks.append(filtered_chunk)
                        silence_chunk_count = 0
                        elapsed = time.monotonic() - lecture_start_time
                        sys.stdout.write(f"\r[Recording {elapsed:5.1f}s] Voice RMS: {energy:.3f} ... ")
                        sys.stdout.flush()
                    else:
                        if not in_speech:
                            pre_roll_buffer.append(filtered_chunk)
                        else:
                            silence_chunk_count += 1
                            current_speech_chunks.append(filtered_chunk)

                            # Speech boundary reached (1.6s silence or max 12s utterance)
                            chunk_duration = len(current_speech_chunks) * 0.1
                            if silence_chunk_count >= 16 or chunk_duration >= 12.0:
                                utterance = np.concatenate(current_speech_chunks)
                                t_end_rel = time.monotonic() - lecture_start_time
                                t_start_rel = max(0.0, t_end_rel - chunk_duration)

                                # Transcribe chunk via Whisper
                                spoken_text = query_whisper_stream(utterance, no_speech_thold=0.40)
                                norm_spoken = clean_trigger_text(spoken_text)

                                # Check for STOP command
                                is_stop = any(stop_t in norm_spoken for stop_t in STOP_TRIGGERS) or \
                                          ("stop" in norm_spoken and ("recording" in norm_spoken or "lecture" in norm_spoken)) or \
                                          ("jetson" in norm_spoken and "stop" in norm_spoken) or \
                                          (norm_spoken.startswith("stop") and len(norm_spoken.split()) <= 3)

                                if is_stop:
                                    print(f"\n[Agent] Stop-Command Detected: '{spoken_text}'", flush=True)
                                    self.state = AgentState.PROCESSING
                                    stream.stop_stream()
                                    play_prompt("recording_stopped")
                                    current_speech_chunks.clear()
                                    pre_roll_buffer.clear()
                                    in_speech = False
                                    vad.reset()
                                    # Process immediately
                                    self.process_lecture(lecture_chunks, raw_audio_frames, lecture_start_time)
                                    # Re-enter standby
                                    self.state = AgentState.STANDBY
                                    stream.start_stream()
                                    print("\n[Agent] Standby mode active. Say 'Hey Jetson start recording' to begin...", flush=True)
                                    continue

                                # Standard Technical Speech Chunk
                                if spoken_text and len(spoken_text) > 2:
                                    # Run Qwen SLM math translation
                                    latex_output = self.slm.translate(spoken_text)
                                    chunk_obj = TranscriptChunk(
                                        timestamp_start=t_start_rel,
                                        timestamp_end=t_end_rel,
                                        spoken_text=spoken_text,
                                        latex_text=latex_output
                                    )
                                    lecture_chunks.append(chunk_obj)
                                    print(f"\n[Live {t_end_rel:.1f}s]: {spoken_text}", flush=True)
                                    print(f"       -> [LaTeX]: {latex_output}", flush=True)

                                current_speech_chunks.clear()
                                pre_roll_buffer.clear()
                                in_speech = False
                                vad.reset()

        except KeyboardInterrupt:
            print("\n[Agent] Shutting down headless agent...", flush=True)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def process_lecture(
        self,
        chunks: List[TranscriptChunk],
        raw_audio_frames: List[bytes],
        start_time: float
    ):
        """
        Post-processes lecture, formats textbook, compiles PDF, and exports to USB drive.
        """
        print("\n[Agent] Synthesizing textbook monograph with TikZ vector graphics...")
        ts = time.strftime("%Y%m%d_%H%M%S")
        lecture_title = f"Lecture_{ts}"

        # 1. Build Textbook TeX, Markdown, and Subtitles
        tex_content, md_content = self.book_builder.build_book(lecture_title, chunks)
        srt_content, vtt_content = generate_subtitles(chunks)

        # Temporary working directory for generation
        temp_dir = Path("/tmp") / f"lecture_{ts}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        tex_file = temp_dir / f"{lecture_title}.tex"
        tex_file.write_text(tex_content, encoding="utf-8")

        md_file = temp_dir / f"{lecture_title}.md"
        md_file.write_text(md_content, encoding="utf-8")

        srt_file = temp_dir / f"{lecture_title}.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        vtt_file = temp_dir / f"{lecture_title}.vtt"
        vtt_file.write_text(vtt_content, encoding="utf-8")

        # Save raw audio WAV file
        wav_file = temp_dir / f"{lecture_title}_raw.wav"
        if raw_audio_frames:
            with wave.open(str(wav_file), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"".join(raw_audio_frames))

        # 2. Global AI Review & PDF Compilation Pass
        print("[Agent] Running Global AI review and PDF compiler pass...")
        post_res = self.post_processor.process_and_compile(
            lecture_title=lecture_title,
            tex_path=tex_file,
            output_dir=temp_dir,
            raw_transcript="\n".join([c.spoken_text for c in chunks])
        )

        pdf_file = post_res.get("pdf")

        # 3. Export to USB Drive AYUSH & Local SSD
        print("[Agent] Exporting deliverables to USB drive AYUSH...")
        export_map = {
            "tex": tex_file,
            "md": md_file,
            "srt": srt_file,
            "vtt": vtt_file,
            "wav": wav_file if wav_file.is_file() else None,
            "pdf": pdf_file if (pdf_file and pdf_file.is_file()) else None
        }

        export_res = self.exporter.export(lecture_title, export_map, timestamp=ts)
        print(f"[Agent] Export Complete:")
        print(f"       Local SSD: {export_res['local_path']}")
        print(f"       USB Drive: {export_res['usb_path']} (Success: {export_res['usb_success']})")

        # 4. Spoken Shutdown Signal
        print("[Agent] Playing final announcement...")
        play_prompt("transcription_saved")
        # Clean up temp
        shutil.rmtree(temp_dir, ignore_errors=True)

    def run_file_simulation(self, wav_path: str, max_duration: float = 60.0):
        """
        Simulate the complete headless workflow on an existing WAV file.
        """
        print(f"\n[Agent Simulation] Testing on WAV file: {wav_path}", flush=True)
        ensure_edge_daemons()
        self.slm = QwenMathTranslator.get_instance(backend="server")

        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            framerate = wf.getframerate()
            total_frames = wf.getnframes()
            frames_to_read = int(min(total_frames, max_duration * framerate))
            raw_bytes = wf.readframes(frames_to_read)

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
        filtered_audio = self.hpf.process(float_audio)

        print(f"[Agent Simulation] Processing {len(filtered_audio)/16000.0:.1f}s of lecture audio...", flush=True)
        chunk_sec = 6.0
        samples_per_chunk = int(chunk_sec * 16000)
        chunks: List[TranscriptChunk] = []

        total_audio_sec = len(filtered_audio) / 16000.0
        for i in range(0, len(filtered_audio), samples_per_chunk):
            segment = filtered_audio[i : i + samples_per_chunk]
            t_start = i / 16000.0
            t_end = min((i + samples_per_chunk) / 16000.0, total_audio_sec)

            spoken = query_whisper_stream(segment)
            if spoken.strip():
                latex = self.slm.translate(spoken)
                chunks.append(TranscriptChunk(t_start, t_end, spoken, latex))
                print(f"[{t_start:.1f}s - {t_end:.1f}s]: {spoken}", flush=True)
                print(f"       -> [LaTeX]: {latex}", flush=True)

        # Post-process and export
        raw_frames = [np.clip(filtered_audio * 32767, -32768, 32767).astype(np.int16).tobytes()]
        self.process_lecture(chunks, raw_frames, time.monotonic())


def main():
    parser = argparse.ArgumentParser(description="Headless Autonomous Speech-to-LaTeX Edge Agent")
    parser.add_argument("--sim-file", type=str, default=None, help="Test simulation on a WAV audio file")
    args = parser.parse_args()

    agent = HeadlessVoiceAgent()
    if args.sim_file:
        agent.run_file_simulation(args.sim_file)
    else:
        agent.run_continuous_loop()


if __name__ == "__main__":
    main()
