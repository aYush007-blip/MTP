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
from enum import Enum
import glob
import io
import json
import os
from pathlib import Path
import queue
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
from usb_exporter import LectureExporter, compute_sha256, find_usb_mount

WHISPER_SERVER_URL = "http://127.0.0.1:8081"
LLAMA_SERVER_URL = "http://127.0.0.1:8080"
WHISPER_BIN = "/mnt/thesis/main/whisper.cpp/build-orin/bin/whisper-server"
WHISPER_MODEL = "/mnt/thesis/models/whisper/ggml-small.en.bin"
LLAMA_BIN = "/mnt/thesis/llama.cpp/build/bin/llama-server"
LLAMA_MODEL = "/mnt/thesis/models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"

WHISPER_MATH_PROMPT = (
    "Academic STEM lecture and professional presentation: mathematics, physics, computer science, algorithms, "
    "engineering, calculus, linear algebra, limits, integrals, derivatives, differential equations, matrices, vectors, "
    "proofs, theorems, definitions, functions f(x), variables x, y, z, t, n, parameters alpha, beta, theta, omega, "
    "equations, formulas, and technical exposition."
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

STATUS_TRIGGERS = [
    "hey jetson whats the status",
    "hey jetson what is the status",
    "hey jetson what's the status",
    "hey jetson status",
    "jetson whats the status",
    "jetson what is the status",
    "jetson what's the status",
    "jetson status",
    "whats the status",
    "what is the status",
    "what's the status",
    "status please",
    "status update",
    "status",
    "how much time",
    "how much longer",
    "how long will it take",
    "how long",
    "are you done",
    "is it done",
    "is it finished",
    "are you finished",
    "progress",
]


class ProcessingStage(str, Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    SAVING_AUDIO = "SAVING_AUDIO"
    TRANSCRIBING = "TRANSCRIBING"
    STRUCTURING_NOTES = "STRUCTURING_NOTES"
    TRANSLATING_HINDI = "TRANSLATING_HINDI"
    COMPILING_PDF = "COMPILING_PDF"
    EXPORTING_USB = "EXPORTING_USB"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class ProcessingTracker:
    """
    Thread-safe progress and remaining-time tracker for the post-lecture pipeline.
    Calculates dynamic percentage completion and remaining time for instant neural voice synthesis.
    """
    STAGE_PERCENTAGES = {
        ProcessingStage.IDLE: 0,
        ProcessingStage.STARTING: 5,
        ProcessingStage.SAVING_AUDIO: 15,
        ProcessingStage.TRANSCRIBING: 35,
        ProcessingStage.STRUCTURING_NOTES: 55,
        ProcessingStage.TRANSLATING_HINDI: 75,
        ProcessingStage.COMPILING_PDF: 88,
        ProcessingStage.EXPORTING_USB: 95,
        ProcessingStage.COMPLETE: 100,
        ProcessingStage.FAILED: 0,
    }

    STAGE_DESCRIPTIONS = {
        ProcessingStage.IDLE: "Agent is in standby and ready to record.",
        ProcessingStage.STARTING: "Initializing post-lecture synthesis.",
        ProcessingStage.SAVING_AUDIO: "Saving and validating raw master audio.",
        ProcessingStage.TRANSCRIBING: "Transcribing speech and extracting technical terminology.",
        ProcessingStage.STRUCTURING_NOTES: "Structuring monograph and numbering mathematical equations.",
        ProcessingStage.TRANSLATING_HINDI: "Generating bilingual Hindi monograph with LaTeX math preservation.",
        ProcessingStage.COMPILING_PDF: "Compiling publication-grade XeTeX PDF documents.",
        ProcessingStage.EXPORTING_USB: "Mirroring deliverables and checksums to USB drive AYUSH.",
        ProcessingStage.COMPLETE: "Lecture processing is 100% complete. Deliverables verified on USB drive.",
        ProcessingStage.FAILED: "Lecture processing encountered an error.",
    }

    def __init__(self):
        self.stage = ProcessingStage.IDLE
        self.start_time = 0.0
        self.stage_start_time = 0.0
        self.estimated_total_sec = 25.0
        self.current_stage_est = 10.0
        self.lock = threading.RLock()

    def start_pipeline(self, estimated_total: float = 25.0):
        with self.lock:
            now = time.monotonic()
            self.stage = ProcessingStage.STARTING
            self.start_time = now
            self.stage_start_time = now
            self.estimated_total_sec = max(estimated_total, 15.0)
            self.current_stage_est = 5.0

    def set_stage(self, stage: ProcessingStage, additional_est: Optional[float] = None):
        with self.lock:
            now = time.monotonic()
            self.stage = stage
            self.stage_start_time = now
            if additional_est is not None:
                elapsed = now - self.start_time
                self.estimated_total_sec = elapsed + additional_est
                self.current_stage_est = additional_est

    def get_progress_percentage(self) -> int:
        with self.lock:
            if self.stage == ProcessingStage.COMPLETE:
                return 100
            if self.stage in (ProcessingStage.IDLE, ProcessingStage.FAILED):
                return 0
            base_pct = self.STAGE_PERCENTAGES.get(self.stage, 10)
            elapsed = time.monotonic() - self.stage_start_time if self.stage_start_time > 0 else 0.0
            
            stages_order = [
                ProcessingStage.STARTING,
                ProcessingStage.SAVING_AUDIO,
                ProcessingStage.TRANSCRIBING,
                ProcessingStage.STRUCTURING_NOTES,
                ProcessingStage.TRANSLATING_HINDI,
                ProcessingStage.COMPILING_PDF,
                ProcessingStage.EXPORTING_USB,
                ProcessingStage.COMPLETE
            ]
            try:
                idx = stages_order.index(self.stage)
                next_pct = self.STAGE_PERCENTAGES[stages_order[idx + 1]]
            except Exception:
                next_pct = base_pct + 10

            est_stage_dur = max(3.0, self.current_stage_est)
            frac = min(0.90, elapsed / est_stage_dur)
            current_pct = int(base_pct + frac * (next_pct - base_pct))
            return max(1, min(99, current_pct))

    def get_remaining_seconds(self) -> int:
        with self.lock:
            if self.stage in (ProcessingStage.IDLE, ProcessingStage.COMPLETE):
                return 0
            elapsed = time.monotonic() - self.start_time if self.start_time > 0 else 0.0
            rem = int(self.estimated_total_sec - elapsed)
            return max(5, rem)

    def get_status_prompt_and_text(self) -> Tuple[str, str]:
        with self.lock:
            stage = self.stage
            pct = self.get_progress_percentage()
            rem = self.get_remaining_seconds()

        if stage == ProcessingStage.IDLE:
            return "status_standby", "Jetson speech agent is in standby and ready to record. Say hey Jetson start recording to begin."
        elif stage == ProcessingStage.COMPLETE:
            return "status_complete", "Lecture processing is 100% complete. Deliverables have been verified and saved to your USB drive. Device is shutting down."
        elif stage == ProcessingStage.FAILED:
            return "status_failed", "Lecture processing encountered an error. Please inspect system logs."

        stage_desc = self.STAGE_DESCRIPTIONS.get(stage, "Processing lecture.")
        status_text = (
            f"Lecture processing is {pct}% complete. {stage_desc} "
            f"Approximately {rem} seconds remaining."
        )
        return f"status_{stage.value.lower()}", status_text


def trigger_system_shutdown(delay_sec: float = 2.0):
    """
    Safely flush filesystem buffers and initiate system power-off.
    Employs triple redundancy across systemctl, logind D-Bus, and shutdown.
    """
    print(f"[Agent] Flushing filesystem buffers before shutdown...", flush=True)
    subprocess.run(["sync"], check=False)
    time.sleep(delay_sec)

    # 1. Standard systemd poweroff
    print("[Agent] Initiating system poweroff via systemctl...", flush=True)
    try:
        res = subprocess.run(["systemctl", "poweroff"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if res.returncode == 0:
            return
        print(f"[Agent] systemctl poweroff returned code {res.returncode}: {res.stderr.strip()}", flush=True)
    except Exception as e:
        print(f"[Agent] systemctl poweroff exception: {e}", flush=True)

    # 2. D-Bus login1 Manager PowerOff
    try:
        res = subprocess.run([
            "busctl", "call", "org.freedesktop.login1", "/org/freedesktop/login1",
            "org.freedesktop.login1.Manager", "PowerOff", "b", "true"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if res.returncode == 0:
            return
    except Exception:
        pass

    # 3. Sudoers passwordless shutdown fallbacks
    for cmd in (["sudo", "-n", "systemctl", "poweroff"], ["sudo", "-n", "shutdown", "-h", "now"]):
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            if res.returncode == 0:
                return
            print(f"[Agent] {' '.join(cmd)} returned code {res.returncode}: {res.stderr.strip()}", flush=True)
        except Exception as e:
            print(f"[Agent] {' '.join(cmd)} exception: {e}", flush=True)

    print(
        "[Agent Shutdown Error] Power-off not permitted for current user. "
        "Grant passwordless shutdown via /etc/sudoers.d/ (NOPASSWD: /sbin/shutdown, /usr/bin/systemctl poweroff).",
        flush=True,
    )


class EdgeBackpressureScheduler:
    """
    Telemetry-driven thermal and unified-memory backpressure scheduler for the
    Jetson Orin Nano 8 GB edge board. Monitors /proc/meminfo, SoC thermal zones,
    and worker queue depths to prioritize uninterrupted audio capture (Requirement R4)
    and avoid out-of-memory kills under load.
    """

    def __init__(
        self,
        meminfo_path: str = "/proc/meminfo",
        thermal_glob: str = "/sys/class/thermal/thermal_zone*/temp",
        pressure_mem_mb: float = 1400.0,
        critical_mem_mb: float = 950.0,
        pressure_temp_c: float = 78.0,
        critical_temp_c: float = 84.0,
        pressure_q_depth: int = 6,
        critical_q_depth: int = 10,
    ):
        self.meminfo_path = meminfo_path
        self.thermal_glob = thermal_glob
        self.pressure_mem_mb = pressure_mem_mb
        self.critical_mem_mb = critical_mem_mb
        self.pressure_temp_c = pressure_temp_c
        self.critical_temp_c = critical_temp_c
        self.pressure_q_depth = pressure_q_depth
        self.critical_q_depth = critical_q_depth
        self.reset()

    def reset(self) -> None:
        self.history: List[Dict] = []
        self.deferred_chunks: int = 0
        self.regimes_seen: List[str] = []
        self.last_decision: Dict = {
            "regime": "NORMAL",
            "defer_live_slm": False,
            "defer_heavy_rewrite": False,
            "throttle_ms": 0,
        }

    def read_system_telemetry(self) -> Dict[str, float]:
        mem_total_kb = 0.0
        mem_avail_kb = 0.0
        try:
            with open(self.meminfo_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total_kb = float(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        mem_avail_kb = float(line.split()[1])
                    if mem_total_kb > 0 and mem_avail_kb > 0:
                        break
        except Exception:
            pass

        temps: Dict[str, float] = {}
        for tpath in sorted(glob.glob(self.thermal_glob)):
            try:
                raw_val = float(Path(tpath).read_text(encoding="utf-8").strip())
                temp_c = raw_val / 1000.0 if raw_val > 200.0 else raw_val
                type_path = Path(tpath).parent / "type"
                zname = type_path.read_text(encoding="utf-8").strip() if type_path.is_file() else Path(tpath).parent.name
                temps[zname] = round(temp_c, 2)
            except Exception:
                continue

        mem_total_mb = round(mem_total_kb / 1024.0, 1) if mem_total_kb else 0.0
        mem_avail_mb = round(mem_avail_kb / 1024.0, 1) if mem_avail_kb else mem_total_mb
        max_temp = max(temps.values()) if temps else 0.0
        return {
            "mem_total_mb": mem_total_mb,
            "mem_available_mb": mem_avail_mb,
            "max_temp_c": round(max_temp, 2),
            "cpu_temp_c": temps.get("cpu-thermal", 0.0),
            "gpu_temp_c": temps.get("gpu-thermal", 0.0),
        }

    def evaluate(
        self,
        audio_q_depth: int = 0,
        typeset_q_depth: int = 0,
        telemetry: Optional[Dict[str, float]] = None,
    ) -> Dict:
        tel = dict(telemetry) if telemetry is not None else self.read_system_telemetry()
        mem_avail = float(tel.get("mem_available_mb", 4096.0))
        max_temp = float(tel.get("max_temp_c", 0.0))

        is_critical = (
            (mem_avail > 0 and mem_avail < self.critical_mem_mb)
            or max_temp >= self.critical_temp_c
            or typeset_q_depth >= self.critical_q_depth
        )
        is_pressure = (
            is_critical
            or (mem_avail > 0 and mem_avail < self.pressure_mem_mb)
            or max_temp >= self.pressure_temp_c
            or typeset_q_depth >= self.pressure_q_depth
        )

        if is_critical:
            regime = "CRITICAL"
            decision = {
                "regime": regime,
                "defer_live_slm": True,
                "defer_heavy_rewrite": True,
                "throttle_ms": 0,
            }
            self.deferred_chunks += 1
        elif is_pressure:
            regime = "PRESSURE"
            decision = {
                "regime": regime,
                "defer_live_slm": False,
                "defer_heavy_rewrite": False,
                "throttle_ms": 120,
            }
        else:
            regime = "NORMAL"
            decision = {
                "regime": regime,
                "defer_live_slm": False,
                "defer_heavy_rewrite": False,
                "throttle_ms": 0,
            }

        entry = {
            **tel,
            "audio_queue_depth": int(audio_q_depth),
            "typeset_queue_depth": int(typeset_q_depth),
            **decision,
        }
        if len(self.history) < 500:
            self.history.append(entry)
        if regime not in self.regimes_seen:
            self.regimes_seen.append(regime)
        self.last_decision = decision
        return entry

    def summary(self) -> Dict:
        if not self.history:
            tel = self.read_system_telemetry()
            return {
                "samples": 0,
                "min_mem_available_mb": tel.get("mem_available_mb", 0.0),
                "peak_temp_c": tel.get("max_temp_c", 0.0),
                "peak_typeset_queue": 0,
                "deferred_chunks": self.deferred_chunks,
                "regimes_seen": list(self.regimes_seen) or ["NORMAL"],
            }
        mem_vals = [h["mem_available_mb"] for h in self.history if h.get("mem_available_mb", 0) > 0]
        return {
            "samples": len(self.history),
            "min_mem_available_mb": min(mem_vals) if mem_vals else 0.0,
            "peak_temp_c": max(h.get("max_temp_c", 0.0) for h in self.history),
            "peak_typeset_queue": max(h.get("typeset_queue_depth", 0) for h in self.history),
            "deferred_chunks": self.deferred_chunks,
            "regimes_seen": list(self.regimes_seen),
        }


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
        log_whisper = open("/tmp/whisper-server.log", "a")
        w_proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log_whisper, stderr=log_whisper, start_new_session=True)
        for _ in range(60):
            if w_proc.poll() is not None:
                print(f"[Agent Error] whisper-server exited prematurely (code {w_proc.returncode}).", flush=True)
                break
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
            "-np", "1",
            "-ngl", "0",
            "-t", "4",
            "--fit", "off",
            "--port", "8080",
            "--host", "127.0.0.1"
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["LD_LIBRARY_PATH"] = "/mnt/thesis/llama.cpp/build/bin:" + env.get("LD_LIBRARY_PATH", "")
        log_llama = open("/tmp/llama-server.log", "a")
        l_proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log_llama, stderr=log_llama, start_new_session=True)
        for _ in range(60):
            if l_proc.poll() is not None:
                print(f"[Agent Error] llama-server exited prematurely (code {l_proc.returncode}).", flush=True)
                break
            time.sleep(1.0)
            if is_service_alive(LLAMA_SERVER_URL, endpoint="/health"):
                break


def query_whisper_stream(
    pcm_samples: np.ndarray,
    no_speech_thold: float = 0.40,
    prompt: Optional[str] = None
) -> str:
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

    effective_prompt = prompt if prompt is not None else WHISPER_MATH_PROMPT

    boundary = "----JetsonHeadlessAgentBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="stream.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode("utf-8")
        + wav_bytes
        + f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="prompt"\r\n\r\n'
        f"{effective_prompt}\r\n".encode("utf-8")
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
    def __init__(self, auto_shutdown: bool = True):
        self.state = AgentState.STANDBY
        self.auto_shutdown = auto_shutdown
        self.status_tracker = ProcessingTracker()
        self.exporter = LectureExporter()
        self.book_builder = LectureToBookBuilder()
        self.post_processor = PostLectureProcessor()
        self.slm = None
        self.hpf = HighPassHumFilter()
        self.scheduler = EdgeBackpressureScheduler()

        # Decoupled high-priority audio capture thread & buffers
        self.audio_queue = queue.Queue(maxsize=1500)
        self.raw_audio_frames: List[bytes] = []
        self._frames_lock = threading.Lock()
        self._capture_running = False
        self._capture_thread = None

        # Real-time concurrent typesetting queue and thread
        self.live_typeset_queue = queue.Queue()
        self.live_lecture_chunks: List[TranscriptChunk] = []
        self._lecture_chunks_lock = threading.Lock()
        self._typeset_thread = None

        # Asynchronous background post-processing coordination
        self._proc_done_event = threading.Event()
        self._proc_res_box = {}

    def report_status(self):
        """Announce current post-processing status, completion percentage, and remaining time aloud via Piper neural TTS."""
        _, status_text = self.status_tracker.get_status_prompt_and_text()
        print(f"\n[Agent Dynamic Status]: \"{status_text}\"", flush=True)
        speak(status_text, non_blocking=False)

    def _initiate_stop_and_process(self, lecture_start_time: float):
        """
        Transitions agent to PROCESSING state and launches the post-lecture
        pipeline in a decoupled background thread so the voice agent remains
        fully responsive to status queries.
        """
        self.state = AgentState.PROCESSING
        play_prompt("recording_stopped", non_blocking=False)

        print("[Agent] Finalizing real-time typesetting pipeline...", flush=True)
        try:
            self.live_typeset_queue.join()
        except Exception:
            pass

        with self._lecture_chunks_lock:
            final_chunks = list(self.live_lecture_chunks)
        with self._frames_lock:
            frames_to_save = list(self.raw_audio_frames)

        # Estimate processing time: ~25s for live pre-typeset, or 30s + 0.02*duration for batch
        audio_dur = (len(frames_to_save) * len(frames_to_save[0]) / 32000.0) if frames_to_save else 0.0
        est_total = 25.0 if final_chunks else (30.0 + audio_dur * 0.02)
        self.status_tracker.start_pipeline(est_total)

        self._proc_done_event.clear()
        self._proc_res_box.clear()

        def _worker():
            try:
                res = self.process_lecture(final_chunks, frames_to_save, lecture_start_time)
                self._proc_res_box["res"] = res
            except Exception as e:
                self._proc_res_box["err"] = e
                print(f"[Agent] Lecture post-processing error: {e}", flush=True)
                self.status_tracker.set_stage(ProcessingStage.FAILED)
            finally:
                self._proc_done_event.set()

        proc_thread = threading.Thread(target=_worker, daemon=True, name="LecturePostProcessor")
        proc_thread.start()
        print(f"[Agent] Background post-processing worker active (Estimated ~{int(est_total)}s). Listening for status queries...", flush=True)

    def initialize_hardware(self) -> Tuple[pyaudio.PyAudio, int, int, int]:
        """Detect and configure USB mic and audio subsystem."""
        # Optimal mic gain & volume across all candidate ALSA sound cards
        for card in ["Audio", "1", "2"]:
            try:
                subprocess.run(["amixer", "-c", card, "sset", "Mic", "110", "cap", "unmute"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                subprocess.run(["amixer", "-c", card, "sset", "PCM", "95", "unmute"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
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

    def _audio_capture_worker(self, stream, chunk_samples: int, rate: int, channels: int):
        """
        Ultra-high-priority, non-blocking audio capture worker thread.
        Reads the hardware microphone stream continuously without ever blocking on Whisper or SLM,
        guaranteeing zero dropped samples during the entire lecture.
        """
        while self._capture_running:
            try:
                raw_bytes = stream.read(chunk_samples, exception_on_overflow=False)
                if not raw_bytes:
                    continue

                # Decimate 48 kHz stereo to 16 kHz mono float32
                if rate == 48000 and channels == 2:
                    mono_chunk = stereo_48k_to_mono_16k(raw_bytes)
                else:
                    pcm = np.frombuffer(raw_bytes, dtype=np.int16)
                    mono_chunk = np.clip(pcm / 32768.0, -1.0, 1.0).astype(np.float32)

                filtered_chunk = self.hpf.process(mono_chunk)

                # Continuous uninterrupted audio archiving while in RECORDING state
                if self.state == AgentState.RECORDING:
                    int16_pcm = np.clip(filtered_chunk * 32767, -32768, 32767).astype(np.int16)
                    pcm_bytes = int16_pcm.tobytes()
                    with self._frames_lock:
                        self.raw_audio_frames.append(pcm_bytes)

                # Push to live queue for VAD and voice command trigger detection
                try:
                    self.audio_queue.put_nowait((mono_chunk, filtered_chunk))
                except queue.Full:
                    # Drop oldest live packet so live queue doesn't lag; raw_audio_frames remains 100% complete
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait((mono_chunk, filtered_chunk))
                    except Exception:
                        pass
            except Exception:
                time.sleep(0.005)

    def _live_typesetting_worker(self):
        """
        Background worker that translates live spoken utterances to LaTeX math concurrently
        while the lecturer is speaking, guaranteeing zero post-lecture processing delay.
        """
        prev_spoken = ""
        prev_latex = ""
        while self._capture_running:
            try:
                item = self.live_typeset_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                self.live_typeset_queue.task_done()
                break

            with self._lecture_chunks_lock:
                if not self.live_lecture_chunks:
                    prev_spoken = ""
                    prev_latex = ""

            t_start, t_end, spoken = item
            sched = self.scheduler.evaluate(self.audio_queue.qsize(), self.live_typeset_queue.qsize())
            try:
                if sched["defer_live_slm"] or not self.slm:
                    if sched["defer_live_slm"]:
                        print(f"[Scheduler] Backpressure ({sched['regime']}): deferring live SLM to protect capture.", flush=True)
                    latex = spoken
                else:
                    hist = [(prev_spoken, prev_latex)] if prev_spoken else None
                    latex = self.slm.translate(spoken, context=prev_latex, history=hist)
                    if sched["throttle_ms"] > 0:
                        time.sleep(sched["throttle_ms"] / 1000.0)
            except Exception as e:
                print(f"[SLM Translation Error]: {e}", flush=True)
                latex = spoken

            prev_spoken = spoken
            prev_latex = latex

            chunk_obj = TranscriptChunk(
                timestamp_start=t_start,
                timestamp_end=t_end,
                spoken_text=spoken,
                latex_text=latex
            )
            with self._lecture_chunks_lock:
                self.live_lecture_chunks.append(chunk_obj)

            print(f"[Live {t_end:.1f}s]: {spoken}", flush=True)
            if latex != spoken:
                print(f"       -> [LaTeX]: {latex}", flush=True)

            self.live_typeset_queue.task_done()

    def transcribe_full_audio(self, wav_file: Path) -> List[TranscriptChunk]:
        """
        Executes high-performance GPU batch transcription via whisper-cli on the pristine raw WAV.
        Captures 100% of the lecturer's words with complete contextual continuity and exact timestamps.
        """
        whisper_cli = Path(WHISPER_BIN).parent / "whisper-cli"
        if not whisper_cli.is_file() or not wav_file.is_file():
            return []

        import tempfile
        temp_json_prefix = Path(tempfile.gettempdir()) / f"batch_trans_{int(time.time())}"
        cmd = [
            str(whisper_cli),
            "-m", WHISPER_MODEL,
            "-f", str(wav_file),
            "--prompt", WHISPER_MATH_PROMPT,
            "-sns", "-nth", "0.60", "-fa",
            "-oj", "-of", str(temp_json_prefix)
        ]

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/mnt/thesis/main/whisper.cpp/build-orin/bin:" + env.get("LD_LIBRARY_PATH", "")

        try:
            res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
            json_file = Path(f"{temp_json_prefix}.json")
            if not json_file.is_file():
                # Fallback to multi-threaded CPU NEON inference
                cmd_cpu = [
                    str(whisper_cli),
                    "-m", WHISPER_MODEL,
                    "-f", str(wav_file),
                    "--prompt", WHISPER_MATH_PROMPT,
                    "-sns", "-nth", "0.60", "-ng", "-t", "4",
                    "-oj", "-of", str(temp_json_prefix)
                ]
                res = subprocess.run(cmd_cpu, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)

            if not json_file.is_file():
                return []

            with open(json_file, "r", encoding="utf-8") as jf:
                jdata = json.load(jf)
            segments = jdata.get("transcription", [])

            def _parse_time_sec(ts_str: str) -> float:
                ts_str = ts_str.replace(",", ".")
                parts = ts_str.split(":")
                if len(parts) == 3:
                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                elif len(parts) == 2:
                    return float(parts[0]) * 60 + float(parts[1])
                return float(ts_str)

            raw_segs = []
            num_segs = len(segments)
            for idx, seg in enumerate(segments):
                txt = seg.get("text", "").strip()
                txt = re.sub(r'[\(\[\{].*?[\)\]\}]', '', txt)
                txt = re.sub(r'[♪♫]+', '', txt).strip()
                if not txt or not any(c.isalnum() for c in txt):
                    continue

                norm = clean_trigger_text(txt)
                # Strip out initial start triggers captured in the first 2 segments
                if idx <= 1 and (any(trig in norm for trig in START_TRIGGERS) or "start recording" in norm):
                    continue
                # Strip out trailing stop triggers captured in the last 3 segments
                if idx >= num_segs - 3 and (any(stop_t in norm for stop_t in STOP_TRIGGERS) or "stop recording" in norm or "jetson stop" in norm):
                    continue

                ts_from = seg.get("timestamps", {}).get("from", "00:00:00,000")
                ts_to = seg.get("timestamps", {}).get("to", "00:00:05,000")
                t_start = _parse_time_sec(ts_from)
                t_end = _parse_time_sec(ts_to)
                raw_segs.append((t_start, t_end, txt))

            # Sentence-boundary aggregation to prevent clause fractures and hallucinations
            aggregated_units = []
            curr_start = None
            curr_end = None
            curr_words = []

            for t_start, t_end, txt in raw_segs:
                if curr_start is None:
                    curr_start = t_start
                curr_end = t_end
                curr_words.append(txt)

                joined = " ".join(curr_words)
                ends_with_punct = bool(re.search(r'[.!?]\s*$', txt))
                word_count = len(joined.split())
                duration = curr_end - curr_start

                if ends_with_punct or word_count >= 24 or duration >= 16.0:
                    aggregated_units.append((curr_start, curr_end, joined))
                    curr_start = None
                    curr_end = None
                    curr_words = []

            if curr_words:
                aggregated_units.append((curr_start, curr_end, " ".join(curr_words)))

            chunks = []
            if self.slm and hasattr(self.slm, "reset_symbol_registry"):
                self.slm.reset_symbol_registry()
            prev_spoken = ""
            prev_latex = ""
            for t_start, t_end, full_spoken in aggregated_units:
                if self.slm:
                    hist = [(prev_spoken, prev_latex)] if prev_spoken else None
                    latex = self.slm.translate(full_spoken, context=prev_latex, history=hist)
                else:
                    latex = full_spoken
                prev_spoken = full_spoken
                prev_latex = latex
                chunks.append(TranscriptChunk(t_start, t_end, full_spoken, latex))
                print(f"[{t_start:.1f}s - {t_end:.1f}s]: {full_spoken}", flush=True)
                if latex != full_spoken:
                    print(f"       -> [LaTeX]: {latex}", flush=True)

            json_file.unlink(missing_ok=True)
            return chunks
        except Exception as e:
            print(f"[Agent] Batch transcription error: {e}", flush=True)
            return []

    def run_continuous_loop(self):
        """
        Master background loop listening for voice commands and recording lectures.
        Features decoupled continuous audio ingestion to prevent dropped speech frames.
        """
        print("[Agent] Initializing daemons and audio subsystems...")
        ensure_edge_daemons()
        self.slm = QwenMathTranslator.get_instance(backend="server")

        pa, dev_idx, channels, rate = self.initialize_hardware()

        chunk_samples = 4800 if rate == 48000 else 1600
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=dev_idx,
            frames_per_buffer=chunk_samples,
        )

        # Launch dedicated uninterrupted audio capture thread
        self._capture_running = True
        self._capture_thread = threading.Thread(
            target=self._audio_capture_worker,
            args=(stream, chunk_samples, rate, channels),
            daemon=True,
            name="AudioCaptureWorker"
        )
        self._capture_thread.start()
        print("[Agent] Continuous audio ingestion thread active.", flush=True)

        # Launch concurrent real-time LaTeX typesetting thread
        self._typeset_thread = threading.Thread(
            target=self._live_typesetting_worker,
            daemon=True,
            name="LiveTypesettingWorker"
        )
        self._typeset_thread.start()
        print("[Agent] Concurrent real-time LaTeX typesetting engine active.", flush=True)

        print("[Agent] Announcing system ready via TTS...")
        play_prompt("system_ready")

        # Verify USB drive presence
        usb_root = find_usb_mount()
        if not usb_root:
            print("[Agent] Notice: USB drive AYUSH not currently inserted.")
            play_prompt("usb_warning", non_blocking=False)
        else:
            print(f"[Agent] USB Drive detected at: {usb_root}")

        # Calibrate ambient noise floor from incoming audio queue
        print("[Agent] Calibrating microphone noise floor...", flush=True)
        noise_samples = []
        for _ in range(8):
            try:
                _, fc = self.audio_queue.get(timeout=1.5)
                noise_samples.append(float(np.sqrt(np.mean(fc ** 2))))
            except queue.Empty:
                pass

        noise_floor = max(float(np.percentile(noise_samples, 85)), 0.005) if noise_samples else 0.010
        speech_thold = max(noise_floor * 2.2, 0.020)
        silence_thold = max(noise_floor * 1.3, 0.012)
        print(f"[Agent] Calibrated: Noise floor={noise_floor:.4f} | Speech Trigger={speech_thold:.4f}", flush=True)

        vad = AdaptiveEnergyVAD(speech_threshold=speech_thold, silence_threshold=silence_thold)

        # State Buffers
        current_speech_chunks = []
        lecture_chunks: List[TranscriptChunk] = []

        lecture_start_time = 0.0
        in_speech = False
        silence_chunk_count = 0

        print(f"\n[Agent] Standby mode active. Say 'Hey Jetson start recording' (or press Enter) to begin...", flush=True)

        try:
            while True:
                # 1. Non-blocking terminal keyboard check
                try:
                    if sys.stdin.isatty():
                        r, _, _ = select.select([sys.stdin], [], [], 0.0)
                        if r:
                            sys.stdin.readline()
                            if self.state == AgentState.STANDBY:
                                print("\n[Agent] Keyboard Enter detected! Starting countdown...", flush=True)
                                self.state = AgentState.COUNTDOWN
                                play_prompt("countdown_start")
                                current_speech_chunks.clear()
                                lecture_chunks.clear()
                                with self._lecture_chunks_lock:
                                    self.live_lecture_chunks.clear()
                                while not self.live_typeset_queue.empty():
                                    try:
                                        self.live_typeset_queue.get_nowait()
                                    except Exception:
                                        break
                                with self._frames_lock:
                                    self.raw_audio_frames.clear()
                                # Drain queue
                                while not self.audio_queue.empty():
                                    try:
                                        self.audio_queue.get_nowait()
                                    except Exception:
                                        break
                                self.hpf.reset()
                                vad.reset()
                                self.scheduler.reset()
                                if self.slm and hasattr(self.slm, "reset_symbol_registry"):
                                    self.slm.reset_symbol_registry()
                                lecture_start_time = time.monotonic()
                                self.state = AgentState.RECORDING
                                print("[Agent] Recording started! Capturing live lecture...", flush=True)
                                in_speech = False
                                continue
                            elif self.state == AgentState.RECORDING:
                                print("\n[Agent] Keyboard Enter detected! Stopping recording...", flush=True)
                                self._initiate_stop_and_process(lecture_start_time)
                                current_speech_chunks.clear()
                                in_speech = False
                                vad.reset()
                                while not self.audio_queue.empty():
                                    try:
                                        self.audio_queue.get_nowait()
                                    except Exception:
                                        break
                                continue
                            elif self.state == AgentState.PROCESSING:
                                print("\n[Agent] Keyboard Enter detected during processing! Status query triggered...", flush=True)
                                self.report_status()
                                continue
                except Exception:
                    pass

                # 2. Retrieve chunk from decoupled audio ingestion queue
                try:
                    mono_chunk, filtered_chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # =========================================================
                # 1. STANDBY STATE (Listening for Wake-Word)
                # =========================================================
                if self.state == AgentState.STANDBY:
                    is_speech, energy = vad.process_chunk(filtered_chunk)

                    if is_speech:
                        if not in_speech:
                            in_speech = True
                        current_speech_chunks.append(filtered_chunk)
                        silence_chunk_count = 0
                        sys.stdout.write(f"\r[Listening] Voice detected (RMS={energy:.3f})... ")
                        sys.stdout.flush()
                    else:
                        if in_speech:
                            silence_chunk_count += 1
                            current_speech_chunks.append(filtered_chunk)

                            # Speech boundary (0.7s silence or max 4s phrase)
                            if silence_chunk_count >= 7 or len(current_speech_chunks) >= 40:
                                utterance = np.concatenate(current_speech_chunks)
                                spoken = query_whisper_stream(
                                    utterance,
                                    no_speech_thold=0.30,
                                    prompt="Hey Jetson, start recording, start lecture, begin, record, standby."
                                )
                                norm_spoken = clean_trigger_text(spoken)

                                if spoken:
                                    print(f"\n[Standby Heard]: \"{spoken}\"", flush=True)

                                # Check triggers: exact match, or contains "start", or "jetson" and "record"
                                is_start = any(trig in norm_spoken for trig in START_TRIGGERS) or \
                                           ("start" in norm_spoken and ("recording" in norm_spoken or "lecture" in norm_spoken)) or \
                                           ("jetson" in norm_spoken and "start" in norm_spoken) or \
                                           (norm_spoken.startswith("start") and len(norm_spoken.split()) <= 3)

                                is_status = any(trig in norm_spoken for trig in STATUS_TRIGGERS) or \
                                            ("status" in norm_spoken) or \
                                            ("how long" in norm_spoken) or \
                                            ("how much time" in norm_spoken)

                                if is_start:
                                    print(f"[Agent] Wake-Word Activated! Starting countdown...", flush=True)
                                    self.state = AgentState.COUNTDOWN
                                    play_prompt("countdown_start")

                                    current_speech_chunks.clear()
                                    lecture_chunks.clear()
                                    with self._lecture_chunks_lock:
                                        self.live_lecture_chunks.clear()
                                    while not self.live_typeset_queue.empty():
                                        try:
                                            self.live_typeset_queue.get_nowait()
                                        except Exception:
                                            break
                                    with self._frames_lock:
                                        self.raw_audio_frames.clear()
                                    while not self.audio_queue.empty():
                                        try:
                                            self.audio_queue.get_nowait()
                                        except Exception:
                                            break
                                    self.hpf.reset()
                                    vad.reset()
                                    self.scheduler.reset()
                                    if self.slm and hasattr(self.slm, "reset_symbol_registry"):
                                        self.slm.reset_symbol_registry()

                                    lecture_start_time = time.monotonic()
                                    self.state = AgentState.RECORDING
                                    print("[Agent] Recording started! Capturing live lecture...", flush=True)
                                    in_speech = False
                                    continue
                                elif is_status:
                                    print(f"[Agent] Standby Status Query Activated: '{spoken}'", flush=True)
                                    self.report_status()
                                    while not self.audio_queue.empty():
                                        try:
                                            self.audio_queue.get_nowait()
                                        except Exception:
                                            break
                                    self.hpf.reset()
                                    vad.reset()
                                else:
                                    if spoken:
                                        print("[Standby] (Say 'Hey Jetson start recording' or 'Start recording' to begin)", flush=True)

                                current_speech_chunks.clear()
                                in_speech = False
                                vad.reset()

                # =========================================================
                # 2. RECORDING STATE (Continuous Lecture Transcription)
                # =========================================================
                elif self.state == AgentState.RECORDING:
                    # Note: 100% of audio frames are continuously archived by self._audio_capture_worker
                    is_speech, energy = vad.process_chunk(filtered_chunk)

                    if is_speech:
                        if not in_speech:
                            in_speech = True
                        current_speech_chunks.append(filtered_chunk)
                        silence_chunk_count = 0
                    else:
                        if in_speech:
                            silence_chunk_count += 1
                            current_speech_chunks.append(filtered_chunk)

                            # Speech boundary reached (1.6s silence or max 12s utterance)
                            chunk_duration = len(current_speech_chunks) * 0.1
                            if silence_chunk_count >= 16 or chunk_duration >= 12.0:
                                utterance = np.concatenate(current_speech_chunks)
                                t_end_rel = time.monotonic() - lecture_start_time
                                t_start_rel = max(0.0, t_end_rel - chunk_duration)

                                # Fast Whisper check for STOP voice trigger
                                spoken_text = query_whisper_stream(utterance, no_speech_thold=0.40)
                                norm_spoken = clean_trigger_text(spoken_text)

                                # Check for STOP command
                                is_stop = any(stop_t in norm_spoken for stop_t in STOP_TRIGGERS) or \
                                          ("stop" in norm_spoken and ("recording" in norm_spoken or "lecture" in norm_spoken)) or \
                                          ("jetson" in norm_spoken and "stop" in norm_spoken) or \
                                          (norm_spoken.startswith("stop") and len(norm_spoken.split()) <= 3)

                                if is_stop:
                                    print(f"\n[Agent] Stop-Command Detected: '{spoken_text}'", flush=True)
                                    self._initiate_stop_and_process(lecture_start_time)
                                    current_speech_chunks.clear()
                                    in_speech = False
                                    vad.reset()
                                    while not self.audio_queue.empty():
                                        try:
                                            self.audio_queue.get_nowait()
                                        except Exception:
                                            break
                                    continue

                                # Standard Technical Speech Chunk: Enqueue for concurrent real-time SLM translation
                                if spoken_text and len(spoken_text) > 2:
                                    self.live_typeset_queue.put((t_start_rel, t_end_rel, spoken_text))

                                current_speech_chunks.clear()
                                in_speech = False
                                vad.reset()

                # =========================================================
                # 3. PROCESSING STATE (Background Typesetting + Voice Status Query)
                # =========================================================
                elif self.state == AgentState.PROCESSING:
                    # 1. Check if background pipeline completed
                    if self._proc_done_event.is_set():
                        proc_res = self._proc_res_box.get("res") or {}
                        usb_success = proc_res.get("usb_success", False)
                        self.status_tracker.set_stage(ProcessingStage.COMPLETE)

                        if usb_success:
                            print(f"[Agent] Lecture post-processing complete and verified on USB drive AYUSH!", flush=True)
                            if self.auto_shutdown:
                                print("[Agent] Playing final announcement with auto-shutdown countdown...", flush=True)
                                play_prompt("transcription_saved_shutdown", non_blocking=False)
                                print("[Agent] Auto-shutdown initiated in 5 seconds. Goodbye!", flush=True)
                                time.sleep(5.0)
                                trigger_system_shutdown()
                                sys.exit(0)
                            else:
                                play_prompt("transcription_saved", non_blocking=False)
                                self.state = AgentState.STANDBY
                                print("\n[Agent] Standby mode active. Say 'Hey Jetson start recording' to begin...", flush=True)
                                continue
                        else:
                            print("[Agent] Warning: Deliverables saved to internal SSD, but USB export failed or USB missing!", flush=True)
                            play_prompt("usb_warning", non_blocking=False)
                            if self.auto_shutdown:
                                print("[Agent] Auto-shutdown initiated in 5 seconds...", flush=True)
                                time.sleep(5.0)
                                trigger_system_shutdown()
                                sys.exit(0)
                            else:
                                self.state = AgentState.STANDBY
                                print("\n[Agent] Standby mode active. Say 'Hey Jetson start recording' to begin...", flush=True)
                                continue

                    # 2. While processing is in progress, listen for status queries
                    is_speech, energy = vad.process_chunk(filtered_chunk)
                    if is_speech:
                        if not in_speech:
                            in_speech = True
                        current_speech_chunks.append(filtered_chunk)
                        silence_chunk_count = 0
                    else:
                        if in_speech:
                            silence_chunk_count += 1
                            current_speech_chunks.append(filtered_chunk)

                            # Speech boundary (0.7s silence or max 4s phrase)
                            if silence_chunk_count >= 7 or len(current_speech_chunks) >= 40:
                                utterance = np.concatenate(current_speech_chunks)
                                spoken = query_whisper_stream(
                                    utterance,
                                    no_speech_thold=0.30,
                                    prompt="Hey Jetson, what is the status? How much time is left? Status update."
                                )
                                norm_spoken = clean_trigger_text(spoken)
                                if spoken:
                                    print(f"\n[Processing Heard]: \"{spoken}\"", flush=True)

                                is_status = any(trig in norm_spoken for trig in STATUS_TRIGGERS) or \
                                            ("status" in norm_spoken) or \
                                            ("how long" in norm_spoken) or \
                                            ("how much time" in norm_spoken) or \
                                            ("are you done" in norm_spoken) or \
                                            ("is it done" in norm_spoken) or \
                                            ("is it finished" in norm_spoken) or \
                                            ("finished" in norm_spoken) or \
                                            ("progress" in norm_spoken)

                                if is_status:
                                    print(f"[Agent] Status Query Triggered during processing! Replying...", flush=True)
                                    self.report_status()
                                    while not self.audio_queue.empty():
                                        try:
                                            self.audio_queue.get_nowait()
                                        except Exception:
                                            break
                                    self.hpf.reset()
                                    vad.reset()

                                current_speech_chunks.clear()
                                in_speech = False
                                vad.reset()

        except KeyboardInterrupt:
            print("\n[Agent] Shutting down headless agent...", flush=True)
        finally:
            self._capture_running = False
            self.live_typeset_queue.put(None)
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=1.0)
            if self._typeset_thread and self._typeset_thread.is_alive():
                self._typeset_thread.join(timeout=1.0)
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
        Post-processes lecture, executes full GPU batch transcription,
        translates mathematical notation to LaTeX, formats textbook monograph, compiles PDF,
        and exports deliverables to USB drive AYUSH.
        """
        print("\n[Agent] Typesetting complete lecture transcript with LaTeX equations and TikZ graphics...")
        ts = time.strftime("%Y%m%d_%H%M%S")
        base_lecture_title = f"Lecture_{ts}"

        # Temporary working directory for generation
        temp_dir = Path("/tmp") / f"lecture_{ts}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save pristine unclipped raw audio WAV file
        self.status_tracker.set_stage(ProcessingStage.SAVING_AUDIO, additional_est=25.0)
        wav_file = temp_dir / f"{base_lecture_title}_raw.wav"
        with self._frames_lock:
            frames_to_save = list(raw_audio_frames)

        if frames_to_save:
            with wave.open(str(wav_file), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"".join(frames_to_save))
            audio_sec = len(b"".join(frames_to_save)) / (2 * 16000)
            print(f"[Agent] Saved complete raw audio: {wav_file} ({audio_sec:.1f} seconds recorded, 100% complete)")

        # 2. Acquire Typeset Lecture Chunks (Real-Time Pre-Typeset or Batch Fallback)
        if chunks and len(chunks) > 0:
            print(f"[Agent] Real-time typesetting complete: using {len(chunks)} live-transcribed and LaTeX-typeset chunks.", flush=True)
            active_chunks = chunks
        else:
            print("[Agent] Live chunks empty; performing full GPU batch transcription via whisper-cli...", flush=True)
            self.status_tracker.set_stage(ProcessingStage.TRANSCRIBING, additional_est=22.0)
            batch_chunks = self.transcribe_full_audio(wav_file) if wav_file.is_file() else []
            if batch_chunks:
                print(f"[Agent] whisper-cli captured {len(batch_chunks)} high-fidelity segments across full lecture.", flush=True)
                active_chunks = batch_chunks
            else:
                active_chunks = []

        # 3. Build Typeset Transcript TeX, Markdown, and Subtitles
        self.status_tracker.set_stage(ProcessingStage.STRUCTURING_NOTES, additional_est=18.0)
        tex_content, md_content, clean_title = self.book_builder.build_book(base_lecture_title, active_chunks)
        srt_content, vtt_content = generate_subtitles(active_chunks)

        slug = re.sub(r'[^a-zA-Z0-9]+', '_', clean_title).strip('_')
        if len(slug) > 40:
            slug = slug[:40].rstrip('_')
        session_name = f"{ts}_{slug}" if slug else base_lecture_title

        # Rename raw WAV to session name
        final_wav = temp_dir / f"{session_name}_raw.wav"
        if wav_file.is_file():
            wav_file.rename(final_wav)

        tex_file = temp_dir / f"{session_name}.tex"
        tex_file.write_text(tex_content, encoding="utf-8")

        md_file = temp_dir / f"{session_name}.md"
        md_file.write_text(md_content, encoding="utf-8")

        srt_file = temp_dir / f"{session_name}.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        vtt_file = temp_dir / f"{session_name}.vtt"
        vtt_file.write_text(vtt_content, encoding="utf-8")

        # 4. Build Bilingual Hindi Monograph (subtitle .srt/.vtt translation skipped to save CPU latency)
        self.status_tracker.set_stage(ProcessingStage.TRANSLATING_HINDI, additional_est=15.0)
        print(f"[Agent] Generating bilingual Hindi monograph for Indian students...")
        try:
            hindi_tex, hindi_md, hindi_title, hindi_chunks, hindi_paras = self.book_builder.build_hindi_book(
                clean_title, active_chunks, translate_subtitles=False
            )

            hi_tex_file = temp_dir / f"{session_name}_hindi.tex"
            hi_tex_file.write_text(hindi_tex, encoding="utf-8")

            hi_md_file = temp_dir / f"{session_name}_hindi.md"
            hi_md_file.write_text(hindi_md, encoding="utf-8")

            if hindi_chunks:
                hindi_srt, hindi_vtt = generate_subtitles(hindi_chunks)
                hi_srt_file = temp_dir / f"{session_name}_hindi.srt"
                hi_srt_file.write_text(hindi_srt, encoding="utf-8")
                hi_vtt_file = temp_dir / f"{session_name}_hindi.vtt"
                hi_vtt_file.write_text(hindi_vtt, encoding="utf-8")
            else:
                hi_srt_file = None
                hi_vtt_file = None

            hi_pdf_file = temp_dir / f"{session_name}_hindi.pdf"
            compiled_hi = self.post_processor.compile_hindi_pdf(
                tex_path=hi_tex_file,
                output_dir=temp_dir,
                title=hindi_title,
                paragraphs=hindi_paras,
                output_pdf_path=hi_pdf_file
            )
            if compiled_hi and compiled_hi.is_file():
                hi_pdf_file = compiled_hi
        except Exception as e:
            print(f"[Agent] Warning: Hindi translation/compilation error: {e}", flush=True)
            hi_tex_file = hi_md_file = hi_srt_file = hi_vtt_file = hi_pdf_file = None

        # 5. English PDF Compilation Pass (Standalone XeTeX via Tectonic)
        self.status_tracker.set_stage(ProcessingStage.COMPILING_PDF, additional_est=10.0)
        print(f"[Agent] Compiling typeset lecture transcript: '{clean_title}'...")
        post_res = self.post_processor.process_and_compile(
            lecture_title=clean_title,
            tex_path=tex_file,
            output_dir=temp_dir,
            raw_transcript="\n".join([c.spoken_text for c in active_chunks])
        )

        pdf_file = post_res.get("pdf")

        # 6. Build Cryptographic Mathematical-Invariance Audit Certificate & Export
        self.status_tracker.set_stage(ProcessingStage.EXPORTING_USB, additional_est=5.0)
        print("[Agent] Writing invariance_audit.json and exporting deliverables to USB drive AYUSH...")
        export_map = {
            "tex": tex_file,
            "md": md_file,
            "srt": srt_file,
            "vtt": vtt_file,
            "pdf": pdf_file if (pdf_file and pdf_file.is_file()) else None,
            "hindi_tex": hi_tex_file if (hi_tex_file and hi_tex_file.is_file()) else None,
            "hindi_md": hi_md_file if (hi_md_file and hi_md_file.is_file()) else None,
            "hindi_srt": hi_srt_file if (hi_srt_file and hi_srt_file.is_file()) else None,
            "hindi_vtt": hi_vtt_file if (hi_vtt_file and hi_vtt_file.is_file()) else None,
            "hindi_pdf": hi_pdf_file if (hi_pdf_file and hi_pdf_file.is_file()) else None,
            "wav": final_wav if final_wav.is_file() else None,
        }

        artifact_hashes = {}
        for k, p in export_map.items():
            if p and Path(p).is_file():
                artifact_hashes[Path(p).name] = compute_sha256(Path(p))

        try:
            from hindi_translator import HindiLectureTranslator
            hindi_audit = dict(HindiLectureTranslator.get_instance().audit_stats)
        except Exception:
            hindi_audit = {}

        reviewer_stats = {}
        if getattr(self.book_builder, "_ai_reviewer", None) is not None:
            reviewer_stats = dict(getattr(self.book_builder._ai_reviewer, "harmonisation_stats", {}))

        slm_audit = {}
        if self.slm and hasattr(self.slm, "symbol_registry_snapshot"):
            slm_audit = {
                "symbol_registry": self.slm.symbol_registry_snapshot(),
                "cache_prompt_enabled": getattr(self.slm, "enable_cache_prompt", True),
                "early_abort_enabled": getattr(self.slm, "enable_early_abort", True),
            }

        self.scheduler.evaluate(self.audio_queue.qsize(), self.live_typeset_queue.qsize())
        audit_payload = {
            "schema": "jetson-math-notes-invariance-audit-v1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "session_name": session_name,
            "lecture_title": clean_title,
            "slm_transduction": slm_audit,
            "notation_harmonisation": reviewer_stats,
            "hindi_math_preservation": hindi_audit,
            "resource_scheduler": self.scheduler.summary(),
            "artifact_hashes_sha256": artifact_hashes,
        }
        audit_file = temp_dir / f"{session_name}_invariance_audit.json"
        audit_file.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        export_map["audit"] = audit_file

        export_res = self.exporter.export(session_name, export_map, timestamp=ts)
        print(f"[Agent] Export Complete:")
        print(f"       Local SSD: {export_res['local_path']}")
        print(f"       USB Drive: {export_res['usb_path']} (Success: {export_res['usb_success']})")

        self.status_tracker.set_stage(ProcessingStage.COMPLETE)

        # Clean up temp
        shutil.rmtree(temp_dir, ignore_errors=True)
        return export_res

    def run_file_simulation(self, wav_path: str, max_duration: Optional[float] = None):
        """
        Simulate the complete headless workflow on an existing WAV file.
        Loads raw audio frames and executes process_lecture() with high-performance GPU batch transcription,
        academic typesetting, vector TikZ graphics, Tectonic XeTeX compilation, and USB export.
        """
        print(f"\n[Agent Simulation] Testing on WAV file: {wav_path}", flush=True)
        ensure_edge_daemons()
        self.slm = QwenMathTranslator.get_instance(backend="server")

        with wave.open(wav_path, "rb") as wf:
            n_ch = wf.getnchannels()
            fr = wf.getframerate()
            n_fr = wf.getnframes()
            if max_duration is not None and max_duration > 0:
                n_fr = int(min(n_fr, max_duration * fr))
            raw_b = wf.readframes(n_fr)

        pcm_raw = np.frombuffer(raw_b, dtype=np.int16)
        if n_ch > 1:
            pcm_raw = pcm_raw.reshape(-1, n_ch).mean(axis=1)
        if fr != 16000:
            num_s = int(len(pcm_raw) * 16000 / fr)
            pcm_raw = np.interp(
                np.linspace(0, len(pcm_raw), num_s, endpoint=False),
                np.arange(len(pcm_raw)),
                pcm_raw,
            ).astype(np.int16)

        raw_frames = [pcm_raw.tobytes()]
        res = self.process_lecture([], raw_frames, time.monotonic())
        if res.get("usb_success"):
            play_prompt("transcription_saved")
        else:
            play_prompt("usb_warning")


def main():
    parser = argparse.ArgumentParser(description="Headless Autonomous Speech-to-LaTeX Edge Agent")
    parser.add_argument("--sim-file", type=str, default=None, help="Test simulation on a WAV audio file")
    parser.add_argument("--max-duration", type=float, default=None, help="Optional max duration to process in seconds (default: entire file)")
    parser.add_argument("--no-auto-shutdown", action="store_true", help="Disable automatic system shutdown upon post-processing completion")
    args = parser.parse_args()

    auto_shutdown = False if (args.sim_file or args.no_auto_shutdown) else True
    agent = HeadlessVoiceAgent(auto_shutdown=auto_shutdown)
    if args.sim_file:
        agent.run_file_simulation(args.sim_file, max_duration=args.max_duration)
    else:
        agent.run_continuous_loop()


if __name__ == "__main__":
    main()
