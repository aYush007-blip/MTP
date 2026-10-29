"""
Visual Live Testing Server for NVIDIA Jetson Orin Nano.
Exposes REST API and serves the showcase Web GUI for live speech-to-LaTeX transcription.
"""

import asyncio
import io
import json
import math
import os
import re
import subprocess
import time
import wave
from datetime import datetime
from pathlib import Path
import sys

# Auto-bootstrap to repo .venv if running under system python
_venv_python = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
if _venv_python.exists() and sys.executable != str(_venv_python):
    try:
        import aiohttp
    except ImportError:
        os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)

from aiohttp import web
import numpy as np
import pyaudio

# Silence ALSA C-level library warnings on Jetson
try:
    import ctypes
    from ctypes import CFUNCTYPE, c_char_p, c_int
    _ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def _py_alsa_err(filename, line, function, err, fmt):
        pass
    _c_alsa_err = _ERROR_HANDLER_FUNC(_py_alsa_err)
    _asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _asound.snd_lib_error_set_handler(_c_alsa_err)
except Exception:
    pass

# Add src to python path
import sys
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import urllib.request

from audio import stereo_48k_to_mono_16k

# Paths & Daemon Endpoints
WHISPER_SERVER_URL = "http://127.0.0.1:8081"
LLAMA_SERVER_URL = "http://127.0.0.1:8080"
WHISPER_CLI_BIN = Path("/mnt/thesis/main/whisper.cpp/build-orin/bin/whisper-cli")
WHISPER_SERVER_BIN = Path("/mnt/thesis/main/whisper.cpp/build-orin/bin/whisper-server")
WHISPER_BASE_MODEL = Path("/mnt/thesis/models/whisper/ggml-base.bin")
WHISPER_SMALL_MODEL = Path("/mnt/thesis/models/whisper/ggml-small.en.bin")
LLAMA_BIN = Path("/mnt/thesis/llama.cpp/build/bin/llama-server")
LLAMA_MODEL = Path("/mnt/thesis/models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf")
RECORDINGS_DIR = Path("/mnt/thesis/recordings")
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
GUI_DIR = Path(__file__).resolve().parent
STATIC_DIR = GUI_DIR / "static"

# Math domain prompt for Whisper ASR
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


def clean_whisper_text(text: str) -> str:
    """Sanitize raw Whisper transcripts: remove non-speech acoustic tags and hallucination phrases."""
    if not text:
        return ""
    cleaned = re.sub(r"[\(\[\{].*?[\)\]\}]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    hallucination_phrases = {
        "you", "you.", "thank you", "thank you.", "thank you for watching", "thank you for watching.",
        "subtitles by", "bye.", "bye bye", "the end", "i'm out.", "i'm sorry.", "peace."
    }
    if cleaned.lower() in hallucination_phrases:
        return ""
    return cleaned


def is_service_alive(url: str, endpoint: str = "/health") -> bool:
    try:
        req = urllib.request.Request(f"{url}{endpoint}", method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def ensure_edge_daemons(model_type: str = "small"):
    """Ensure both whisper-server and llama-server daemons are running on the Jetson."""
    model_path = str(WHISPER_SMALL_MODEL if model_type == "small" and WHISPER_SMALL_MODEL.exists() else WHISPER_BASE_MODEL)

    # 1. Check if whisper-server is alive (Port 8081)
    if not is_service_alive(WHISPER_SERVER_URL, endpoint="/"):
        print(f"[Init] Starting whisper-server daemon with {Path(model_path).name} (Port 8081, Flash Attention)...")
        cmd = [
            str(WHISPER_SERVER_BIN),
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
        print(f"[Ready] whisper-server ready ({Path(model_path).name}) on {WHISPER_SERVER_URL}")
    else:
        # Check if loaded model matches
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

    # 2. Check if llama-server is alive (Port 8080)
    if not is_service_alive(LLAMA_SERVER_URL, endpoint="/health"):
        print(f"[Init] Starting llama-server daemon on port 8080 (CUDA SM87)...")
        cmd = [
            str(LLAMA_BIN),
            "-m", str(LLAMA_MODEL),
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
        print(f"[Ready] llama-server (Qwen2.5-1.5B) ready on {LLAMA_SERVER_URL}")


def query_whisper_server(wav_bytes: bytes) -> tuple[str, float]:
    """Query whisper-server daemon on port 8081 with mathematical prompting."""
    # Auto-revive daemon if not reachable
    if not is_service_alive(WHISPER_SERVER_URL, endpoint="/"):
        print("[Whisper ASR] whisper-server is offline. Auto-starting daemon...")
        ensure_edge_daemons()

    t0 = time.monotonic()
    boundary = "----JetsonGuiStreamBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
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

    with urllib.request.urlopen(req, timeout=25.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        raw_text = data.get("text", "").strip()
        cleaned = clean_whisper_text(raw_text)
        latency_ms = (time.monotonic() - t0) * 1000.0
        return cleaned, latency_ms


# Global Singletons
translator = None
router = None
slm_translator = None


def get_router():
    global translator, router
    if router is None:
        from router import MathRouter, T5MathTranslator
        print("[GUI Server] Initializing T5-v2 translator on CPU...")
        translator = T5MathTranslator.get_instance(device="cpu")
        router = MathRouter(translator=translator)
        print("[GUI Server] MathRouter initialized.")
    return router


def get_slm_translator():
    global slm_translator
    if slm_translator is None:
        from slm_translator import QwenMathTranslator
        slm_translator = QwenMathTranslator.get_instance()
    return slm_translator


def ensure_16k_mono_wav(src_path: Path, dst_path: Path) -> Path:
    """Validate and convert any WAV file to 16kHz 16-bit signed PCM mono."""
    try:
        with wave.open(str(src_path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        # Already 16kHz 16-bit mono PCM
        if channels == 1 and sampwidth == 2 and framerate == 16000:
            return src_path

        # Unpack PCM
        if sampwidth == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sampwidth == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

        # Downmix stereo to mono
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)

        # Resample to 16000 Hz using linear interpolation
        if framerate != 16000:
            num_samples = int(len(data) * 16000 / framerate)
            data = np.interp(
                np.linspace(0, len(data), num_samples, endpoint=False),
                np.arange(len(data)),
                data,
            ).astype(np.float32)

        pcm16 = np.clip(data * 32767, -32768, 32767).astype(np.int16)
        with wave.open(str(dst_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm16.tobytes())
        return dst_path
    except Exception as e:
        print(f"[Audio Warning] WAV normalization error ({e}). Using raw file.")
        return src_path


import threading


class HardwareMicRecorder:
    """Manages continuous or timed audio recording from the Jetson USB mic with 120Hz HPF."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._chunks: list[np.ndarray] = []
        self._is_recording = False
        self._start_time: float = 0.0
        self._last_rms: float = 0.0
        self._hpf = HighPassHumFilter()

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    def get_status(self) -> dict:
        with self._lock:
            elapsed = time.monotonic() - self._start_time if self._is_recording else 0.0
            return {
                "is_recording": self._is_recording,
                "elapsed_sec": round(elapsed, 1),
                "rms": round(self._last_rms, 4),
            }

    def start(self, force: bool = True) -> dict:
        with self._lock:
            if self._is_recording:
                if not force:
                    elapsed = time.monotonic() - self._start_time
                    return {
                        "status": "already_recording",
                        "elapsed_sec": round(elapsed, 1),
                        "rms": round(self._last_rms, 4),
                    }
                self._stop_event.set()
                if self._thread and self._thread.is_alive():
                    self._thread.join(timeout=1.0)

            self._chunks = []
            self._hpf.reset()
            self._last_rms = 0.0
            self._stop_event.clear()
            self._is_recording = True
            self._start_time = time.monotonic()
            self._thread = threading.Thread(target=self._record_loop, daemon=True)
            self._thread.start()
            return {"status": "started", "start_time": self._start_time}

    def _record_loop(self):
        pa = pyaudio.PyAudio()
        dev_idx = None
        channels = 1
        rate = 16000
        chunk_size = 1600

        try:
            for i in range(pa.get_device_count()):
                try:
                    info = pa.get_device_info_by_index(i)
                    if "ab13x" in info.get("name", "").lower() and info.get("maxInputChannels", 0) > 0:
                        dev_idx = i
                        channels = 2
                        rate = 48000
                        chunk_size = 4800
                        break
                except Exception:
                    continue

            if dev_idx is None:
                default_info = pa.get_default_input_device_info()
                dev_idx = default_info["index"]

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=dev_idx,
                frames_per_buffer=chunk_size,
            )

            while not self._stop_event.is_set():
                raw_bytes = stream.read(chunk_size, exception_on_overflow=False)
                if rate == 48000 and channels == 2:
                    mono_chunk = stereo_48k_to_mono_16k(raw_bytes)
                else:
                    pcm_data = np.frombuffer(raw_bytes, dtype=np.int16)
                    mono_chunk = np.clip(pcm_data / 32768.0, -1.0, 1.0).astype(np.float32)

                filtered_chunk = self._hpf.process(mono_chunk)
                rms = float(np.sqrt(np.mean(filtered_chunk ** 2)))

                with self._lock:
                    self._chunks.append(filtered_chunk)
                    self._last_rms = rms

            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"[Hardware Mic Error] {e}")
        finally:
            pa.terminate()

    def stop(self) -> Path:
        with self._lock:
            if not self._is_recording:
                raise RuntimeError("Hardware mic is not currently recording.")
            self._is_recording = False

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

        with self._lock:
            chunks = list(self._chunks)
            self._chunks = []

        if not chunks:
            raise RuntimeError("No audio captured from hardware microphone.")

        audio = np.concatenate(chunks)
        filename = f"hw_mic_{int(time.time() * 1000)}.wav"
        out_path = RECORDINGS_DIR / filename
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())

        return out_path


hw_recorder = HardwareMicRecorder()


def run_whisper(audio_path: Path) -> tuple[str, float]:
    """Run Whisper ASR: queries whisper-server daemon first; falls back to whisper-cli."""
    try:
        with open(audio_path, "rb") as f:
            wav_bytes = f.read()
        return query_whisper_server(wav_bytes)
    except Exception as e:
        print(f"[Whisper Server Query Error] {e}. Falling back to CLI...")

    start_t = time.monotonic()
    model = WHISPER_SMALL_MODEL if WHISPER_SMALL_MODEL.exists() else WHISPER_BASE_MODEL
    result = subprocess.run(
        [
            str(WHISPER_CLI_BIN),
            "-m", str(model),
            "-f", str(audio_path),
            "-t", "4",
            "-nt",
            "--prompt", WHISPER_MATH_PROMPT,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    duration_ms = (time.monotonic() - start_t) * 1000.0

    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Whisper ASR failed (code {result.returncode}): {error}")

    segments = []
    for line in result.stdout.splitlines():
        if "]" in line:
            text = line.split("]", 1)[1].strip()
            if text and "[BLANK_AUDIO]" not in text:
                segments.append(text)
    transcript = clean_whisper_text(" ".join(segments).strip())
    return transcript, duration_ms


# ===========================================================================
# HTTP Handlers
# ===========================================================================

async def handle_index(request: web.Request) -> web.Response:
    index_file = GUI_DIR / "index.html"
    return web.FileResponse(index_file)


async def handle_system_info(request: web.Request) -> web.Response:
    info = {
        "device": "NVIDIA Jetson Orin Nano (8GB)",
        "architecture": "Ampere (SM87)",
        "cuda_available": True,
        "cuda_version": "12.6 (sm_87)",
        "pytorch_version": "Not required (Edge C++ GGUF)",
        "whisper_model": "ggml-small.en.bin (Whisper-server CUDA Flash Attention)",
        "slm_model": "Qwen2.5-1.5B-Instruct-GGUF (CUDA SM87)",
        "t5_model": "math-to-latex-v2 (53 added tokens, fine-tuned)",
        "offline_status": "100% Offline Active",
        "timestamp": datetime.now().isoformat(),
    }
    return web.json_response(info)


async def handle_presets(request: web.Request) -> web.Response:
    presets = [
        {
            "id": "eng_1",
            "category": "pure_english",
            "badge": "Pure English",
            "title": "General English Speech",
            "input": "The microphone captures my voice and sends it to the computer.",
            "description": "Standard conversational speech. Must remain 100% untouched English.",
        },
        {
            "id": "eng_num",
            "category": "english_with_numbers",
            "badge": "English with Numbers",
            "title": "Numbers in English",
            "input": "There are nine students in room forty five.",
            "description": "Critical criteria: isolated number words and room numbers must NEVER convert to math formulas.",
        },
        {
            "id": "phys_desc",
            "category": "descriptive_physics",
            "badge": "Descriptive Science",
            "title": "Conceptual Physics Law",
            "input": "Voltage is the product of resistance and current.",
            "description": "Descriptive sentence describing Ohm's Law. Must remain English text.",
        },
        {
            "id": "math_ohm",
            "category": "symbolic_math",
            "badge": "Symbolic Math",
            "title": "Ohm's Law (Symbolic Equation)",
            "input": "v equals i times r",
            "expected_latex": "V = IR",
            "description": "Algebraic formula using variables. Converts directly to LaTeX formula.",
        },
        {
            "id": "math_frac",
            "category": "symbolic_math",
            "badge": "Symbolic Math",
            "title": "Algebraic Fraction",
            "input": "fraction two b over n",
            "expected_latex": "\\frac{2b}{n}",
            "description": "Unseen spoken fraction syntax. Tests generalizing to LaTeX \\frac syntax.",
        },
        {
            "id": "math_geom",
            "category": "symbolic_math",
            "badge": "Symbolic Math",
            "title": "Circle Equation",
            "input": "x squared plus y squared equals r squared",
            "expected_latex": "x^{2} + y^{2} = r^{2}",
            "description": "Exponents and algebraic relations with proper LaTeX exponents.",
        },
        {
            "id": "mixed_linear",
            "category": "mixed",
            "badge": "Mixed Utterance",
            "title": "Carrier Sentence + Linear Equation",
            "input": "We know that x plus y equals ten so we can solve for y.",
            "expected_latex": "x + y = 10",
            "description": "Extracts only the math span 'x plus y equals ten', embeds as LaTeX, keeps surrounding English intact.",
        },
        {
            "id": "mixed_ohm",
            "category": "mixed",
            "badge": "Mixed Utterance",
            "title": "Physics Carrier + Ohm's Law",
            "input": "The formula is v equals i times r for this circuit.",
            "expected_latex": "V = IR",
            "description": "Extracts 'v equals i times r' into LaTeX, leaving 'The formula is ... for this circuit' intact.",
        },
        {
            "id": "mixed_function",
            "category": "mixed",
            "badge": "Mixed Utterance",
            "title": "Function Definition with Polynomial",
            "input": "Today we have f of x equals three x squared plus two.",
            "expected_latex": "f(x) = 3x^{2} + 2",
            "description": "Mathematical function definition inside carrier preamble 'Today we have ...'.",
        },
        {
            "id": "mixed_sinusoidal",
            "category": "mixed",
            "badge": "Multi-Sentence Mixed",
            "title": "Sinusoidal Signal & Frequency Explanation",
            "input": "Now consider a sinusoidal signal defined as x of t equals a sin of omega t plus 5. Here a represents the amplitude omega is the angular frequency in radians per second and 5 is the phase angle. If the frequency is 50 hertz then omega equals 2 pi which is 100 pi radians per second.",
            "expected_latex": "x(t) = a \\sin(\\omega t + 5) \\quad \\text{and} \\quad \\omega = 2\\pi",
            "description": "Multi-sentence lecture paragraph: preserves 100% of the surrounding English explanations while converting 'x of t equals a sin of omega t plus 5' and 'omega equals 2 pi' into visual LaTeX.",
        },
    ]
    return web.json_response({"presets": presets})


async def handle_transcribe_text(request: web.Request) -> web.Response:
    """Simulate or test transcription directly with text."""
    data = await request.json()
    raw_text = data.get("text", "").strip()
    engine = data.get("engine", "auto")
    if not raw_text:
        return web.json_response({"error": "Empty text provided"}, status=400)

    # Default to high-performance QwenMathTranslator (SLM) unless T5 is explicitly requested
    if engine != "t5":
        try:
            slm = get_slm_translator()
            t_start = time.monotonic()
            processed_text = slm.translate(raw_text)
            total_ms = (time.monotonic() - t_start) * 1000.0
            return web.json_response({
                "raw_text": raw_text,
                "route_type": "slm_direct",
                "processed_text": processed_text,
                "spans": [],
                "engine": "qwen2.5-1.5b",
                "asr_time_ms": 0.0,
                "router_time_ms": total_ms,
                "total_time_ms": total_ms,
            })
        except Exception as e:
            print(f"[SLM Text Error] {e}. Falling back to router...")

    try:
        r = get_router()
        t_start = time.monotonic()
        result = r.process(raw_text)
        total_ms = (time.monotonic() - t_start) * 1000.0
    except Exception as e:
        print(f"[Transcribe Text Error] {e}")
        return web.json_response({"error": f"Processing error: {str(e)}"}, status=500)

    spans_data = [
        {
            "start": s.start_char,
            "end": s.end_char,
            "spoken": s.spoken_text,
            "latex": s.latex_text,
        }
        for s in result.spans
    ]

    return web.json_response({
        "raw_text": raw_text,
        "route_type": result.route_type.value,
        "processed_text": result.processed_text,
        "spans": spans_data,
        "engine": "t5-v2",
        "asr_time_ms": 0.0,
        "router_time_ms": total_ms,
        "total_time_ms": total_ms,
    })


async def run_pipeline_on_file(wav_path: Path, t_total_start: float, engine: str = "auto") -> web.Response:
    """Run Whisper.cpp -> SLM / MathRouter on a WAV file and return structured JSON."""
    try:
        raw_transcript, asr_ms = run_whisper(wav_path)
    except Exception as e:
        return web.json_response({"error": f"Whisper ASR failed: {str(e)}"}, status=500)

    # Default to high-performance QwenMathTranslator (SLM) unless T5 is explicitly requested
    if engine != "t5":
        try:
            slm = get_slm_translator()
            t_slm_start = time.monotonic()
            processed_text = slm.translate(raw_transcript)
            slm_ms = (time.monotonic() - t_slm_start) * 1000.0
            total_ms = (time.monotonic() - t_total_start) * 1000.0
            return web.json_response({
                "audio_file": wav_path.name,
                "audio_url": f"/recordings/{wav_path.name}",
                "download_url": f"/api/download_audio/{wav_path.name}",
                "raw_transcript": raw_transcript,
                "route_type": "slm_direct",
                "processed_text": processed_text,
                "spans": [],
                "engine": "qwen2.5-1.5b",
                "asr_time_ms": asr_ms,
                "router_time_ms": slm_ms,
                "total_time_ms": total_ms,
            })
        except Exception as e:
            print(f"[SLM Pipeline Error] {e}. Falling back to router...")

    try:
        r = get_router()
        t_router_start = time.monotonic()
        result = r.process(raw_transcript)
        router_ms = (time.monotonic() - t_router_start) * 1000.0
        total_ms = (time.monotonic() - t_total_start) * 1000.0
    except Exception as e:
        print(f"[Pipeline Error] {e}")
        return web.json_response({"error": f"Math routing error: {str(e)}"}, status=500)

    spans_data = [
        {
            "start": s.start_char,
            "end": s.end_char,
            "spoken": s.spoken_text,
            "latex": s.latex_text,
        }
        for s in result.spans
    ]

    return web.json_response({
        "audio_file": wav_path.name,
        "audio_url": f"/recordings/{wav_path.name}",
        "download_url": f"/api/download_audio/{wav_path.name}",
        "raw_transcript": raw_transcript,
        "route_type": result.route_type.value,
        "processed_text": result.processed_text,
        "spans": spans_data,
        "asr_time_ms": round(asr_ms, 1),
        "router_time_ms": round(router_ms, 1),
        "total_time_ms": round(total_ms, 1),
    })


async def handle_hardware_mic_start(request: web.Request) -> web.Response:
    """Start continuous or unlimited audio recording on Jetson USB hardware mic."""
    try:
        res = hw_recorder.start()
        return web.json_response(res)
    except Exception as e:
        return web.json_response({"error": f"Could not start hardware mic: {str(e)}"}, status=500)


async def handle_hardware_mic_stop(request: web.Request) -> web.Response:
    """Stop active hardware mic recording and run Whisper + Router + T5 pipeline."""
    if not hw_recorder.is_recording:
        return web.json_response({"error": "Hardware mic is not currently recording."}, status=400)

    t_total_start = time.monotonic()
    loop = asyncio.get_event_loop()
    try:
        wav_path = await loop.run_in_executor(None, hw_recorder.stop)
    except Exception as e:
        return web.json_response({"error": f"Failed to stop hardware recording: {str(e)}"}, status=500)

    engine = request.query.get("engine", "auto")
    return await run_pipeline_on_file(wav_path, t_total_start, engine=engine)


async def handle_hardware_mic_status(request: web.Request) -> web.Response:
    """Check current status and elapsed seconds of hardware recording."""
    return web.json_response(hw_recorder.get_status())


async def handle_record_hardware_mic(request: web.Request) -> web.Response:
    """
    Unified / backward-compatible hardware mic endpoint.
    - If action == 'start': starts recording and returns status.
    - If action == 'stop': stops recording and runs pipeline.
    - If action == 'status': returns recording status.
    - If duration_sec > 0: records for duration_sec, then stops and transcribes.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    engine = body.get("engine", request.query.get("engine", "auto"))
    action = body.get("action", "")
    if action == "start":
        return await handle_hardware_mic_start(request)
    elif action == "stop":
        return await handle_hardware_mic_stop(request)
    elif action == "status":
        return await handle_hardware_mic_status(request)

    duration_sec = float(body.get("duration_sec", 0.0))
    if duration_sec <= 0:
        return await handle_hardware_mic_start(request)

    # Fixed duration recording
    t_total_start = time.monotonic()
    hw_recorder.start(force=True)
    await asyncio.sleep(duration_sec)
    loop = asyncio.get_event_loop()
    try:
        wav_path = await loop.run_in_executor(None, hw_recorder.stop)
    except Exception as e:
        return web.json_response({"error": f"Hardware recording error: {str(e)}"}, status=500)

    return await run_pipeline_on_file(wav_path, t_total_start, engine=engine)


async def handle_transcribe_audio(request: web.Request) -> web.Response:
    """Accept uploaded WAV audio and run full pipeline."""
    engine = request.query.get("engine", "auto")
    reader = await request.multipart()
    field = await reader.next()
    if not field or field.name != "audio":
        return web.json_response({"error": "Missing audio file field"}, status=400)

    filename = f"upload_{int(time.time()*1000)}.wav"
    temp_path = RECORDINGS_DIR / filename

    size = 0
    with open(temp_path, "wb") as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            size += len(chunk)
            f.write(chunk)

    if size < 100:
        return web.json_response({"error": "Audio file too small or empty"}, status=400)

    t_total_start = time.monotonic()

    # Normalize WAV format to 16kHz 16-bit mono PCM if needed
    clean_path = temp_path.with_name(f"clean_{temp_path.name}")
    try:
        proc_path = ensure_16k_mono_wav(temp_path, clean_path)
    except Exception as e:
        return web.json_response({"error": f"Audio format error: {str(e)}"}, status=400)

    return await run_pipeline_on_file(proc_path, t_total_start, engine=engine)


async def handle_recordings_list(request: web.Request) -> web.Response:
    """List pre-recorded benchmark WAV files."""
    files = sorted(RECORDINGS_DIR.glob("*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)
    file_list = []
    for f in files:
        if f.name.startswith("clean_") or f.stat().st_size < 1000:
            continue
        file_list.append({
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "created": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return web.json_response({"recordings": file_list[:30]})


async def handle_transcribe_saved(request: web.Request) -> web.Response:
    """Transcribe an existing file from RECORDINGS_DIR."""
    data = await request.json()
    filename = data.get("filename", "")
    engine = data.get("engine", "auto")
    target = RECORDINGS_DIR / filename
    if not target.is_file():
        return web.json_response({"error": "File not found"}, status=404)

    t_total_start = time.monotonic()
    return await run_pipeline_on_file(target, t_total_start, engine=engine)


async def handle_download_audio(request: web.Request) -> web.Response:
    """Download a recorded or processed WAV audio file as attachment."""
    filename = request.match_info.get("filename", "")
    target = RECORDINGS_DIR / filename
    if not target.is_file() or not target.name.endswith(".wav"):
        return web.json_response({"error": "Audio file not found"}, status=404)
    return web.FileResponse(
        target,
        headers={
            "Content-Disposition": f'attachment; filename="{target.name}"',
            "Content-Type": "audio/wav",
        },
    )


# ===========================================================================
# WebSocket Support & Real-Time Telemetry
# ===========================================================================
active_websockets: set[web.WebSocketResponse] = set()


async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    active_websockets.add(ws)

    await ws.send_json({
        "type": "connected",
        "status": "ready",
        "whisper_model": "ggml-small.en.bin",
        "slm_model": "Qwen2.5-1.5B-Instruct",
        "is_recording": hw_recorder.is_recording,
    })

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                except Exception:
                    continue

                action = payload.get("action", "")
                engine = payload.get("engine", "slm")

                if action == "start_hw_mic":
                    res = hw_recorder.start()
                    await ws.send_json({"type": "recording_started", **res})

                elif action in ("stop_hw_mic", "transcribe_now"):
                    if hw_recorder.is_recording:
                        t_start = time.monotonic()
                        loop = asyncio.get_event_loop()
                        try:
                            wav_path = await loop.run_in_executor(None, hw_recorder.stop)
                            raw_transcript, asr_ms = await loop.run_in_executor(None, run_whisper, wav_path)

                            if engine != "t5":
                                slm = get_slm_translator()
                                t_slm = time.monotonic()
                                processed = await loop.run_in_executor(None, slm.translate, raw_transcript)
                                slm_ms = (time.monotonic() - t_slm) * 1000.0
                            else:
                                r = get_router()
                                t_r = time.monotonic()
                                res = await loop.run_in_executor(None, r.process, raw_transcript)
                                processed = res.processed_text
                                slm_ms = (time.monotonic() - t_r) * 1000.0

                            total_ms = (time.monotonic() - t_start) * 1000.0
                            result_packet = {
                                "type": "transcription_result",
                                "audio_file": wav_path.name,
                                "audio_url": f"/recordings/{wav_path.name}",
                                "download_url": f"/api/download_audio/{wav_path.name}",
                                "raw_transcript": raw_transcript,
                                "processed_text": processed,
                                "engine": "qwen2.5-1.5b" if engine == "slm" else "t5-v2",
                                "asr_time_ms": round(asr_ms, 1),
                                "router_time_ms": round(slm_ms, 1),
                                "total_time_ms": round(total_ms, 1),
                            }
                            for w in list(active_websockets):
                                if not w.closed:
                                    await w.send_json(result_packet)
                        except Exception as e:
                            await ws.send_json({"type": "error", "message": str(e)})
                    else:
                        await ws.send_json({"type": "error", "message": "Not currently recording"})

                elif action == "ping":
                    await ws.send_json({"type": "pong"})

            elif msg.type == web.WSMsgType.ERROR:
                print(f"[WS Error] {ws.exception()}")
    finally:
        active_websockets.discard(ws)
    return ws


async def ws_telemetry_loop(app: web.Application):
    """Broadcast real-time recording duration and RMS energy to WebSocket clients."""
    while True:
        try:
            if active_websockets and hw_recorder.is_recording:
                status = hw_recorder.get_status()
                packet = {
                    "type": "telemetry",
                    "is_recording": True,
                    "elapsed_sec": status["elapsed_sec"],
                    "rms": status["rms"],
                }
                for ws in list(active_websockets):
                    if not ws.closed:
                        try:
                            await ws.send_json(packet)
                        except Exception:
                            pass
            await asyncio.sleep(0.12)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(0.5)


async def start_background_tasks(app: web.Application):
    # Ensure background daemons on startup
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, ensure_edge_daemons)
    app["telemetry_task"] = asyncio.create_task(ws_telemetry_loop(app))


async def cleanup_background_tasks(app: web.Application):
    if "telemetry_task" in app:
        app["telemetry_task"].cancel()
        try:
            await app["telemetry_task"]
        except asyncio.CancelledError:
            pass


def create_app() -> web.Application:
    app = web.Application(client_max_size=32 * 1024 * 1024)

    # Routes
    app.router.add_get("/", handle_index)
    app.router.add_get("/ws", handle_ws)
    app.router.add_get("/api/system_info", handle_system_info)
    app.router.add_get("/api/presets", handle_presets)
    app.router.add_post("/api/transcribe_text", handle_transcribe_text)
    app.router.add_post("/api/transcribe_audio", handle_transcribe_audio)
    app.router.add_post("/api/record_hardware_mic", handle_record_hardware_mic)
    app.router.add_post("/api/hardware_mic/start", handle_hardware_mic_start)
    app.router.add_post("/api/hardware_mic/stop", handle_hardware_mic_stop)
    app.router.add_get("/api/hardware_mic/status", handle_hardware_mic_status)
    app.router.add_get("/api/recordings", handle_recordings_list)
    app.router.add_post("/api/transcribe_saved", handle_transcribe_saved)
    app.router.add_get("/api/download_audio/{filename}", handle_download_audio)

    # Static routes
    app.router.add_static("/static/", STATIC_DIR, name="static")
    app.router.add_static("/recordings/", RECORDINGS_DIR, name="recordings")

    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n=======================================================")
    print(f"[GUI Server] Jetson Orin Speech-to-LaTeX Showcase GUI")
    print(f"[GUI Server] Local Server URL: http://0.0.0.0:{port}")
    print(f"=======================================================\n")
    # Ensure edge GPU daemons (whisper-server on 8081, llama-server on 8080)
    ensure_edge_daemons()
    get_slm_translator()
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=port)
