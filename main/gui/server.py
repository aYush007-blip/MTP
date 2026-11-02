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
import zipfile
import collections
from collections import Counter
import queue
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional, Dict, Any, List, Tuple
import urllib.request

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


def to_16k_mono(raw_bytes: bytes, rate: int, channels: int) -> np.ndarray:
    """Safely convert any incoming raw PCM byte buffer into 16000Hz mono float32 normalized in [-1.0, 1.0]."""
    if rate == 48000 and channels == 2:
        return stereo_48k_to_mono_16k(raw_bytes)
    pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if rate == 16000:
        return np.clip(pcm, -1.0, 1.0).astype(np.float32)
    elif rate == 48000:
        return np.clip(pcm[::3], -1.0, 1.0).astype(np.float32)
    else:
        num_target = int(len(pcm) * 16000 / rate)
        resampled = np.interp(np.linspace(0, len(pcm), num_target, endpoint=False), np.arange(len(pcm)), pcm)
        return np.clip(resampled, -1.0, 1.0).astype(np.float32)


def get_best_input_stream_config(pa: pyaudio.PyAudio, requested_idx: Optional[int] = None, requested_name: Optional[str] = None) -> tuple[int, int, int, int]:
    """Negotiate the optimal (device_index, channels, rate, chunk_size) supported by hardware."""
    target_idx = requested_idx

    # If requested_name was specified, attempt to resolve it to a PyAudio device index
    if target_idx is None and requested_name:
        req_low = requested_name.lower()
        if "hw:" in req_low:
            hw_tag = req_low[req_low.find("hw:"):req_low.find("hw:")+4]
            for i in range(pa.get_device_count()):
                try:
                    info = pa.get_device_info_by_index(i)
                    if hw_tag in info.get("name", "").lower() and info.get("maxInputChannels", 0) > 0:
                        target_idx = i
                        break
                except Exception:
                    continue

        if target_idx is None:
            for i in range(pa.get_device_count()):
                try:
                    info = pa.get_device_info_by_index(i)
                    name_low = info.get("name", "").lower()
                    if ("ab13x" in req_low or "usb" in req_low) and ("ab13x" in name_low or "usb" in name_low) and info.get("maxInputChannels", 0) > 0:
                        target_idx = i
                        break
                    elif ("platform" in req_low or "built-in" in req_low) and "platform" in name_low and info.get("maxInputChannels", 0) > 0:
                        target_idx = i
                        break
                except Exception:
                    continue

    if target_idx is None:
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
                name_low = info.get("name", "").lower()
                if ("ab13x" in name_low or "usb" in name_low) and info.get("maxInputChannels", 0) > 0:
                    target_idx = i
                    break
            except Exception:
                continue

    if target_idx is None:
        try:
            default_info = pa.get_default_input_device_info()
            target_idx = default_info["index"]
        except Exception:
            target_idx = 0

    info = pa.get_device_info_by_index(target_idx)
    name = info.get("name", "")
    max_ch = max(1, int(info.get("maxInputChannels", 1)))
    def_rate = int(info.get("defaultSampleRate", 16000))

    if "ab13x" in name.lower() or "hw:2" in name.lower() or "hw:" in name.lower():
        candidates = [(48000, 2), (48000, 1), (16000, 1)]
    else:
        candidates = [(16000, 1), (48000, 2), (48000, 1), (def_rate, min(max_ch, 2))]

    for rate, ch in candidates:
        try:
            if pa.is_format_supported(rate, input_device=target_idx, input_channels=ch, input_format=pyaudio.paInt16):
                chunk = int(rate * 0.1)  # 100ms chunks
                return target_idx, ch, rate, chunk
        except Exception:
            continue

    return target_idx, 1, 16000, 1600


# Paths & Daemon Endpoints
WHISPER_SERVER_URL = "http://127.0.0.1:8081"
LLAMA_SERVER_URL = "http://127.0.0.1:8080"
WHISPER_CLI_BIN = Path("/mnt/thesis/main/whisper.cpp/build-orin/bin/whisper-cli")
WHISPER_SERVER_BIN = Path("/mnt/thesis/main/whisper.cpp/build-orin/bin/whisper-server")
WHISPER_BASE_MODEL = Path("/mnt/thesis/models/whisper/ggml-base.bin")
WHISPER_SMALL_MODEL = Path("/mnt/thesis/models/whisper/ggml-small.en.bin")
LLAMA_BIN = Path("/mnt/thesis/llama.cpp/build/bin/llama-server")
LLAMA_MODEL = (
    Path("/home/ayush02/models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    if Path("/home/ayush02/models/qwen2.5-1.5b-instruct-q4_k_m.gguf").exists()
    else Path("/mnt/thesis/models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf")
)
RECORDINGS_DIR = Path("/mnt/thesis/recordings")
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
LIVE_TESTS_DIR = Path("/mnt/thesis/live_tests")
LIVE_TESTS_DIR.mkdir(parents=True, exist_ok=True)
GUI_DIR = Path(__file__).resolve().parent
STATIC_DIR = GUI_DIR / "static"

# Universal STEM domain prompt for Whisper ASR across all engineering fields
WHISPER_MATH_PROMPT = (
    "Engineering lecture: calculus, linear algebra, differential equations, "
    "mechanics, thermodynamics, fluid dynamics, circuits, control systems, "
    "electromagnetics, algorithms, stress, strain, shear, entropy, enthalpy, heat flux, "
    "curl, divergence, gradient, partial derivative, Navier-Stokes, Reynolds number, "
    "Kirchhoff's laws, KCL, KVL, inductance, resistance, capacitance, flux, current i(t), voltage v(t), "
    "transfer function, eigenvalues, vectors, tensors, matrices, summations, integrals, "
    "state-space, transient, steady state, impulse response, step response, convolution, Bode plot, Nyquist, "
    "discrete-time sequences x[n], y[n], downsampling, upsampling, difference equations, "
    "units: Pascals, Newtons, Joules, Watts, Hertz, Farads, Ohms, Kelvin, meters per second."
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


def clean_speech_disfluencies(text: str) -> str:
    """Pre-clean speech stutters, repeated false starts, and filler hitches before SLM translation."""
    if not text:
        return ""
    s = text.strip()
    # 1. Collapse immediate repeated words: "if if" -> "if", "so so" -> "so", "to to" -> "to", "is is" -> "is"
    s = re.sub(r"\b([a-zA-Z]+)\s+\1\b", r"\1", s, flags=re.IGNORECASE)
    # 2. Collapse repeated two-word phrases: "y of n y of n" -> "y of n", "x of x of" -> "x of"
    s = re.sub(r"\b([a-zA-Z]+\s+[a-zA-Z]+)\s+\1\b", r"\1", s, flags=re.IGNORECASE)
    # 3. Clean conversational filler hesitations when surrounded by mathematical discourse
    s = re.sub(r"\b(?:you know|basically|so basically|like I said|as I said)\b", "", s, flags=re.IGNORECASE)
    # 4. Normalize multiple punctuation and spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()


DANGLING_OPERATOR_PATTERN = re.compile(
    r"(\b(?:equals?|is\s+equal\s+to|is\s+greater\s+than|is\s+less\s+than|"
    r"plus|minus|times|multiplied\s+by|divided\s+by|over|"
    r"raised\s+to|to\s+the\s+power\s+of|"
    r"partial\s+of|derivative\s+of|integral\s+of|sum\s+of|summation\s+from|"
    r"with\s+respect\s+to|evaluated\s+at|"
    r"(?:[a-zA-Z]|theta|omega|phi|psi|sigma|alpha|beta|lambda|mu)\s+of|"
    r"such\s+that|where|defined\s+as|given\s+by)\b|[=+\-*/<>^])$",
    re.IGNORECASE
)


def is_dangling_math_clause(text: str) -> bool:
    """Detect whether a spoken transcript ends abruptly with an open/incomplete mathematical clause or operator."""
    clean = text.strip().rstrip(".,;:!?")
    return bool(DANGLING_OPERATOR_PATTERN.search(clean))


# ===========================================================================
# Fast Linguistic Gating & SLM Math Pre-Checks (Sub-millisecond Latency)
# ===========================================================================

MATH_INDICATOR_PATTERN = re.compile(
    r"\b("
    # Equality, relations, and comparison
    r"equals?|is\s+equal\s+to|greater\s+than|less\s+than|approaches|tends\s+to|"
    # Arithmetic operators
    r"plus|minus|times|multiplied\s+by|divided\s+by|"
    r"(?:[0-9]+|[a-z]|zero|one|two|three|four|five|six|seven|eight|nine|ten|pi|numerator|\))\s+over\s+(?:[0-9]+|[a-z]|zero|one|two|three|four|five|six|seven|eight|nine|ten|pi|denominator)|"
    # Powers and roots
    r"squared|cubed|to\s+the\s+power|square\s+root|cube\s+root|roots?\s+of|"
    # Fractions & polynomials
    r"fraction|numerator|denominator|polynomials?|"
    # Trigonometry & transcendentals
    r"sine|cosine|tangent|cotangent|secant|cosecant|logarithm|\bln\b|exponential|"
    # Calculus & Partials
    r"integral|derivative|dy\s+by\s+d[tx]|partial\s+derivative|partial\s+of|gradient|divergence|curl|"
    # Sequences, series, indexing & discrete time
    r"summation|sum\s+from|sum\s+of|discrete[\s-]time|downsampl\w+|upsampl\w+|sampling\s+rate|difference\s+equation|"
    # Greek letters
    r"sigma|omega|theta|alpha|beta|gamma|lambda|phi|psi|delta|epsilon|tau|mu|rho|kappa|zeta|"
    # Linear algebra & transforms
    r"matrix|matrices|determinant|eigenvalues?|eigenvectors?|transpose|"
    r"z[\s-]transform|laplace|fourier|dtft|"
    # Control systems & signals
    r"transfer\s+function|root\s+locus|poles?|zeros?|characteristic\s+equation|"
    r"frequency\s+response|impulse\s+response|step\s+response|state\s+space|"
    r"bode\s+plot|nyquist|open\s+loop|closed\s+loop|feedback|"
    # Fluid mechanics, thermodynamics & physics
    r"navier[\s-]stokes|reynolds|bernoulli|mach\s+number|prandtl|enthalpy|entropy|internal\s+energy|"
    r"heat\s+transfer|conduction|convection|radiation|stress|strain|shear|viscosity|young\'?s\s+modulus|"
    # Computer science & probability
    r"big\s+o|recurrence|bayes|posterior|prior|probability|variance|standard\s+deviation|"
    # Electrical machines & power systems
    r"synchronous|rotor|stator|flux\s+linkage|direct\s+axis|quadrature\s+axis|d[\s-]axis|q[\s-]axis|"
    r"field\s+winding|damper|reactance|impedance|park\'?s\s+transform\w*|"
    # Functional notation: universal [var] of [arg] (e.g. h of s, f of x, g of t, y of t, x of n, T of r)
    r"(?:[a-z]|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|rho|sigma|tau|phi|psi|omega)\s+of\s+(?:j\s+omega|omega|[a-z0-9]+)|"
    r"sinusoid|kcl|kvl|ohm\'?s\s+law"
    r")\b|"
    r"\b[0-9]+(?:\.[0-9]+)?\s*(?:hz|khz|mhz|ghz|volts?|amps?|amperes?|ohms?|farads?|henrys?|watts?|kev|mev|ev|db|pa|kpa|mpa|gpa|n|kn|j|kj|rad/s)\b|"
    r"[=+\-*/^<>]",
    re.IGNORECASE,
)


def has_math_indicators(text: str) -> bool:
    """Fast linguistic pre-check to distinguish mathematical speech from pure conversational discourse."""
    if not text:
        return False
    return bool(MATH_INDICATOR_PATTERN.search(text))


def merge_text_clauses(prev: str, curr: str) -> str:
    """Merge two consecutive transcript clauses, deduplicating any overlapping words at boundary."""
    if not prev or not prev.strip():
        return curr.strip()
    if not curr or not curr.strip():
        return prev.strip()
    words_p = prev.strip().split()
    words_c = curr.strip().split()
    max_k = min(len(words_p), len(words_c), 6)
    for k in range(max_k, 0, -1):
        if [w.lower().strip(".,;:!?") for w in words_p[-k:]] == [w.lower().strip(".,;:!?") for w in words_c[:k]]:
            return (" ".join(words_p) + " " + " ".join(words_c[k:])).strip()
    return f"{prev.strip()} {curr.strip()}"


def clean_slm_hallucinations(latex_text: str, raw_text: str, context_text: str = "") -> str:
    """Safeguard against conversational assistant filler, prompt leakage, context hijacking, or repetition loops."""
    s = latex_text.strip()
    # Strip leaked prompt tags
    s = re.sub(r"(?:^|\n)\s*\[?(?:Spoken transcript|Preceding context|Translation|Output|Text|Transcription)\]?:\s*", "\n", s, flags=re.IGNORECASE).strip()

    if re.search(r"^(?:I apologize|Thank you for sharing|As an AI|Could you please provide|Here is the)", s, re.IGNORECASE):
        return raw_text

    # Strip verbatim context echo prefix
    if context_text:
        ctx_clean = context_text.strip()
        if s.startswith(ctx_clean):
            rem = s[len(ctx_clean):].strip()
            if rem:
                s = rem
        for line in [l.strip() for l in ctx_clean.split("\n") if len(l.strip()) > 10]:
            if s.startswith(line):
                rem = s[len(line):].strip()
                if rem:
                    s = rem

    words_out = re.findall(r"\b[a-zA-Z0-9_]{3,}\b", s.lower())
    words_raw = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", raw_text.lower()))

    # Length explosion rejection
    if len(words_out) > max(20, len(words_raw) * 3 + 10):
        return raw_text

    # Repetition loop detection
    if len(words_out) >= 9:
        tri_grams = [" ".join(words_out[i:i+3]) for i in range(len(words_out)-2)]
        counts = Counter(tri_grams)
        if any(cnt >= 3 for cnt in counts.values()):
            return raw_text

    # Context echo hijacking rejection
    if context_text and len(words_out) >= 5:
        words_ctx = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", context_text.lower()))
        spoken_overlap = len(set(words_out) & words_raw) / max(1, len(set(words_out)))
        context_overlap = len(set(words_out) & words_ctx) / max(1, len(set(words_out)))
        if context_overlap > 0.6 and spoken_overlap < 0.2:
            return raw_text

    return s


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
        subprocess.Popen(cmd, env=env, stdout=log_whisper, stderr=log_whisper, start_new_session=True)
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
            "-c", "4096",
            "-ngl", "28",
            "-t", "2",
            "-np", "1",
            "--port", "8080",
            "--host", "127.0.0.1"
        ]
        log_llama = open("/tmp/llama-server.log", "w")
        subprocess.Popen(cmd, stdout=log_llama, stderr=log_llama, start_new_session=True)
        for _ in range(50):
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

    with urllib.request.urlopen(req, timeout=90.0) as resp:
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

active_websockets: set[web.WebSocketResponse] = set()
main_event_loop: Optional[asyncio.AbstractEventLoop] = None


def broadcast_ws_message(msg_dict: dict):
    """Safely broadcast JSON payload to all active WebSocket clients."""
    if not active_websockets:
        return
    msg_str = json.dumps(msg_dict)
    for ws in list(active_websockets):
        if not ws.closed:
            try:
                if main_event_loop and main_event_loop.is_running():
                    asyncio.run_coroutine_threadsafe(ws.send_str(msg_str), main_event_loop)
            except Exception:
                pass


# ===========================================================================
# Audio Hardware Device Enumeration & Control (PulseAudio / ALSA)
# ===========================================================================

def get_usb_alsa_card() -> Optional[str]:
    """Return ALSA card index for connected USB audio devices (e.g. AB13X)."""
    try:
        with open("/proc/asound/cards", "r") as f:
            for line in f:
                if "usb" in line.lower() or "ab13x" in line.lower():
                    return line.strip().split()[0]
    except Exception:
        pass
    return None


def ensure_usb_audio_duplex(force_rescan: bool = False) -> None:
    """Ensure any connected USB audio device (e.g. AB13X earphones) is configured in Full Duplex mode.
    
    In PulseAudio, connecting a USB audio device can default to single-direction profiles
    (e.g. output-only analog-stereo or input-only stereo-fallback), which makes input and
    output mutually exclusive in Ubuntu system settings. This function scans cards and enforces
    the simultaneous duplex profile (output:analog-stereo+input:analog-stereo).
    """
    try:
        # Check /proc/asound/cards for USB audio card index
        usb_alsa_card = get_usb_alsa_card()

        # 1. Clean up duplicate module-alsa-card instances if any were previously loaded
        if usb_alsa_card is not None:
            mod_res = subprocess.run(["pactl", "list", "modules", "short"], capture_output=True, text=True, timeout=2)
            if mod_res.returncode == 0:
                loaded_mods = [
                    line.split()[0] for line in mod_res.stdout.splitlines()
                    if f"device_id={usb_alsa_card}" in line or f'device_id="{usb_alsa_card}"' in line
                ]
                if len(loaded_mods) > 1:
                    print(f"[Audio Devices] Unloading {len(loaded_mods)-1} duplicate module-alsa-card instances...")
                    for extra_mod in loaded_mods[1:]:
                        subprocess.run(["pactl", "unload-module", extra_mod], capture_output=True, timeout=2)

        res = subprocess.run(["pactl", "list", "cards"], capture_output=True, text=True, timeout=3)
        if res.returncode != 0:
            return

        cur_card = ""
        cur_desc = ""
        cur_owner_mod = ""
        active_profile = ""
        duplex_candidates = []
        found_usb_card = False

        def _is_usb_card(card_name: str, desc: str) -> bool:
            if not card_name:
                return False
            c_low = card_name.lower()
            d_low = desc.lower()
            if "usb" in c_low or "ab13x" in c_low or "usb" in d_low or "ab13x" in d_low:
                return True
            if usb_alsa_card is not None:
                if c_low == f"alsa_card.{usb_alsa_card}" or c_low.startswith(f"alsa_card.{usb_alsa_card}."):
                    return True
            return False

        def _apply_duplex(card_name: str, card_desc: str, active_p: str, cands: list):
            nonlocal found_usb_card
            if not _is_usb_card(card_name, card_desc):
                return
            found_usb_card = True
            if not force_rescan and (("output:" in active_p and "input:" in active_p) or ("duplex" in active_p.lower())):
                return
            chosen = None
            for cand in cands:
                if "analog-stereo+input:analog-stereo" in cand:
                    chosen = cand
                    break
            if not chosen and cands:
                chosen = cands[0]
            if chosen:
                print(f"[Audio Devices] Auto-enforcing duplex profile '{chosen}' for card {card_name} (was: {active_p})")
                subprocess.run(["pactl", "set-card-profile", card_name, chosen], capture_output=True, timeout=2)

        for line in res.stdout.splitlines():
            line_str = line.strip()
            if line_str.startswith("Name:"):
                _apply_duplex(cur_card, cur_desc, active_profile, duplex_candidates)
                cur_card = line_str.split(":", 1)[1].strip()
                cur_desc = ""
                cur_owner_mod = ""
                active_profile = ""
                duplex_candidates = []
            elif line_str.startswith("device.description ="):
                cur_desc = line_str.split("=", 1)[1].strip().strip('"')
            elif line_str.startswith("Owner Module:"):
                cur_owner_mod = line_str.split(":", 1)[1].strip()
            elif line_str.startswith("Active Profile:"):
                active_profile = line_str.split(":", 1)[1].strip()
            elif "output:" in line_str and "input:" in line_str:
                m = re.match(r"^\s*([a-zA-Z0-9_\-\+:]+):\s+(.*)$", line_str)
                if m:
                    pname = m.group(1)
                    pdesc = m.group(2)
                    if "sinks: 0" not in pdesc and "sources: 0" not in pdesc:
                        duplex_candidates.append(pname)

        _apply_duplex(cur_card, cur_desc, active_profile, duplex_candidates)

        # If USB card exists in ALSA, verify PulseAudio has an active NON-MONITOR input source for it
        if usb_alsa_card is not None:
            src_res = subprocess.run(["pactl", "list", "sources", "short"], capture_output=True, text=True, timeout=2)
            has_usb_input_source = False
            if src_res.returncode == 0:
                for s_line in src_res.stdout.splitlines():
                    parts = s_line.split()
                    if len(parts) >= 2:
                        s_id = parts[1].lower()
                        # Exclude sink monitors (e.g. *.monitor)
                        if not s_id.endswith(".monitor") and ("ab13x" in s_id or "usb" in s_id):
                            has_usb_input_source = True
                            break

            if not has_usb_input_source:
                print(f"[Audio Devices] Loading PulseAudio ALSA source for AB13X card {usb_alsa_card}...")
                subprocess.run(
                    [
                        "pactl", "load-module", "module-alsa-source",
                        f"device=hw:{usb_alsa_card},0", "rate=48000", "channels=2",
                        "source_name=ab13x_earphones_mic",
                        "source_properties=device.description=AB13X_USB_Earphones_Mic"
                    ],
                    capture_output=True,
                    timeout=2
                )
                subprocess.run(["pactl", "set-default-source", "ab13x_earphones_mic"], capture_output=True, timeout=1)

        # If USB card exists in ALSA but was completely absent in PulseAudio cards, load it ONCE
        if not found_usb_card and usb_alsa_card is not None:
            mod_res = subprocess.run(["pactl", "list", "modules", "short"], capture_output=True, text=True, timeout=2)
            has_mod = any(f"device_id={usb_alsa_card}" in l or f'device_id="{usb_alsa_card}"' in l for l in mod_res.stdout.splitlines())
            if not has_mod:
                print(f"[Audio Devices] USB card {usb_alsa_card} in ALSA but not in PulseAudio. Loading module-alsa-card...")
                subprocess.run(["pactl", "load-module", "module-alsa-card", f"device_id={usb_alsa_card}"], capture_output=True, timeout=2)

    except Exception as e:
        print(f"[Audio Devices] Error checking USB duplex: {e}")


def get_pulse_audio_devices(force_rescan: bool = False) -> dict:
    """Retrieve available PulseAudio and ALSA audio sources (microphones) and sinks (earphones/speakers)."""
    # 1. Guarantee that any connected USB earphones have duplex mode unlocked
    ensure_usb_audio_duplex(force_rescan=force_rescan)

    inputs = []
    outputs = []
    default_source = ""
    default_sink = ""

    try:
        res = subprocess.run(["pactl", "get-default-source"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            default_source = res.stdout.strip()
    except Exception:
        pass

    try:
        res = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            default_sink = res.stdout.strip()
    except Exception:
        pass

    try:
        res = subprocess.run(["pactl", "list", "sources"], capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            cur_source = {}
            seen_src_ids = set()
            for line in res.stdout.splitlines():
                line = line.strip()
                if line.startswith("Name:"):
                    name = line.split(":", 1)[1].strip()
                    cur_source = {"id": name}
                elif line.startswith("Description:") and cur_source:
                    desc = line.split(":", 1)[1].strip()
                    cur_source["name"] = desc
                    if not cur_source["id"].endswith(".monitor") and cur_source["id"] not in seen_src_ids:
                        is_ab13x = "ab13x" in cur_source["id"].lower() or "ab13x" in desc.lower() or "usb" in cur_source["id"].lower() or "usb" in desc.lower()
                        cur_source["is_earphones"] = is_ab13x
                        cur_source["is_default"] = (cur_source["id"] == default_source)
                        inputs.append(cur_source)
                        seen_src_ids.add(cur_source["id"])
                    cur_source = {}
    except Exception as e:
        print(f"[Audio Devices] Error listing sources: {e}")

    try:
        res = subprocess.run(["pactl", "list", "sinks"], capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            cur_sink = {}
            seen_sink_ids = set()
            for line in res.stdout.splitlines():
                line = line.strip()
                if line.startswith("Name:"):
                    name = line.split(":", 1)[1].strip()
                    cur_sink = {"id": name}
                elif line.startswith("Description:") and cur_sink:
                    desc = line.split(":", 1)[1].strip()
                    cur_sink["name"] = desc
                    if cur_sink["id"] not in seen_sink_ids:
                        is_ab13x = "ab13x" in cur_sink["id"].lower() or "ab13x" in desc.lower() or "usb" in cur_sink["id"].lower() or "usb" in desc.lower()
                        cur_sink["is_earphones"] = is_ab13x
                        cur_sink["is_default"] = (cur_sink["id"] == default_sink)
                        outputs.append(cur_sink)
                        seen_sink_ids.add(cur_sink["id"])
                    cur_sink = {}
    except Exception as e:
        print(f"[Audio Devices] Error listing sinks: {e}")

    # Fallback: if ALSA detects a USB card, but PulseAudio has no earphones/USB source in inputs, add ALSA direct device
    usb_card = get_usb_alsa_card()
    if not any(s.get("is_earphones") for s in inputs) and usb_card is not None:
        alsa_src = {
            "id": f"hw:{usb_card},0",
            "name": f"AB13X USB Audio Direct (ALSA hw:{usb_card},0)",
            "is_earphones": True,
            "is_default": True,
        }
        inputs.append(alsa_src)
        if not default_source:
            default_source = alsa_src["id"]

    # Auto-switch defaults to USB earphones if present and current default is internal platform audio
    usb_source = next((s["id"] for s in inputs if s.get("is_earphones")), None)
    if usb_source and (not default_source or "platform" in default_source.lower()):
        try:
            subprocess.run(["pactl", "set-default-source", usb_source], capture_output=True, timeout=1)
            default_source = usb_source
            for s in inputs:
                s["is_default"] = (s["id"] == default_source)
        except Exception:
            pass

    usb_sink = next((s["id"] for s in outputs if s.get("is_earphones")), None)
    if usb_sink and (not default_sink or "platform" in default_sink.lower()):
        try:
            subprocess.run(["pactl", "set-default-sink", usb_sink], capture_output=True, timeout=1)
            default_sink = usb_sink
            for s in outputs:
                s["is_default"] = (s["id"] == default_sink)
        except Exception:
            pass

    return {
        "sources": inputs,
        "sinks": outputs,
        "default_source": default_source,
        "default_sink": default_sink,
    }


def set_pulse_default_device(source_id: Optional[str] = None, sink_id: Optional[str] = None) -> bool:
    """Set the system-wide active PulseAudio input (mic) and output (earphones) device."""
    ensure_usb_audio_duplex()
    success = True
    if source_id:
        try:
            r = subprocess.run(["pactl", "set-default-source", source_id], capture_output=True, timeout=2)
            if r.returncode != 0:
                success = False
        except Exception:
            success = False
    if sink_id:
        try:
            r = subprocess.run(["pactl", "set-default-sink", sink_id], capture_output=True, timeout=2)
            if r.returncode != 0:
                success = False
        except Exception:
            success = False
    return success


# ===========================================================================
# Live Tests & Session Manager (SRT, VTT, Markdown, LaTeX, TXT, ZIP)
# ===========================================================================

def format_srt_timestamp(seconds: float) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    secs = int(seconds) % 60
    mins = (int(seconds) // 60) % 60
    hours = int(seconds) // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    secs = int(seconds) % 60
    mins = (int(seconds) // 60) % 60
    hours = int(seconds) // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def to_inline_speech(text: str) -> str:
    """Format speech and math into natural sequential flowing transcript:
    timestamp + text + inline latex code ($...$) + text...
    """
    if not text:
        return ""
    # Strip accidental \text{...} wrappers around non-math speech
    s = re.sub(r"\$\$(?:\\text\{([^\}]+)\}\s*(?:\\quad\s*)?)+\$\$", r"\1", text)
    s = re.sub(r"\$(?:\\text\{([^\}]+)\}\s*(?:\\quad\s*)?)+\$", r"\1", s)
    s = re.sub(r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", r"$\1$", s, flags=re.DOTALL)
    s = re.sub(r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}", r"$\1$", s, flags=re.DOTALL)
    s = re.sub(r"\\\[(.*?)\\\]", r"$\1$", s, flags=re.DOTALL)
    s = re.sub(r"\$\$(.*?)\$\$", r"$\1$", s, flags=re.DOTALL)
    s = re.sub(r"\\\((.*?)\\\)", r"$\1$", s, flags=re.DOTALL)
    s = re.sub(r"\$\s*\$", "", s)
    s = re.sub(r"\n+", " ", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


class LiveTestSession:
    """Manages an active testing or lecture session, saving all WAVs, JSONs, SRT, MD, and TEX."""

    def __init__(self, base_dir: Path = LIVE_TESTS_DIR):
        self.base_dir = base_dir
        self.lock = threading.Lock()
        self.session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / self.session_id
        self.recordings_dir = self.session_dir / "audio"
        self.metadata_dir = self.session_dir / "metadata"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = time.time()
        self.items: list[dict] = []
        self._init_files()

    def _init_files(self):
        md_file = self.session_dir / "lecture_notes.md"
        if not md_file.exists():
            md_file.write_text(
                f"# Live Test & Lecture Transcription Notes\n\n"
                f"- **Session ID:** `{self.session_id}`\n"
                f"- **Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"- **Platform:** NVIDIA Jetson Orin Nano (8GB) | 100% Offline Edge\n"
                f"- **Models:** Whisper small.en (CUDA Flash Attention) + Qwen2.5-1.5B (CUDA SM87)\n\n"
                f"---\n\n## Transcribed Lecture Content\n\n",
                encoding="utf-8"
            )

        tex_file = self.session_dir / "lecture_paper.tex"
        if not tex_file.exists():
            tex_file.write_text(
                "\\documentclass[11pt,a4paper]{article}\n"
                "\\usepackage[utf8]{inputenc}\n"
                "\\usepackage{amsmath,amssymb,amsfonts}\n"
                "\\usepackage{geometry}\n"
                "\\geometry{margin=1in}\n"
                f"\\title{{Lecture Notes and Mathematical Formulas}}\n"
                f"\\author{{Jetson Edge Speech-to-LaTeX}}\n"
                f"\\date{{{datetime.now().strftime('%B %d, %Y')}}}\n"
                "\\begin{document}\n"
                "\\maketitle\n\n"
                "\\section{Lecture Transcript}\n\n",
                encoding="utf-8"
            )

        txt_file = self.session_dir / "lecture_transcript.txt"
        if not txt_file.exists():
            txt_file.write_text(
                f"=======================================================\n"
                f" Live Test & Lecture Transcript - {self.session_id}\n"
                f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"=======================================================\n\n",
                encoding="utf-8"
            )

        srt_file = self.session_dir / "lecture_subtitles.srt"
        if not srt_file.exists():
            srt_file.write_text("", encoding="utf-8")

        vtt_file = self.session_dir / "lecture_subtitles.vtt"
        if not vtt_file.exists():
            vtt_file.write_text("WEBVTT - Jetson Speech-to-LaTeX Lecture Subtitles\n\n", encoding="utf-8")

    def record_utterance(self, wav_bytes: bytes, raw_transcript: str, processed_latex: str, duration_sec: float, start_offset: Optional[float] = None) -> dict:
        with self.lock:
            idx = len(self.items) + 1
            if start_offset is None:
                start_offset = max(0.0, time.time() - self.start_epoch - duration_sec)
            end_offset = start_offset + duration_sec

            wav_name = f"utterance_{idx:03d}.wav"
            wav_path = self.recordings_dir / wav_name
            wav_path.write_bytes(wav_bytes)

            item = {
                "index": idx,
                "timestamp": datetime.now().isoformat(),
                "start_sec": round(start_offset, 2),
                "end_sec": round(end_offset, 2),
                "duration_sec": round(duration_sec, 2),
                "start_formatted": format_srt_timestamp(start_offset),
                "end_formatted": format_srt_timestamp(end_offset),
                "raw_transcript": raw_transcript,
                "processed_latex": processed_latex,
                "wav_file": wav_name,
                "wav_path": str(wav_path),
            }
            self.items.append(item)

            # 1. Save utterance JSON
            json_name = f"utterance_{idx:03d}.json"
            (self.metadata_dir / json_name).write_text(json.dumps(item, indent=2), encoding="utf-8")

            inline_speech = to_inline_speech(processed_latex)

            # 2. Append to SRT
            srt_block = (
                f"{idx}\n"
                f"{format_srt_timestamp(start_offset)} --> {format_srt_timestamp(end_offset)}\n"
                f"{inline_speech}\n\n"
            )
            with open(self.session_dir / "lecture_subtitles.srt", "a", encoding="utf-8") as f:
                f.write(srt_block)

            # 3. Append to VTT
            vtt_block = (
                f"{idx}\n"
                f"{format_vtt_timestamp(start_offset)} --> {format_vtt_timestamp(end_offset)}\n"
                f"{inline_speech}\n\n"
            )
            with open(self.session_dir / "lecture_subtitles.vtt", "a", encoding="utf-8") as f:
                f.write(vtt_block)

            # 4. Append to Markdown (Sequential: timestamp + text + latex code + text...)
            time_tag = f"[{int(start_offset//60):02d}:{int(start_offset%60):02d}]"
            md_block = f"**{time_tag}** {inline_speech}\n\n"
            with open(self.session_dir / "lecture_notes.md", "a", encoding="utf-8") as f:
                f.write(md_block)

            # 5. Append to LaTeX
            tex_block = f"% {time_tag}\n{inline_speech}\n\n"
            with open(self.session_dir / "lecture_paper.tex", "a", encoding="utf-8") as f:
                f.write(tex_block)

            # 6. Append to Plain Text
            txt_block = f"{time_tag} {inline_speech}\n"
            with open(self.session_dir / "lecture_transcript.txt", "a", encoding="utf-8") as f:
                f.write(txt_block)

            # 7. Update Session Manifest
            manifest = {
                "session_id": self.session_id,
                "total_utterances": len(self.items),
                "started": datetime.fromtimestamp(self.start_epoch).isoformat(),
                "last_update": datetime.now().isoformat(),
                "utterances": self.items,
            }
            (self.session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            return item

    def get_summary(self) -> dict:
        with self.lock:
            elapsed = time.time() - self.start_epoch
            return {
                "session_id": self.session_id,
                "total_utterances": len(self.items),
                "elapsed_sec": round(elapsed, 1),
                "elapsed_formatted": f"{int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}",
                "session_dir": str(self.session_dir),
                "items": list(self.items),
            }

    def create_zip_archive(self) -> bytes:
        with self.lock:
            # Ensure LaTeX document finishes cleanly
            tex_file = self.session_dir / "lecture_paper.tex"
            tex_content = tex_file.read_text(encoding="utf-8") if tex_file.exists() else ""
            if "\\end{document}" not in tex_content:
                tex_content += "\n\\end{document}\n"
                tex_file.write_text(tex_content, encoding="utf-8")

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in self.session_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(self.session_dir)
                        zf.write(file_path, arcname)
            zip_buf.seek(0)
            return zip_buf.getvalue()


_active_session: Optional[LiveTestSession] = None
_session_lock = threading.Lock()


def get_active_session() -> LiveTestSession:
    global _active_session
    with _session_lock:
        if _active_session is None:
            _active_session = LiveTestSession()
        return _active_session


def reset_active_session() -> LiveTestSession:
    global _active_session
    with _session_lock:
        _active_session = LiveTestSession()
        return _active_session


# ===========================================================================
# Hardware Microphone Recorder & Continuous Lecture Manager
# ===========================================================================

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
        self.selected_device_name: Optional[str] = None
        self.selected_device_index: Optional[int] = None

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
                "selected_device": self.selected_device_name or "Default (AB13X Earphones)",
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
        try:
            dev_idx, channels, rate, chunk_size = get_best_input_stream_config(
                pa, self.selected_device_index, self.selected_device_name
            )
            print(f"[Hardware Mic] Opening stream on device {dev_idx} ({channels}ch, {rate}Hz, chunk={chunk_size})...")
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
                mono_chunk = to_16k_mono(raw_bytes, rate, channels)
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


class ContinuousLectureManager:
    """Manages continuous 1-hour lecture recording with automated pause detection, non-blocking queue processing, and live streaming."""

    def __init__(self, recorder: HardwareMicRecorder):
        self.recorder = recorder
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.queue: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.start_epoch = 0.0
        self.utterance_count = 0
        self.hpf = HighPassHumFilter()
        self.recent_chunks: list[dict] = []
        # Universal Syntactic Clause Stitcher & Rolling Context
        self.pending_text: str = ""
        self.pending_wav: list[bytes] = []
        self.pending_duration: float = 0.0
        self.pending_start_offset: float = 0.0
        self.pending_count: int = 0
        self.context_history: collections.deque = collections.deque(maxlen=2)

    def get_status(self) -> dict:
        with self.lock:
            elapsed = time.time() - self.start_epoch if self.is_running else 0.0
            return {
                "is_running": self.is_running,
                "elapsed_sec": round(elapsed, 1),
                "elapsed_formatted": f"{int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}",
                "utterance_count": self.utterance_count,
            }

    def start(self) -> dict:
        with self.lock:
            if self.is_running:
                return {"status": "already_running", "start_time": self.start_epoch}
            self.stop_event.clear()
            self.is_running = True
            self.start_epoch = time.time()
            self.utterance_count = 0
            self.recent_chunks.clear()
            self.pending_text = ""
            self.pending_wav.clear()
            self.pending_duration = 0.0
            self.pending_start_offset = 0.0
            self.pending_count = 0
            self.context_history.clear()
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except Exception:
                    break
            self.hpf.reset()
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            self.thread = threading.Thread(target=self._lecture_loop, daemon=True)
            self.thread.start()
            print(f"[Continuous Lecture] Started manager at {datetime.now().strftime('%H:%M:%S')}")
            return {"status": "started", "start_time": self.start_epoch}

    def stop(self) -> dict:
        with self.lock:
            if not self.is_running:
                return {"status": "not_running"}
            self.is_running = False
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        # Flush any pending dangling clause before stopping
        self._flush_pending()
        if self.worker_thread:
            self.worker_thread.join(timeout=3.0)
        print(f"[Continuous Lecture] Stopped manager. Captured {self.utterance_count} utterances.")
        return self.get_status()

    def _flush_pending(self):
        with self.lock:
            if not self.pending_text:
                return
            p_text = self.pending_text
            p_wav = b"".join(self.pending_wav)
            p_dur = self.pending_duration
            p_offset = self.pending_start_offset
            self.pending_text = ""
            self.pending_wav.clear()
            self.pending_duration = 0.0
            self.pending_start_offset = 0.0
            self.pending_count = 0
        self._translate_and_broadcast(p_text, p_wav, p_dur, p_offset, asr_ms=0.0)

    def _worker_loop(self):
        """Dedicated background worker: processes speech chunks via Whisper & Qwen SLM without blocking audio input."""
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            wav_bytes, dur_sec, start_offset = item
            try:
                self._process_and_broadcast_chunk(wav_bytes, dur_sec, start_offset)
            except Exception as e:
                print(f"[Continuous Lecture Worker Error] {e}")
            finally:
                self.queue.task_done()

    def _lecture_loop(self):
        """High-priority audio capture loop: continuously streams hardware mic and applies real-time adaptive VAD."""
        pa = pyaudio.PyAudio()
        try:
            dev_idx, channels, rate, chunk_size = get_best_input_stream_config(
                pa, self.recorder.selected_device_index, self.recorder.selected_device_name
            )
            print(f"[Continuous Lecture] Recording stream opened: device={dev_idx}, ch={channels}, rate={rate}, chunk={chunk_size}")
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=dev_idx,
                frames_per_buffer=chunk_size,
            )

            noise_floor = 0.003
            pre_roll_buffer = collections.deque(maxlen=4)  # 400ms pre-speech audio buffer
            speech_buffer: list[np.ndarray] = []
            silence_chunks = 0
            is_in_speech = False
            speech_start_sec = 0.0

            while not self.stop_event.is_set():
                raw_bytes = stream.read(chunk_size, exception_on_overflow=False)
                mono_chunk = to_16k_mono(raw_bytes, rate, channels)
                filtered_chunk = self.hpf.process(mono_chunk)
                rms = float(np.sqrt(np.mean(filtered_chunk ** 2)))
                self.recorder._last_rms = rms
                cur_offset = time.time() - self.start_epoch

                if not is_in_speech:
                    noise_floor = 0.95 * noise_floor + 0.05 * rms
                    pre_roll_buffer.append(filtered_chunk)
                    # Adaptive trigger threshold: down to 0.004 RMS for sensitive earphone pickup
                    speech_thold = max(0.004, min(0.020, noise_floor * 1.8))

                    if rms >= speech_thold:
                        is_in_speech = True
                        speech_start_sec = max(0.0, cur_offset - len(pre_roll_buffer) * 0.1)
                        speech_buffer = list(pre_roll_buffer)
                        speech_buffer.append(filtered_chunk)
                        silence_chunks = 0
                        print(f"[Continuous Lecture] Speech triggered (RMS={rms:.4f} >= {speech_thold:.4f})")
                else:
                    speech_buffer.append(filtered_chunk)
                    speech_thold = max(0.004, min(0.020, noise_floor * 1.8))
                    if rms < speech_thold * 0.70:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0

                    # Natural lecture pause: 8 chunks = 0.8s, or max chunk = 90 chunks (9.0s)
                    if silence_chunks >= 8 or len(speech_buffer) >= 90:
                        if len(speech_buffer) >= 3:  # At least 300ms
                            audio_arr = np.concatenate(speech_buffer)
                            dur_sec = len(audio_arr) / 16000.0
                            print(f"[Continuous Lecture] Utterance ready ({dur_sec:.2f}s, {len(speech_buffer)} chunks). Enqueueing...")

                            wav_buf = io.BytesIO()
                            with wave.open(wav_buf, "wb") as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(16000)
                                audio_int16 = np.clip(audio_arr * 32767, -32768, 32767).astype(np.int16)
                                wf.writeframes(audio_int16.tobytes())

                            self.queue.put((wav_buf.getvalue(), dur_sec, speech_start_sec))

                        speech_buffer = []
                        silence_chunks = 0
                        is_in_speech = False
                        pre_roll_buffer.clear()

            # Flush any pending utterance on stop
            if is_in_speech and len(speech_buffer) >= 3:
                audio_arr = np.concatenate(speech_buffer)
                dur_sec = len(audio_arr) / 16000.0
                wav_buf = io.BytesIO()
                with wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    audio_int16 = np.clip(audio_arr * 32767, -32768, 32767).astype(np.int16)
                    wf.writeframes(audio_int16.tobytes())
                self.queue.put((wav_buf.getvalue(), dur_sec, speech_start_sec))

            stream.stop_stream()
            stream.close()
            print("[Continuous Lecture] Audio capture stream stopped cleanly.")
        except Exception as e:
            print(f"[Continuous Lecture Loop Error] {e}")
        finally:
            pa.terminate()

    def _process_and_broadcast_chunk(self, wav_bytes: bytes, duration_sec: float, start_offset: float):
        try:
            raw_text, asr_ms = query_whisper_server(wav_bytes)
            raw_text = clean_whisper_text(raw_text)
            if not raw_text.strip():
                print(f"[Continuous Lecture] Whisper returned blank/silent speech ({duration_sec:.2f}s).")
                return

            cleaned_speech = clean_speech_disfluencies(raw_text)

            # Syntactic Clause Stitcher:
            # If we had a previously buffered open/dangling clause, merge it with current speech using word-boundary deduplication
            if self.pending_text:
                full_text = merge_text_clauses(self.pending_text, cleaned_speech)
                self.pending_wav.append(wav_bytes)
                total_dur = self.pending_duration + duration_sec
                start_off = self.pending_start_offset
            else:
                full_text = cleaned_speech
                self.pending_wav = [wav_bytes]
                total_dur = duration_sec
                start_off = start_offset

            # If the assembled clause still ends in a dangling operator, buffer it up to 3 chunks
            if is_dangling_math_clause(full_text) and self.pending_count < 3:
                self.pending_text = full_text
                self.pending_duration = total_dur
                self.pending_start_offset = start_off
                self.pending_count += 1
                print(f"[Syntactic Stitcher] Incomplete math clause detected ('{full_text}'). Buffering for completion...")
                return

            # Complete clause reached (or max buffer limit reached)
            stitched_wav = b"".join(self.pending_wav)
            self.pending_text = ""
            self.pending_wav.clear()
            self.pending_duration = 0.0
            self.pending_start_offset = 0.0
            self.pending_count = 0

            self._translate_and_broadcast(full_text, stitched_wav, total_dur, start_off, asr_ms)
        except Exception as e:
            print(f"[Process Chunk Error] {e}")

    def _translate_and_broadcast(self, text: str, wav_bytes: bytes, duration_sec: float, start_offset: float, asr_ms: float):
        # Fast linguistic gating: bypass SLM in 0.05ms for conversational speech
        if has_math_indicators(text):
            if not is_service_alive(LLAMA_SERVER_URL, endpoint="/health"):
                print("[SLM] llama-server is offline. Auto-starting daemon...")
                ensure_edge_daemons()
            slm = get_slm_translator()
            t0 = time.monotonic()
            num_words = len(text.split())
            dyn_tokens = min(384, max(80, int(num_words * 3.5)))
            recent_context = " ".join(t[1] for t in self.context_history) if self.context_history else ""
            try:
                raw_latex = slm.translate(text, history=list(self.context_history), max_new_tokens=dyn_tokens)
                latex_text = clean_slm_hallucinations(raw_latex, text, context_text=recent_context)
            except Exception as e:
                print(f"[SLM Translation Error] {e}")
                latex_text = text
            slm_ms = (time.monotonic() - t0) * 1000.0
        else:
            latex_text = text
            slm_ms = 0.05

        # Update rolling context for continuous cross-chunk awareness (store raw & latex pair)
        self.context_history.append((text, latex_text))

        session = get_active_session()
        item = session.record_utterance(
            wav_bytes=wav_bytes,
            raw_transcript=text,
            processed_latex=latex_text,
            duration_sec=duration_sec,
            start_offset=start_offset,
        )

        packet = {
            "type": "lecture_chunk",
            "index": item["index"],
            "start_time": format_srt_timestamp(item["start_sec"]),
            "end_time": format_srt_timestamp(item["end_sec"]),
            "start_sec": item["start_sec"],
            "end_sec": item["end_sec"],
            "duration_sec": item["duration_sec"],
            "raw_transcript": text,
            "processed_text": latex_text,
            "asr_time_ms": round(asr_ms, 1),
            "router_time_ms": round(slm_ms, 1),
            "audio_url": f"/live_tests/{session.session_id}/audio/{item['wav_file']}",
        }

        with self.lock:
            self.utterance_count += 1
            self.recent_chunks.append(packet)

        broadcast_ws_message(packet)
        print(f"[Live Lecture Broadcast] Chunk #{item['index']}: '{text}' -> '{latex_text}' (ASR: {asr_ms:.1f}ms, SLM: {slm_ms:.1f}ms)")


lecture_mgr = ContinuousLectureManager(hw_recorder)



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
            if has_math_indicators(raw_text):
                if not is_service_alive(LLAMA_SERVER_URL, endpoint="/health"):
                    ensure_edge_daemons()
                slm = get_slm_translator()
                t_start = time.monotonic()
                num_words = len(raw_text.split())
                dyn_tokens = min(256, max(64, int(num_words * 3.5)))
                raw_latex = slm.translate(raw_text, max_new_tokens=dyn_tokens)
                processed_text = clean_slm_hallucinations(raw_latex, raw_text)
                total_ms = (time.monotonic() - t_start) * 1000.0
            else:
                processed_text = raw_text
                total_ms = 0.05

            return web.json_response({
                "raw_text": raw_text,
                "route_type": "slm_direct" if processed_text != raw_text else "PURE_ENGLISH",
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
            raw_clean = clean_whisper_text(raw_transcript)
            if has_math_indicators(raw_clean):
                if not is_service_alive(LLAMA_SERVER_URL, endpoint="/health"):
                    ensure_edge_daemons()
                slm = get_slm_translator()
                t_slm_start = time.monotonic()
                num_words = len(raw_clean.split())
                dyn_tokens = min(256, max(64, int(num_words * 3.5)))
                raw_latex = slm.translate(raw_clean, max_new_tokens=dyn_tokens)
                processed_text = clean_slm_hallucinations(raw_latex, raw_clean)
                slm_ms = (time.monotonic() - t_slm_start) * 1000.0
            else:
                processed_text = raw_clean
                slm_ms = 0.05
            total_ms = (time.monotonic() - t_total_start) * 1000.0

            # Record into active LiveTestSession
            live_item = None
            try:
                duration_sec = 0.0
                try:
                    with wave.open(str(wav_path), "rb") as wf:
                        duration_sec = wf.getnframes() / float(wf.getframerate())
                except Exception:
                    duration_sec = asr_ms / 1000.0
                live_item = get_active_session().record_utterance(
                    wav_bytes=wav_path.read_bytes(),
                    raw_transcript=raw_transcript,
                    processed_latex=processed_text,
                    duration_sec=duration_sec,
                )
            except Exception as e:
                print(f"[Live Tests Log Error] {e}")

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
                "live_test": live_item,
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

    # Record into active LiveTestSession
    live_item = None
    try:
        duration_sec = asr_ms / 1000.0
        live_item = get_active_session().record_utterance(
            wav_bytes=wav_path.read_bytes(),
            raw_transcript=raw_transcript,
            processed_latex=result.processed_text,
            duration_sec=duration_sec,
        )
    except Exception as e:
        print(f"[Live Tests Log Error] {e}")

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
        "live_test": live_item,
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
        "is_lecture_running": lecture_mgr.is_running,
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
                            cleaned_raw = clean_whisper_text(raw_transcript)

                            if engine != "t5":
                                slm = get_slm_translator()
                                t_slm = time.monotonic()
                                if has_math_indicators(cleaned_raw):
                                    processed = await loop.run_in_executor(None, slm.translate, cleaned_raw)
                                    processed = clean_slm_hallucinations(processed, cleaned_raw)
                                else:
                                    processed = cleaned_raw
                                slm_ms = (time.monotonic() - t_slm) * 1000.0
                            else:
                                r = get_router()
                                t_r = time.monotonic()
                                res = await loop.run_in_executor(None, r.process, cleaned_raw)
                                processed = res.processed_text
                                slm_ms = (time.monotonic() - t_r) * 1000.0

                            total_ms = (time.monotonic() - t_start) * 1000.0

                            # Record into LiveTestSession
                            live_item = None
                            try:
                                duration_sec = 0.0
                                try:
                                    with wave.open(str(wav_path), "rb") as wf:
                                        duration_sec = wf.getnframes() / float(wf.getframerate())
                                except Exception:
                                    duration_sec = asr_ms / 1000.0
                                live_item = get_active_session().record_utterance(
                                    wav_bytes=wav_path.read_bytes(),
                                    raw_transcript=cleaned_raw,
                                    processed_latex=processed,
                                    duration_sec=duration_sec,
                                )
                            except Exception as e:
                                print(f"[Live Tests WS Log Error] {e}")

                            result_packet = {
                                "type": "transcription_result",
                                "audio_file": wav_path.name,
                                "audio_url": f"/recordings/{wav_path.name}",
                                "download_url": f"/api/download_audio/{wav_path.name}",
                                "raw_transcript": cleaned_raw,
                                "processed_text": processed,
                                "engine": "qwen2.5-1.5b" if engine != "t5" else "t5-v2",
                                "asr_time_ms": round(asr_ms, 1),
                                "router_time_ms": round(slm_ms, 1),
                                "total_time_ms": round(total_ms, 1),
                                "live_test": live_item,
                            }
                            for w in list(active_websockets):
                                if not w.closed:
                                    await w.send_json(result_packet)
                        except Exception as e:
                            await ws.send_json({"type": "error", "message": str(e)})
                    else:
                        await ws.send_json({"type": "error", "message": "Not currently recording"})

                elif action == "start_lecture":
                    res = lecture_mgr.start()
                    await ws.send_json({"type": "lecture_started", **res})

                elif action == "stop_lecture":
                    res = lecture_mgr.stop()
                    await ws.send_json({"type": "lecture_stopped", **res})

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
            if active_websockets:
                if hw_recorder.is_recording:
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

                if lecture_mgr.is_running:
                    l_status = lecture_mgr.get_status()
                    packet = {
                        "type": "lecture_telemetry",
                        "is_lecture_running": True,
                        "elapsed_sec": l_status["elapsed_sec"],
                        "elapsed_formatted": l_status["elapsed_formatted"],
                        "utterance_count": l_status["utterance_count"],
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


# ===========================================================================
# Audio Devices & Export HTTP Handlers
# ===========================================================================

async def handle_audio_devices(request: web.Request) -> web.Response:
    """Return available PulseAudio/ALSA input microphones and output earphones devices."""
    rescan = request.query.get("rescan", "false").lower() in ("true", "1", "yes")
    loop = asyncio.get_event_loop()
    devs = await loop.run_in_executor(None, get_pulse_audio_devices, rescan)
    devs["active_mic_name"] = hw_recorder.selected_device_name or "Default (AB13X USB Earphones Mic)"
    return web.json_response(devs)


async def handle_select_audio_device(request: web.Request) -> web.Response:
    """Select the active microphone input or earphones output device."""
    data = await request.json()
    source_id = data.get("source_id")
    sink_id = data.get("sink_id")
    dev_type = data.get("type")
    dev_val = data.get("name") or data.get("id")
    device_name = data.get("device_name")
    device_index = data.get("device_index")

    if dev_type == "source" and not source_id:
        source_id = dev_val
    elif dev_type == "sink" and not sink_id:
        sink_id = dev_val

    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(None, set_pulse_default_device, source_id, sink_id)

    if device_name:
        hw_recorder.selected_device_name = device_name
    elif source_id:
        hw_recorder.selected_device_name = source_id

    if device_index is not None:
        try:
            hw_recorder.selected_device_index = int(device_index)
        except Exception:
            pass
    elif hw_recorder.selected_device_name:
        try:
            pa = pyaudio.PyAudio()
            idx, _, _, _ = get_best_input_stream_config(pa, requested_name=hw_recorder.selected_device_name)
            hw_recorder.selected_device_index = idx
            pa.terminate()
        except Exception:
            pass

    return web.json_response({
        "status": "ok" if ok else "warning",
        "source": source_id,
        "sink": sink_id,
        "active_mic": hw_recorder.selected_device_name,
    })


async def handle_live_tests_summary(request: web.Request) -> web.Response:
    """Return summary and history of utterances in current live test session."""
    session = get_active_session()
    return web.json_response(session.get_summary())


async def handle_live_tests_new_session(request: web.Request) -> web.Response:
    """Start a fresh new live test or lecture session."""
    new_sess = reset_active_session()
    return web.json_response({"status": "created", **new_sess.get_summary()})


async def handle_export_srt(request: web.Request) -> web.Response:
    """Export current session as SubRip (.srt) subtitle file."""
    session = get_active_session()
    srt_file = session.session_dir / "lecture_subtitles.srt"
    content = srt_file.read_text(encoding="utf-8") if srt_file.exists() else ""
    return web.Response(
        text=content,
        content_type="text/plain",
        charset="utf-8",
        headers={"Content-Disposition": f'attachment; filename="{session.session_id}_subtitles.srt"'},
    )


async def handle_export_vtt(request: web.Request) -> web.Response:
    """Export current session as WebVTT (.vtt) subtitle file."""
    session = get_active_session()
    vtt_file = session.session_dir / "lecture_subtitles.vtt"
    content = vtt_file.read_text(encoding="utf-8") if vtt_file.exists() else "WEBVTT\n\n"
    return web.Response(
        text=content,
        content_type="text/vtt",
        charset="utf-8",
        headers={"Content-Disposition": f'attachment; filename="{session.session_id}_subtitles.vtt"'},
    )


async def handle_export_md(request: web.Request) -> web.Response:
    """Export current session as Markdown lecture notes with LaTeX math equations."""
    session = get_active_session()
    md_file = session.session_dir / "lecture_notes.md"
    content = md_file.read_text(encoding="utf-8") if md_file.exists() else "# Lecture Notes\n\n"
    return web.Response(
        text=content,
        content_type="text/markdown",
        charset="utf-8",
        headers={"Content-Disposition": f'attachment; filename="{session.session_id}_notes.md"'},
    )


async def handle_export_tex(request: web.Request) -> web.Response:
    """Export current session as full compilable LaTeX document (.tex)."""
    session = get_active_session()
    tex_file = session.session_dir / "lecture_paper.tex"
    content = tex_file.read_text(encoding="utf-8") if tex_file.exists() else ""
    if "\\end{document}" not in content:
        content += "\n\\end{document}\n"
    return web.Response(
        text=content,
        content_type="application/x-tex",
        charset="utf-8",
        headers={"Content-Disposition": f'attachment; filename="{session.session_id}_paper.tex"'},
    )


async def handle_export_txt(request: web.Request) -> web.Response:
    """Export current session as clean formatted plain text (.txt)."""
    session = get_active_session()
    txt_file = session.session_dir / "lecture_transcript.txt"
    content = txt_file.read_text(encoding="utf-8") if txt_file.exists() else ""
    return web.Response(
        text=content,
        content_type="text/plain",
        charset="utf-8",
        headers={"Content-Disposition": f'attachment; filename="{session.session_id}_transcript.txt"'},
    )


async def handle_export_zip(request: web.Request) -> web.Response:
    """Export entire live test session (all WAV audio files, metadata, subtitles, and notes) as a ZIP archive."""
    session = get_active_session()
    loop = asyncio.get_event_loop()
    zip_bytes = await loop.run_in_executor(None, session.create_zip_archive)
    return web.Response(
        body=zip_bytes,
        content_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{session.session_id}_complete_export.zip"'},
    )


async def handle_lecture_start(request: web.Request) -> web.Response:
    """Start continuous 1-hour lecture recording mode with automated incremental VAD streaming."""
    res = lecture_mgr.start()
    return web.json_response(res)


async def handle_lecture_stop(request: web.Request) -> web.Response:
    """Stop continuous lecture mode and finalize session exports."""
    res = lecture_mgr.stop()
    return web.json_response(res)


async def handle_lecture_status(request: web.Request) -> web.Response:
    """Get active continuous lecture streaming status."""
    return web.json_response(lecture_mgr.get_status())


async def handle_lecture_chunks(request: web.Request) -> web.Response:
    """Return recently streamed chunks from continuous lecture for resilient fallback polling."""
    try:
        since_idx = int(request.query.get("since_index", 0))
    except Exception:
        since_idx = 0
    with lecture_mgr.lock:
        new_chunks = [c for c in lecture_mgr.recent_chunks if c.get("index", 0) > since_idx]
        return web.json_response({
            "is_running": lecture_mgr.is_running,
            "utterance_count": lecture_mgr.utterance_count,
            "chunks": new_chunks,
        })


async def start_background_tasks(app: web.Application):
    # Ensure background daemons and hardware duplex on startup
    global main_event_loop
    main_event_loop = asyncio.get_event_loop()
    await main_event_loop.run_in_executor(None, ensure_edge_daemons)
    await main_event_loop.run_in_executor(None, ensure_usb_audio_duplex)
    app["telemetry_task"] = asyncio.create_task(ws_telemetry_loop(app))


async def cleanup_background_tasks(app: web.Application):
    if "telemetry_task" in app:
        app["telemetry_task"].cancel()
        try:
            await app["telemetry_task"]
        except asyncio.CancelledError:
            pass


@web.middleware
async def no_cache_middleware(request: web.Request, handler) -> web.StreamResponse:
    if not request.path.startswith("/static/") and request.path != "/ws":
        print(f"[HTTP {request.method}] {request.path}")
    if request.path == "/ws":
        return await handler(request)
    response = await handler(request)
    if isinstance(response, web.Response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def create_app() -> web.Application:
    app = web.Application(client_max_size=32 * 1024 * 1024, middlewares=[no_cache_middleware])

    # Core Routes
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

    # Audio Device & Earphones Selection Routes
    app.router.add_get("/api/audio_devices", handle_audio_devices)
    app.router.add_post("/api/audio_devices/select", handle_select_audio_device)

    # Live Tests & Session Export Routes (.srt, .vtt, .md, .tex, .txt, .zip)
    app.router.add_get("/api/live_tests/summary", handle_live_tests_summary)
    app.router.add_post("/api/live_tests/new_session", handle_live_tests_new_session)
    app.router.add_get("/api/live_tests/export/srt", handle_export_srt)
    app.router.add_get("/api/live_tests/export/vtt", handle_export_vtt)
    app.router.add_get("/api/live_tests/export/md", handle_export_md)
    app.router.add_get("/api/live_tests/export/tex", handle_export_tex)
    app.router.add_get("/api/live_tests/export/txt", handle_export_txt)
    app.router.add_get("/api/live_tests/export/zip", handle_export_zip)

    # Continuous 1-Hour Lecture Mode Routes
    app.router.add_post("/api/lecture/start", handle_lecture_start)
    app.router.add_post("/api/lecture/stop", handle_lecture_stop)
    app.router.add_get("/api/lecture/status", handle_lecture_status)
    app.router.add_get("/api/lecture/chunks", handle_lecture_chunks)

    # Static routes
    app.router.add_static("/static/", STATIC_DIR, name="static")
    app.router.add_static("/recordings/", RECORDINGS_DIR, name="recordings")
    app.router.add_static("/live_tests/", LIVE_TESTS_DIR, name="live_tests")

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
    ensure_usb_audio_duplex()
    get_slm_translator()
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=port)
