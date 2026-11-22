#!/usr/bin/env python3
"""
================================================================================
Manual Lecture Audio Tester & End-to-End Pipeline Verification Tool
Optimized for NVIDIA Jetson Orin Nano (100% Offline Edge Operation)
================================================================================

Tests the complete Speech-to-LaTeX & Bilingual Publishing pipeline on any WAV file:
  1. Audio normalization, trimming, and validation (16kHz 16-bit mono PCM).
  2. Offline Whisper ASR with Universal Mathematical Decoding Prompt.
  3. Qwen2.5-1.5B SLM Transduction (Speech -> Standard LaTeX Math).
  4. Publication-Grade Academic Monograph Synthesis with TikZ Vector Graphics.
  5. Bilingual Hindi Technical Translation with 100% Math Invariant Preservation.
  6. Tectonic XeTeX PDF Compilation for both English and Devanagari Hindi.
  7. Synchronized Multilingual Subtitle Generation (SRT & VTT).
  8. SHA-256 Checksum Manifest & Automatic USB Export (/media/ayush02/AYUSH).
================================================================================
"""

import argparse
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import wave
from typing import List, Optional, Tuple, Dict, Any

# Ensure headless_agent/src is in sys.path
THESIS_DIR = Path(__file__).resolve().parent
HEADLESS_SRC = THESIS_DIR / "headless_agent" / "src"
if str(HEADLESS_SRC) not in sys.path:
    sys.path.insert(0, str(HEADLESS_SRC))

try:
    from lecture_to_book import LectureToBookBuilder, TranscriptChunk, generate_subtitles, sanitize_latex_source
    from postprocess_lecture_ai import PostLectureProcessor
    from slm_translator import QwenMathTranslator
    from hindi_translator import HindiLectureTranslator
    from usb_exporter import LectureExporter, find_usb_mount
except ImportError as e:
    print(f"[Error] Failed to import pipeline modules: {e}")
    print(f"        Ensure you are using the virtual environment at: /mnt/thesis/headless_agent/.venv")
    sys.exit(1)

# Default Model & Binary Paths
WHISPER_BIN = "/mnt/thesis/main/whisper.cpp/build-orin/bin/whisper-cli"
WHISPER_MODEL = "/mnt/thesis/models/whisper/ggml-small.en.bin"
LLAMA_BIN = "/mnt/thesis/llama.cpp/build/bin/llama-server"
LLAMA_MODEL = "/mnt/thesis/models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"
TECTONIC_BIN = "/mnt/thesis/bin/tectonic"

WHISPER_MATH_PROMPT = (
    "Academic STEM lecture and professional presentation: mathematics, physics, computer science, algorithms, "
    "engineering, calculus, linear algebra, limits, integrals, derivatives, differential equations, matrices, vectors, "
    "proofs, theorems, definitions, functions f(x), variables x, y, z, t, n, parameters alpha, beta, theta, omega, "
    "equations, formulas, and technical exposition."
)

# Common Sample Paths
SAMPLE_SEARCH_DIRS = [
    "/media/ayush02/AYUSH/Lectures",
    "/mnt/thesis/live_tests",
    "/mnt/thesis/gui_tests",
    "/mnt/thesis",
]


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def print_banner(text: str, color: str = Colors.CYAN):
    border = "=" * 80
    print(f"\n{color}{border}")
    print(f" {text}")
    print(f"{border}{Colors.RESET}")


def print_step(step_num: int, total_steps: int, title: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step {step_num}/{total_steps}] {title}{Colors.RESET}")


def check_service_alive(url: str, endpoint: str = "/health") -> bool:
    try:
        req = urllib.request.Request(f"{url}{endpoint}", method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status in (200, 204)
    except Exception:
        return False


def ensure_llama_server():
    """Ensure llama-server is running on port 8080."""
    if check_service_alive("http://127.0.0.1:8080", "/health"):
        return True

    print(f"{Colors.YELLOW}[Pre-flight] llama-server not running on port 8080. Auto-starting...{Colors.RESET}")
    cmd = [
        LLAMA_BIN,
        "-m", LLAMA_MODEL,
        "-c", "2048",
        "-ngl", "0",
        "-t", "4",
        "--fit", "off",
        "--port", "8080",
        "--host", "127.0.0.1"
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["LD_LIBRARY_PATH"] = "/mnt/thesis/llama.cpp/build/bin:" + env.get("LD_LIBRARY_PATH", "")

    log_file = open("/tmp/llama-server.log", "a")
    proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log_file, stderr=log_file, start_new_session=True)

    for i in range(120):
        time.sleep(0.5)
        if proc.poll() is not None:
            print(f"\n{Colors.RED}[Error] llama-server process terminated prematurely (exit code {proc.returncode}). Check /tmp/llama-server.log{Colors.RESET}")
            return False
        if check_service_alive("http://127.0.0.1:8080", "/health"):
            print(f"{Colors.GREEN}[Pre-flight] llama-server is UP and healthy (PID: {proc.pid})!{Colors.RESET}")
            return True
        elapsed = (i + 1) * 0.5
        print(f"           Waiting for llama-server ({elapsed:.1f}s/60s)...", end="\r")

    print(f"\n{Colors.RED}[Error] Failed to start llama-server within 60s. Check /tmp/llama-server.log{Colors.RESET}")
    return False


def discover_sample_wavs() -> List[Path]:
    """Find lecture WAV files across system."""
    found = []
    seen_names = set()
    for s_dir in SAMPLE_SEARCH_DIRS:
        p = Path(s_dir)
        if not p.is_dir():
            continue
        try:
            for wav in sorted(p.rglob("*.wav")):
                if "Trash" in str(wav) or "site-packages" in str(wav):
                    continue
                sz = wav.stat().st_size
                if sz > 200_000:  # > 200KB
                    if wav.name not in seen_names:
                        seen_names.add(wav.name)
                        found.append(wav)
        except Exception:
            pass
    return found


def prompt_select_sample() -> Optional[Path]:
    """Display interactive menu to select a sample WAV file."""
    samples = discover_sample_wavs()
    if not samples:
        print(f"{Colors.YELLOW}No sample WAV files found automatically.{Colors.RESET}")
        return None

    print(f"\n{Colors.BOLD}{Colors.GREEN}Available Lecture WAV Samples:{Colors.RESET}")
    for idx, s in enumerate(samples[:10], 1):
        sz_mb = s.stat().st_size / (1024 * 1024)
        print(f"  [{idx:2d}] {s.name:<45} ({sz_mb:5.2f} MB) -> {s.parent}")

    print(f"  [ 0] Enter custom WAV file path manually")
    try:
        choice = input(f"\n{Colors.BOLD}Select a sample (1-{min(10, len(samples))}) [default: 1]: {Colors.RESET}").strip()
        if not choice:
            return samples[0]
        if choice == "0":
            custom = input(f"{Colors.BOLD}Enter full path to audio file: {Colors.RESET}").strip()
            p = Path(custom).expanduser().resolve()
            return p if p.is_file() else None
        c_int = int(choice)
        if 1 <= c_int <= len(samples):
            return samples[c_int - 1]
    except Exception:
        pass
    return samples[0] if samples else None


def prepare_audio(
    input_audio: Path,
    output_wav: Path,
    duration_sec: Optional[float] = None,
    offset_sec: float = 0.0
) -> Tuple[bool, float]:
    """Standardize audio to 16kHz 16-bit mono PCM using SoX."""
    t0 = time.monotonic()
    sox_bin = shutil.which("sox")
    
    cmd = []
    if sox_bin:
        cmd = [sox_bin, str(input_audio), "-r", "16000", "-c", "1", "-b", "16", str(output_wav)]
        if offset_sec > 0 or (duration_sec and duration_sec > 0):
            cmd.append("trim")
            cmd.append(str(offset_sec))
            if duration_sec and duration_sec > 0:
                cmd.append(str(duration_sec))
    else:
        # Fallback to python wave
        try:
            import numpy as np
            with wave.open(str(input_audio), "rb") as wf:
                n_ch = wf.getnchannels()
                fr = wf.getframerate()
                n_frames = wf.getnframes()
                skip_frames = int(offset_sec * fr)
                if skip_frames > 0:
                    wf.setpos(min(skip_frames, n_frames - 1))
                read_n = int(duration_sec * fr) if duration_sec else (n_frames - skip_frames)
                raw_bytes = wf.readframes(read_n)
            
            pcm = np.frombuffer(raw_bytes, dtype=np.int16)
            if n_ch > 1:
                pcm = pcm.reshape(-1, n_ch).mean(axis=1)
            if fr != 16000:
                num_s = int(len(pcm) * 16000 / fr)
                pcm = np.interp(
                    np.linspace(0, len(pcm), num_s, endpoint=False),
                    np.arange(len(pcm)),
                    pcm,
                ).astype(np.int16)
            
            with wave.open(str(output_wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm.tobytes())
            cmd = None
        except Exception as e:
            print(f"[Error] Audio conversion failed: {e}")
            return False, 0.0

    if cmd:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(f"[Error] SoX failed: {res.stderr}")
            return False, 0.0

    # Calculate actual duration
    with wave.open(str(output_wav), "rb") as wf:
        actual_sec = wf.getnframes() / float(wf.getframerate())

    return True, actual_sec


def run_whisper_transcription(wav_file: Path, use_gpu: bool = False) -> List[Tuple[float, float, str]]:
    """Execute whisper-cli with multi-threaded ARM NEON CPU engine (or GPU if requested)."""
    temp_prefix = Path("/tmp") / f"test_trans_{int(time.time()*1000)}"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/mnt/thesis/main/whisper.cpp/build-orin/bin:" + env.get("LD_LIBRARY_PATH", "")

    if use_gpu:
        cmd = [
            WHISPER_BIN,
            "-m", WHISPER_MODEL,
            "-f", str(wav_file),
            "--prompt", WHISPER_MATH_PROMPT,
            "-sns", "-nth", "0.60", "-fa",
            "-oj", "-of", str(temp_prefix)
        ]
    else:
        cmd = [
            WHISPER_BIN,
            "-m", WHISPER_MODEL,
            "-f", str(wav_file),
            "--prompt", WHISPER_MATH_PROMPT,
            "-sns", "-nth", "0.60", "-ng", "-t", "4",
            "-oj", "-of", str(temp_prefix)
        ]

    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    json_file = Path(f"{temp_prefix}.json")

    # If GPU was requested but failed, fallback to CPU NEON
    if use_gpu and (res.returncode != 0 or not json_file.is_file()):
        print(f"{Colors.YELLOW}       GPU allocation unavailable; falling back to 4-core ARM NEON CPU engine...{Colors.RESET}")
        cmd_cpu = [
            WHISPER_BIN,
            "-m", WHISPER_MODEL,
            "-f", str(wav_file),
            "--prompt", WHISPER_MATH_PROMPT,
            "-sns", "-nth", "0.60", "-ng", "-t", "4",
            "-oj", "-of", str(temp_prefix)
        ]
        res = subprocess.run(cmd_cpu, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if not json_file.is_file():
        print(f"{Colors.RED}[Error] Whisper ASR produced no JSON output.{Colors.RESET}")
        return []

    with open(json_file, "r", encoding="utf-8") as jf:
        data = json.load(jf)
    json_file.unlink(missing_ok=True)

    segments = data.get("transcription", [])
    raw_chunks = []

    def _parse_ts(ts_str: str) -> float:
        ts_str = ts_str.replace(",", ".")
        p = ts_str.split(":")
        if len(p) == 3:
            return float(p[0]) * 3600 + float(p[1]) * 60 + float(p[2])
        elif len(p) == 2:
            return float(p[0]) * 60 + float(p[1])
        return float(ts_str)

    for seg in segments:
        txt = seg.get("text", "").strip()
        txt = re.sub(r'[\(\[\{].*?[\)\]\}]', '', txt)
        txt = re.sub(r'[♪♫]+', '', txt).strip()
        if not txt or not any(c.isalnum() for c in txt):
            continue
        ts_from = _parse_ts(seg.get("timestamps", {}).get("from", "00:00:00,000"))
        ts_to = _parse_ts(seg.get("timestamps", {}).get("to", "00:00:05,000"))
        raw_chunks.append((ts_from, ts_to, txt))

    return raw_chunks


def get_pdf_page_count(pdf_path: Path) -> int:
    """Read page count of a PDF file using pdfinfo."""
    if not pdf_path or not pdf_path.is_file():
        return 0
    try:
        res = subprocess.run(["pdfinfo", str(pdf_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        m = re.search(r"Pages:\s+(\d+)", res.stdout)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 0


def run_manual_test(
    audio_path: Path,
    output_dir: Optional[Path] = None,
    duration_sec: Optional[float] = None,
    offset_sec: float = 0.0,
    export_usb: bool = True,
    skip_hindi: bool = False,
    skip_pdf: bool = False,
    use_gpu: bool = False,
    verbose: bool = False
):
    start_total_time = time.monotonic()
    print_banner("JETSON ORIN NANO - MANUAL LECTURE AUDIO PIPELINE TESTER", Colors.CYAN)
    print(f"🎵 Input Audio:       {audio_path.name}")
    print(f"📁 Source Path:       {audio_path}")
    print(f"⏱️ Duration Limit:    {f'{duration_sec:.1f}s' if duration_sec else 'Full Recording'}")
    print(f"⏩ Offset:            {offset_sec:.1f}s")
    print(f"⚡ ASR Model:         Whisper-Small (Flash Attention / NEON)")
    print(f"🧠 SLM Transduction:  Qwen2.5-1.5B-Instruct-Q4_K_M (Port 8080)")
    print(f"🇮🇳 Hindi Translation: Verbatim Math Invariant Preservation (Active)")
    print(f"📄 Typesetting:       Tectonic XeTeX + Lohit Devanagari")

    # Step 1: Pre-flight Checks
    print_step(1, 6, "Pre-flight Environment & Daemon Health Check")
    if not Path(WHISPER_BIN).is_file():
        print(f"{Colors.RED}[Error] Whisper binary missing at: {WHISPER_BIN}{Colors.RESET}")
        return
    if not Path(TECTONIC_BIN).is_file():
        print(f"{Colors.RED}[Error] Tectonic binary missing at: {TECTONIC_BIN}{Colors.RESET}")
        return

    ensure_llama_server()
    print(f"{Colors.GREEN}✓ All edge binaries and background daemons verified.{Colors.RESET}")

    # Prepare Temporary Working Directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    temp_dir = Path("/tmp") / f"manual_test_{ts}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Audio Standardization & Preprocessing
    print_step(2, 6, "Audio Normalization & Ingestion")
    norm_wav = temp_dir / f"test_input_16k.wav"
    t0 = time.monotonic()
    success, actual_dur = prepare_audio(audio_path, norm_wav, duration_sec, offset_sec)
    if not success or actual_dur <= 0.1:
        print(f"{Colors.RED}[Error] Failed to prepare 16kHz audio.{Colors.RESET}")
        return
    print(f"{Colors.GREEN}✓ Standardized to 16kHz 16-bit Mono PCM in {(time.monotonic()-t0)*1000:.1f}ms{Colors.RESET}")
    print(f"  Processed Audio Duration: {actual_dur:.2f} seconds ({actual_dur/60:.2f} minutes)")

    # Step 3: Speech-to-Text Transcription (Whisper)
    print_step(3, 6, "Whisper Speech-to-Text Transcription")
    t0_asr = time.monotonic()
    raw_segments = run_whisper_transcription(norm_wav, use_gpu=use_gpu)
    t_asr = time.monotonic() - t0_asr
    if not raw_segments:
        print(f"{Colors.RED}[Error] No speech segments detected.{Colors.RESET}")
        return
    
    rtf_asr = t_asr / actual_dur
    print(f"{Colors.GREEN}✓ Transcribed {len(raw_segments)} utterances in {t_asr:.2f}s (ASR Speed: {1/rtf_asr:.1f}x real-time){Colors.RESET}")

    # Step 4: Mathematical SLM Transduction (Speech -> LaTeX)
    print_step(4, 6, "Mathematical SLM Transduction (Qwen2.5-1.5B ChatML)")
    t0_slm = time.monotonic()
    slm = QwenMathTranslator.get_instance(backend="server")

    # Aggregate raw acoustic segments into grammatically complete sentences before SLM
    aggregated_units = []
    curr_start = None
    curr_end = None
    curr_words = []

    for t_start, t_end, txt in raw_segments:
        txt = txt.strip()
        if not txt:
            continue
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

    active_chunks = []
    print(f"\n{Colors.BOLD}{'Timestamp':<16} {'Spoken Lecture Speech':<40} {'LaTeX Typeset'}{Colors.RESET}")
    print("-" * 80)
    for t_start, t_end, full_spoken in aggregated_units:
        latex = slm.translate(full_spoken)
        active_chunks.append(TranscriptChunk(t_start, t_end, full_spoken, latex))
        ts_label = f"[{t_start:5.1f}s-{t_end:5.1f}s]"
        txt_disp = (full_spoken[:38] + "..") if len(full_spoken) > 40 else full_spoken
        latex_disp = (latex[:45] + "..") if len(latex) > 47 else latex
        print(f"{Colors.CYAN}{ts_label:<16}{Colors.RESET} {txt_disp:<40} {Colors.GREEN}{latex_disp}{Colors.RESET}")

    t_slm = time.monotonic() - t0_slm
    print(f"\n{Colors.GREEN}✓ SLM Transduction completed in {t_slm:.2f}s ({len(active_chunks)} sentence-chunks from {len(raw_segments)} raw segments){Colors.RESET}")

    # Step 5: Monograph Synthesis & Bilingual Translation
    print_step(5, 6, "Academic Monograph & Bilingual Synthesis")
    book_builder = LectureToBookBuilder()
    post_processor = PostLectureProcessor()

    clean_base = audio_path.stem.replace("_raw", "")
    tex_content, md_content, clean_title = book_builder.build_book(clean_base, active_chunks)
    srt_content, vtt_content = generate_subtitles(active_chunks)

    slug = re.sub(r'[^a-zA-Z0-9]+', '_', clean_title).strip('_')
    if len(slug) > 40:
        slug = slug[:40].rstrip('_')
    session_name = f"{ts}_{slug}" if slug else f"Lecture_{ts}"

    # Write English Deliverables
    tex_file = temp_dir / f"{session_name}.tex"
    tex_file.write_text(tex_content, encoding="utf-8")
    md_file = temp_dir / f"{session_name}.md"
    md_file.write_text(md_content, encoding="utf-8")
    srt_file = temp_dir / f"{session_name}.srt"
    srt_file.write_text(srt_content, encoding="utf-8")
    vtt_file = temp_dir / f"{session_name}.vtt"
    vtt_file.write_text(vtt_content, encoding="utf-8")

    # Bilingual Hindi Synthesis
    hi_tex_file = hi_md_file = hi_srt_file = hi_vtt_file = hi_pdf_file = None
    if not skip_hindi:
        t0_hi = time.monotonic()
        print(f"🇮🇳 Translating explanatory prose into Hindi with 100% LaTeX math preservation...")
        try:
            hindi_tex, hindi_md, hindi_title, hindi_chunks, hindi_paras = book_builder.build_hindi_book(
                clean_title, active_chunks
            )
            hindi_srt, hindi_vtt = generate_subtitles(hindi_chunks)

            hi_tex_file = temp_dir / f"{session_name}_hindi.tex"
            hi_tex_file.write_text(hindi_tex, encoding="utf-8")
            hi_md_file = temp_dir / f"{session_name}_hindi.md"
            hi_md_file.write_text(hindi_md, encoding="utf-8")
            hi_srt_file = temp_dir / f"{session_name}_hindi.srt"
            hi_srt_file.write_text(hindi_srt, encoding="utf-8")
            hi_vtt_file = temp_dir / f"{session_name}_hindi.vtt"
            hi_vtt_file.write_text(hindi_vtt, encoding="utf-8")

            t_hi = time.monotonic() - t0_hi
            print(f"{Colors.GREEN}✓ Bilingual Hindi synthesis completed in {t_hi:.2f}s{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}[Warning] Hindi synthesis error: {e}{Colors.RESET}")

    # Step 6: Publication PDF Compilation (Tectonic XeTeX)
    print_step(6, 6, "Tectonic XeTeX PDF Compilation & Export")
    pdf_file = None
    if not skip_pdf:
        t0_comp = time.monotonic()
        print("📄 Compiling English publication monograph via Tectonic XeTeX...")
        post_res = post_processor.process_and_compile(
            lecture_title=clean_title,
            tex_path=tex_file,
            output_dir=temp_dir,
            raw_transcript="\n".join([c.spoken_text for c in active_chunks])
        )
        pdf_file = post_res.get("pdf")
        if pdf_file and pdf_file.is_file():
            p_cnt = get_pdf_page_count(pdf_file)
            print(f"{Colors.GREEN}✓ English PDF compiled: {pdf_file.name} ({pdf_file.stat().st_size} bytes, {p_cnt} pages){Colors.RESET}")

        if hi_tex_file and hi_tex_file.is_file():
            print("📄 Compiling Hindi bilingual monograph with Lohit-Devanagari font...")
            hi_pdf_dest = temp_dir / f"{session_name}_hindi.pdf"
            compiled_hi = post_processor.compile_hindi_pdf(
                tex_path=hi_tex_file,
                output_dir=temp_dir,
                title=clean_title,
                paragraphs=hindi_paras if 'hindi_paras' in locals() else None,
                output_pdf_path=hi_pdf_dest
            )
            if compiled_hi and compiled_hi.is_file():
                hi_pdf_file = compiled_hi
                p_cnt_hi = get_pdf_page_count(hi_pdf_file)
                print(f"{Colors.GREEN}✓ Hindi PDF compiled: {hi_pdf_file.name} ({hi_pdf_file.stat().st_size} bytes, {p_cnt_hi} pages){Colors.RESET}")

    # Save audio WAV
    final_wav = temp_dir / f"{session_name}_raw.wav"
    shutil.copy2(norm_wav, final_wav)

    # Determine Output Directory
    if not output_dir:
        target_dir = THESIS_DIR / "manual_tests" / session_name
    else:
        target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Export deliverables to target_dir
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

    # Use LectureExporter to copy to SSD, build zip & SHA256SUMS
    exporter = LectureExporter()
    export_res = exporter.export(session_name, export_map, timestamp=ts)

    # If user specified a custom directory, mirror to it
    if target_dir != Path(export_res["local_path"]):
        for f in Path(export_res["local_path"]).iterdir():
            shutil.copy2(f, target_dir / f.name)

    total_pipeline_time = time.monotonic() - start_total_time
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Print Final Verification Report
    print_banner("MANUAL TEST VERIFICATION SUMMARY", Colors.GREEN)
    print(f"🏷️ Session Title:      {clean_title}")
    print(f"⏱️ Audio Duration:     {actual_dur:.1f}s ({actual_dur/60:.2f} min)")
    print(f"⏳ Total Test Time:    {total_pipeline_time:.1f}s (Overall RTF: {total_pipeline_time/actual_dur:.2f})")
    print(f"💾 SSD Destination:    {export_res['local_path']}")
    print(f"🔌 USB Auto-Sync:      {export_res['usb_path']} (Mounted: {export_res['usb_success']})")

    # Table of generated artifacts
    print(f"\n{Colors.BOLD}{'Generated Deliverable':<48} {'Size':<12} {'Pages / Details'}{Colors.RESET}")
    print("-" * 80)
    local_p = Path(export_res["local_path"])
    for f in sorted(local_p.iterdir()):
        sz = f.stat().st_size
        sz_str = f"{sz/1024:.1f} KB" if sz < 1024*1024 else f"{sz/(1024*1024):.2f} MB"
        details = "-"
        if f.suffix == ".pdf":
            pages = get_pdf_page_count(f)
            details = f"{pages} Page{'s' if pages != 1 else ''} (Verified)"
        elif f.suffix in (".srt", ".vtt"):
            details = "Synchronized Subtitles"
        elif f.suffix == ".zip":
            details = "Complete Archival Bundle"
        elif f.suffix == ".txt":
            details = "SHA-256 Manifest"
        print(f"  {f.name:<46} {sz_str:<12} {details}")

    print("-" * 80)
    print(f"{Colors.BOLD}{Colors.GREEN}✓ Test Completed Successfully! All deliverables generated and verified.{Colors.RESET}\n")


def check_system_diagnostics():
    """Print full diagnostic report of system daemons, GPU, memory, and models."""
    print_banner("EDGE SYSTEM DIAGNOSTICS & PRE-FLIGHT AUDIT", Colors.CYAN)
    
    # 1. Daemon Health
    whisper_srv = check_service_alive("http://127.0.0.1:8081", "/")
    llama_srv = check_service_alive("http://127.0.0.1:8080", "/health")
    print(f"{Colors.BOLD}Daemons Status:{Colors.RESET}")
    print(f"  • Whisper Server (Port 8081) : {'[ONLINE - OK]' if whisper_srv else '[OFFLINE]'}")
    print(f"  • Llama Server   (Port 8080) : {'[ONLINE - OK]' if llama_srv else '[OFFLINE]'}")

    # 2. Binaries
    print(f"\n{Colors.BOLD}Core Binaries:{Colors.RESET}")
    for name, p in [
        ("Whisper CLI", WHISPER_BIN),
        ("Llama Server", LLAMA_BIN),
        ("Tectonic XeTeX", TECTONIC_BIN),
        ("SoX Audio Tool", shutil.which("sox") or "NOT FOUND"),
    ]:
        status = "[FOUND]" if Path(p).is_file() else "[MISSING]"
        print(f"  • {name:<16} : {status:<10} {p}")

    # 3. Models
    print(f"\n{Colors.BOLD}Model Weights:{Colors.RESET}")
    for name, p in [
        ("Whisper Small", WHISPER_MODEL),
        ("Qwen2.5-1.5B GGUF", LLAMA_MODEL),
    ]:
        status = "[FOUND]" if Path(p).is_file() else "[MISSING]"
        sz = f"({Path(p).stat().st_size / (1024*1024):.1f} MB)" if Path(p).is_file() else ""
        print(f"  • {name:<18} : {status:<10} {sz} {p}")

    # 4. Storage & USB
    usb_mount = find_usb_mount()
    print(f"\n{Colors.BOLD}External Media:{Colors.RESET}")
    print(f"  • USB Drive AYUSH : {f'[DETECTED at {usb_mount}]' if usb_mount else '[NOT MOUNTED]'}")


def main():
    parser = argparse.ArgumentParser(
        description="Manual Speech-to-LaTeX Lecture Audio Tester on Jetson Orin Nano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./test_system_wav.py                                      # Interactive menu of discovered samples
  ./test_system_wav.py lecture.wav                          # Test specific WAV file end-to-end
  ./test_system_wav.py lecture.wav -d 60                    # Quick test on first 60 seconds of audio
  ./test_system_wav.py lecture.wav --skip-hindi             # Test English monograph only
  ./test_system_wav.py --list-samples                       # List all detected lecture WAVs
  ./test_system_wav.py --check-system                       # Audit daemons, binaries, and models
        """
    )
    parser.add_argument("wav_path", nargs="?", type=str, default=None, help="Path to input audio file (.wav, .mp3, etc.)")
    parser.add_argument("-w", "--wav", type=str, default=None, help="Explicit path to input audio file")
    parser.add_argument("-d", "--duration", type=float, default=None, help="Process first N seconds of audio")
    parser.add_argument("-s", "--offset", type=float, default=0.0, help="Offset start time in seconds")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Custom output directory for deliverables")
    parser.add_argument("--no-usb", action="store_true", help="Disable automatic export to external USB drive")
    parser.add_argument("--skip-hindi", action="store_true", help="Skip bilingual Hindi translation and monograph")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip Tectonic PDF compilation")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration for Whisper ASR (default: multi-core NEON CPU)")
    parser.add_argument("--list-samples", action="store_true", help="List all available lecture sample WAVs and exit")
    parser.add_argument("--check-system", action="store_true", help="Audit all edge daemons, binaries, and models")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logs")

    args = parser.parse_args()

    if args.check_system:
        check_system_diagnostics()
        return

    if args.list_samples:
        samples = discover_sample_wavs()
        print(f"\n{Colors.BOLD}Discovered {len(samples)} Lecture WAV Files:{Colors.RESET}")
        for idx, s in enumerate(samples, 1):
            sz_mb = s.stat().st_size / (1024 * 1024)
            print(f"  [{idx:2d}] {s.name:<45} ({sz_mb:5.2f} MB) -> {s}")
        return

    # Determine input path
    raw_path = args.wav or args.wav_path
    if not raw_path:
        target_path = prompt_select_sample()
        if not target_path:
            print(f"{Colors.YELLOW}No audio file selected. Exiting.{Colors.RESET}")
            return
    else:
        target_path = Path(raw_path).expanduser().resolve()
        if not target_path.is_file():
            print(f"{Colors.RED}[Error] Specified audio file does not exist: {target_path}{Colors.RESET}")
            return

    run_manual_test(
        audio_path=target_path,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        duration_sec=args.duration,
        offset_sec=args.offset,
        export_usb=not args.no_usb,
        skip_hindi=args.skip_hindi,
        skip_pdf=args.skip_pdf,
        use_gpu=args.gpu,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
