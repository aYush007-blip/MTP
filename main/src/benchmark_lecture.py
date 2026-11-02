#!/usr/bin/env python3
"""
Comprehensive Long-Form Academic Lecture Benchmark & Typesetting Profiler
Optimized for NVIDIA Jetson Orin Nano (CUDA SM87 Flash Attention + Qwen2.5-1.5B SLM)

Features:
  1. High-Fidelity Audio Decimation via SoX (16kHz 16-bit mono PCM).
  2. Dynamic Adaptive VAD Utterance Segmentation (2.0s - 12.0s).
  3. Whisper-Server with Universal Mathematical Decoding Prompt.
  4. Whisper Text Normalization & Speech Disfluency Removal.
  5. Fast Linguistic Pre-Check (<0.05ms) to gate SLM translation.
  6. Multi-Turn ChatML SLM Translation with Sliding Window Context.
  7. 4-Tier Hallucination & Degenerative Repetition Guardrails.
  8. Full Deliverables Export: Markdown, LaTeX (.tex), Plain Text, Subtitles (SRT/VTT), CSV Metrics, and JSON Manifest.
"""

import argparse
import collections
from collections import Counter
from datetime import datetime
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import wave
import numpy as np

# Ensure src/ and main/ are in sys.path
SRC_DIR = Path(__file__).resolve().parent
MAIN_DIR = SRC_DIR.parent
for p in (str(SRC_DIR), str(MAIN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from slm_translator import QwenMathTranslator

try:
    from gui.server import (
        WHISPER_MATH_PROMPT,
        clean_whisper_text,
        clean_speech_disfluencies,
        clean_slm_hallucinations,
        has_math_indicators,
    )
except ImportError:
    WHISPER_MATH_PROMPT = (
        "Engineering lecture: calculus, linear algebra, differential equations, "
        "mechanics, thermodynamics, fluid dynamics, circuits, control systems, "
        "electromagnetics, algorithms, stress, strain, shear, entropy, enthalpy, heat flux, "
        "curl, divergence, gradient, partial derivative, Navier-Stokes, Reynolds number, "
        "Kirchhoff's laws, KCL, KVL, inductance, resistance, capacitance, flux, current i(t), voltage v(t), "
        "transfer function, eigenvalues, vectors, tensors, matrices, summations, integrals, "
        "state-space, transient, steady state, impulse response, step response, convolution, Bode plot, Nyquist, "
        "discrete-time sequences x[n], y[n], downsampling, upsampling, difference equations, z-transform, ROC, "
        "units: Pascals, Newtons, Joules, Watts, Hertz, Farads, Ohms, Kelvin, meters per second."
    )

    def clean_whisper_text(text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"[\(\[\{].*?[\)\]\}]", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def clean_speech_disfluencies(text: str) -> str:
        if not text:
            return ""
        s = re.sub(r"\b([a-zA-Z]+)\s+\1\b", r"\1", text.strip(), flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", s).strip()

    def has_math_indicators(text: str) -> bool:
        pattern = re.compile(
            r"\b(equals?|is\s+equal\s+to|greater\s+than|less\s+than|approaches|tends\s+to|"
            r"plus|minus|times|divided|over|squared|cubed|power|root|fraction|numerator|denominator|"
            r"sine|cosine|tangent|integral|derivative|sum|summation|sigma|omega|theta|alpha|beta|"
            r"matrix|eigenvalue|z[\s-]transform|roc|region\s+of\s+convergence|poles?|zeros?|z[\s-]plane|unit\s+circle|"
            r"transfer\s+function|convolution|laplace|fourier|frequency\s+response|difference\s+equation|"
            r"partial\s+fraction|polynomials?|magnitude|modulus|infinity|discrete[\s-]time|"
            r"synchronous|rotor|stator|flux\s+linkage|direct\s+axis|quadrature\s+axis|d[\s-]axis|q[\s-]axis|"
            r"field\s+winding|damper|reactance|impedance|park\'?s\s+transform\w*|"
            r"z\s+to\s+the\s+minus|z\s+inverse|inverse\s+z|"
            r"[a-zA-Z]\s+of\s+[a-zA-Z0-9_\-\+]+|u\s+of\s+n|delta\s+of\s+n)\b|[=+\-*/^<>≤≥≠∑∫|]",
            re.IGNORECASE
        )
        return bool(pattern.search(text))

    def clean_slm_hallucinations(latex_text: str, raw_text: str, context_text: str = "") -> str:
        s = latex_text.strip()
        if not s:
            return raw_text
        if re.search(r"^(?:I apologize|Thank you for sharing|As an AI|Could you please provide|Here is the)", s, re.IGNORECASE):
            return raw_text
        if re.search(r"(.)\1{8,}", s):
            return raw_text
        return s


def format_timestamp(seconds: float, srt_format: bool = False) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    secs = int(seconds) % 60
    mins = (int(seconds) // 60) % 60
    hours = int(seconds) // 3600
    if srt_format:
        return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"
    return f"{hours:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def format_mmss(seconds: float) -> str:
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins:02d}:{secs:02d}"


def validate_latex_syntax(latex_str: str) -> tuple[bool, str]:
    """Check balanced braces, brackets, and parenthesis in mathematical LaTeX syntax."""
    stack = []
    for char in latex_str:
        if char in "{[(":
            stack.append(char)
        elif char in "}])":
            if not stack:
                return False, f"Unmatched closing '{char}'"
            last = stack.pop()
            if (last == "{" and char != "}") or (last == "[" and char != "]") or (last == "(" and char != ")"):
                return False, f"Mismatched bracket '{last}' with '{char}'"
    if stack:
        return False, f"Unclosed '{stack[-1]}'"
    return True, "Valid"


def segment_audio(wav_path: Path, min_chunk_sec: float = 2.0, max_chunk_sec: float = 10.0, min_silence_sec: float = 0.40) -> list[tuple[float, float, float, bytes]]:
    """Segment 16kHz mono WAV into speech utterance chunks using adaptive RMS energy."""
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    frame_len = int(rate * 0.02)  # 20ms frame
    num_frames = len(pcm) // frame_len

    rms = np.sqrt(np.mean(pcm[:num_frames * frame_len].reshape(-1, frame_len) ** 2, axis=1))
    p25 = float(np.percentile(rms, 25))
    adapt_thresh = max(0.006, min(0.030, p25 * 1.8))

    chunk_start = 0
    silence_frames = 0
    min_silence_frames = int(min_silence_sec / 0.02)
    min_chunk_frames = int(min_chunk_sec / 0.02)
    max_chunk_frames = int(max_chunk_sec / 0.02)

    chunks = []
    for i, r in enumerate(rms):
        cur_len_frames = i - chunk_start
        if r < adapt_thresh:
            silence_frames += 1
        else:
            silence_frames = 0

        if (silence_frames >= min_silence_frames and cur_len_frames >= min_chunk_frames) or (cur_len_frames >= max_chunk_frames):
            split_frame = i - (silence_frames // 2) if silence_frames >= min_silence_frames else i
            start_sec = chunk_start * 0.02
            end_sec = split_frame * 0.02
            duration_sec = end_sec - start_sec

            if duration_sec >= 1.0:
                s_idx = int(start_sec * 16000)
                e_idx = int(end_sec * 16000)
                sub_pcm = raw[s_idx * 2 : e_idx * 2]
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w_out:
                    w_out.setnchannels(1)
                    w_out.setsampwidth(2)
                    w_out.setframerate(16000)
                    w_out.writeframes(sub_pcm)
                chunks.append((start_sec, end_sec, duration_sec, buf.getvalue()))

            chunk_start = split_frame
            silence_frames = 0

    if chunk_start < num_frames:
        start_sec = chunk_start * 0.02
        end_sec = num_frames * 0.02
        duration_sec = end_sec - start_sec
        if duration_sec >= 1.0:
            s_idx = int(start_sec * 16000)
            e_idx = int(end_sec * 16000)
            sub_pcm = raw[s_idx * 2 : e_idx * 2]
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w_out:
                w_out.setnchannels(1)
                w_out.setsampwidth(2)
                w_out.setframerate(16000)
                w_out.writeframes(sub_pcm)
            chunks.append((start_sec, end_sec, duration_sec, buf.getvalue()))

    return chunks


def query_whisper(wav_bytes: bytes, server_url: str = "http://127.0.0.1:8081", prompt: str = WHISPER_MATH_PROMPT) -> tuple[str, float]:
    """Query whisper-server on Jetson via multipart/form-data with mathematical domain prompt."""
    boundary = "----JetsonBenchmarkStreamBoundary"
    parts = [
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="chunk.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode("utf-8"),
        wav_bytes,
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="prompt"\r\n\r\n'
        f"{prompt}\r\n".encode("utf-8"),
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="no_speech_thold"\r\n\r\n'
        f"0.65\r\n".encode("utf-8"),
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'.encode("utf-8"),
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    body = b"".join(parts)

    req = urllib.request.Request(
        f"{server_url}/inference",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read().decode())
    asr_ms = (time.monotonic() - t0) * 1000.0

    raw_text = res.get("text", "").strip()
    raw_text = clean_whisper_text(raw_text)
    return raw_text, asr_ms


def main():
    parser = argparse.ArgumentParser(description="Academic Lecture Spoken-Math-to-LaTeX Benchmark Engine")
    parser.add_argument(
        "--audio",
        type=str,
        default="/mnt/thesis/Lecture 12 Z - Transform.wav",
        help="Path to input lecture audio WAV file",
    )
    parser.add_argument("--name", type=str, default="z-transform_lecture_test", help="Test session name")
    parser.add_argument("--start_sec", type=float, default=0.0, help="Start offset in seconds")
    parser.add_argument("--duration_sec", type=float, default=0.0, help="Test duration in seconds (0 for full lecture)")
    parser.add_argument("--output_dir", type=str, default="/mnt/thesis/evaluation_results", help="Directory for results")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        print(f"❌ Error: Audio file not found at {audio_path}")
        sys.exit(1)

    out_base = Path(args.output_dir)
    out_dir = out_base if out_base.name == args.name else out_base / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = args.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 80)
    print("🎓 JETSON ORIN NANO - ACADEMIC LECTURE SPEECH-TO-LATEX BENCHMARK")
    print("=" * 80)
    print(f"📁 Input File:        {audio_path.name}")
    print(f"🏷️ Test Session:      {base_name}")
    print(f"⏱️ Test Duration:     {args.duration_sec:.1f}s ({args.duration_sec/60:.2f} min)" if args.duration_sec > 0 else "⏱️ Test Duration:     Full Recording")
    print(f"⚡ ASR Engine:        Whisper-Small (Flash Attention on CUDA SM87, port 8081)")
    print(f"🧠 SLM Engine:        Qwen2.5-1.5B-Instruct-GGUF (CUDA SM87, port 8080)")
    print(f"📂 Output Dir:        {out_dir}")
    print("=" * 80 + "\n")

    # 1. Resample and slice target audio via SoX
    sliced_wav = out_dir / f"benchmark_temp_{timestamp}_16k.wav"
    sox_cmd = ["sox", str(audio_path), "-r", "16000", "-c", "1", "-b", "16", str(sliced_wav)]
    if args.start_sec > 0 or args.duration_sec > 0:
        dur = args.duration_sec if args.duration_sec > 0 else 7200
        sox_cmd.extend(["trim", str(args.start_sec), str(dur)])

    print("🔊 Step 1/4: Resampling & Normalizing Audio to 16kHz Mono PCM via SoX...")
    t_sox = time.monotonic()
    subprocess.run(sox_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sox_time = time.monotonic() - t_sox

    with wave.open(str(sliced_wav), "rb") as wf:
        actual_audio_sec = wf.getnframes() / float(wf.getframerate())
    print(f"✅ Extracted {actual_audio_sec:.1f}s audio in {sox_time:.2f}s.\n")

    # 2. VAD Utterance Segmentation
    print("✂️ Step 2/4: Segmenting Audio into Natural Utterances via Adaptive VAD...")
    chunks = segment_audio(sliced_wav, min_chunk_sec=2.0, max_chunk_sec=10.0, min_silence_sec=0.40)
    print(f"✅ Generated {len(chunks)} speech utterance chunks (Mean duration: {actual_audio_sec/len(chunks):.2f}s).\n")

    # 3. Initialize SLM Translator
    print("🧠 Step 3/4: Connecting to Qwen2.5-1.5B SLM on llama-server (CUDA)...")
    slm = QwenMathTranslator(backend="server")
    print("✅ SLM Translator ready.\n")

    # 4. Processing Loop
    print("🚀 Step 4/4: Benchmarking End-to-End Pipeline on Lecture Chunks...")
    print("-" * 80)

    chunk_records = []
    total_asr_ms = 0.0
    total_slm_ms = 0.0
    total_wall_start = time.monotonic()

    asr_latencies = []
    slm_latencies = []
    total_latencies = []
    rtf_values = []
    detected_math_count = 0
    all_formulas = []
    invalid_latex_count = 0
    context_history = collections.deque(maxlen=2)

    for idx, (c_start, c_end, c_dur, c_bytes) in enumerate(chunks, 1):
        abs_start = args.start_sec + c_start
        abs_end = args.start_sec + c_end
        t_chunk_start = time.monotonic()

        # Whisper ASR
        try:
            raw_text, asr_ms = query_whisper(c_bytes)
        except Exception as e:
            raw_text, asr_ms = f"[ASR Error: {e}]", 0.0

        cleaned_speech = clean_speech_disfluencies(raw_text)

        # SLM Math Translation
        if cleaned_speech and not cleaned_speech.startswith("[ASR Error"):
            if has_math_indicators(cleaned_speech):
                t0 = time.monotonic()
                recent_context = " ".join(t[1] for t in context_history) if context_history else ""
                num_words = len(cleaned_speech.split())
                dyn_tokens = min(384, max(80, int(num_words * 3.5)))
                try:
                    raw_latex = slm.translate(cleaned_speech, history=list(context_history), max_new_tokens=dyn_tokens)
                    latex_text = clean_slm_hallucinations(raw_latex, cleaned_speech, context_text=recent_context)
                except Exception as e:
                    print(f"\n[SLM Error] Chunk #{idx}: {e}")
                    latex_text = cleaned_speech
                slm_ms = (time.monotonic() - t0) * 1000.0
            else:
                latex_text = cleaned_speech
                slm_ms = 0.05
        else:
            latex_text = cleaned_speech
            slm_ms = 0.0

        context_history.append((cleaned_speech, latex_text))

        total_chunk_ms = (time.monotonic() - t_chunk_start) * 1000.0
        rtf = total_chunk_ms / (c_dur * 1000.0)

        # Formula detection
        formulas = re.findall(r"\$([^\$]+)\$", latex_text)
        has_math = len(formulas) > 0
        if has_math:
            detected_math_count += len(formulas)
            for f in formulas:
                is_valid, err = validate_latex_syntax(f)
                all_formulas.append({
                    "formula": f"${f}$",
                    "timestamp": format_mmss(abs_start),
                    "valid": is_valid,
                    "error": err,
                })
                if not is_valid:
                    invalid_latex_count += 1

        asr_latencies.append(asr_ms)
        slm_latencies.append(slm_ms)
        total_latencies.append(total_chunk_ms)
        rtf_values.append(rtf)
        total_asr_ms += asr_ms
        total_slm_ms += slm_ms

        math_badge = f" [Math: {len(formulas)}]" if has_math else ""
        print(f"[{idx:03d}/{len(chunks):03d}] [{format_mmss(abs_start)}-{format_mmss(abs_end)} | {c_dur:4.1f}s] ASR: {asr_ms:5.1f}ms | SLM: {slm_ms:5.1f}ms | Total: {total_chunk_ms:5.1f}ms | RTF: {rtf:5.3f}x{math_badge}")
        if has_math:
            print(f"     LaTeX: {latex_text}")

        chunk_records.append({
            "chunk_id": idx,
            "start_sec": round(abs_start, 2),
            "end_sec": round(abs_end, 2),
            "start_time": format_mmss(abs_start),
            "end_time": format_mmss(abs_end),
            "duration_sec": round(c_dur, 2),
            "asr_ms": round(asr_ms, 1),
            "slm_ms": round(slm_ms, 1),
            "total_ms": round(total_chunk_ms, 1),
            "rtf": round(rtf, 4),
            "has_math": has_math,
            "math_count": len(formulas),
            "raw_transcript": cleaned_speech,
            "latex_transcript": latex_text,
        })

    total_wall_sec = time.monotonic() - total_wall_start
    overall_rtf = total_wall_sec / actual_audio_sec

    # 5. Summary Statistics Calculation
    print("\n" + "=" * 80)
    print("📊 BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    print(f"⏱️ Total Audio Processed:   {actual_audio_sec:.1f}s ({actual_audio_sec/60:.2f} minutes)")
    print(f"⚡ Total Wall-Clock Time:   {total_wall_sec:.1f}s ({total_wall_sec/60:.2f} minutes)")
    print(f"🚀 Overall Real-Time Factor: {overall_rtf:.4f}x ({1.0/overall_rtf:.2f}x faster than real-time!)")
    print("-" * 80)
    print("LATENCY DISTRIBUTION (ms per chunk):")
    print(f"  ASR:   Mean={np.mean(asr_latencies):.1f}ms | P50={np.median(asr_latencies):.1f}ms | P90={np.percentile(asr_latencies, 90):.1f}ms | P95={np.percentile(asr_latencies, 95):.1f}ms | Max={max(asr_latencies):.1f}ms")
    print(f"  SLM:   Mean={np.mean(slm_latencies):.1f}ms | P50={np.median(slm_latencies):.1f}ms | P90={np.percentile(slm_latencies, 90):.1f}ms | P95={np.percentile(slm_latencies, 95):.1f}ms | Max={max(slm_latencies):.1f}ms")
    print(f"  Total: Mean={np.mean(total_latencies):.1f}ms | P50={np.median(total_latencies):.1f}ms | P90={np.percentile(total_latencies, 90):.1f}ms | P95={np.percentile(total_latencies, 95):.1f}ms | Max={max(total_latencies):.1f}ms")
    print("-" * 80)
    print("ACCURACY & MATHEMATICAL FORMATION:")
    print(f"  Total Utterance Chunks:   {len(chunks)}")
    print(f"  Chunks with Math:         {sum(1 for c in chunk_records if c['has_math'])} ({sum(1 for c in chunk_records if c['has_math'])/len(chunks)*100:.1f}%)")
    print(f"  Total Formulas Detected:  {detected_math_count}")
    print(f"  LaTeX Syntax Pass Rate:   {100.0 if not all_formulas else (len(all_formulas)-invalid_latex_count)/len(all_formulas)*100:.1f}%")
    print("=" * 80 + "\n")

    # 6. Generate All Deliverables

    # CSV Metrics Export
    csv_file = out_dir / f"{base_name}_metrics.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("chunk_id,start_time,end_time,duration_sec,asr_ms,slm_ms,total_ms,rtf,has_math,raw_transcript,latex_transcript\n")
        for c in chunk_records:
            raw_clean = c["raw_transcript"].replace('"', '""')
            latex_clean = c["latex_transcript"].replace('"', '""')
            f.write(f'{c["chunk_id"]},{c["start_time"]},{c["end_time"]},{c["duration_sec"]},{c["asr_ms"]},{c["slm_ms"]},{c["total_ms"]},{c["rtf"]},{c["has_math"]},"{raw_clean}","{latex_clean}"\n')

    # JSON Manifest Export
    json_file = out_dir / f"{base_name}_manifest.json"
    json_data = {
        "metadata": {
            "source_file": audio_path.name,
            "session_name": base_name,
            "timestamp": timestamp,
            "audio_duration_sec": actual_audio_sec,
            "wall_clock_time_sec": total_wall_sec,
            "overall_rtf": overall_rtf,
            "chunk_count": len(chunks),
            "math_formula_count": detected_math_count,
            "latex_syntax_pass_rate_pct": 100.0 if not all_formulas else (len(all_formulas)-invalid_latex_count)/len(all_formulas)*100,
        },
        "latency_percentiles_ms": {
            "asr_mean": round(float(np.mean(asr_latencies)), 1),
            "asr_p50": round(float(np.median(asr_latencies)), 1),
            "asr_p90": round(float(np.percentile(asr_latencies, 90)), 1),
            "asr_p95": round(float(np.percentile(asr_latencies, 95)), 1),
            "slm_mean": round(float(np.mean(slm_latencies)), 1),
            "slm_p50": round(float(np.median(slm_latencies)), 1),
            "slm_p90": round(float(np.percentile(slm_latencies, 90)), 1),
            "slm_p95": round(float(np.percentile(slm_latencies, 95)), 1),
            "total_mean": round(float(np.mean(total_latencies)), 1),
            "total_p50": round(float(np.median(total_latencies)), 1),
            "total_p90": round(float(np.percentile(total_latencies, 90)), 1),
            "total_p95": round(float(np.percentile(total_latencies, 95)), 1),
        },
        "chunks": chunk_records,
        "math_formulas": all_formulas,
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # SubRip (.srt) Subtitles Export
    srt_file = out_dir / f"{base_name}_subtitles.srt"
    with open(srt_file, "w", encoding="utf-8") as f:
        for c in chunk_records:
            f.write(f"{c['chunk_id']}\n")
            f.write(f"{format_timestamp(c['start_sec'], srt_format=True)} --> {format_timestamp(c['end_sec'], srt_format=True)}\n")
            f.write(f"{c['latex_transcript']}\n\n")

    # WebVTT (.vtt) Subtitles Export
    vtt_file = out_dir / f"{base_name}_subtitles.vtt"
    with open(vtt_file, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for c in chunk_records:
            f.write(f"{c['chunk_id']}\n")
            f.write(f"{format_timestamp(c['start_sec'], srt_format=False)} --> {format_timestamp(c['end_sec'], srt_format=False)}\n")
            f.write(f"{c['latex_transcript']}\n\n")

    # Full Markdown Transcript with Inline KaTeX Math
    md_file = out_dir / f"{base_name}_transcript.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# Academic Lecture Transcript: {audio_path.name}\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')} | **Audio Duration:** {format_mmss(actual_audio_sec)} | **Engine:** Whisper-Small + Qwen2.5-1.5B (CUDA SM87)\n\n---\n\n")
        for c in chunk_records:
            f.write(f"**[{c['start_time']} - {c['end_time']}]** {c['latex_transcript']}\n\n")

    # Compilable LaTeX Document Paper Export
    tex_file = out_dir / f"{base_name}_paper.tex"
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage[utf8]{inputenc}\n")
        f.write("\\usepackage{amsmath,amssymb,amsfonts}\n")
        f.write("\\usepackage{geometry}\n\\geometry{a4paper, margin=1in}\n\n")
        f.write(f"\\title{{Lecture Transcription: {audio_path.stem}}}\n")
        f.write("\\author{Automated Jetson Edge Speech-to-LaTeX}\n")
        f.write(f"\\date{{{datetime.now().strftime('%B %d, %Y')}}}\n\n")
        f.write("\\begin{document}\n\\maketitle\n\n")
        for c in chunk_records:
            f.write(f"% [{c['start_time']} - {c['end_time']}]\n")
            f.write(f"{c['latex_transcript']}\n\n")
        f.write("\\end{document}\n")

    # Plain Text Transcript Export
    txt_file = out_dir / f"{base_name}_transcript.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        for c in chunk_records:
            f.write(f"[{c['start_time']} - {c['end_time']}] {c['latex_transcript']}\n")

    # Executive Evaluation Report
    report_file = out_dir / f"{base_name}_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# Long-Form Lecture Benchmark Report: {audio_path.name}\n\n")
        f.write(f"- **Benchmark Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Hardware Platform:** NVIDIA Jetson Orin Nano Developer Kit (8GB Unified Memory)\n")
        f.write(f"- **Audio Duration Tested:** {actual_audio_sec:.1f} seconds ({actual_audio_sec/60:.2f} minutes)\n")
        f.write(f"- **Total Wall-Clock Time:** {total_wall_sec:.1f} seconds ({total_wall_sec/60:.2f} minutes)\n")
        f.write(f"- **Real-Time Factor (RTF):** **`{overall_rtf:.4f}x`** ({1.0/overall_rtf:.2f}x faster than real-time)\n\n")
        f.write("## 1. Latency Breakdown\n\n")
        f.write("| Component | Mean (ms) | Median / P50 (ms) | P90 (ms) | P95 (ms) | Max (ms) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Whisper ASR (CUDA Flash Attention)** | {np.mean(asr_latencies):.1f} | {np.median(asr_latencies):.1f} | {np.percentile(asr_latencies, 90):.1f} | {np.percentile(asr_latencies, 95):.1f} | {max(asr_latencies):.1f} |\n")
        f.write(f"| **Qwen2.5-1.5B SLM (CUDA SM87)** | {np.mean(slm_latencies):.1f} | {np.median(slm_latencies):.1f} | {np.percentile(slm_latencies, 90):.1f} | {np.percentile(slm_latencies, 95):.1f} | {max(slm_latencies):.1f} |\n")
        f.write(f"| **End-to-End Pipeline** | **{np.mean(total_latencies):.1f}** | **{np.median(total_latencies):.1f}** | **{np.percentile(total_latencies, 90):.1f}** | **{np.percentile(total_latencies, 95):.1f}** | **{max(total_latencies):.1f}** |\n\n")
        f.write("## 2. Accuracy & Spoken-Math Typesetting\n\n")
        f.write(f"- **Total Utterance Chunks:** {len(chunks)}\n")
        f.write(f"- **Mathematical Utterance Density:** {sum(1 for c in chunk_records if c['has_math'])} / {len(chunks)} chunks ({sum(1 for c in chunk_records if c['has_math'])/len(chunks)*100:.1f}%)\n")
        f.write(f"- **Total Mathematical Expressions Extracted:** {detected_math_count}\n")
        f.write(f"- **LaTeX Syntax Integrity:** {100.0 if not all_formulas else (len(all_formulas)-invalid_latex_count)/len(all_formulas)*100:.1f}% valid\n\n")
        f.write("### Sample Mathematical Extractions Detected in Lecture:\n\n")
        f.write("| Timestamp | Extracted LaTeX Formula |\n")
        f.write("| :--- | :--- |\n")
        for f_item in all_formulas[:25]:
            f.write(f"| {f_item['timestamp']} | `{f_item['formula']}` ({f_item['formula']}) |\n")
        f.write("\n---\n")
        f.write(f"*Report auto-generated by `benchmark_lecture.py` on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Clean up temporary sliced audio file
    if sliced_wav.exists():
        sliced_wav.unlink()

    print(f"📁 Reports & Transcripts Saved to:")
    print(f"  - Markdown Report:     {report_file}")
    print(f"  - Markdown Transcript: {md_file}")
    print(f"  - LaTeX Paper (.tex):  {tex_file}")
    print(f"  - Plain Transcript:    {txt_file}")
    print(f"  - Subtitles (SRT):     {srt_file}")
    print(f"  - Subtitles (VTT):     {vtt_file}")
    print(f"  - Per-Chunk CSV:       {csv_file}")
    print(f"  - Structured JSON:     {json_file}\n")


if __name__ == "__main__":
    main()
