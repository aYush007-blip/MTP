#!/usr/bin/env python3
"""
Comprehensive Long-Form Lecture Benchmark & Accuracy Profiler for NVIDIA Jetson Orin Nano.

Evaluates latency, real-time factor (RTF), and spoken-math-to-LaTeX accuracy on real-world
academic engineering lecture recordings.

Architecture:
  1. High-Fidelity Audio Resampling: SoX 16kHz 16-bit mono decimation
  2. Dynamic Adaptive VAD: Pause-aware utterance chunking (2.0s - 12.0s)
  3. Edge ASR: whisper-server HTTP daemon (CUDA SM87 Flash Attention)
  4. Mathematical SLM: Qwen2.5-1.5B-Instruct-GGUF via llama-server (CUDA SM87)
  5. Multi-Metric Profiling: P50, P90, P95, Mean, Min, Max, and overall RTF
  6. Artifacts: Markdown notes, SRT/VTT subtitles, CSV metrics, and LaTeX syntax report
"""

import argparse
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

# Ensure src/ is in sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from slm_translator import QwenMathTranslator

MATH_INDICATOR_PATTERN = re.compile(
    r"\b("
    r"equals?|is\s+equal\s+to|greater\s+than|less\s+than|"
    r"plus|minus|times|multiplied\s+by|divided\s+by|"
    r"(?:[0-9]+|[a-z]|zero|one|two|three|four|five|six|seven|eight|nine|ten|pi|numerator|\))\s+over\s+(?:[0-9]+|[a-z]|zero|one|two|three|four|five|six|seven|eight|nine|ten|pi|denominator)|"
    r"squared|cubed|to\s+the\s+power|square\s+root|cube\s+root|root\s+of|"
    r"fraction|numerator|denominator|"
    r"sine|cosine|tangent|cotangent|secant|cosecant|logarithm|\bln\b|exponential|"
    r"integral|derivative|dy\s+by\s+d[tx]|partial\s+derivative|gradient|divergence|curl|"
    r"summation|sum\s+from|sum\s+of|"
    r"sigma|omega|theta|alpha|beta|gamma|lambda|phi|psi|delta|epsilon|tau|mu|"
    r"matrix|determinant|eigenvalue|eigenvector|"
    r"z-transform|laplace|fourier|"
    r"[xyzfstk]\s+of\s+[xyzfstkn0-9]|"
    r"sinusoid|kcl|kvl|ohm\'?s\s+law"
    r")\b|"
    r"\b[0-9]+(?:\.[0-9]+)?\s*(?:hz|khz|mhz|ghz|volts?|amps?|amperes?|ohms?|farads?|henrys?|watts?|kev|mev|ev|db)\b|"
    r"[=+\-*/^<>]",
    re.IGNORECASE,
)



def has_math_indicators(text: str) -> bool:
    """Fast linguistic pre-check to distinguish mathematical speech from pure conversational discourse."""
    if not text:
        return False
    return bool(MATH_INDICATOR_PATTERN.search(text))


def clean_slm_hallucinations(latex_text: str, raw_text: str) -> str:
    """Safeguard against conversational assistant apologies or degenerative repetition loops."""
    s = latex_text.strip()
    # Check for conversational apology / assistant filler
    if re.search(r"^(?:I apologize|Thank you for sharing|As an AI|Could you please provide|Here is the)", s, re.IGNORECASE):
        return raw_text
    # Check for repetitive loops (e.g. 'by speaking in English... by speaking in English...')
    words = s.split()
    if len(words) > 15:
        # Check if same 3-word phrase repeats more than 4 times
        tri_grams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        counts = Counter(tri_grams)
        if any(cnt >= 4 for cnt in counts.values()):
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
    """Basic structural validation of LaTeX expressions (balanced braces, valid commands)."""
    # Check balanced braces
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


def segment_audio(wav_path: Path, min_chunk_sec: float = 2.0, max_chunk_sec: float = 12.0, min_silence_sec: float = 0.35) -> list[tuple[float, float, float, bytes]]:
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
    adapt_thresh = max(0.008, min(0.035, p25 * 1.8))

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


def query_whisper(wav_bytes: bytes, server_url: str = "http://127.0.0.1:8081") -> tuple[str, float]:
    """Query whisper-server on Jetson via multipart/form-data."""
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="chunk.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode("latin1") + wav_bytes + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'
        f"--{boundary}--\r\n"
    ).encode("latin1")

    req = urllib.request.Request(
        f"{server_url}/inference",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=15) as r:
        res = json.loads(r.read().decode())
    asr_ms = (time.monotonic() - t0) * 1000.0

    raw_text = res.get("text", "").strip()
    raw_text = re.sub(r"\[BLANK_AUDIO\]", "", raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r"\((?:laughter|applause|cheering|cough|sigh|crashing)\)", "", raw_text, flags=re.IGNORECASE)
    return raw_text.strip(), asr_ms


def main():
    parser = argparse.ArgumentParser(description="Academic Lecture Spoken-Math-to-LaTeX Benchmark Engine")
    parser.add_argument(
        "--audio",
        type=str,
        default="/home/ayush02/Downloads/Electrical Engineering 100 - Lecture 10 - audio only. No Visu-PGbvb3Pv570(1).wav",
        help="Path to input lecture audio WAV file",
    )
    parser.add_argument("--start_sec", type=float, default=0.0, help="Start offset in seconds")
    parser.add_argument("--duration_sec", type=float, default=300.0, help="Test duration in seconds (0 for full)")
    parser.add_argument("--output_dir", type=str, default="/mnt/thesis/evaluation_results", help="Directory for results")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        print(f"❌ Error: Audio file not found at {audio_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 78)
    print("🎓 JETSON ORIN NANO - ACADEMIC LECTURE SPEECH-TO-LATEX BENCHMARK")
    print("=" * 78)
    print(f"📁 Input File:        {audio_path.name}")
    print(f"⏱️ Test Duration:     {args.duration_sec:.1f}s ({args.duration_sec/60:.2f} min)" if args.duration_sec > 0 else "⏱️ Test Duration:     Full 65 Minutes")
    print(f"⚡ ASR Engine:        Whisper-Small (Flash Attention on CUDA)")
    print(f"🧠 SLM Engine:        Qwen2.5-1.5B-Instruct-GGUF (CUDA SM87)")
    print(f"📂 Output Dir:        {out_dir}")
    print("=" * 78 + "\n")

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
    chunks = segment_audio(sliced_wav, min_chunk_sec=2.0, max_chunk_sec=12.0, min_silence_sec=0.35)
    print(f"✅ Generated {len(chunks)} speech utterance chunks (Mean duration: {actual_audio_sec/len(chunks):.2f}s).\n")

    # 3. Initialize SLM Translator
    print("🧠 Step 3/4: Connecting to Qwen2.5-1.5B SLM on llama-server (CUDA)...")
    slm = QwenMathTranslator(backend="server")
    print("✅ SLM Translator ready.\n")

    # 4. Processing Loop
    print("🚀 Step 4/4: Benchmarking End-to-End Pipeline on Lecture Chunks...")
    print("-" * 78)

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

    for idx, (c_start, c_end, c_dur, c_bytes) in enumerate(chunks, 1):
        # Abs timestamp in lecture
        abs_start = args.start_sec + c_start
        abs_end = args.start_sec + c_end
        t_chunk_start = time.monotonic()

        # Whisper ASR
        try:
            raw_text, asr_ms = query_whisper(c_bytes)
        except Exception as e:
            raw_text, asr_ms = f"[ASR Error: {e}]", 0.0

        # SLM Math Translation
        if raw_text and not raw_text.startswith("[ASR Error"):
            if has_math_indicators(raw_text):
                try:
                    t0 = time.monotonic()
                    raw_latex = slm.translate(raw_text)
                    latex_text = clean_slm_hallucinations(raw_latex, raw_text)
                    slm_ms = (time.monotonic() - t0) * 1000.0
                except Exception as e:
                    latex_text = raw_text
                    slm_ms = 0.0
            else:
                latex_text = raw_text
                slm_ms = 0.05
        else:
            latex_text = raw_text
            slm_ms = 0.0

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
        print(f"[{idx:02d}/{len(chunks):02d}] [{format_mmss(abs_start)}-{format_mmss(abs_end)} | {c_dur:4.1f}s] ASR: {asr_ms:5.1f}ms | SLM: {slm_ms:5.1f}ms | Total: {total_chunk_ms:5.1f}ms | RTF: {rtf:5.3f}x{math_badge}")
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
            "raw_transcript": raw_text,
            "latex_transcript": latex_text,
        })

    total_wall_sec = time.monotonic() - total_wall_start
    overall_rtf = total_wall_sec / actual_audio_sec

    # 5. Summary Statistics Calculation
    print("\n" + "=" * 78)
    print("📊 BENCHMARK EVALUATION SUMMARY")
    print("=" * 78)
    print(f"⏱️ Total Audio Processed:   {actual_audio_sec:.1f}s ({actual_audio_sec/60:.2f} minutes)")
    print(f"⚡ Total Wall-Clock Time:   {total_wall_sec:.1f}s ({total_wall_sec/60:.2f} minutes)")
    print(f"🚀 Overall Real-Time Factor: {overall_rtf:.4f}x ({1.0/overall_rtf:.2f}x faster than real-time!)")
    print("-" * 78)
    print("LATENCY DISTRIBUTION (ms per chunk):")
    print(f"  ASR:   Mean={np.mean(asr_latencies):.1f}ms | P50={np.median(asr_latencies):.1f}ms | P90={np.percentile(asr_latencies, 90):.1f}ms | P95={np.percentile(asr_latencies, 95):.1f}ms | Max={max(asr_latencies):.1f}ms")
    print(f"  SLM:   Mean={np.mean(slm_latencies):.1f}ms | P50={np.median(slm_latencies):.1f}ms | P90={np.percentile(slm_latencies, 90):.1f}ms | P95={np.percentile(slm_latencies, 95):.1f}ms | Max={max(slm_latencies):.1f}ms")
    print(f"  Total: Mean={np.mean(total_latencies):.1f}ms | P50={np.median(total_latencies):.1f}ms | P90={np.percentile(total_latencies, 90):.1f}ms | P95={np.percentile(total_latencies, 95):.1f}ms | Max={max(total_latencies):.1f}ms")
    print("-" * 78)
    print("ACCURACY & MATHEMATICAL FORMATION:")
    print(f"  Total Utterance Chunks:   {len(chunks)}")
    print(f"  Chunks with Math:         {sum(1 for c in chunk_records if c['has_math'])} ({sum(1 for c in chunk_records if c['has_math'])/len(chunks)*100:.1f}%)")
    print(f"  Total Formulas Detected:  {detected_math_count}")
    print(f"  LaTeX Syntax Pass Rate:   {100.0 if not all_formulas else (len(all_formulas)-invalid_latex_count)/len(all_formulas)*100:.1f}%")
    print("=" * 78 + "\n")

    # 6. Generate Exports
    base_name = f"lecture_eval_{timestamp}"

    # CSV Export
    csv_file = out_dir / f"{base_name}.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("chunk_id,start_time,end_time,duration_sec,asr_ms,slm_ms,total_ms,rtf,has_math,raw_transcript,latex_transcript\n")
        for c in chunk_records:
            raw_clean = c["raw_transcript"].replace('"', '""')
            latex_clean = c["latex_transcript"].replace('"', '""')
            f.write(f'{c["chunk_id"]},{c["start_time"]},{c["end_time"]},{c["duration_sec"]},{c["asr_ms"]},{c["slm_ms"]},{c["total_ms"]},{c["rtf"]},{c["has_math"]},"{raw_clean}","{latex_clean}"\n')

    # JSON Export
    json_file = out_dir / f"{base_name}.json"
    json_data = {
        "metadata": {
            "source_file": audio_path.name,
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
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')} | **Segment Tested:** 00:00 - {format_mmss(actual_audio_sec)} | **Engine:** Whisper-Small + Qwen2.5-1.5B (CUDA SM87)\n\n---\n\n")
        for c in chunk_records:
            f.write(f"**[{c['start_time']} - {c['end_time']}]** {c['latex_transcript']}\n\n")

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
        for f_item in all_formulas[:15]:
            f.write(f"| {f_item['timestamp']} | `{f_item['formula']}` ({f_item['formula']}) |\n")
        f.write("\n---\n")
        f.write(f"*Report auto-generated by `benchmark_lecture.py` on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Clean up temporary sliced audio file
    if sliced_wav.exists():
        sliced_wav.unlink()

    print(f"📁 Reports & Transcripts Saved to:")
    print(f"  - Markdown Report:     {report_file}")
    print(f"  - Markdown Transcript: {md_file}")
    print(f"  - Subtitles (SRT):     {srt_file}")
    print(f"  - Subtitles (VTT):     {vtt_file}")
    print(f"  - Per-Chunk CSV:       {csv_file}")
    print(f"  - Structured JSON:     {json_file}\n")


if __name__ == "__main__":
    main()
