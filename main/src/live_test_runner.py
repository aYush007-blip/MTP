import argparse
import csv
import re
import subprocess
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudio

from audio import stereo_48k_to_mono_16k
from router import MathRouter, RouteType, T5MathTranslator
from vad import SileroVAD


RATE = 48000
CHANNELS = 2
FORMAT = pyaudio.paInt16
CHUNK = 4800
DEVICE_NAME = "AB13X USB Audio"
SAMPLE_RATE = 16000
PRE_BUFFER_CHUNKS = 5
MAX_UTTERANCE_SECONDS = 60

OUTPUT_DIR = Path("/mnt/thesis/recordings")
WHISPER_BIN = Path(
    "/mnt/thesis/main/whisper.cpp/build-orin/bin/whisper-cli"
)
WHISPER_MODEL = Path("/mnt/thesis/models/whisper/ggml-base.bin")

PROMPTS = {
    "english": [
        "I am testing the live speech transcription system.",
        "The microphone captures my voice and sends it to the computer.",
        "The system converts spoken sentences into written text.",
        "This project runs speech recognition offline on a Jetson device.",
        "The voltage across a resistor depends on the current and the resistance.",
        "The experiment measures the output signal at different frequencies.",
        "Please record this sentence clearly and transcribe every word.",
        "A stable power supply is important for reliable measurements.",
        "The real-time audio pipeline detects speech and saves each utterance.",
        "Today I am checking the accuracy and response time of the system.",
    ],
    "english_math": [
        "Voltage is the product of resistance and current.",
        "Power is the product of voltage and current.",
        "Force is the product of mass and acceleration.",
        "The current through the resistor increases when the applied voltage increases.",
        "The total resistance in a series circuit is the sum of the individual resistances.",
        "The output voltage depends on the input signal and the gain of the amplifier.",
        "The frequency is measured in hertz and the phase is measured in degrees.",
        "The energy stored in a capacitor depends on its capacitance and voltage.",
    ],
    "symbolic_math": [
        {
            "spoken": "V equals I times R.",
            "latex": "V=IR",
        },
        {
            "spoken": "P equals V times I.",
            "latex": "P=VI",
        },
        {
            "spoken": "F equals m times a.",
            "latex": "F=ma",
        },
        {
            "spoken": (
                "Sigma equals the summation from i equals 1 to n "
                "of x sub i."
            ),
            "latex": "\\sigma=\\sum_{i=1}^{n}x_{i}",
        },
    ],
    "mixed": [
        {
            "spoken": "We know that x plus y equals ten so we can solve for y.",
            "latex": "x + y = 10",
        },
        {
            "spoken": "Suppose f of x equals three x squared plus two.",
            "latex": "f(x) = 3x^{2} + 2",
        },
        {
            "spoken": "Let y equal sine of theta.",
            "latex": "y = \\sin(\\theta)",
        },
        {
            "spoken": "Therefore a squared plus b squared equals c squared.",
            "latex": "a^{2} + b^{2} = c^{2}",
        },
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run guided live microphone tests for English transcription "
            "and spoken-math-to-LaTeX."
        )
    )
    parser.add_argument(
        "--category",
        choices=("all", *PROMPTS),
        default="all",
        help="Evaluation track to run (default: all).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for each utterance to finish (default: 30).",
    )
    parser.add_argument(
        "--batch_audio_dir",
        type=str,
        default=None,
        help="Optional directory of recorded WAV files to evaluate non-interactively.",
    )
    return parser.parse_args()


def normalize_text(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def normalize_latex(text):
    text = text.strip().strip("$")
    return re.sub(r"\s+", "", text)


def word_error_rate(expected, actual):
    reference = normalize_text(expected).split()
    hypothesis = normalize_text(actual).split()
    if not reference:
        return 0.0 if not hypothesis else 1.0

    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_word in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_word in enumerate(hypothesis, start=1):
            substitution = previous[hyp_index - 1] + (ref_word != hyp_word)
            insertion = current[hyp_index - 1] + 1
            deletion = previous[hyp_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] / len(reference)


def transcribe(audio_file):
    result = subprocess.run(
        [
            str(WHISPER_BIN),
            "-m",
            str(WHISPER_MODEL),
            "-f",
            str(audio_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Whisper failed: {error}")

    segments = []
    for line in result.stdout.splitlines():
        if "]" in line:
            text = line.split("]", 1)[1].strip()
            if text:
                segments.append(text)
    return " ".join(segments)


def find_microphone(pa):
    for index in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(index)
        if (
            info["maxInputChannels"] >= CHANNELS
            and DEVICE_NAME.lower() in info["name"].lower()
        ):
            print(f"Using microphone: device {index} - {info['name']}")
            return index
    raise RuntimeError(f"Microphone '{DEVICE_NAME}' not found.")


def capture_utterance(stream, vad, timeout):
    pre_buffer = []
    speech_buffer = []
    in_speech = False
    started_at = time.monotonic()

    vad.reset()
    for _ in range(3):
        stream.read(CHUNK, exception_on_overflow=False)

    while time.monotonic() - started_at < timeout:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio = stereo_48k_to_mono_16k(data)
        events = vad.process(audio)

        pre_buffer.append(audio)
        if len(pre_buffer) > PRE_BUFFER_CHUNKS:
            pre_buffer.pop(0)

        for event in events:
            if "start" in event and not in_speech:
                print(">>> Speech detected")
                in_speech = True
                speech_buffer = list(pre_buffer)
            elif "end" in event and in_speech:
                print("<<< Speech ended")
                return np.concatenate(speech_buffer)

        if in_speech:
            speech_buffer.append(audio)

    return None


def save_audio(audio, audio_path):
    audio_int16 = np.clip(
        audio * 32767,
        -32768,
        32767,
    ).astype(np.int16)
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_int16.tobytes())


def selected_prompts(category):
    categories = tuple(PROMPTS) if category == "all" else (category,)
    selected = []
    for name in categories:
        for prompt in PROMPTS[name]:
            if name in ("symbolic_math", "mixed"):
                selected.append((name, prompt["spoken"], prompt["latex"]))
            else:
                selected.append((name, prompt, ""))
    return selected


def run():
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero.")
    if not WHISPER_BIN.is_file():
        raise FileNotFoundError(f"Whisper executable not found: {WHISPER_BIN}")
    if not WHISPER_MODEL.is_file():
        raise FileNotFoundError(f"Whisper model not found: {WHISPER_MODEL}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    results_path = OUTPUT_DIR / f"live_test_{session_id}.csv"
    prompts = selected_prompts(args.category)

    translator = T5MathTranslator.get_instance()
    router = MathRouter(translator=translator)

    pa = pyaudio.PyAudio() if not args.batch_audio_dir else None
    stream = None
    try:
        if not args.batch_audio_dir:
            device_index = find_microphone(pa)
            stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK,
            )
            vad = SileroVAD()
        else:
            vad = None

        with results_path.open("w", newline="", encoding="utf-8") as csv_file:
            fields = (
                "number",
                "category",
                "expected_transcript",
                "raw_transcript",
                "transcript_exact_match",
                "transcript_wer",
                "route_type",
                "processed_output",
                "expected_latex",
                "formula_latex",
                "formula_exact_match",
                "status",
                "error",
                "audio_seconds",
                "transcription_seconds",
                "audio_file",
            )
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            csv_file.flush()

            print()
            print(f"Running {len(prompts)} prompts ({args.category}).")
            print("English formula descriptions remain English.")
            print("Symbolic math and mixed formula spans are converted to LaTeX via T5-v2.")
            print(f"Results will be saved to: {results_path}")

            task_results = {
                name: {
                    "count": 0,
                    "exact": 0,
                    "wer": 0.0,
                    "formula_count": 0,
                    "formula_exact": 0,
                }
                for name in PROMPTS
            }

            batch_files = []
            if args.batch_audio_dir:
                b_dir = Path(args.batch_audio_dir)
                batch_files = sorted(b_dir.glob("*.wav"))

            for number, (category, expected, expected_latex) in enumerate(
                prompts, start=1
            ):
                print()
                print(f"[{number}/{len(prompts)}] {category.upper()}")
                print(f"Say/Expected: {expected}")

                if not args.batch_audio_dir:
                    action = input("Press Enter to record, s to skip, q to quit: ")
                    if action.strip().lower() == "q":
                        break
                else:
                    action = ""

                row = {
                    "number": number,
                    "category": category,
                    "expected_transcript": expected,
                    "raw_transcript": "",
                    "transcript_exact_match": "",
                    "transcript_wer": "",
                    "route_type": "",
                    "processed_output": "",
                    "expected_latex": expected_latex,
                    "formula_latex": "",
                    "formula_exact_match": "",
                    "status": "",
                    "error": "",
                    "audio_seconds": "",
                    "transcription_seconds": "",
                    "audio_file": "",
                }

                if action.strip().lower() == "s":
                    row["status"] = "skipped"
                    writer.writerow(row)
                    csv_file.flush()
                    continue

                task_results[category]["count"] += 1
                if expected_latex:
                    task_results[category]["formula_count"] += 1

                audio_path = None
                if args.batch_audio_dir:
                    # Look for matching file by number or category
                    matched = None
                    for bf in batch_files:
                        if f"_{number:02d}_" in bf.name or f"_{number:d}_" in bf.name:
                            matched = bf
                            break
                    if not matched and number <= len(batch_files):
                        matched = batch_files[number - 1]

                    if matched and matched.is_file():
                        audio_path = matched
                        row["audio_file"] = str(audio_path)
                    else:
                        print(f"No audio file found for prompt {number}.")
                        row["status"] = "file_not_found"
                        writer.writerow(row)
                        continue
                else:
                    print("Listening...")
                    audio = capture_utterance(stream, vad, args.timeout)
                    if audio is None:
                        print("No completed utterance detected before timeout.")
                        row["status"] = "timeout"
                        row["transcript_wer"] = "1.0000"
                        task_results[category]["wer"] += 1.0
                        if expected_latex:
                            row["formula_exact_match"] = "False"
                        writer.writerow(row)
                        csv_file.flush()
                        continue
                    else:
                        audio_path = OUTPUT_DIR / (
                            f"live_test_{session_id}_{number:02d}_{category}.wav"
                        )
                        save_audio(audio, audio_path)
                        row["audio_file"] = str(audio_path)
                        row["audio_seconds"] = f"{len(audio) / SAMPLE_RATE:.2f}"
                        print(f"Saved audio: {audio_path}")

                print("Transcribing...")
                transcription_started = time.monotonic()
                try:
                    transcript = transcribe(audio_path)
                except RuntimeError as error:
                    row["status"] = "transcription_error"
                    row["error"] = str(error)
                    row["transcript_wer"] = "1.0000"
                    task_results[category]["wer"] += 1.0
                    if expected_latex:
                        row["formula_exact_match"] = "False"
                    print(f"Transcription failed: {error}")
                else:
                    transcription_seconds = (
                        time.monotonic() - transcription_started
                    )
                    transcript_match = (
                        normalize_text(expected) == normalize_text(transcript)
                    )
                    transcript_wer = word_error_rate(expected, transcript)
                    row["raw_transcript"] = transcript
                    row["transcript_exact_match"] = str(transcript_match)
                    row["transcript_wer"] = f"{transcript_wer:.4f}"
                    row["status"] = "transcribed"
                    row["transcription_seconds"] = (
                        f"{transcription_seconds:.2f}"
                    )
                    task_results[category]["exact"] += int(transcript_match)
                    task_results[category]["wer"] += transcript_wer

                    print(f"Expected transcript: {expected}")
                    print(f"Raw transcript:      {transcript or '(empty)'}")

                    # Run MathRouter end-to-end
                    route_res = router.process(transcript)
                    row["route_type"] = route_res.route_type.value
                    row["processed_output"] = route_res.processed_text
                    print(f"Router Decision:     {route_res.route_type.value}")
                    print(f"Routed Output:       {route_res.processed_text}")

                    if expected_latex:
                        if category == "symbolic_math":
                            formula = route_res.processed_text
                        elif category == "mixed" and route_res.spans:
                            formula = route_res.spans[0].latex_text
                        else:
                            formula = route_res.processed_text

                        formula_match = (
                            normalize_latex(formula)
                            == normalize_latex(expected_latex)
                        )
                        row["formula_latex"] = formula
                        row["formula_exact_match"] = str(formula_match)
                        task_results[category]["formula_exact"] += int(
                            formula_match
                        )
                        print(f"Expected LaTeX:      {expected_latex}")
                        print(f"Formula output:      {formula}")
                        print(f"Formula match:       {formula_match}")

                    print(f"Transcript WER:      {transcript_wer:.1%}")
                    print(
                        f"ASR time:            "
                        f"{transcription_seconds:.2f} seconds"
                    )

                writer.writerow(row)
                csv_file.flush()

        print()
        print("Per-track scores:")
        for category, result in task_results.items():
            if not result["count"]:
                continue
            exact_percent = 100 * result["exact"] / result["count"]
            mean_wer = 100 * result["wer"] / result["count"]
            print(
                f"  {category}: transcript exact "
                f"{result['exact']}/{result['count']} ({exact_percent:.1f}%), "
                f"mean WER {mean_wer:.1f}%"
            )
            if result["formula_count"]:
                formula_percent = (
                    100 * result["formula_exact"] / result["formula_count"]
                )
                print(
                    f"    LaTeX exact {result['formula_exact']}/"
                    f"{result['formula_count']} ({formula_percent:.1f}%)"
                )
        print(f"Test run finished. Results: {results_path}")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        if pa is not None:
            pa.terminate()


if __name__ == "__main__":
    run()
