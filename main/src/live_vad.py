import pyaudio
import wave
import time
import subprocess
import numpy as np

from audio import stereo_48k_to_mono_16k
from router import MathRouter, RouteType, T5MathTranslator
from vad import SileroVAD


RATE = 48000
CHANNELS = 2
FORMAT = pyaudio.paInt16
CHUNK = 4800
DEVICE_NAME = "AB13X USB Audio"

OUTPUT_DIR = "/mnt/thesis/recordings"
TRANSCRIPT_DIR = "/mnt/thesis/transcripts"

WHISPER_BIN = "/mnt/thesis/main/whisper.cpp/build-orin/bin/whisper-cli"
WHISPER_MODEL = "/mnt/thesis/models/whisper/ggml-base.bin"


def transcribe(audio_file):
    """Run Whisper.cpp on a saved utterance."""

    cmd = [
        WHISPER_BIN,
        "-m",
        WHISPER_MODEL,
        "-f",
        audio_file,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Whisper error:")
        print(result.stderr)
        return None

    lines = []

    for line in result.stdout.splitlines():
        if "]" in line:
            text = line.split("]", 1)[1].strip()
            if text:
                lines.append(text)

    return " ".join(lines)


pa = pyaudio.PyAudio()

DEVICE_INDEX = None

for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if (
        info["maxInputChannels"] >= CHANNELS
        and DEVICE_NAME.lower() in info["name"].lower()
    ):
        DEVICE_INDEX = i
        print(f"Using microphone: device {i} - {info['name']}")
        break

if DEVICE_INDEX is None:
    raise RuntimeError(f"Microphone '{DEVICE_NAME}' not found.")

stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK,
)

vad = SileroVAD()
print("Loading Math-to-LaTeX Router and T5-v2 model...")
translator = T5MathTranslator.get_instance()
router = MathRouter(translator=translator)

pre_buffer = []
speech_buffer = []
in_speech = False

MAX_PRE_BUFFER = 5

print()
print("Listening...")
print("Speak normally. English speech stays English.")
print("Spoken math is converted to LaTeX.")
print("Press Ctrl+C to stop.")
print()

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio = stereo_48k_to_mono_16k(data)

        events = vad.process(audio)

        pre_buffer.append(audio)
        if len(pre_buffer) > MAX_PRE_BUFFER:
            pre_buffer.pop(0)

        for event in events:
            if "start" in event and not in_speech:
                print(">>> SPEECH START")
                in_speech = True
                speech_buffer = list(pre_buffer)

            elif "end" in event and in_speech:
                print("<<< SPEECH END")
                in_speech = False

                full_audio = np.concatenate(speech_buffer)
                timestamp = int(time.time())

                audio_file = f"{OUTPUT_DIR}/utterance_{timestamp}.wav"
                transcript_file = f"{TRANSCRIPT_DIR}/transcript_{timestamp}.txt"

                audio_int16 = np.clip(
                    full_audio * 32767,
                    -32768,
                    32767,
                ).astype(np.int16)

                with wave.open(audio_file, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_int16.tobytes())

                print(f"Saved: {audio_file}")
                print("Transcribing...")

                transcript = transcribe(audio_file)

                if transcript:
                    print(f"RAW TEXT: {transcript}")
                    res = router.process(transcript)
                    print(f"ROUTE:    [{res.route_type.value}]")
                    print(f"OUTPUT:   {res.processed_text}")
                    if res.spans:
                        for s in res.spans:
                            print(f"  Span: '{s.spoken_text}' -> '{s.latex_text}'")

                    with open(transcript_file, "w", encoding="utf-8") as f:
                        f.write(f"RAW: {transcript}\n")
                        f.write(f"ROUTE: {res.route_type.value}\n")
                        f.write(f"OUTPUT: {res.processed_text}\n")

                    print(f"Transcript saved: {transcript_file}")

                speech_buffer = []

        if in_speech:
            speech_buffer.append(audio)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
