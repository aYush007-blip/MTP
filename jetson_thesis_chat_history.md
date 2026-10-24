# Complete Chat History — Jetson Thesis / Live Speech Transcription Project

**Compiled:** 2026-10-08

## Project Goal

The project is a master's thesis based on an **NVIDIA Jetson Orin Nano Developer Kit 8GB**.

The goal is to build a **fully offline live speech transcription system**, eventually extending it to:

- Multilingual speech recognition
- Language identification
- Code-switching
- Engineering/scientific terminology
- Recognition of engineering formulae and mathematical notation
- Conversion of spoken equations into structured notation / LaTeX

High-level architecture:

```text
Microphone
    ↓
Audio preprocessing
    ↓
VAD
    ↓
LID
    ↓
ASR
    ↓
Domain NLP / Notation Parser
    ├── Normal text
    └── Engineering expression → LaTeX / Unicode
```

## Hardware / OS

- NVIDIA Jetson Orin Nano Developer Kit 8GB
- Ubuntu 22.04.5 LTS
- aarch64
- JetPack 6.2
- L4T R36.4.3
- CUDA 12.6
- cuDNN 9.3
- TensorRT 10.3
- Power mode 25W
- Hostname: `jetson-thesis`
- Username: `ayush02`
- RDP access is now working.

Previous Jetson IPs included `10.117.166.63` and `10.117.166.61`.

Example SSH:

```bash
ssh ayush02@10.117.166.61
```

Important precautions established:

- Do not reflash unnecessarily.
- Do not manually replace NVIDIA drivers.
- Do not blindly run `apt upgrade`.
- Keep the working JetPack/CUDA stack intact.

## External SSD

SSD:

```text
/dev/sda2
```

Filesystem:

```text
ext4
```

Label:

```text
THESIS_SSD
```

UUID:

```text
fa855082-7efd-4055-be99-4c3490dbd3a6
```

Mount:

```text
/mnt/thesis
```

`/etc/fstab`:

```text
UUID=fa855082-7efd-4055-be99-4c3490dbd3a6 /mnt/thesis ext4 defaults,nofail 0 2
```

Project structure:

```text
/mnt/thesis/
├── logs/
├── lost+found/
├── main/
├── models/
├── recordings/
└── transcripts/
```

Ownership:

```bash
sudo chown -R ayush02:ayush02 /mnt/thesis
```

Verified mount:

```text
/dev/sda2  117G  6.0G  105G  6%  /mnt/thesis
```

If disconnected:

```bash
df -h /mnt/thesis
lsblk -f
```

Manual mount:

```bash
sudo mount /dev/sda2 /mnt/thesis
```

## Python Environment

Virtual environment:

```text
/mnt/thesis/main/.venv
```

Python:

```text
3.10.12
```

Activation:

```bash
cd /mnt/thesis/main
source .venv/bin/activate
```

VS Code is installed directly on the Jetson using ARM64 support.

Recommended interpreter:

```text
/mnt/thesis/main/.venv/bin/python
```

## Audio

USB Type-C earphones/mic:

```text
AB13X USB Audio
```

ALSA:

```text
card 2: Audio [AB13X USB Audio]
device 0: USB Audio [USB Audio]
```

Native hardware parameters:

```text
FORMAT   S16_LE
CHANNELS 2
RATE     48000
```

Mic test:

```bash
arecord -d 5 -f cd test.wav
aplay test.wav
```

worked.

PyAudio:

```text
0.2.14
```

The USB device index changes after re-enumeration. One verified index was `24`.

Direct PyAudio test:

```text
MIC OPEN OK
```

Direct 16 kHz capture failed with:

```text
OSError: [Errno -9997] Invalid sample rate
```

Therefore the system captures at 48 kHz stereo and converts internally to 16 kHz mono.

## Audio Preprocessing

File:

```text
/mnt/thesis/main/src/audio.py
```

Function:

```python
stereo_48k_to_mono_16k(audio_bytes: bytes) -> np.ndarray
```

Process:

1. Stereo int16 input
2. Average channels
3. `scipy.signal.resample_poly(..., up=1, down=3)`
4. Clip to `[-1, 1]`
5. Return float32

Test:

```text
239,616 stereo samples
→ 79,872 mono 16 kHz samples
≈ 4.992 s
```

Installed:

```text
NumPy 1.26.4
SciPy 1.15.3
```

NumPy was downgraded from 2.2.6 to maintain PyTorch compatibility.

## PyTorch / CUDA

Initially generic:

```text
torch 2.13.0+cu130
```

was installed, but CUDA was unavailable.

It was removed.

Working Jetson-specific version:

```text
torch 2.8.0
```

from:

```text
https://pypi.jetson-ai-lab.io/jp6/cu126
```

Verified:

```text
PyTorch 2.8.0
Torch CUDA 12.6
CUDA available: True
GPU: Orin
```

Do not replace this installation.

`pypi.jetson-ai-lab.dev` returned NXDOMAIN on the college DNS, while `.io` worked.

## Silero VAD

Version:

```text
silero-vad 6.2.1
```

A `torchaudio 2.11.0` installation pulled CUDA 13 and caused:

```text
libcudart.so.13
```

errors.

It was removed.

Working torchaudio:

```text
torchaudio 2.8.0
```

from:

```text
https://pypi.jetson-ai-lab.io/jp6/cu126
```

### Original Silero implementation

```python
import torch
from silero_vad import load_silero_vad, VADIterator

class SileroVAD:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_silero_vad()
        self.model = self.model.to(self.device)

        self.vad = VADIterator(
            self.model,
            sampling_rate=16000,
            threshold=0.5,
            min_silence_duration_ms=500,
            speech_pad_ms=100,
        )

        self.buffer = torch.empty(0, device=self.device)

    def process(self, audio):
        audio_tensor = torch.from_numpy(audio).to(self.device)
        self.buffer = torch.cat((self.buffer, audio_tensor))
        results = []
        while self.buffer.numel() >= 512:
            chunk = self.buffer[:512]
            self.buffer = self.buffer[512:]
            result = self.vad(chunk)
            if result is not None:
                results.append(result)
        return results

    def reset(self):
        self.vad.reset_states()
        self.buffer = torch.empty(0, device=self.device)
```

It had previously worked and produced:

```python
[{'start': 6080}, {'end': 11328}, {'start': 16832}]
```

and live:

```text
>>> SPEECH START
<<< SPEECH END
```

During the latest test, `python src/live_vad.py` segfaulted immediately.

Isolation showed:

- PyAudio opens successfully.
- PyTorch CUDA works.
- `load_silero_vad()` works.
- `model.to("cuda")` works.
- `VADIterator(...)` segfaults in `reset_states`.
- `model.reset_states()` segfaults on CUDA.
- `model.reset_states()` also segfaults on CPU.
- Direct forward inference `model(torch.zeros(512), 16000)` also segfaults on CPU.

The loaded model is:

```text
torch.jit._script.RecursiveScriptModule
```

The `reset_states` graph contains `aten::zeros` and `prim::SetAttr` operations.

Current evidence supports:

> Silero VAD 6.2.1's TorchScript model crashes in this environment independently of CPU/GPU execution.

Therefore the decision was to stop modifying Silero blindly and use a lightweight fallback for the live demo.

## Whisper.cpp

Repository:

```text
/mnt/thesis/main/whisper.cpp
```

Generic CUDA build:

```bash
cmake -B build -DGGML_CUDA=ON
```

The generic build targeted several architectures.

Binary:

```text
/mnt/thesis/main/whisper.cpp/build/bin/whisper-cli
```

Approximate binary size:

```text
922K
```

CUDA library:

```text
~512M
```

## Whisper Model

Downloaded with:

```bash
bash ./models/download-ggml-model.sh base
```

Moved to:

```text
/mnt/thesis/models/whisper/ggml-base.bin
```

Size:

```text
142M
```

It is multilingual:

```text
n_langs = 99
```

and is not `base.en`.

## Orin SM87 Optimization

Orin Nano:

```text
Compute Capability = 8.7
CUDA architecture = sm_87
```

`SM` means Streaming Multiprocessor.

Optimized build:

```bash
cd /mnt/thesis/main/whisper.cpp

cmake -B build-orin     -DGGML_CUDA=ON     -DCMAKE_CUDA_ARCHITECTURES=87

cmake --build build-orin -j$(nproc)
```

Binary:

```text
/mnt/thesis/main/whisper.cpp/build-orin/bin/whisper-cli
```

## Whisper Performance

Generic CUDA build on a 3.1-second clip:

```text
73.31 s
```

CPU:

```text
2.631 s
```

Optimized SM87 CUDA:

```text
transcript: Can you hear me on your left?

load    431.22 ms
mel      12.18 ms
sample   58.65 ms
encode  259.03 ms
batchd  139.52 ms
total   951.55 ms
```

Later warm run:

```text
[00:00:00.000 --> 00:00:02.000] And can you hear me?

load    515.03 ms
mel      10.06 ms
sample   44.12 ms
encode  250.09 ms
batchd   99.79 ms
total   965.99 ms
```

Rough RTF for the 2-second sample:

```text
~0.48
```

Do not attribute the entire ~77× improvement solely to SM87 without controlled identical-audio benchmarking.

Professor-safe wording:

> The Orin Nano has Compute Capability 8.7, so its native CUDA target is sm_87. The first whisper.cpp CUDA build was generic and was extremely slow. Since the deployment hardware is fixed, whisper.cpp was rebuilt with `-DCMAKE_CUDA_ARCHITECTURES=87`, which tells nvcc to generate device code for SM87. Measured inference time fell from approximately 73 seconds to approximately 1 second for short test audio. This is a strong preliminary result, but identical audio and repeated controlled benchmarks should be used before attributing the entire improvement solely to architecture targeting.

Keep both:

```text
build/
build-orin/
```

for benchmarking.

## Direct Whisper Test

Optimized executable:

```text
/mnt/thesis/main/whisper.cpp/build-orin/bin/whisper-cli
```

Model:

```text
/mnt/thesis/models/whisper/ggml-base.bin
```

Command:

```bash
/mnt/thesis/main/whisper.cpp/build-orin/bin/whisper-cli   -m /mnt/thesis/models/whisper/ggml-base.bin   -f /mnt/thesis/recordings/utterance_1789327509.wav
```

This direct test **runs successfully**.

Previous expected engineering transcript:

```text
The voltage is equal to current multiplied by resistance and power is equal to voltage multiplied by current.
```

This confirmed that Whisper itself is healthy.

## Whisper Language Identification

Whisper supports:

```text
-l LANG
```

and:

```text
-l auto
```

for automatic language detection.

Translation to English:

```text
-tr
```

No-GPU:

```text
-ng
```

`-ngl` was invalid.

English test:

```text
Detected: en
p ≈ 0.720603
Transcript: And can you hear me?
```

Hindi test:

```text
Detected: hi
p ≈ 0.867440
Transcript: Today we will do Samikaran.
```

Explicit `-l hi` produced the same output.

Decision:

> Do not add a dedicated LID model yet. Use Whisper's built-in LID first.

## Previous Live Pipeline

Before the Silero crash, the complete pipeline worked:

```text
Microphone
↓
48 kHz stereo
↓
48 kHz → 16 kHz mono
↓
Silero VAD
↓
Save utterance WAV
↓
Whisper.cpp
↓
Transcript
```

Previous outputs included:

```text
I am testing the live speech transcription system.
```

```text
The system captures audio from the microphone, detects speech activity and converts the audio into text using whisper.
```

```text
The voltage across the resistor depends on the current through the resistor and its resistance.
```

```text
The voltage is equal to current multiplied by resistance and power is equal to voltage multiplied by current.
```

## Energy-Based VAD

Replacement chosen:

```text
EnergyVAD
```

File:

```text
/mnt/thesis/main/src/energy_vad.py
```

Code:

```python
import numpy as np


class EnergyVAD:
    def __init__(
        self,
        threshold=0.01,
        min_speech_chunks=2,
        min_silence_chunks=10,
    ):
        self.threshold = threshold
        self.min_speech_chunks = min_speech_chunks
        self.min_silence_chunks = min_silence_chunks

        self.speech_count = 0
        self.silence_count = 0
        self.in_speech = False

    def process(self, audio):
        rms = np.sqrt(np.mean(audio ** 2))

        if rms > self.threshold:
            self.speech_count += 1
            self.silence_count = 0
        else:
            self.silence_count += 1
            self.speech_count = 0

        if not self.in_speech and self.speech_count >= self.min_speech_chunks:
            self.in_speech = True
            self.speech_count = 0
            return "start"

        if self.in_speech and self.silence_count >= self.min_silence_chunks:
            self.in_speech = False
            self.silence_count = 0
            return "end"

        return None

    def reset(self):
        self.speech_count = 0
        self.silence_count = 0
        self.in_speech = False
```

Test:

```bash
cd /mnt/thesis/main
source .venv/bin/activate
python -c "from src.energy_vad import EnergyVAD; print('EnergyVAD OK')"
```

Expected:

```text
EnergyVAD OK
```

## Current live_vad.py

The key change from Silero is that EnergyVAD returns:

```text
"start"
"end"
None
```

rather than a list of event dictionaries.

Therefore the old:

```python
for event in result:
```

was replaced with:

```python
if result == "start" and not in_speech:
```

and:

```python
elif result == "end" and in_speech:
```

Current complete file:

```python
import pyaudio
import wave
import time
import subprocess
import numpy as np

from audio import stereo_48k_to_mono_16k
from energy_vad import EnergyVAD


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
    cmd = [
        WHISPER_BIN,
        "-m", WHISPER_MODEL,
        "-f", audio_file,
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
        print(
            f"Using microphone: device {i} - "
            f"{info['name']}"
        )
        break

if DEVICE_INDEX is None:
    raise RuntimeError(
        f"Microphone '{DEVICE_NAME}' not found."
    )


stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK,
)

vad = EnergyVAD()

pre_buffer = []
speech_buffer = []
in_speech = False

MAX_PRE_BUFFER = 5

print()
print("Listening...")
print("Speak normally. Each utterance will")
print("be transcribed.")
print("Press Ctrl+C to stop.")
print()


try:
    while True:

        data = stream.read(
            CHUNK,
            exception_on_overflow=False
        )

        audio = stereo_48k_to_mono_16k(data)

        result = vad.process(audio)

        pre_buffer.append(audio)

        if len(pre_buffer) > MAX_PRE_BUFFER:
            pre_buffer.pop(0)

        if result == "start" and not in_speech:

            print(">>> SPEECH START")

            in_speech = True
            speech_buffer = list(pre_buffer)

        elif result == "end" and in_speech:

            print("<<< SPEECH END")

            in_speech = False

            full_audio = np.concatenate(speech_buffer)

            timestamp = int(time.time())

            audio_file = (
                f"{OUTPUT_DIR}/"
                f"utterance_{timestamp}.wav"
            )

            transcript_file = (
                f"{TRANSCRIPT_DIR}/"
                f"transcript_{timestamp}.txt"
            )

            audio_int16 = np.clip(
                full_audio * 32767,
                -32768,
                32767
            ).astype(np.int16)

            with wave.open(audio_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(
                    audio_int16.tobytes()
                )

            print(f"Saved: {audio_file}")

            print("Transcribing...")

            transcript = transcribe(audio_file)

            if transcript:

                print(f"TEXT: {transcript}")

                with open(
                    transcript_file,
                    "w",
                    encoding="utf-8"
                ) as f:
                    f.write(transcript + "
")

                print(
                    f"Transcript saved: "
                    f"{transcript_file}"
                )

            speech_buffer = []

        if in_speech:
            speech_buffer.append(audio)


except KeyboardInterrupt:

    print("
Stopping...")


finally:

    stream.stop_stream()
    stream.close()
    pa.terminate()
```

Run:

```bash
cd /mnt/thesis/main
source .venv/bin/activate
python src/live_vad.py
```

Test sentence:

```text
The voltage is equal to current multiplied by resistance.
```

## Engineering Notation Parser

File:

```text
/mnt/thesis/main/src/notation_parser.py
```

Greek vocabulary includes:

```text
alpha beta gamma delta epsilon theta lambda mu sigma omega
```

Engineering dictionary includes:

```text
voltage        V
current        I
resistance     R
power          P
force          F
mass           m
acceleration   a
velocity       v
displacement   d
frequency      f
time           t
charge         Q
capacitance    C
inductance     L
```

Summation example:

```text
sigma equals summation from i equals 1 to n of x i
```

→ approximately:

```latex
\sigma = \sum_{i=1}^{n} x_{i}
```

Engineering patterns:

```text
voltage is equal to current multiplied by resistance
```

→

```text
V = IR
```

```text
power is equal to voltage multiplied by current
```

→

```text
P = VI
```

```text
force is equal to mass multiplied by acceleration
```

→

```text
F = ma
```

The rule-based parser is a baseline, not the final research solution.

## MathSpeech

MathSpeech paper:

> Speech-to-Math: Leveraging Small LMs for Accurate Conversion in Mathematical Speech-to-Formula

`MathSpeech.xlsx` contains:

```text
transcription
LaTeX
Source
```

Approximate size:

```text
1101 rows
89 unique sources
```

Example:

```text
ax plus by plus cz equals d
```

→

```latex
$ax + by + cz = d$
```

Architecture:

```text
ASR #1 ─┐
        ├→ T5-small error corrector
ASR #2 ─┘
              ↓
       Clean spoken math
              ↓
       T5-small LaTeX translator
              ↓
             LaTeX
```

Two T5-small models:

```text
~120M total parameters
~60M each
```

The paper describes approximately:

```text
23M spoken-English/LaTeX pairs
```

for the translator.

Reported paper latency:

```text
~0.45 s for a 5-second segment on V100
```

This is not a Jetson measurement.

## MathSpeech Jetson Experiment

Only the MathSpeech LaTeX translator T5-small checkpoint was tested.

Parameters:

```text
~60.495M
```

Mathematical input:

```text
sigma equals summation from i equals 1 to n of x i
```

produced meaningful LaTeX-like output.

Engineering input:

```text
voltage is equal to current multiplied by resistance
```

produced poor output similar to:

```text
textvolt = text current cdot text PET
```

This shows domain mismatch.

## EquationGPT-like Dataset

Explored domains:

```text
chemistry
economics
financial
geometry
machine_learning
math
physics
statistics
```

Approximate:

```text
1674 rows
1592 with equations
82 without equations
```

## ElectricPy

Initial extraction:

```text
~150 LaTeX math blocks
```

One cleaning stage:

```text
~130 unique raw blocks
```

Examples:

```latex
C = rac{1}{\omega*Z}

C = rac{VAR}{2*\pi*freq*V^2}

C = rac{arepsilon A}{d}

E=rac{1}{2}CV^2

E=rac{1}{2}LI^2

I_1 = rac{V_{th}}{Z_0+Z_1+Z_2+3R_f}

I_L=I_0e^{-Rt/L}

Z_{eq}=(1/Z_1+...+1/Z_n)^{-1}
```

A later count of 135 clean unique equations appeared but was considered a preprocessing anomaly and should not be treated as validated without rechecking.

## Other Research

### Speech-to-LaTeX

2025 paper:

```text
Speech-to-LaTeX
arXiv:2508.03542
```

Reported:

- >66k human-annotated audio samples
- English and Russian
- Equations and sentences
- Diverse scientific domains

Reported equation CER comparisons included approximately:

```text
28% vs 30%
27% vs 64%
```

and a sentence benchmark around:

```text
40%
```

These are paper results, not Jetson results.

### MathBridge

```text
arXiv:2408.07081
```

Approximately:

```text
23M LaTeX formulas
```

paired with spoken mathematical sentences.

Reported T5-large sacreBLEU:

```text
4.77 → 46.8
```

The work emphasizes that BLEU/ROUGE/WER/CER can be inadequate for mathematical equivalence.

## Multilingual / Code-Switching

Intended capabilities:

- Multilingual ASR
- Language identification
- Code-switching
- Indian language + English speech

A lightweight CLD approach was considered:

```text
https://github.com/pilancilab/CLD
```

It uses a lightweight convex language-detection head on a frozen Whisper encoder.

Indian resources considered included:

- VITB-HEBiC Hindi-English
- Microsoft Indian multilingual/code-switching challenge
- Hindi-English and Bengali-English resources

Current decision:

> Use Whisper's built-in LID first rather than adding a dedicated LID model.

## Evaluation Plan

Candidate baselines:

1. Rule-based notation parser
2. Original MathSpeech T5-small
3. Fine-tuned engineering-domain model

Potential metrics:

- Normalized exact match
- CER
- BLEU
- ROUGE
- Latency
- Structure-aware metrics
- Symbolic / AST equivalence

Reason:

> LaTeX is non-unique, so exact string equality is not always mathematical correctness.

## Professor Presentation

PPT was prepared and sent to the professor.

It included:

- Jetson environment
- PyTorch 2.8.0
- Torch CUDA 12.6
- CUDA available
- Orin GPU
- Proposed architecture
- Live speech transcription
- ASR/VAD/preprocessing theory
- Engineering fine-tuning proposal

Good progress summary:

> The complete offline Jetson ASR pipeline is operational, VAD and multilingual Whisper have been validated, architecture-targeted CUDA optimization has brought warm ASR inference below one second for a short utterance, and preliminary experiments show that engineering notation is the remaining domain-specific research challenge.

Do not present as completed:

- Engineering T5 fine-tuning
- Final multilingual/code-switching
- Final engineering LaTeX accuracy
- Superiority of one T5 model
- Entire ~77× improvement as solely caused by SM87

## Overall Thesis Story

```text
Build offline edge ASR
        ↓
Deploy on Jetson
        ↓
Optimize Whisper for SM87
        ↓
Achieve sub-second warm short-utterance ASR
        ↓
Add multilingual/LID
        ↓
Live speech segmentation
        ↓
Engineering-domain speech
        ↓
Spoken equation → structured notation
        ↓
Engineering-domain fine-tuning
        ↓
Accuracy + latency benchmarking
```

The central research challenge is:

> **Reliable recognition and structural conversion of spoken engineering/mathematical expressions under the compute and latency constraints of an edge device.**

## Current Project Status

```text
Jetson Orin Nano                 ✅
RDP desktop access               ✅
External SSD                     ✅
Project venv                     ✅
USB microphone                   ✅
48k stereo capture               ✅
48k → 16k preprocessing          ✅
PyTorch 2.8 CUDA                 ✅
Whisper.cpp                      ✅
Whisper Base multilingual        ✅
SM87 optimized Whisper           ✅
Direct Whisper transcription     ✅
Whisper LID tests                ✅
Silero VAD 6.2.1                 ❌ crashing
Energy-based VAD                 ✅ implemented
Live EnergyVAD pipeline          🔄 next test
Engineering parser               ✅ baseline
MathSpeech translator            ✅ tested
Engineering MathSpeech           ❌ poor without adaptation
Engineering fine-tuning          ⏳ future
Final multilingual pipeline      ⏳ future
```

## Key Commands

### Activate environment

```bash
cd /mnt/thesis/main
source .venv/bin/activate
```

### Check SSD

```bash
df -h /mnt/thesis
lsblk -f
```

### Mount SSD

```bash
sudo mount /dev/sda2 /mnt/thesis
```

### Test PyTorch

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Build optimized Whisper

```bash
cd /mnt/thesis/main/whisper.cpp

cmake -B build-orin     -DGGML_CUDA=ON     -DCMAKE_CUDA_ARCHITECTURES=87

cmake --build build-orin -j$(nproc)
```

### Direct Whisper

```bash
/mnt/thesis/main/whisper.cpp/build-orin/bin/whisper-cli   -m /mnt/thesis/models/whisper/ggml-base.bin   -f /mnt/thesis/recordings/utterance_1789327509.wav
```

### Test EnergyVAD

```bash
cd /mnt/thesis/main
source .venv/bin/activate
python -c "from src.energy_vad import EnergyVAD; print('EnergyVAD OK')"
```

### Run live pipeline

```bash
cd /mnt/thesis/main
source .venv/bin/activate
python src/live_vad.py
```

## Latest Conversation State

The latest direct Whisper test was confirmed to run successfully.

The next step was replacing the crashing Silero VAD with EnergyVAD.

The full corrected `live_vad.py` was provided.

The immediate next test is:

```bash
cd /mnt/thesis/main
source .venv/bin/activate
python src/live_vad.py
```

Then speak:

```text
The voltage is equal to current multiplied by resistance.
```

The expected flow is:

```text
Microphone
   ↓
Energy VAD
   ↓
Utterance WAV
   ↓
Whisper SM87
   ↓
Transcript
```
