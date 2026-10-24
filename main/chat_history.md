# Chat History / Project Handoff

## Goal

Build an offline live speech transcription system on an NVIDIA Jetson
Orin Nano Developer Kit 8GB:

**Microphone → audio preprocessing/VAD → Whisper ASR → English/math
routing → spoken-math-to-LaTeX model → output**

Critical behavior: ordinary English must remain English, even when it
contains numbers or math-related words. A bare "nine" should output
`nine`, not `9` or LaTeX. "There are nine students" should remain
English. "x squared equals nine" should convert to `x^{2} = 9`. Mixed
English/math should ideally preserve English and convert only math
spans.

The user wants a model that generalizes to unseen equations, not
hard-coded formula mappings or regex-only conversion.

## Hardware and working environment

-   Jetson Orin Nano Developer Kit, 8GB; Ubuntu 22.04.5, aarch64.

-   L4T R36.4.3 / JetPack 6.2; CUDA 12.6, cuDNN 9.3, TensorRT 10.3.

-   Python 3.10.12; PyTorch 2.8.0; `torch.cuda.is_available()` is
    `True`.

-   Main venv: `/mnt/thesis/main/.venv`

    ``` bash
    cd /mnt/thesis/main
    source .venv/bin/activate
    ```

-   SSD mounted at `/mnt/thesis`; verify mount/free space before large
    writes.

-   Do not create another venv, replace PyTorch, or broadly update
    CUDA/NVIDIA packages.

-   Hugging Face cache: `/mnt/thesis/huggingface_cache`.

## Existing speech pipeline

-   USB mic: `AB13X USB Audio`; hardware capture is 48 kHz stereo
    S16_LE.
-   Capture at 48 kHz, average channels, downsample to 16 kHz mono.
-   `/mnt/thesis/main/src/audio.py`:
    `stereo_48k_to_mono_16k(audio_bytes)` uses
    `scipy.signal.resample_poly(up=1, down=3)`.
-   `/mnt/thesis/main/src/energy_vad.py`: energy-based VAD.
-   `/mnt/thesis/main/src/live_vad.py`: records audio, resamples, uses
    energy VAD, saves utterances, transcribes with whisper.cpp, writes
    transcripts.
-   Silero VAD TorchScript segfaulted in this environment; continue with
    energy VAD.
-   whisper.cpp clone: `/mnt/thesis/main/whisper.cpp`
-   Orin-optimized CUDA build:
    `/mnt/thesis/main/whisper.cpp/build-orin/`, built with
    `GGML_CUDA=ON`, CUDA architecture 87.
-   Whisper multilingual Base model:
    `/mnt/thesis/models/whisper/ggml-base.bin` (\~142 MB).
-   A 3.1-second clip took about 0.95 seconds with the optimized Whisper
    CUDA build in a prior test.

## Datasets

### MathSpeech

-   `/mnt/thesis/datasets/MathSpeech`, originally 1,101 rows.
-   Arrow file:
    `/mnt/thesis/datasets/MathSpeech/data-00000-of-00001.arrow`.
-   Columns include `audio`, `transcription`, `LaTeX`, `Source`.
-   Audio is embedded MP3 bytes; stored relative paths may not exist
    locally.
-   Read Arrow directly with PyArrow; do not depend on the currently
    broken `datasets` import.
-   Examples: `ax plus by plus cz equals d` → `$ax + by + cz = d$`;
    `x plus 5y plus 10z equals zero` → `$x + 5y + 10z = 0$`.

### TIE_shorts

-   `/mnt/thesis/datasets/TIE_shorts_train`, about 12 GB, 7,884 rows.
-   Engineering-focused speech data with `Transcript`,
    `Normalised_Transcript`, and discipline metadata.
-   Useful for technical speech/ASR evaluation, not paired LaTeX
    targets.

### Existing split

`/mnt/thesis/datasets/MathSpeech/splits/` - `train.jsonl`: 850 rows -
`validation.jsonl`: 106 rows - `test.jsonl`: 107 rows - 1,063 unique
transcript inputs total; fields `input`, `target`. - Seeded random split
only; it does not establish compositional generalization. - Keep the
current test split unchanged for the initial in-distribution baseline.

## T5 baseline

-   Base model: `google-t5/t5-small`, saved at
    `/mnt/thesis/models/t5-small-baseline`.
-   \~60.5M parameters. Transformers 4.37.2, tokenizers 0.15.2, Hugging
    Face Hub 0.34.4.
-   Fine-tuned checkpoint: `/mnt/thesis/models/math-to-latex`.
-   Training script: `/mnt/thesis/main/src/train_math_latex.py`.
-   Custom PyTorch loop; 5 epochs, batch size 4, LR `2e-4`, max sequence
    length 128.
-   AdamW must use `foreach=False`; one optimizer update and the full
    run succeeded with this setting.
-   Completed 5 epochs: train loss `0.5157`, validation loss `0.5356`.
-   T5 emits a warning about added special tokens / embeddings.
    Investigate tokenizer/model vocabulary compatibility before future
    changes, without disrupting the working setup.

### Evaluation

-   Held-out test accuracy: **2/107 = 1.87% normalized exact match**.
-   Common errors: `\frac{2b}{n}` becomes `frac2bn`; exponents lose
    braces; `\ln` becomes `ln`; parentheses may be missing; "times" may
    remain literal.
-   Conclusion: baseline is not reliable. Do not simply increase epochs.
    Improve data and evaluation first.
-   Exact match is strict; eventually add LaTeX structural validity and
    mathematical equivalence checks where feasible.

## Synthetic generator (not yet approved for training)

-   Script: `/mnt/thesis/main/src/generate_math_data.py`
-   Output: `/mnt/thesis/datasets/MathSpeech/synthetic.jsonl`
-   Intended size 6,000, seed 42.
-   Grammar includes variables, number words, pi, theta, powers,
    fractions, functions (sine, cosine, tangent, natural log,
    exponential), binary operations, grouping, equations.
-   Examples:
    -   `cosine of four times theta` →
        `\cos\left(4 \times \theta\right)`
    -   `pi over open parenthesis two plus pi close parenthesis` →
        `\frac{\pi}{\left(2 + \pi\right)}`
-   **Do not train on this data yet.** Some spoken phrases are
    unnatural/ambiguous, and the generated data contained `ninesquared`
    while the visible code appears to insert a space between base and
    power. The cause is unresolved; the sample may come from stale
    output or a mismatch between the actual on-disk code and the
    displayed version.
-   A whitespace normalization patch was applied but did not remove the
    observed `ninesquared` sample.
-   Inspect the actual file and reproduce the issue from a fresh
    generation before patching. Validate a random sample, not only the
    first few examples.
-   The source pasted in chat had HTML/Markdown artifacts; inspect the
    on-disk Python file rather than copying the rendered snippet.

## Environment caveats

-   `datasets==5.1.0` import is broken with current
    `huggingface-hub==0.34.4` (`BucketNotFoundError` missing);
    `pip check` reports version incompatibility. Avoid casual Hub
    changes because Transformers currently loads T5.
-   SciPy `scipy.io.loadmat` has had an import issue; `scipy.signal`
    works.
-   SymPy 1.14.0 is now installed and imports successfully.
-   Avoid further package changes unless necessary.

## Next steps (agreed direction)

1.  **Audit generator and data.** Inspect actual on-disk script and
    saved JSONL. Reproduce `ninesquared` on a fresh generation; ensure
    spaces between spoken tokens and correct spoken/target alignment.
    Add validation for empty fields, duplicates, malformed braces, and
    suspicious concatenated words.
2.  **Improve synthetic grammar.** Start with simple, natural,
    unambiguous constructions. Add variables, constants, coefficients,
    powers, fractions, parentheses, operators, functions, equations,
    derivatives, and subscripts incrementally. Keep target LaTeX paired
    by construction; avoid excessive nesting initially.
3.  **Create compositional evaluation.** Hold out operator
    combinations/patterns (e.g. train on powers and fractions
    separately, test fractions containing powers). Keep MathSpeech's
    existing test split separate. Measure exact match, structural
    correctness, and equivalence where practical.
4.  **Build English-vs-math routing data.** Include ordinary English
    with number words, mathematical vocabulary used non-mathematically,
    pure math utterances, and mixed English/math. A bare "nine" and
    "There are nine students" must remain English. Consider a small
    classifier/router and, for mixed utterances, a span detector.
5.  **Train a new versioned checkpoint.** Combine vetted synthetic pairs
    with real MathSpeech data. Do not overwrite
    `/mnt/thesis/models/math-to-latex`. Use the existing PyTorch loop
    and `foreach=False`; avoid new dependencies unless needed.
6.  **Test Whisper-style and live speech.** Evaluate text examples
    first, then actual Whisper transcripts, then integrate routing and
    conversion into `live_vad.py`. Preserve ordinary English and convert
    only intended math spans.

## Coding guardrails

-   Work step by step; prefer small diagnostics and targeted patches.
-   Do not overwrite the current baseline checkpoint.
-   Do not install more packages without a clear need.
-   Preserve the working Jetson CUDA/PyTorch/Whisper setup.
-   Verify `/mnt/thesis` is mounted and has enough free space before
    large writes.
-   The final goal is a practical offline live speech system, not merely
    a good score on the small MathSpeech test set.
