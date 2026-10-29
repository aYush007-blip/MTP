#!/usr/bin/env python3
"""
Balanced 20,000-Sample Dataset Generator & Packager for Qwen2.5-1.5B LoRA Fine-Tuning.

Produces train.jsonl and val.jsonl in ChatML format:
  <|im_start|>system
  You are an expert lecture transcriber and mathematical typesetter...
  <|im_end|>
  <|im_start|>user
  Spoken transcript:
  ...
  <|im_end|>
  <|im_start|>assistant
  ...
  <|im_end|>

Dataset Distribution (The 'Golden 20k' Split):
  - 40% (8,000): Mixed technical lecture sentences (Laplace, Z-transform, DSP, circuits, calculus)
  - 25% (5,000): Complex spoken mathematical operators (summations, integrals, fractions, derivatives)
  - 20% (4,000): Signal, variable, and symbol disambiguation (x[n] vs x(t), Greek letters, transfer functions)
  - 15% (3,000): Negative / Pure conversational English anchors (introductions, countdowns, questions)
"""

import json
from pathlib import Path
import random
import re

SEED = 42
TOTAL_SAMPLES = 20000
TRAIN_RATIO = 0.90
rng = random.Random(SEED)

DATA_DIR = Path("/mnt/thesis/datasets/Qwen_FineTune_20k")
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"

SYSTEM_PROMPT = (
    "You are an expert lecture transcriber and mathematical typesetter.\n"
    "Your task is to take a spoken lecture transcript and convert it into a natural, continuous transcript where spoken mathematical formulas and expressions are converted into standard inline LaTeX ($...$).\n\n"
    "STRICT RULES:\n"
    "1. When mathematical operators, definitions, or equations are spoken in words (e.g. summations, integrals, fractions, derivatives, signals), convert the complete spoken mathematical phrase into its standard LaTeX formula:\n"
    "   - 'the integral from zero to infinity of f of t times e to the minus s t dt' -> '$\\int_{0}^{\\infty} f(t) e^{-st} \\, dt$'\n"
    "   - 'the sum over all integers n of x of n times z to the power minus n' -> '$\\sum_{n=-\\infty}^{\\infty} x[n] z^{-n}$'\n"
    "   - 'sum from n equals 0 to infinity of a to the n' -> '$\\sum_{n=0}^{\\infty} a^n$'\n"
    "   - 'dy by dt plus two y equals one' -> '$\\frac{dy}{dt} + 2y = 1$'\n"
    "   - 'x of n' -> '$x[n]$', 'x of t' -> '$x(t)$', 'f of t' -> '$f(t)$', 'z-plane' -> '$z$-plane'\n"
    "2. Preserve 100% of the spoken conversational explanations, surrounding context, and English words. Do not summarize or omit explanations.\n"
    "3. NEVER wrap non-math words, conversational phrases, numbers, or counting in LaTeX or \\text{}. (e.g. 'Five, four, three, two, one, zero. Let us get technical.' must remain as plain text).\n"
    "4. Use ONLY inline math ($...$). NEVER use display math ($$...$$ or \\( ... \\)).\n"
    "5. Output ONLY the resulting transcript. Do not output preamble, notes, or explanations."
)

# ---------------------------------------------------------------------------
# Templates & Generators
# ---------------------------------------------------------------------------

GREEK_LETTERS = [
    ("alpha", r"\alpha"), ("beta", r"\beta"), ("gamma", r"\gamma"),
    ("delta", r"\delta"), ("theta", r"\theta"), ("lambda", r"\lambda"),
    ("mu", r"\mu"), ("sigma", r"\sigma"), ("omega", r"\omega"), ("phi", r"\phi"),
    ("pi", r"\pi"), ("epsilon", r"\epsilon"), ("tau", r"\tau")
]

VARIABLES = ["x", "y", "z", "s", "t", "n", "k", "a", "b", "c", "u", "v"]


def gen_mixed_lecture_sample() -> tuple[str, str]:
    """Category 1: Continuous conversational lecture sentence with embedded math formulas (40%)."""
    templates = [
        (
            "Today we are looking at the Laplace transform of a continuous signal {fn} of t, which maps into a complex frequency variable s.",
            "Today we are looking at the Laplace transform of a continuous signal ${fn}(t)$, which maps into a complex frequency variable $s$."
        ),
        (
            "The definition is the integral from zero to infinity of {fn} of t times e to the minus s t dt.",
            "The definition is $\\int_{{0}}^{{\\infty}} {fn}(t) e^{{-st}} \\, dt$."
        ),
        (
            "Here s equals sigma plus j omega where sigma is the attenuation and omega is the angular frequency in radians per second.",
            "Here $s = \\sigma + j\\omega$ where $\\sigma$ is the attenuation and $\\omega$ is the angular frequency in radians per second."
        ),
        (
            "If the input is {fn} of t equals e to the {a} t, the Laplace transform evaluates to 1 over s minus {a} for real part of s greater than {a}.",
            "If the input is ${fn}(t) = e^{{{a}t}}$, the Laplace transform evaluates to $\\frac{{1}}{{s - {a}}}$ for real part of $s > {a}$."
        ),
        (
            "For the differential equation d {y} by d t plus {c} {y} equals {d} with initial condition {y} of zero equals zero, the s-domain representation is s {Y} of s plus {c} {Y} of s equals {d} over s.",
            "For the differential equation $\\frac{{d{y}}}{{dt}} + {c}{y} = {d}$ with initial condition ${y}(0) = 0$, the s-domain representation is $s {Y}(s) + {c} {Y}(s) = \\frac{{{d}}}{{s}}$."
        ),
        (
            "The bilateral Z-transform of a discrete-time sequence {x} of n is the sum over all integers n of {x} of n times z to the power minus n.",
            "The bilateral Z-transform of a discrete-time sequence ${x}[n]$ is $\\sum_{{n=-\\infty}}^{{\\infty}} {x}[n] z^{{-n}}$."
        ),
        (
            "In digital filter design, if the unit circle with radius 1 in the z-plane lies entirely inside the region of convergence, the filter is bounded-input bounded-output stable.",
            "In digital filter design, if the unit circle with radius $1$ in the $z$-plane lies entirely inside the region of convergence, the filter is bounded-input bounded-output stable."
        ),
        (
            "Applying the Fourier transform to {x} of t gives the continuous spectrum capital X of omega equals integral from minus infinity to infinity of {x} of t e to the minus j omega t dt.",
            "Applying the Fourier transform to ${x}(t)$ gives the continuous spectrum $X(\\omega) = \\int_{{-\\infty}}^{{\\infty}} {x}(t) e^{{-j\\omega t}} \\, dt$."
        ),
        (
            "The transfer function h of s equals {num} over s squared plus {two_zeta} omega n s plus omega n squared represents a second-order underdamped system.",
            "The transfer function $h(s) = \\frac{{{num}}}{{s^2 + {two_zeta} \\omega_n s + \\omega_n^2}}$ represents a second-order underdamped system."
        ),
        (
            "According to Ohm's law, the voltage v across the resistor equals current i multiplied by resistance r.",
            "According to Ohm's law, the voltage $v$ across the resistor equals current $i$ multiplied by resistance $r$ ($V = IR$)."
        )
    ]
    t_in, t_out = rng.choice(templates)
    fn = rng.choice(["f", "g", "x", "y"])
    y = rng.choice(["y", "v", "x"])
    Y = y.upper()
    x = rng.choice(["x", "h", "y"])
    a = rng.choice(["a", "2", "3", "alpha"])
    c = rng.choice(["2", "3", "4", "5"])
    d = rng.choice(["1", "2", "10"])
    num = rng.choice(["1", "omega n squared", "k"])
    two_zeta = rng.choice(["2 zeta", "2", "4"])
    
    spoken = t_in.format(fn=fn, y=y, Y=Y, x=x, a=a, c=c, d=d, num=num, two_zeta=two_zeta)
    target = t_out.format(fn=fn, y=y, Y=Y, x=x, a=a, c=c, d=d, num=num, two_zeta=two_zeta)
    return spoken, target


def gen_operator_sample() -> tuple[str, str]:
    """Category 2: Spoken mathematical operator phrases converting to full formulas (25%)."""
    idx_var = rng.choice(["n", "k", "i", "m"])
    var = rng.choice(["x", "y", "a", "f"])
    lower = rng.choice(["0", "1", "minus infinity"])
    upper = rng.choice(["infinity", "capital N", "capital N minus 1", "10"])
    
    lower_tex = "-\\infty" if lower == "minus infinity" else lower
    upper_tex = "\\infty" if upper == "infinity" else ("N - 1" if "minus 1" in upper else ("N" if "capital N" in upper else upper))
    
    ops = [
        (
            f"sum from {idx_var} equals {lower} to {upper} of {var} of {idx_var}",
            f"$\\sum_{{{idx_var}={lower_tex}}}^{{{upper_tex}}} {var}[{idx_var}]$"
        ),
        (
            f"the sum over all integers {idx_var} of {var} of {idx_var} times z to the minus {idx_var}",
            f"$\\sum_{{{idx_var}=-\\infty}}^{{\\infty}} {var}[{idx_var}] z^{{-{idx_var}}}$"
        ),
        (
            f"integral from zero to infinity of e to the minus s t times f of t dt",
            f"$\\int_{{0}}^{{\\infty}} e^{{-st}} f(t) \\, dt$"
        ),
        (
            f"integral from minus infinity to infinity of absolute value of x of t squared dt is finite",
            f"$\\int_{{-\\infty}}^{{\\infty}} |x(t)|^2 \\, dt < \\infty$"
        ),
        (
            f"derivative of {var} of t with respect to t equals {rng.choice(['zero', 'constant k', 'minus a times f of t'])}",
            f"\\frac{{d{var}}}{{dt}} = 0"
        ),
        (
            f"partial of u with respect to x plus partial of v with respect to y equals zero",
            r"\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0"
        ),
        (
            f"fraction 1 over s plus a",
            r"\frac{1}{s + a}"
        ),
        (
            f"limit as {idx_var} approaches infinity of 1 plus 1 over {idx_var} to the power {idx_var} equals e",
            f"\\lim_{{{idx_var} \\to \\infty}} \\left(1 + \\frac{{1}}{{{idx_var}}}\\right)^{{{idx_var}}} = e"
        )
    ]
    spk, tex = rng.choice(ops)
    if not tex.startswith("$"):
        tex = f"${tex}$"
    return spk, tex


def gen_disambiguation_sample() -> tuple[str, str]:
    """Category 3: Discrete vs continuous signals, Greek letters, transfer functions (20%)."""
    gk_name, gk_sym = rng.choice(GREEK_LETTERS)
    v = rng.choice(VARIABLES)
    
    samples = [
        (
            f"the discrete sequence x of n is convolved with h of n to produce y of n",
            f"the discrete sequence $x[n]$ is convolved with $h[n]$ to produce $y[n]$"
        ),
        (
            f"the continuous impulse response h of t is the inverse Laplace transform of capital H of s",
            f"the continuous impulse response $h(t)$ is the inverse Laplace transform of $H(s)$"
        ),
        (
            f"the angular frequency omega equals two pi f where f is frequency in hertz",
            f"the angular frequency $\\omega = 2\\pi f$ where $f$ is frequency in hertz"
        ),
        (
            f"let {gk_name} denote the damping ratio and omega n be the natural frequency",
            f"let ${gk_sym}$ denote the damping ratio and $\\omega_n$ be the natural frequency"
        ),
        (
            f"the eigenvalues lambda 1 and lambda 2 satisfy determinant of a minus lambda i equals zero",
            f"the eigenvalues $\\lambda_1$ and $\\lambda_2$ satisfy $\\det(A - \\lambda I) = 0$"
        ),
        (
            f"the complex variable s equals sigma plus j omega resides in the s-plane",
            f"the complex variable $s = \\sigma + j\\omega$ resides in the $s$-plane"
        ),
        (
            f"the input impedance z in of s equals r plus 1 over c s plus l s",
            f"the input impedance $Z_{{in}}(s) = R + \\frac{{1}}{{Cs}} + Ls$"
        ),
        (
            f"the matrix equation a times x equals b has a unique solution if determinant of a is non zero",
            f"the matrix equation $A\\mathbf{{x}} = \\mathbf{{b}}$ has a unique solution if $\\det(A) \\neq 0$"
        )
    ]
    return rng.choice(samples)


def gen_negative_sample() -> tuple[str, str]:
    """Category 4: Pure conversational / English lecture speech (15%). Must NEVER wrap in LaTeX."""
    negatives = [
        (
            "Five, four, three, two, one, zero. Okay, let us begin today's lecture.",
            "Five, four, three, two, one, zero. Okay, let us begin today's lecture."
        ),
        (
            "Good evening everyone. Please turn to chapter four of the course syllabus.",
            "Good evening everyone. Please turn to chapter four of the course syllabus."
        ),
        (
            "Before we dive into the proofs, are there any administrative questions regarding homework three?",
            "Before we dive into the proofs, are there any administrative questions regarding homework three?"
        ),
        (
            "The midterm exam will be held next Tuesday in room four hundred and ten.",
            "The midterm exam will be held next Tuesday in room four hundred and ten."
        ),
        (
            "Notice how the intuitive behavior matches our real world laboratory observations.",
            "Notice how the intuitive behavior matches our real world laboratory observations."
        ),
        (
            "That is the kind of practical result we use over and over in circuits, control, and signal processing.",
            "That is the kind of practical result we use over and over in circuits, control, and signal processing."
        ),
        (
            "Make sure to review the reading assignment before our next laboratory session on Thursday.",
            "Make sure to review the reading assignment before our next laboratory session on Thursday."
        ),
        (
            "Remember that attendance in the recitation sessions counts toward fifteen percent of your overall grade.",
            "Remember that attendance in the recitation sessions counts toward fifteen percent of your overall grade."
        )
    ]
    return rng.choice(negatives)


def format_chatml(spoken: str, target: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Spoken transcript:\n{spoken}"},
            {"role": "assistant", "content": target}
        ]
    }


def main():
    print("=" * 65)
    print(f"Generating Balanced 20k LoRA Dataset for Spoken-Math-to-LaTeX")
    print(f"Output Directory: {DATA_DIR}")
    print("=" * 65)

    n_mixed = int(TOTAL_SAMPLES * 0.40)      # 8,000
    n_ops = int(TOTAL_SAMPLES * 0.25)        # 5,000
    n_disambig = int(TOTAL_SAMPLES * 0.20)   # 4,000
    n_neg = int(TOTAL_SAMPLES * 0.15)        # 3,000

    all_pairs = []

    # 1. Real MIT OCW MathSpeech samples if present
    mathspeech_file = Path("/mnt/thesis/datasets/MathSpeech/splits_v2/train.json")
    if mathspeech_file.exists():
        with open(mathspeech_file, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                spk = row.get("input", "").strip()
                tgt = row.get("target", "").strip()
                if spk and tgt:
                    # Wrap single targets in $...$ if not already
                    if not tgt.startswith("$") and not tgt.endswith("$"):
                        tgt = f"${tgt}$"
                    all_pairs.append((spk, tgt))
        print(f"[Real OCW Data] Loaded {len(all_pairs)} verified real lecture samples.")

    # 2. Generate categories
    print(f"[Generating] {n_mixed} mixed technical lecture sentences...")
    for _ in range(n_mixed):
        all_pairs.append(gen_mixed_lecture_sample())

    print(f"[Generating] {n_ops} complex spoken operator expressions...")
    for _ in range(n_ops):
        all_pairs.append(gen_operator_sample())

    print(f"[Generating] {n_disambig} signal and variable disambiguation pairs...")
    for _ in range(n_disambig):
        all_pairs.append(gen_disambiguation_sample())

    print(f"[Generating] {n_neg} non-math pure English negative anchors...")
    for _ in range(n_neg):
        all_pairs.append(gen_negative_sample())

    rng.shuffle(all_pairs)
    final_pairs = all_pairs[:TOTAL_SAMPLES]

    # Split into train / val
    n_train = int(len(final_pairs) * TRAIN_RATIO)
    train_set = final_pairs[:n_train]
    val_set = final_pairs[n_train:]

    print(f"\n[Writing] {len(train_set)} train samples to {TRAIN_FILE}...")
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for spk, tgt in train_set:
            f.write(json.dumps(format_chatml(spk, tgt), ensure_ascii=False) + "\n")

    print(f"[Writing] {len(val_set)} validation samples to {VAL_FILE}...")
    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for spk, tgt in val_set:
            f.write(json.dumps(format_chatml(spk, tgt), ensure_ascii=False) + "\n")

    print("\n" + "=" * 65)
    print(f"Dataset Generation Complete!")
    print(f"Train File:      {TRAIN_FILE} ({TRAIN_FILE.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Validation File: {VAL_FILE} ({VAL_FILE.stat().st_size / 1024 / 1024:.2f} MB)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
