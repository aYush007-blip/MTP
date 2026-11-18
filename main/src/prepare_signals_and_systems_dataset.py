#!/usr/bin/env python3
"""
High-Diversity, Zero-Duplication Signals & Systems Dataset Generator
for Qwen2.5-1.5B-Instruct LoRA/QLoRA Fine-Tuning.

Guarantees enforced on every run:
  1. 100% unique user prompts (zero duplicated examples).
  2. 0% train/validation leakage (strictly disjoint splits; MathSpeech official
     train/val splits respected).
  3. Balanced inline LaTeX ($...$) on every single target (verified programmatically).
  4. Both single-sentence equations AND multi-sentence lecture derivations (300-850 chars)
     matching live Whisper chunk lengths.
  5. Inference-aligned SYSTEM_PROMPT matching headless_agent/src/slm_translator.py.
  6. Automatic gzip packaging into /mnt/thesis/training/data/{train,val}.jsonl.gz.
"""

import gzip
import json
from pathlib import Path
import random
import re
import shutil
from typing import Callable, Dict, List, Set, Tuple

SEED = 42
rng = random.Random(SEED)

DATA_DIR = Path("/mnt/thesis/datasets/Qwen_Signals_And_Systems_FineTune")
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"

TRAINING_DATA_DIR = Path("/mnt/thesis/training/data")
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

HARVESTED_DIR = Path("/mnt/thesis/datasets/nptel_signals_and_systems/raw_transcripts")
MATHSPEECH_TRAIN = Path("/mnt/thesis/datasets/MathSpeech/splits_v2/train.jsonl")
MATHSPEECH_VAL = Path("/mnt/thesis/datasets/MathSpeech/splits_v2/validation.jsonl")

# Concise, inference-aligned system prompt (matches slm_translator.py)
SYSTEM_PROMPT = (
    "You are an expert Electrical Engineering lecture transcriber and mathematical typesetter "
    "specializing in Signals and Systems.\n"
    "Convert spoken mathematical formulas, signals, transforms, variables, and equations into "
    "clean inline LaTeX ($...$) while preserving all surrounding English lecture prose verbatim.\n"
    "Rules:\n"
    "- Continuous-time signals use parentheses: $x(t)$, $y(t)$, $h(t)$, $u(t)$, $\\delta(t)$, $x_2(t)$.\n"
    "- Discrete-time sequences use brackets: $x[n]$, $y[n]$, $h[n]$, $u[n]$, $\\delta[n]$, $x[2n]$, $x[n-k]$.\n"
    "- Transforms use uppercase: $X(s)$, $H(s)$, $X(z)$, $H(z)$, $X(j\\omega)$, $X(e^{j\\Omega})$, $X[k]$.\n"
    "- Use ONLY inline math ($...$). Never wrap ordinary conversational English in LaTeX.\n"
    "- Output ONLY the typeset lecture transcript with no labels or commentary."
)

# ---------------------------------------------------------------------------
# Spoken Phrasing Helpers (Realistic Engineering / NPTEL / Whisper Variation)
# ---------------------------------------------------------------------------

EQ_PHRASES = ["equals", "is equal to", "equal to", "is given by", "="]
DIV_PHRASES = ["divided by", "over", "by", "upon"]
EXP_PHRASES = ["e to the power", "e power", "e to the", "e raised to", "exponential of"]
OPENERS = [
    "",
    "Now ",
    "Notice that ",
    "Recall that ",
    "Observe that ",
    "In this case, ",
    "Therefore, ",
    "From this, ",
    "For our analysis, ",
    "Consider that ",
    "As we saw earlier, ",
    "Let us note that ",
    "In general, ",
    "Specifically, ",
    "Here, ",
]
SIG_CT = ["x", "y", "h", "g", "r", "v", "s", "f"]
SIG_DT = ["x", "y", "h", "g", "w", "v"]
SUBS = ["0", "1", "2", "3", "k", "m"]
POS_INTS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12"]
CONST_SYMS = [
    ("a", "a"),
    ("b", "b"),
    ("alpha", "\\alpha"),
    ("beta", "\\beta"),
    ("gamma", "\\gamma"),
    ("sigma", "\\sigma"),
    ("2", "2"),
    ("3", "3"),
    ("4", "4"),
    ("5", "5"),
    ("6", "6"),
    ("8", "8"),
]
W0_SPOKEN = ["omega 0", "omega zero", "omega naught", "ω0", "omega_0"]
BIG_W_SPOKEN = ["capital omega", "big omega", "Omega", "uppercase omega"]
NUM_WORDS = {
    "1": ["1", "one"],
    "2": ["2", "two"],
    "3": ["3", "three"],
    "4": ["4", "four"],
    "5": ["5", "five"],
    "6": ["6", "six"],
    "7": ["7", "seven"],
    "8": ["8", "eight"],
    "9": ["9", "nine"],
    "10": ["10", "ten"],
    "12": ["12", "twelve"],
}


def spk_num(n: str) -> str:
    return rng.choice(NUM_WORDS.get(n, [n]))


def spk_ct(sig: str, var: str = "t") -> str:
    return rng.choice([f"{sig} of {var}", f"{sig}({var})", f"{sig} {var}"])


def spk_dt(sig: str, var: str = "n") -> str:
    return rng.choice([f"{sig} of {var}", f"{sig}[{var}]", f"{sig}({var})"])


def spk_sub_ct(sig: str, sub: str, var: str = "t") -> str:
    return rng.choice([
        f"{sig} {sub} of {var}",
        f"{sig}{sub} of {var}",
        f"{sig}{sub}({var})",
        f"{sig} sub {sub} of {var}",
    ])


def spk_sub_dt(sig: str, sub: str, var: str = "n") -> str:
    return rng.choice([
        f"{sig} {sub} of {var}",
        f"{sig}{sub} of {var}",
        f"{sig} sub {sub} of {var}",
    ])


def cap_first(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    return s[0].upper() + s[1:]


def with_opener(spoken_body: str, target_body: str) -> Tuple[str, str]:
    op = rng.choice(OPENERS)
    if not op:
        return cap_first(spoken_body), cap_first(target_body)
    return op + spoken_body, op + target_body


# ---------------------------------------------------------------------------
# Category 1: Signal Basics, Operations, Disambiguation (Continuous & Discrete)
# ---------------------------------------------------------------------------

def gen_signal_basics() -> Tuple[str, str]:
    kind = rng.randint(0, 13)
    eq = rng.choice(EQ_PHRASES)
    div = rng.choice(DIV_PHRASES)
    exp = rng.choice(EXP_PHRASES)
    w0 = rng.choice(W0_SPOKEN)

    if kind == 0:
        sig = rng.choice(SIG_CT)
        sub = rng.choice(SUBS)
        trig_spk, trig_tex = rng.choice([("cos", "\\cos"), ("cosine", "\\cos"), ("sin", "\\sin"), ("sine", "\\sin")])
        amp = rng.choice(["", "2", "3", "4", "5", "A", "B"])
        amp_spk = f"{amp} " if amp else ""
        amp_tex = f"{amp} " if amp else ""
        has_t = rng.choice([True, False])
        t_spk = " t" if has_t else ""
        spk = f"the signal {spk_sub_ct(sig, sub)} {eq} {amp_spk}{trig_spk} {w0}{t_spk} for all values of t."
        tgt = f"the signal ${sig}_{sub}(t) = {amp_tex}{trig_tex}(\\omega_0 t)$ for all values of $t$."
        return with_opener(spk, tgt)

    elif kind == 1:
        sig = rng.choice(SIG_CT)
        a = rng.choice(["2", "3", "4", "5", "1/2", "1/3"])
        b = rng.choice(["1", "2", "3", "4", "5", "t 0"])
        sign_spk, sign_tex = rng.choice([("minus", "-"), ("plus", "+")])
        b_tex = "t_0" if b == "t 0" else b
        a_spk = "half" if a == "1/2" else ("one third" if a == "1/3" else a)
        a_tex = "\\frac{1}{2}" if a == "1/2" else ("\\frac{1}{3}" if a == "1/3" else a)
        y = rng.choice([s for s in SIG_CT if s != sig])
        spk = f"after time scaling and shifting we obtain {spk_ct(y)} {eq} {sig} of {a_spk} t {sign_spk} {b}."
        tgt = f"after time scaling and shifting, we obtain ${y}(t) = {sig}({a_tex}t {sign_tex} {b_tex})$."
        return with_opener(spk, tgt)

    elif kind == 2:
        sig = rng.choice(SIG_DT)
        y = rng.choice([s for s in SIG_DT if s != sig])
        m = rng.choice(["2", "3", "4", "M", "K"])
        shift = rng.choice(["0", "1", "2", "3", "k", "n 0"])
        if shift == "0":
            spk = f"downsampling the discrete sequence produces {spk_dt(y)} {eq} {sig} of {m} n."
            tgt = f"downsampling the discrete sequence produces ${y}[n] = {sig}[{m}n]$."
        else:
            sign_spk, sign_tex = rng.choice([("minus", "-"), ("plus", "+")])
            sh_tex = "n_0" if shift == "n 0" else shift
            spk = f"the shifted and scaled discrete sequence is {spk_dt(y)} {eq} {sig} of {m} n {sign_spk} {shift}."
            tgt = f"the shifted and scaled discrete sequence is ${y}[n] = {sig}[{m}n {sign_tex} {sh_tex}]$."
        return with_opener(spk, tgt)

    elif kind == 3:
        var = rng.choice(["e", "t", "z", "s"])
        k = rng.choice(["1", "2", "3", "k", "m"])
        sign_spk, sign_tex = rng.choice([("minus", "-"), ("plus", "+")])
        if rng.random() < 0.5:
            spk = f"the general power term is {var} to the power n {sign_spk} {k}."
            tgt = f"the general power term is ${var}^{{n {sign_tex} {k}}}$."
        else:
            spk = f"the normalized term is t to the power n {sign_spk} {k} {div} n {sign_spk} {k} factorial."
            tgt = f"the normalized term is $\\frac{{t^{{n {sign_tex} {k}}}}}{{(n {sign_tex} {k})!}}$."
        return with_opener(spk, tgt)

    elif kind == 4:
        sig = rng.choice(SIG_CT)
        a_spk, a_tex = rng.choice(CONST_SYMS)
        sign_spk, sign_tex = rng.choice([("minus", "-"), ("", "")])
        causal = rng.choice([True, False])
        if causal:
            spk = f"the causal exponential signal {spk_ct(sig)} {eq} {exp} {sign_spk} {a_spk} t times u of t."
            tgt = f"the causal exponential signal ${sig}(t) = e^{{{sign_tex}{a_tex}t}} u(t)$."
        else:
            spk = f"the two-sided exponential signal {spk_ct(sig)} {eq} {exp} minus {a_spk} absolute value of t."
            tgt = f"the two-sided exponential signal ${sig}(t) = e^{{-{a_tex}|t|}}$."
        return with_opener(re.sub(r"\s+", " ", spk), tgt)

    elif kind == 5:
        sig = rng.choice(SIG_CT)
        sign_spk, sign_tex = rng.choice([("", ""), ("minus ", "-")])
        mult = rng.choice(["", "2 ", "3 ", "k "])
        has_t = rng.choice([True, False])
        t_spk = " t" if has_t else ""
        t_tex = "t" if has_t else ""
        spk = f"the complex exponential {spk_ct(sig)} {eq} {exp} {sign_spk}j {mult}{w0}{t_spk} has fundamental period T 0 {eq} 2 pi {div} {w0}."
        tgt = f"the complex exponential ${sig}(t) = e^{{{sign_tex}j{mult.strip()}\\omega_0 {t_tex}}}$ has fundamental period $T_0 = \\frac{{2\\pi}}{{\\omega_0}}$."
        tgt = tgt.replace(" }", "}")
        return with_opener(re.sub(r"\s+", " ", spk), tgt)

    elif kind == 6:
        sig = rng.choice(SIG_CT)
        t0 = rng.choice(["0", "1", "2", "3", "t 0", "tau"])
        t0_tex = "t_0" if t0 == "t 0" else ("\\tau" if t0 == "tau" else t0)
        if t0 == "0":
            spk = f"by the sifting property of the Dirac impulse, the integral from minus infinity to infinity of {spk_ct(sig)} delta of t dt {eq} {sig} of 0."
            tgt = f"by the sifting property of the Dirac impulse, $\\int_{{-\\infty}}^{{\\infty}} {sig}(t) \\delta(t) \\, dt = {sig}(0)$."
        else:
            spk = f"by the sifting property, the integral from minus infinity to infinity of {spk_ct(sig)} delta of t minus {t0} dt {eq} {sig} of {t0}."
            tgt = f"by the sifting property, $\\int_{{-\\infty}}^{{\\infty}} {sig}(t) \\delta(t - {t0_tex}) \\, dt = {sig}({t0_tex})$."
        return with_opener(spk, tgt)

    elif kind == 7:
        sig = rng.choice(SIG_DT)
        k = rng.choice(["0", "1", "2", "3", "k", "n 0"])
        k_tex = "n_0" if k == "n 0" else k
        if k == "0":
            spk = f"the discrete unit impulse delta of n {eq} 1 for n {eq} 0 and 0 for n not equal to 0, so summing {spk_dt(sig)} delta of n gives {sig} of 0."
            tgt = f"the discrete unit impulse $\\delta[n] = 1$ for $n = 0$ and $0$ for $n \\neq 0$, so summing ${sig}[n]\\delta[n]$ gives ${sig}[0]$."
        else:
            spk = f"the shifted discrete impulse delta of n minus {k} picks out the sample {sig} of {k} in the sum over all n."
            tgt = f"the shifted discrete impulse $\\delta[n - {k_tex}]$ picks out the sample ${sig}[{k_tex}]$ in the sum over all $n$."
        return with_opener(spk, tgt)

    elif kind == 8:
        sig = rng.choice(SIG_CT)
        spk = f"the even and odd components of {spk_ct(sig)} are {sig} e of t {eq} 1 {div} 2 open parenthesis {spk_ct(sig)} plus {sig} of minus t close parenthesis and {sig} o of t {eq} 1 {div} 2 open parenthesis {spk_ct(sig)} minus {sig} of minus t close parenthesis."
        tgt = f"the even and odd components of ${sig}(t)$ are ${sig}_e(t) = \\frac{{1}}{{2}}({sig}(t) + {sig}(-t))$ and ${sig}_o(t) = \\frac{{1}}{{2}}({sig}(t) - {sig}(-t))$."
        return with_opener(spk, tgt)

    elif kind == 9:
        sig = rng.choice(SIG_CT)
        metric = rng.choice(["energy", "power"])
        if metric == "energy":
            spk = f"the total energy E {sig} of the continuous signal {spk_ct(sig)} {eq} integral from minus infinity to infinity of magnitude {spk_ct(sig)} squared dt."
            tgt = f"the total energy $E_{sig}$ of the continuous signal ${sig}(t)$ is $E_{sig} = \\int_{{-\\infty}}^{{\\infty}} |{sig}(t)|^2 \\, dt$."
        else:
            spk = f"the average power P {sig} of {spk_ct(sig)} {eq} limit as T goes to infinity of 1 {div} 2 T integral from minus T to T of magnitude {spk_ct(sig)} squared dt."
            tgt = f"the average power $P_{sig}$ of ${sig}(t)$ is $P_{sig} = \\lim_{{T \\to \\infty}} \\frac{{1}}{{2T}} \\int_{{-T}}^{{T}} |{sig}(t)|^2 \\, dt$."
        return with_opener(spk, tgt)

    elif kind == 10:
        sig = rng.choice(SIG_DT)
        metric = rng.choice(["energy", "power"])
        if metric == "energy":
            spk = f"for the discrete sequence {spk_dt(sig)}, the total energy E {sig} {eq} sum from n equals minus infinity to infinity of magnitude {spk_dt(sig)} squared."
            tgt = f"for the discrete sequence ${sig}[n]$, the total energy $E_{sig} = \\sum_{{n=-\\infty}}^{{\\infty}} |{sig}[n]|^2$."
        else:
            spk = f"the average power of discrete signal {spk_dt(sig)} is P {sig} {eq} limit as N tends to infinity of 1 {div} 2 N plus 1 sum from n equals minus N to N of magnitude {spk_dt(sig)} squared."
            tgt = f"the average power of discrete signal ${sig}[n]$ is $P_{sig} = \\lim_{{N \\to \\infty}} \\frac{{1}}{{2N+1}} \\sum_{{n=-N}}^{{N}} |{sig}[n]|^2$."
        return with_opener(spk, tgt)

    elif kind == 11:
        sig = rng.choice(SIG_DT)
        base_spk, base_tex = rng.choice([("half", "\\frac{1}{2}"), ("one third", "\\frac{1}{3}"), ("one fourth", "\\frac{1}{4}"), ("0.5", "0.5"), ("0.8", "0.8"), ("a", "a"), ("alpha", "\\alpha")])
        spk = f"the geometric sequence {spk_dt(sig)} {eq} {base_spk} to the power n times u of n is an energy signal when magnitude of {base_spk} is less than 1."
        tgt = f"the geometric sequence ${sig}[n] = \\left({base_tex}\\right)^n u[n]$ is an energy signal when $|{base_tex}| < 1$."
        return with_opener(spk, tgt)

    elif kind == 12:
        sig = rng.choice(SIG_CT)
        t1 = rng.choice(["1", "2", "3", "T", "T 0"])
        t1_tex = "T_0" if t1 == "T 0" else t1
        spk = f"we can express the rectangular pulse {spk_ct(sig)} {eq} u of t minus u of t minus {t1}, which equals 1 for 0 less than t less than {t1}."
        tgt = f"we can express the rectangular pulse ${sig}(t) = u(t) - u(t - {t1_tex})$, which equals $1$ for $0 < t < {t1_tex}$."
        return with_opener(spk, tgt)

    else:
        sig = rng.choice(SIG_CT)
        a_spk, a_tex = rng.choice(CONST_SYMS)
        spk = f"the unit ramp signal r of t {eq} t times u of t, and its shifted version scaled by {a_spk} is {a_spk} times t minus 1 times u of t minus 1."
        tgt = f"the unit ramp signal $r(t) = t u(t)$, and its shifted version scaled by ${a_tex}$ is ${a_tex}(t-1)u(t-1)$."
        return with_opener(spk, tgt)


# ---------------------------------------------------------------------------
# Category 2: LTI Systems, Convolution, Causality, Stability & Diff Eqs
# ---------------------------------------------------------------------------

def gen_lti_convolution() -> Tuple[str, str]:
    kind = rng.randint(0, 7)
    eq = rng.choice(EQ_PHRASES)
    x = rng.choice(["x", "f", "u_in"])
    if x == "u_in":
        x = "x"
    y = rng.choice(["y", "g", "v"])
    h = rng.choice(["h", "h_1", "h_2"])
    h_spk = h.replace("_", " ")

    if kind == 0:
        spk = f"the output of a continuous-time LTI system is {spk_ct(y)} {eq} {spk_ct(x)} convolved with {h_spk} of t {eq} integral from minus infinity to infinity of {x} of tau {h_spk} of t minus tau d tau."
        tgt = f"the output of a continuous-time LTI system is ${y}(t) = {x}(t) * {h}(t) = \\int_{{-\\infty}}^{{\\infty}} {x}(\\tau) {h}(t - \\tau) \\, d\\tau$."
        return with_opener(spk, tgt)

    elif kind == 1:
        spk = f"for a discrete-time LTI system, the convolution sum gives {spk_dt(y)} {eq} {spk_dt(x)} convolved with {h_spk} of n {eq} sum from k equals minus infinity to infinity of {x} of k {h_spk} of n minus k."
        tgt = f"for a discrete-time LTI system, the convolution sum gives ${y}[n] = {x}[n] * {h}[n] = \\sum_{{k=-\\infty}}^{{\\infty}} {x}[k] {h}[n - k]$."
        return with_opener(spk, tgt)

    elif kind == 2:
        dom = rng.choice(["ct", "dt"])
        if dom == "ct":
            spk = f"a continuous-time LTI system with impulse response {h_spk} of t is causal if and only if {h_spk} of t {eq} 0 for all t less than 0."
            tgt = f"a continuous-time LTI system with impulse response ${h}(t)$ is causal if and only if ${h}(t) = 0$ for all $t < 0$."
        else:
            spk = f"a discrete-time LTI system with impulse response {h_spk} of n is causal if and only if {h_spk} of n {eq} 0 for all n less than 0."
            tgt = f"a discrete-time LTI system with impulse response ${h}[n]$ is causal if and only if ${h}[n] = 0$ for all $n < 0$."
        return with_opener(spk, tgt)

    elif kind == 3:
        dom = rng.choice(["ct", "dt"])
        if dom == "ct":
            spk = f"the continuous LTI system is BIBO stable if and only if its impulse response is absolutely integrable, meaning integral from minus infinity to infinity of absolute {h_spk} of t dt is less than infinity."
            tgt = f"the continuous LTI system is BIBO stable if and only if its impulse response is absolutely integrable, meaning $\\int_{{-\\infty}}^{{\\infty}} |{h}(t)| \\, dt < \\infty$."
        else:
            spk = f"the discrete LTI system is BIBO stable if and only if sum from n equals minus infinity to infinity of magnitude {h_spk} of n is less than infinity."
            tgt = f"the discrete LTI system is BIBO stable if and only if $\\sum_{{n=-\\infty}}^{{\\infty}} |{h}[n]| < \\infty$."
        return with_opener(spk, tgt)

    elif kind == 4:
        a1 = rng.choice(["2", "3", "4", "5", "a"])
        a0 = rng.choice(["1", "2", "4", "6", "9", "b"])
        b0 = rng.choice(["1", "2", "3", "5"])
        spk = f"consider the second-order differential equation d squared {y} by d t squared plus {a1} d {y} by d t plus {a0} {spk_ct(y)} {eq} {b0} {spk_ct(x)}."
        tgt = f"consider the second-order differential equation $\\frac{{d^2 {y}}}{{dt^2}} + {a1} \\frac{{d{y}}}{{dt}} + {a0} {y}(t) = {b0} {x}(t)$."
        return with_opener(spk, tgt)

    elif kind == 5:
        a1 = rng.choice(["0.5", "0.8", "1/2", "3/4", "a"])
        b0 = rng.choice(["1", "2", "3", "b 0"])
        b1 = rng.choice(["1", "2", "0.5", "b 1"])
        a1_spk = "half" if a1 == "1/2" else ("three fourths" if a1 == "3/4" else a1)
        a1_tex = "\\frac{1}{2}" if a1 == "1/2" else ("\\frac{3}{4}" if a1 == "3/4" else a1)
        b0_tex = "b_0" if b0 == "b 0" else b0
        b1_tex = "b_1" if b1 == "b 1" else b1
        spk = f"the first-order difference equation is {spk_dt(y)} minus {a1_spk} {y} of n minus 1 {eq} {b0} {spk_dt(x)} plus {b1} {x} of n minus 1."
        tgt = f"the first-order difference equation is ${y}[n] - {a1_tex} {y}[n-1] = {b0_tex} {x}[n] + {b1_tex} {x}[n-1]$."
        return with_opener(spk, tgt)

    elif kind == 6:
        spk = f"for two LTI systems connected in cascade, the overall impulse response is {h_spk} of t {eq} h 1 of t convolved with h 2 of t, whereas in parallel it is h 1 of t plus h 2 of t."
        tgt = f"for two LTI systems connected in cascade, the overall impulse response is ${h}(t) = h_1(t) * h_2(t)$, whereas in parallel it is $h_1(t) + h_2(t)$."
        return with_opener(spk, tgt)

    else:
        a_spk, a_tex = rng.choice(CONST_SYMS)
        spk = f"the step response s of t is the running integral of the impulse response, so s of t {eq} integral from minus infinity to t of {h_spk} of tau d tau, and {h_spk} of t {eq} d s by d t."
        tgt = f"the step response $s(t)$ is the running integral of the impulse response, so $s(t) = \\int_{{-\\infty}}^{{t}} {h}(\\tau) \\, d\\tau$, and ${h}(t) = \\frac{{ds}}{{dt}}$."
        return with_opener(spk, tgt)


# ---------------------------------------------------------------------------
# Category 3: Laplace Transform, ROC, Poles/Zeros, IVT & FVT
# ---------------------------------------------------------------------------

def gen_laplace() -> Tuple[str, str]:
    kind = rng.randint(0, 11)
    eq = rng.choice(EQ_PHRASES)
    div = rng.choice(DIV_PHRASES)
    exp = rng.choice(EXP_PHRASES)
    w0 = rng.choice(W0_SPOKEN)
    sig = rng.choice(["x", "y", "h", "f", "v"])
    SIG = sig.upper()

    if kind == 0:
        spk = f"the bilateral Laplace transform of {spk_ct(sig)} is {SIG} of s {eq} integral from minus infinity to infinity of {spk_ct(sig)} {exp} minus s t dt, where s {eq} sigma plus j omega."
        tgt = f"the bilateral Laplace transform of ${sig}(t)$ is ${SIG}(s) = \\int_{{-\\infty}}^{{\\infty}} {sig}(t) e^{{-st}} \\, dt$, where $s = \\sigma + j\\omega$."
        return with_opener(spk, tgt)

    elif kind == 1:
        a_spk, a_tex = rng.choice(CONST_SYMS)
        sign_spk, sign_tex, roc_sign = rng.choice([("minus", "-", "-"), ("plus", "+", "")])
        roc_dir_spk, roc_dir_tex = rng.choice([("greater than", ">"), ("less than", "<")])
        if roc_dir_tex == ">":
            spk = f"for the causal signal {spk_ct(sig)} {eq} {exp} {sign_spk} {a_spk} t u of t, the Laplace transform is {SIG} of s {eq} 1 {div} s {'plus' if sign_tex=='-' else 'minus'} {a_spk} with ROC real part of s {roc_dir_spk} {roc_sign}{a_spk}."
            tgt = f"for the causal signal ${sig}(t) = e^{{{sign_tex}{a_tex}t}} u(t)$, the Laplace transform is ${SIG}(s) = \\frac{{1}}{{s {'+' if sign_tex=='-' else '-'} {a_tex}}}$ with ROC $\\text{{Re}}(s) {roc_dir_tex} {roc_sign}{a_tex}$."
        else:
            spk = f"for the anti-causal signal {spk_ct(sig)} {eq} minus {exp} minus {a_spk} t u of minus t, {SIG} of s {eq} 1 {div} s plus {a_spk} with ROC real part of s less than minus {a_spk}."
            tgt = f"for the anti-causal signal ${sig}(t) = -e^{{-{a_tex}t}} u(-t)$, ${SIG}(s) = \\frac{{1}}{{s + {a_tex}}}$ with ROC $\\text{{Re}}(s) < -{a_tex}$."
        return with_opener(spk, tgt)

    elif kind == 2:
        spk = f"by the initial value theorem, if {spk_ct(sig)} is causal with no impulses at the origin, then {sig} of 0 plus {eq} limit as s goes to infinity of s times {SIG} of s."
        tgt = f"by the initial value theorem, if ${sig}(t)$ is causal with no impulses at the origin, then ${sig}(0^+) = \\lim_{{s \\to \\infty}} s {SIG}(s)$."
        return with_opener(spk, tgt)

    elif kind == 3:
        spk = f"by the final value theorem, if all poles of s {SIG} of s lie strictly in the left half plane, then limit as t goes to infinity of {spk_ct(sig)} {eq} limit as s goes to 0 of s times {SIG} of s."
        tgt = f"by the final value theorem, if all poles of $s {SIG}(s)$ lie strictly in the left half-plane, then $\\lim_{{t \\to \\infty}} {sig}(t) = \\lim_{{s \\to 0}} s {SIG}(s)$."
        return with_opener(spk, tgt)

    elif kind == 4:
        p1, p2 = rng.sample(["1", "2", "3", "4", "5", "6", "7", "8"], 2)
        A = rng.choice(["1", "2", "3", "4", "5", "A"])
        B = rng.choice(["1", "2", "3", "4", "5", "B"])
        op_spk, op_tex = rng.choice([("plus", "+"), ("minus", "-")])
        spk = f"applying partial fraction expansion yields {SIG} of s {eq} {A} {div} s plus {p1} {op_spk} {B} {div} s plus {p2}."
        tgt = f"applying partial fraction expansion yields ${SIG}(s) = \\frac{{{A}}}{{s + {p1}}} {op_tex} \\frac{{{B}}}{{s + {p2}}}$."
        return with_opener(spk, tgt)

    elif kind == 5:
        trig_spk, trig_tex, num_spk, num_tex = rng.choice([
            ("cos", "\\cos", "s", "s"),
            ("sin", "\\sin", w0, "\\omega_0"),
        ])
        a_spk, a_tex = rng.choice([("", ""), ("a", "a"), ("2", "2"), ("3", "3"), ("4", "4"), ("alpha", "\\alpha")])
        if not a_tex:
            spk = f"the Laplace transform of {trig_spk} {w0} t u of t {eq} {num_spk} {div} s squared plus {w0} squared with ROC real part of s greater than 0."
            tgt = f"the Laplace transform of ${trig_tex}(\\omega_0 t) u(t)$ is $\\frac{{{num_tex}}}{{s^2 + \\omega_0^2}}$ with ROC $\\text{{Re}}(s) > 0$."
        else:
            num_s_spk = f"s plus {a_spk}" if num_tex == "s" else num_spk
            num_s_tex = f"s + {a_tex}" if num_tex == "s" else num_tex
            spk = f"the Laplace transform of {exp} minus {a_spk} t {trig_spk} {w0} t u of t {eq} {num_s_spk} {div} open parenthesis s plus {a_spk} close parenthesis squared plus {w0} squared with ROC real part of s greater than minus {a_spk}."
            tgt = f"the Laplace transform of $e^{{-{a_tex}t}} {trig_tex}(\\omega_0 t) u(t)$ is $\\frac{{{num_s_tex}}}{{(s + {a_tex})^2 + \\omega_0^2}}$ with ROC $\\text{{Re}}(s) > -{a_tex}$."
        return with_opener(spk, tgt)

    elif kind == 6:
        n = rng.choice(["2", "3", "4", "n"])
        a_spk, a_tex = rng.choice(CONST_SYMS)
        if n == "n":
            spk = f"the Laplace transform of t to the power n minus 1 {exp} minus {a_spk} t u of t {div} n minus 1 factorial {eq} 1 {div} s plus {a_spk} to the power n."
            tgt = f"the Laplace transform of $\\frac{{t^{{n-1}} e^{{-{a_tex}t}} u(t)}}{{(n-1)!}}$ is $\\frac{{1}}{{(s + {a_tex})^n}}$."
        else:
            p = int(n) - 1
            fact = {1: "1", 2: "2", 3: "6"}[p]
            spk = f"the Laplace transform of t to the power {p} {exp} minus {a_spk} t u of t {eq} {fact} {div} open parenthesis s plus {a_spk} close parenthesis to the power {n}."
            tgt = f"the Laplace transform of $t^{p} e^{{-{a_tex}t}} u(t)$ is $\\frac{{{fact}}}{{(s + {a_tex})^{n}}}$."
        return with_opener(spk, tgt)

    elif kind == 7:
        z1 = rng.choice(["1", "2", "3", "4", "5"])
        p1, p2 = rng.sample(["2", "3", "4", "5", "6", "8"], 2)
        K = rng.choice(["1", "2", "4", "5", "10"])
        spk = f"the system transfer function is H of s {eq} Y of s {div} X of s {eq} {K} times s plus {z1} {div} open parenthesis s plus {p1} close parenthesis open parenthesis s plus {p2} close parenthesis."
        tgt = f"the system transfer function is $H(s) = \\frac{{Y(s)}}{{X(s)}} = \\frac{{{K}(s + {z1})}}{{(s + {p1})(s + {p2})}}$."
        return with_opener(spk, tgt)

    elif kind == 8:
        t0 = rng.choice(["1", "2", "3", "4", "t 0", "T"])
        t0_tex = "t_0" if t0 == "t 0" else t0
        spk = f"by the time-shifting property of the Laplace transform, {sig} of t minus {t0} transforms to {exp} minus s {t0} times {SIG} of s."
        tgt = f"by the time-shifting property of the Laplace transform, ${sig}(t - {t0_tex})$ transforms to $e^{{-s{t0_tex}}} {SIG}(s)$."
        return with_opener(spk, tgt)

    elif kind == 9:
        order = rng.choice([1, 2])
        if order == 1:
            spk = f"taking the unilateral Laplace transform of d {sig} by d t yields s {SIG} of s minus {sig} of 0 minus."
            tgt = f"taking the unilateral Laplace transform of $\\frac{{d{sig}}}{{dt}}$ yields $s {SIG}(s) - {sig}(0^-)$."
        else:
            spk = f"taking the unilateral Laplace transform of d squared {sig} by d t squared gives s squared {SIG} of s minus s {sig} of 0 minus minus {sig} prime of 0 minus."
            tgt = f"taking the unilateral Laplace transform of $\\frac{{d^2 {sig}}}{{dt^2}}$ gives $s^2 {SIG}(s) - s {sig}(0^-) - {sig}'(0^-)$."
        return with_opener(spk, tgt)

    elif kind == 10:
        p1, p2 = sorted(rng.sample([1, 2, 3, 4, 5, 6, 7], 2))
        spk = f"for a two-sided signal with poles at s {eq} minus {p2} and s {eq} minus {p1}, the region of convergence is the vertical strip minus {p2} less than real part of s less than minus {p1}."
        tgt = f"for a two-sided signal with poles at $s = -{p2}$ and $s = -{p1}$, the region of convergence is the vertical strip $-{p2} < \\text{{Re}}(s) < -{p1}$."
        return with_opener(spk, tgt)

    else:
        spk = f"a causal LTI system with rational transfer function H of s is BIBO stable if and only if the ROC includes the j omega axis, meaning all poles satisfy real part of p k less than 0."
        tgt = f"a causal LTI system with rational transfer function $H(s)$ is BIBO stable if and only if the ROC includes the $j\\omega$-axis, meaning all poles satisfy $\\text{{Re}}(p_k) < 0$."
        return with_opener(spk, tgt)


# ---------------------------------------------------------------------------
# Category 4: Z-Transform & Discrete-Time LTI Systems
# ---------------------------------------------------------------------------

def gen_ztransform() -> Tuple[str, str]:
    kind = rng.randint(0, 8)
    eq = rng.choice(EQ_PHRASES)
    div = rng.choice(DIV_PHRASES)
    sig = rng.choice(SIG_DT)
    SIG = sig.upper()
    bw = rng.choice(BIG_W_SPOKEN)

    if kind == 0:
        spk = f"the bilateral Z-transform of {spk_dt(sig)} is {SIG} of z {eq} sum from n equals minus infinity to infinity of {spk_dt(sig)} z to the power minus n, where z {eq} r e to the j {bw}."
        tgt = f"the bilateral $Z$-transform of ${sig}[n]$ is ${SIG}(z) = \\sum_{{n=-\\infty}}^{{\\infty}} {sig}[n] z^{{-n}}$, where $z = r e^{{j\\Omega}}$."
        return with_opener(spk, tgt)

    elif kind == 1:
        a_spk, a_tex = rng.choice([("a", "a"), ("alpha", "\\alpha"), ("0.5", "0.5"), ("0.7", "0.7"), ("0.9", "0.9"), ("1/2", "\\frac{1}{2}"), ("1/3", "\\frac{1}{3}"), ("2", "2"), ("3", "3")])
        causal = rng.choice([True, False])
        if causal:
            spk = f"for the right-sided sequence {spk_dt(sig)} {eq} {a_spk} to the power n u of n, {SIG} of z {eq} 1 {div} 1 minus {a_spk} z to the minus 1 with ROC magnitude of z greater than absolute value of {a_spk}."
            tgt = f"for the right-sided sequence ${sig}[n] = {a_tex}^n u[n]$, ${SIG}(z) = \\frac{{1}}{{1 - {a_tex} z^{{-1}}}}$ with ROC $|z| > |{a_tex}|$."
        else:
            spk = f"for the left-sided sequence {spk_dt(sig)} {eq} minus {a_spk} to the power n u of minus n minus 1, {SIG} of z {eq} 1 {div} 1 minus {a_spk} z to the minus 1 with ROC magnitude of z less than absolute value of {a_spk}."
            tgt = f"for the left-sided sequence ${sig}[n] = -{a_tex}^n u[-n-1]$, ${SIG}(z) = \\frac{{1}}{{1 - {a_tex} z^{{-1}}}}$ with ROC $|z| < |{a_tex}|$."
        return with_opener(spk, tgt)

    elif kind == 2:
        r1, r2 = rng.choice([("0.5", "2"), ("0.25", "3"), ("1/3", "2"), ("1/2", "4"), ("r 1", "r 2"), ("a", "b")])
        r1_tex = "r_1" if r1 == "r 1" else ("\\frac{1}{3}" if r1 == "1/3" else ("\\frac{1}{2}" if r1 == "1/2" else r1))
        r2_tex = "r_2" if r2 == "r 2" else r2
        spk = f"for a two-sided discrete sequence with poles at {r1} and {r2}, the ROC is the annular ring {r1} less than magnitude of z less than {r2}."
        tgt = f"for a two-sided discrete sequence with poles at ${r1_tex}$ and ${r2_tex}$, the ROC is the annular ring ${r1_tex} < |z| < {r2_tex}$."
        return with_opener(spk, tgt)

    elif kind == 3:
        spk = f"by the discrete initial value theorem, for a causal sequence {spk_dt(sig)}, {sig} of 0 {eq} limit as z goes to infinity of {SIG} of z."
        tgt = f"by the discrete initial value theorem, for a causal sequence ${sig}[n]$, ${sig}[0] = \\lim_{{z \\to \\infty}} {SIG}(z)$."
        return with_opener(spk, tgt)

    elif kind == 4:
        spk = f"by the discrete final value theorem, limit as n tends to infinity of {spk_dt(sig)} {eq} limit as z tends to 1 of open parenthesis 1 minus z to the minus 1 close parenthesis times {SIG} of z."
        tgt = f"by the discrete final value theorem, $\\lim_{{n \\to \\infty}} {sig}[n] = \\lim_{{z \\to 1}} (1 - z^{{-1}}) {SIG}(z)$."
        return with_opener(spk, tgt)

    elif kind == 5:
        k = rng.choice(["1", "2", "3", "4", "k", "n 0"])
        k_tex = "n_0" if k == "n 0" else k
        spk = f"by the time-shifting property of the Z-transform, {sig} of n minus {k} corresponds to z to the power minus {k} times {SIG} of z."
        tgt = f"by the time-shifting property of the $Z$-transform, ${sig}[n - {k_tex}]$ corresponds to $z^{{-{k_tex}}} {SIG}(z)$."
        return with_opener(spk, tgt)

    elif kind == 6:
        a = rng.choice(["0.5", "0.8", "0.9", "a", "alpha"])
        a_tex = "\\alpha" if a == "alpha" else a
        spk = f"the Z-transform of n times {a} to the power n u of n is {a} z to the minus 1 {div} open parenthesis 1 minus {a} z to the minus 1 close parenthesis squared for magnitude of z greater than absolute {a}."
        tgt = f"the $Z$-transform of $n {a_tex}^n u[n]$ is $\\frac{{{a_tex} z^{{-1}}}}{{(1 - {a_tex} z^{{-1}})^2}}$ for $|z| > |{a_tex}|$."
        return with_opener(spk, tgt)

    elif kind == 7:
        b0 = rng.choice(["1", "2", "3"])
        b1 = rng.choice(["0.5", "1", "2", "3"])
        a1 = rng.choice(["0.5", "0.6", "0.8", "0.9"])
        a2 = rng.choice(["0.1", "0.2", "0.25", "0.4"])
        spk = f"the discrete transfer function is H of z {eq} Y of z {div} X of z {eq} open parenthesis {b0} plus {b1} z to the minus 1 close parenthesis {div} open parenthesis 1 minus {a1} z to the minus 1 plus {a2} z to the minus 2 close parenthesis."
        tgt = f"the discrete transfer function is $H(z) = \\frac{{Y(z)}}{{X(z)}} = \\frac{{{b0} + {b1} z^{{-1}}}}{{1 - {a1} z^{{-1}} + {a2} z^{{-2}}}}$."
        return with_opener(spk, tgt)

    else:
        spk = f"a causal discrete-time LTI system with rational system function H of z is BIBO stable if and only if its ROC includes the unit circle magnitude of z {eq} 1, so all poles satisfy magnitude of p k less than 1."
        tgt = f"a causal discrete-time LTI system with rational system function $H(z)$ is BIBO stable if and only if its ROC includes the unit circle $|z| = 1$, so all poles satisfy $|p_k| < 1$."
        return with_opener(spk, tgt)


# ---------------------------------------------------------------------------
# Category 5: Fourier Series & Transforms (CTFS, DTFS, CTFT, DTFT, DFT)
# ---------------------------------------------------------------------------

def gen_fourier() -> Tuple[str, str]:
    kind = rng.randint(0, 10)
    eq = rng.choice(EQ_PHRASES)
    div = rng.choice(DIV_PHRASES)
    exp = rng.choice(EXP_PHRASES)
    w0 = rng.choice(W0_SPOKEN)
    bw = rng.choice(BIG_W_SPOKEN)
    sig = rng.choice(["x", "y", "h", "g", "f"])
    SIG = sig.upper()

    if kind == 0:
        coeff = rng.choice(["a", "c", "X"])
        spk = f"the continuous-time Fourier series synthesis equation writes {spk_ct(sig)} {eq} sum from k equals minus infinity to infinity of {coeff} k {exp} j k {w0} t."
        tgt = f"the continuous-time Fourier series synthesis equation writes ${sig}(t) = \\sum_{{k=-\\infty}}^{{\\infty}} {coeff}_k e^{{jk\\omega_0 t}}$."
        return with_opener(spk, tgt)

    elif kind == 1:
        coeff = rng.choice(["a", "c"])
        spk = f"the Fourier series analysis equation gives {coeff} k {eq} 1 {div} T 0 integral over T 0 of {spk_ct(sig)} {exp} minus j k {w0} t dt."
        tgt = f"the Fourier series analysis equation gives ${coeff}_k = \\frac{{1}}{{T_0}} \\int_{{T_0}} {sig}(t) e^{{-jk\\omega_0 t}} \\, dt$."
        return with_opener(spk, tgt)

    elif kind == 2:
        coeff = rng.choice(["a", "c"])
        spk = f"by Parseval's theorem for periodic signals, 1 {div} T 0 integral over T 0 of magnitude {spk_ct(sig)} squared dt {eq} sum from k equals minus infinity to infinity of magnitude {coeff} k squared."
        tgt = f"by Parseval's theorem for periodic signals, $\\frac{{1}}{{T_0}} \\int_{{T_0}} |{sig}(t)|^2 \\, dt = \\sum_{{k=-\\infty}}^{{\\infty}} |{coeff}_k|^2$."
        return with_opener(spk, tgt)

    elif kind == 3:
        spk = f"the continuous-time Fourier transform of {spk_ct(sig)} is {SIG} of j omega {eq} integral from minus infinity to infinity of {spk_ct(sig)} {exp} minus j omega t dt."
        tgt = f"the continuous-time Fourier transform of ${sig}(t)$ is ${SIG}(j\\omega) = \\int_{{-\\infty}}^{{\\infty}} {sig}(t) e^{{-j\\omega t}} \\, dt$."
        return with_opener(spk, tgt)

    elif kind == 4:
        spk = f"the inverse CTFT recovers {spk_ct(sig)} {eq} 1 {div} 2 pi integral from minus infinity to infinity of {SIG} of j omega {exp} j omega t d omega."
        tgt = f"the inverse CTFT recovers ${sig}(t) = \\frac{{1}}{{2\\pi}} \\int_{{-\\infty}}^{{\\infty}} {SIG}(j\\omega) e^{{j\\omega t}} \\, d\\omega$."
        return with_opener(spk, tgt)

    elif kind == 5:
        spk = f"by Parseval's energy relation for the CTFT, integral from minus infinity to infinity of magnitude {spk_ct(sig)} squared dt {eq} 1 {div} 2 pi integral from minus infinity to infinity of magnitude {SIG} of j omega squared d omega."
        tgt = f"by Parseval's energy relation for the CTFT, $\\int_{{-\\infty}}^{{\\infty}} |{sig}(t)|^2 \\, dt = \\frac{{1}}{{2\\pi}} \\int_{{-\\infty}}^{{\\infty}} |{SIG}(j\\omega)|^2 \\, d\\omega$."
        return with_opener(spk, tgt)

    elif kind == 6:
        y = rng.choice([c for c in ["y", "g", "v"] if c != sig])
        Y = y.upper()
        spk = f"convolution in the time domain corresponds to multiplication in the frequency domain, so {spk_ct(y)} {eq} {spk_ct(sig)} convolved with h of t implies {Y} of j omega {eq} {SIG} of j omega H of j omega."
        tgt = f"convolution in the time domain corresponds to multiplication in the frequency domain, so ${y}(t) = {sig}(t) * h(t)$ implies ${Y}(j\\omega) = {SIG}(j\\omega) H(j\\omega)$."
        return with_opener(spk, tgt)

    elif kind == 7:
        spk = f"the discrete-time Fourier transform DTFT of {spk_dt(sig)} is {SIG} of e to the j {bw} {eq} sum from n equals minus infinity to infinity of {spk_dt(sig)} {exp} minus j {bw} n."
        tgt = f"the discrete-time Fourier transform (DTFT) of ${sig}[n]$ is ${SIG}(e^{{j\\Omega}}) = \\sum_{{n=-\\infty}}^{{\\infty}} {sig}[n] e^{{-j\\Omega n}}$."
        return with_opener(spk, tgt)

    elif kind == 8:
        spk = f"the inverse DTFT is evaluated over one 2 pi period as {spk_dt(sig)} {eq} 1 {div} 2 pi integral from minus pi to pi of {SIG} of e to the j {bw} {exp} j {bw} n d {bw}."
        tgt = f"the inverse DTFT is evaluated over one $2\\pi$ period as ${sig}[n] = \\frac{{1}}{{2\\pi}} \\int_{{-\\pi}}^{{\\pi}} {SIG}(e^{{j\\Omega}}) e^{{j\\Omega n}} \\, d\\Omega$."
        return with_opener(spk, tgt)

    elif kind == 9:
        N = rng.choice(["N", "8", "16", "32", "64", "256", "1024"])
        spk = f"for a length {N} sequence, the DFT is {SIG} of k {eq} sum from n equals 0 to {N} minus 1 of {spk_dt(sig)} W {N} to the power k n where W {N} {eq} {exp} minus j 2 pi {div} {N}."
        tgt = f"for a length-${N}$ sequence, the DFT is ${SIG}[k] = \\sum_{{n=0}}^{{{N}-1}} {sig}[n] W_{{{N}}}^{{kn}}$ where $W_{{{N}}} = e^{{-j\\frac{{2\\pi}}{{{N}}}}}$."
        return with_opener(spk, tgt)

    else:
        spk = f"modulating {spk_ct(sig)} by cos {w0} t shifts its spectrum to 1 {div} 2 {SIG} of j open parenthesis omega minus {w0} close parenthesis plus 1 {div} 2 {SIG} of j open parenthesis omega plus {w0} close parenthesis."
        tgt = f"modulating ${sig}(t)$ by $\\cos(\\omega_0 t)$ shifts its spectrum to $\\frac{{1}}{{2}} {SIG}(j(\\omega - \\omega_0)) + \\frac{{1}}{{2}} {SIG}(j(\\omega + \\omega_0))$."
        return with_opener(spk, tgt)


# ---------------------------------------------------------------------------
# Category 6: Sampling Theorem, Nyquist Rate, Reconstruction & Aliasing
# ---------------------------------------------------------------------------

def gen_sampling() -> Tuple[str, str]:
    kind = rng.randint(0, 5)
    eq = rng.choice(EQ_PHRASES)
    div = rng.choice(DIV_PHRASES)
    sig = rng.choice(["x", "m", "g", "s"])
    SIG = sig.upper()

    if kind == 0:
        fmax = rng.choice(["100", "200", "500", "1000", "2000", "4000", "f max", "B", "W"])
        if fmax.isdigit():
            nyq = str(2 * int(fmax))
            spk = f"if the band-limited signal {spk_ct(sig)} has maximum frequency f max {eq} {fmax} Hz, then the Nyquist sampling rate must satisfy f s greater than or equal to 2 f max {eq} {nyq} Hz."
            tgt = f"if the band-limited signal ${sig}(t)$ has maximum frequency $f_{{\\max}} = {fmax}\\text{{ Hz}}$, then the Nyquist sampling rate must satisfy $f_s \\ge 2f_{{\\max}} = {nyq}\\text{{ Hz}}$."
        else:
            fmax_tex = "f_{\\max}" if fmax == "f max" else fmax
            spk = f"by the Nyquist-Shannon sampling theorem, a signal band-limited to {fmax} can be perfectly reconstructed if the sampling rate satisfies f s greater than or equal to 2 {fmax}."
            tgt = f"by the Nyquist-Shannon sampling theorem, a signal band-limited to ${fmax_tex}$ can be perfectly reconstructed if the sampling rate satisfies $f_s \\ge 2{fmax_tex}$."
        return with_opener(spk, tgt)

    elif kind == 1:
        wm = rng.choice(["omega m", "omega max", "omega B"])
        wm_tex = {"omega m": "\\omega_m", "omega max": "\\omega_{\\max}", "omega B": "\\omega_B"}[wm]
        spk = f"in angular frequency, avoiding aliasing requires sampling frequency omega s greater than or equal to 2 {wm}, which corresponds to sampling period T s less than or equal to pi {div} {wm}."
        tgt = f"in angular frequency, avoiding aliasing requires sampling frequency $\\omega_s \\ge 2{wm_tex}$, which corresponds to sampling period $T_s \\le \\frac{{\\pi}}{{{wm_tex}}}$."
        return with_opener(spk, tgt)

    elif kind == 2:
        spk = f"impulse-train sampling produces the periodic spectrum {SIG} s of j omega {eq} 1 {div} T s sum from k equals minus infinity to infinity of {SIG} of j open parenthesis omega minus k omega s close parenthesis."
        tgt = f"impulse-train sampling produces the periodic spectrum ${SIG}_s(j\\omega) = \\frac{{1}}{{T_s}} \\sum_{{k=-\\infty}}^{{\\infty}} {SIG}(j(\\omega - k\\omega_s))$."
        return with_opener(spk, tgt)

    elif kind == 3:
        spk = f"passing the sampled signal through an ideal low-pass filter of gain T s and cutoff omega c {eq} omega s {div} 2 recovers {spk_ct(sig)} via Whittaker-Shannon interpolation {spk_ct(sig)} {eq} sum from n equals minus infinity to infinity of {sig} of n T s sinc of open parenthesis t minus n T s close parenthesis {div} T s."
        tgt = f"passing the sampled signal through an ideal low-pass filter of gain $T_s$ and cutoff $\\omega_c = \\frac{{\\omega_s}}{{2}}$ recovers ${sig}(t)$ via Whittaker-Shannon interpolation ${sig}(t) = \\sum_{{n=-\\infty}}^{{\\infty}} {sig}(nT_s) \\text{{sinc}}\\left(\\frac{{t - nT_s}}{{T_s}}\\right)$."
        return with_opener(spk, tgt)

    elif kind == 4:
        f0 = rng.choice([600, 700, 800, 900, 1200, 1500])
        fs = rng.choice([1000, 1100, 1400, 1600, 2000])
        if fs < 2 * f0:
            fa = abs(fs - f0)
            spk = f"sampling a sinusoid of frequency f 0 {eq} {f0} Hz at f s {eq} {fs} Hz violates the Nyquist criterion f s greater than or equal to 2 f 0, causing aliasing at frequency f a {eq} absolute value of f s minus f 0 {eq} {fa} Hz."
            tgt = f"sampling a sinusoid of frequency $f_0 = {f0}\\text{{ Hz}}$ at $f_s = {fs}\\text{{ Hz}}$ violates the Nyquist criterion $f_s \\ge 2f_0$, causing aliasing at frequency $f_a = |f_s - f_0| = {fa}\\text{{ Hz}}$."
        else:
            spk = f"sampling a sinusoid of frequency f 0 {eq} {f0} Hz at f s {eq} {fs} Hz satisfies f s greater than or equal to 2 f 0, so no aliasing occurs."
            tgt = f"sampling a sinusoid of frequency $f_0 = {f0}\\text{{ Hz}}$ at $f_s = {fs}\\text{{ Hz}}$ satisfies $f_s \\ge 2f_0$, so no aliasing occurs."
        return with_opener(spk, tgt)

    else:
        spk = f"the relation between discrete-time angular frequency capital omega and continuous-time frequency omega is capital omega {eq} omega T s {eq} 2 pi f {div} f s."
        tgt = f"the relation between discrete-time angular frequency $\\Omega$ and continuous-time frequency $\\omega$ is $\\Omega = \\omega T_s = \\frac{{2\\pi f}}{{f_s}}$."
        return with_opener(spk, tgt)


# ---------------------------------------------------------------------------
# Category 7: Multi-Sentence Lecture Derivations (300 - 850 chars)
# Matches real Whisper lecture chunk lengths!
# ---------------------------------------------------------------------------

def gen_multi_sentence_derivation() -> Tuple[str, str]:
    kind = rng.randint(0, 5)
    eq = rng.choice(EQ_PHRASES)
    div = rng.choice(DIV_PHRASES)
    exp = rng.choice(EXP_PHRASES)
    w0 = rng.choice(W0_SPOKEN)

    if kind == 0:
        # Full Initial & Final Value Theorem walkthrough
        p1, p2 = rng.sample([1, 2, 3, 4, 5, 6], 2)
        K = rng.choice([2, 3, 4, 5, 6, 8, 10])
        z1 = rng.choice([1, 2, 3, 4])
        # s*X(s) = K*s*(s+z1)/((s+p1)*(s+p2)) -> limit s->inf is K, limit s->0 is 0
        # Or X(s) = K*(s+z1)/(s*(s+p1)*(s+p2)) -> limit s->inf s*X(s)=0, limit s->0 s*X(s) = K*z1/(p1*p2)
        if rng.random() < 0.5:
            spk = (
                f"Suppose we are given the Laplace transform X of s {eq} {K} times s plus {z1} {div} open parenthesis s plus {p1} close parenthesis open parenthesis s plus {p2} close parenthesis. "
                f"Since both poles s {eq} minus {p1} and s {eq} minus {p2} lie strictly in the left half plane, we can apply both the initial value theorem and the final value theorem. "
                f"First, the initial value is x of 0 plus {eq} limit as s goes to infinity of s times X of s {eq} {K}. "
                f"Second, the steady-state value is limit as t goes to infinity of x of t {eq} limit as s goes to 0 of s times X of s {eq} 0."
            )
            tgt = (
                f"Suppose we are given the Laplace transform $X(s) = \\frac{{{K}(s + {z1})}}{{(s + {p1})(s + {p2})}}$. "
                f"Since both poles $s = -{p1}$ and $s = -{p2}$ lie strictly in the left half-plane, we can apply both the initial value theorem and the final value theorem. "
                f"First, the initial value is $x(0^+) = \\lim_{{s \\to \\infty}} s X(s) = {K}$. "
                f"Second, the steady-state value is $\\lim_{{t \\to \\infty}} x(t) = \\lim_{{s \\to 0}} s X(s) = 0$."
            )
        else:
            spk = (
                f"Consider a step response with Laplace transform Y of s {eq} {K} times s plus {z1} {div} s times open parenthesis s plus {p1} close parenthesis open parenthesis s plus {p2} close parenthesis. "
                f"Notice that s times Y of s has poles only at s {eq} minus {p1} and s {eq} minus {p2}, both in the open left half plane. "
                f"Therefore, by the final value theorem, limit as t tends to infinity of y of t {eq} limit as s tends to 0 of s Y of s {eq} {K * z1} {div} {p1 * p2}."
            )
            tgt = (
                f"Consider a step response with Laplace transform $Y(s) = \\frac{{{K}(s + {z1})}}{{s(s + {p1})(s + {p2})}}$. "
                f"Notice that $s Y(s)$ has poles only at $s = -{p1}$ and $s = -{p2}$, both in the open left half-plane. "
                f"Therefore, by the final value theorem, $\\lim_{{t \\to \\infty}} y(t) = \\lim_{{s \\to 0}} s Y(s) = \\frac{{{K * z1}}}{{{p1 * p2}}}$."
            )
        return spk, tgt

    elif kind == 1:
        # Causal vs Non-causal sinusoid & complex exponential (directly addresses live lecture edge cases!)
        idx = rng.choice(["1", "2", "3"])
        a = rng.choice(["2", "3", "4", "5", "a"])
        spk = (
            f"Now let us compare two signals. The first signal is x {idx} of t {eq} {exp} minus {a} t u of t, which is causal because x {idx} of t {eq} 0 for t less than 0. "
            f"Its Laplace transform is X {idx} of s {eq} 1 {div} s plus {a} with ROC real part of s greater than minus {a}. "
            f"The second signal is x of t {eq} cos {w0} t for all values of t, which we can decompose using Euler's formula as 1 {div} 2 {exp} j {w0} t plus 1 {div} 2 {exp} minus j {w0} t."
        )
        tgt = (
            f"Now let us compare two signals. The first signal is $x_{idx}(t) = e^{{-{a}t}} u(t)$, which is causal because $x_{idx}(t) = 0$ for $t < 0$. "
            f"Its Laplace transform is $X_{idx}(s) = \\frac{{1}}{{s + {a}}}$ with ROC $\\text{{Re}}(s) > -{a}$. "
            f"The second signal is $x(t) = \\cos(\\omega_0 t)$ for all values of $t$, which we can decompose using Euler's formula as $\\frac{{1}}{{2}} e^{{j\\omega_0 t}} + \\frac{{1}}{{2}} e^{{-j\\omega_0 t}}$."
        )
        return spk, tgt

    elif kind == 2:
        # LTI Convolution -> Frequency Domain -> Stability derivation
        a = rng.choice(["1", "2", "3", "4", "5"])
        b = rng.choice(["2", "3", "4", "5", "6"])
        if a == b:
            b = str(int(b) + 1)
        spk = (
            f"Suppose a causal LTI system has impulse response h of t {eq} {exp} minus {a} t u of t and receives input x of t {eq} {exp} minus {b} t u of t. "
            f"Rather than computing the convolution integral y of t {eq} integral from 0 to t of x of tau h of t minus tau d tau directly, we transform both signals to the s-domain. "
            f"This gives H of s {eq} 1 {div} s plus {a} and X of s {eq} 1 {div} s plus {b}, so the output transform is Y of s {eq} H of s X of s {eq} 1 {div} open parenthesis s plus {a} close parenthesis open parenthesis s plus {b} close parenthesis."
        )
        tgt = (
            f"Suppose a causal LTI system has impulse response $h(t) = e^{{-{a}t}} u(t)$ and receives input $x(t) = e^{{-{b}t}} u(t)$. "
            f"Rather than computing the convolution integral $y(t) = \\int_0^t x(\\tau) h(t - \\tau) \\, d\\tau$ directly, we transform both signals to the $s$-domain. "
            f"This gives $H(s) = \\frac{{1}}{{s + {a}}}$ and $X(s) = \\frac{{1}}{{s + {b}}}$, so the output transform is $Y(s) = H(s) X(s) = \\frac{{1}}{{(s + {a})(s + {b})}}$."
        )
        return spk, tgt

    elif kind == 3:
        # Discrete Z-transform difference equation + stability
        a1 = rng.choice(["0.4", "0.5", "0.6", "0.7", "0.8"])
        b0 = rng.choice(["1", "2", "3", "4"])
        spk = (
            f"Let us analyze the discrete-time LTI system described by y of n minus {a1} y of n minus 1 {eq} {b0} x of n. "
            f"Taking the Z-transform of both sides gives Y of z minus {a1} z to the minus 1 Y of z {eq} {b0} X of z. "
            f"Solving for the system function yields H of z {eq} Y of z {div} X of z {eq} {b0} {div} 1 minus {a1} z to the minus 1. "
            f"Because the single pole at z {eq} {a1} lies inside the unit circle magnitude of z {eq} 1, the causal system is BIBO stable with ROC magnitude of z greater than {a1}."
        )
        tgt = (
            f"Let us analyze the discrete-time LTI system described by $y[n] - {a1} y[n-1] = {b0} x[n]$. "
            f"Taking the $Z$-transform of both sides gives $Y(z) - {a1} z^{{-1}} Y(z) = {b0} X(z)$. "
            f"Solving for the system function yields $H(z) = \\frac{{Y(z)}}{{X(z)}} = \\frac{{{b0}}}{{1 - {a1} z^{{-1}}}}$. "
            f"Because the single pole at $z = {a1}$ lies inside the unit circle $|z| = 1$, the causal system is BIBO stable with ROC $|z| > {a1}$."
        )
        return spk, tgt

    elif kind == 4:
        # Combine 2-3 independent generators into a coherent multi-sentence lecture chunk
        gens = [gen_signal_basics, gen_lti_convolution, gen_laplace, gen_ztransform, gen_fourier, gen_sampling]
        chosen = rng.sample(gens, rng.choice([2, 3]))
        connectors = ["Furthermore, ", "Next, ", "In addition, ", "Consequently, ", "Building on this, "]
        spk_parts, tgt_parts = [], []
        for idx_g, g in enumerate(chosen):
            s_i, t_i = g()
            if idx_g > 0 and rng.random() < 0.6:
                conn = rng.choice(connectors)
                s_i = conn + s_i[0].lower() + s_i[1:]
                t_i = conn + t_i[0].lower() + t_i[1:]
            spk_parts.append(s_i)
            tgt_parts.append(t_i)
        return " ".join(spk_parts), " ".join(tgt_parts)

    else:
        # Fourier Transform & Sampling end-to-end derivation
        bw = rng.choice(["1000", "2000", "4000", "5000", "8000"])
        nyq = str(2 * int(bw))
        spk = (
            f"Consider a continuous-time message signal x of t whose Fourier transform X of j omega is strictly zero for magnitude of f greater than {bw} Hz. "
            f"When we sample x of t at sampling period T s {eq} 1 {div} f s, the sampled spectrum X s of j omega consists of shifted copies of X of j omega spaced by omega s {eq} 2 pi f s. "
            f"To prevent spectral overlap, we require f s greater than or equal to {nyq} Hz, and we reconstruct x of t using an ideal low-pass filter with cutoff frequency f c {eq} f s {div} 2."
        )
        tgt = (
            f"Consider a continuous-time message signal $x(t)$ whose Fourier transform $X(j\\omega)$ is strictly zero for $|f| > {bw}\\text{{ Hz}}$. "
            f"When we sample $x(t)$ at sampling period $T_s = \\frac{{1}}{{f_s}}$, the sampled spectrum $X_s(j\\omega)$ consists of shifted copies of $X(j\\omega)$ spaced by $\\omega_s = 2\\pi f_s$. "
            f"To prevent spectral overlap, we require $f_s \\ge {nyq}\\text{{ Hz}}$, and we reconstruct $x(t)$ using an ideal low-pass filter with cutoff frequency $f_c = \\frac{{f_s}}{{2}}$."
        )
        return spk, tgt


# ---------------------------------------------------------------------------
# Category 8: MathSpeech Real + Synthetic Spoken Math Wrapped in Lecture Prose
# ---------------------------------------------------------------------------

LECTURE_WRAPPERS = [
    ("From the previous step we obtain {spk}.", "From the previous step, we obtain ${tex}$."),
    ("Simplifying the expression gives {spk}.", "Simplifying the expression gives ${tex}$."),
    ("In our derivation we arrive at {spk}.", "In our derivation, we arrive at ${tex}$."),
    ("Notice that this reduces to {spk}.", "Notice that this reduces to ${tex}$."),
    ("We can write this relation as {spk}.", "We can write this relation as ${tex}$."),
    ("Substituting into the equation yields {spk}.", "Substituting into the equation yields ${tex}$."),
    ("Evaluating the term gives {spk}.", "Evaluating the term gives ${tex}$."),
    ("Recall the algebraic identity {spk}.", "Recall the algebraic identity ${tex}$."),
    ("So on the right-hand side we have {spk}.", "So on the right-hand side we have ${tex}$."),
    ("{spk}", "${tex}$"),
]


def clean_mathspeech_latex(tex: str) -> str:
    tex = tex.strip()
    tex = re.sub(r"\\displaystyle\s*", "", tex)
    tex = tex.strip("$").strip()
    return tex


def load_mathspeech_samples(path: Path, max_synthetic: int) -> List[Tuple[str, str]]:
    if not path.exists():
        return []
    real_rows, synth_rows = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            inp = re.sub(r"\s+", " ", row.get("input", "")).strip().rstrip(".")
            tex = clean_mathspeech_latex(row.get("target", ""))
            if not inp or not tex or "$" in tex or len(inp) < 6 or len(tex) > 220:
                continue
            if row.get("source") == "synthetic_v2":
                synth_rows.append((inp, tex))
            else:
                real_rows.append((inp, tex))

    rng.shuffle(synth_rows)
    selected = real_rows + synth_rows[:max_synthetic]
    out = []
    for inp, tex in selected:
        w_spk, w_tgt = rng.choice(LECTURE_WRAPPERS)
        spk = w_spk.format(spk=inp)
        tgt = w_tgt.format(tex=tex)
        if w_spk == "{spk}":
            spk = cap_first(spk) + "."
            tgt = tgt + "."
        out.append((spk, tgt))
    return out


# ---------------------------------------------------------------------------
# Category 9: Clean Conversational & Real NPTEL Prose Negative Anchors
# (Strictly filtered to contain ZERO math symbols/variables/equations!)
# ---------------------------------------------------------------------------

MATH_EXCLUSION_RE = re.compile(
    r"\b(?:equals?|plus|minus|times|divided|integral|sum|limit|sigma|omega|delta|alpha|beta|"
    r"theta|tau|pi|sin|cos|tan|exp|log|ln|square[d]?|cube[d]?|power|root|fraction|infinity|"
    r"laplace|fourier|transform|roc|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"x|y|z|t|n|k|s|h|f|j|e|r|a|b|c|d|m|p|q|u|v|w)\b|[0-9=+\-*/^<>()\[\]{}|\\]",
    re.IGNORECASE,
)


def load_clean_nptel_prose_negatives(max_samples: int = 700) -> List[Tuple[str, str]]:
    """Extracts strictly non-mathematical spoken English prose windows from harvested NPTEL lectures."""
    if not HARVESTED_DIR.exists():
        return []
    candidates: List[str] = []
    for p in sorted(HARVESTED_DIR.glob("*.json")):
        try:
            lec = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue
        auto_track = lec.get("transcripts", {}).get("en_auto", [])
        # Slide non-overlapping windows of 3-5 snippets
        i = 0
        while i + 3 < len(auto_track):
            step = rng.randint(3, 5)
            chunk = auto_track[i:i + step]
            i += step
            texts = [w.get("text", "").strip() for w in chunk]
            if any("[" in t or "]" in t for t in texts):
                continue
            raw = re.sub(r"\s+", " ", " ".join(texts)).strip()
            if len(raw) < 55 or len(raw) > 280:
                continue
            # Strictly exclude anything that could possibly contain math or standalone variables
            if MATH_EXCLUSION_RE.search(raw):
                continue
            # Normalize capitalization & trailing period
            clean = cap_first(raw)
            if clean[-1] not in ".?!":
                clean += "."
            candidates.append(clean)

    # Deduplicate and shuffle
    unique = list(dict.fromkeys(candidates))
    rng.shuffle(unique)
    return [(s, s) for s in unique[:max_samples]]


def gen_synthetic_conversational_negatives(count: int = 500) -> List[Tuple[str, str]]:
    """Generates diverse non-mathematical lecture prose so model never hallucinates LaTeX."""
    greetings = [
        "Good morning everyone.",
        "Welcome back to the lecture.",
        "Hello class, let us get started.",
        "Thank you for joining today's session.",
        "Before we begin the derivation,",
        "Before wrapping up today,",
        "Just a quick administrative reminder:",
        "Let us pause here for a moment.",
    ]
    topics = [
        "continuous-time signal classification",
        "linear time-invariant system properties",
        "frequency-domain intuition",
        "analog filter design",
        "practical communication channels",
        "digital signal processing architectures",
        "spectral analysis and windowing",
        "causality and physical realizability",
        "feedback stability in engineering systems",
        "sampling and analog-to-digital conversion",
        "modulation schemes in wireless systems",
        "laboratory oscilloscope measurements",
    ]
    actions = [
        "we will review the core concepts behind",
        "please read the textbook chapter covering",
        "the upcoming homework assignment focuses on",
        "the tutorial session tomorrow will practice",
        "we discussed the physical motivation for",
        "our next module introduces the fundamentals of",
        "the midterm syllabus includes all material on",
        "remember to check the course portal for notes on",
        "in practical engineering hardware we rely heavily on",
        "let us build strong geometric intuition for",
    ]
    closers = [
        "",
        " Are there any questions so far?",
        " Make sure to revise your notes before the next class.",
        " We will look at concrete examples right after this.",
        " Please bring your calculators to the tutorial.",
        " Feel free to stop by during office hours if anything is unclear.",
        " This perspective will help throughout the rest of the semester.",
    ]
    out: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    attempts = 0
    while len(out) < count and attempts < count * 10:
        attempts += 1
        g = rng.choice(greetings)
        act = rng.choice(actions)
        act_str = act if g.endswith((",", ":")) else cap_first(act)
        sent = f"{g} {act_str} {rng.choice(topics)}.{rng.choice(closers)}"
        sent = re.sub(r"\s+", " ", sent).strip()
        if sent not in seen:
            seen.add(sent)
            out.append((sent, sent))
    return out


# ---------------------------------------------------------------------------
# Helpers: Validation & Packaging
# ---------------------------------------------------------------------------

def has_balanced_latex(text: str) -> bool:
    if "$$" in text:
        return False
    # Count unescaped $ signs
    dollars = len(re.findall(r"(?<!\\)\$", text))
    if dollars % 2 != 0:
        return False
    # Check brace balance inside each $...$ block
    for m in re.finditer(r"(?<!\\)\$(.+?)(?<!\\)\$", text):
        expr = m.group(1)
        depth = 0
        for ch in expr:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    return False
        if depth != 0:
            return False
    return True


def sample_unique_from(
    gen_fn: Callable[[], Tuple[str, str]],
    target_count: int,
    global_seen: Set[str],
    label: str,
) -> List[Tuple[str, str]]:
    collected: List[Tuple[str, str]] = []
    attempts = 0
    max_attempts = target_count * 25
    while len(collected) < target_count and attempts < max_attempts:
        attempts += 1
        spk, tgt = gen_fn()
        spk = re.sub(r"\s+", " ", spk).strip()
        tgt = re.sub(r"\s+", " ", tgt).strip()
        key = spk.lower()
        if key in global_seen:
            continue
        if not has_balanced_latex(tgt):
            continue
        global_seen.add(key)
        collected.append((spk, tgt))
    print(f"  [{label}] Collected {len(collected)}/{target_count} unique valid samples ({attempts} draws).")
    return collected


def format_chatml(spoken: str, target: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Spoken transcript:\n{spoken}"},
            {"role": "assistant", "content": target},
        ]
    }


def write_jsonl_and_gz(samples: List[dict], jsonl_path: Path, gz_path: Path) -> None:
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(jsonl_path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)


def generate_dataset():
    print("=" * 72)
    print("  Building Zero-Duplication, Leak-Free Signals & Systems Dataset")
    print("=" * 72)

    global_seen: Set[str] = set()

    # 1. Parametric Signals & Systems pools (strictly unique)
    pools = {
        "Signal Basics & Disambiguation": sample_unique_from(gen_signal_basics, 1400, global_seen, "Signal Basics"),
        "LTI Systems & Convolution": sample_unique_from(gen_lti_convolution, 800, global_seen, "LTI & Convolution"),
        "Laplace, ROC, IVT & FVT": sample_unique_from(gen_laplace, 1500, global_seen, "Laplace/IVT/FVT"),
        "Z-Transform & Discrete LTI": sample_unique_from(gen_ztransform, 1000, global_seen, "Z-Transform"),
        "Fourier Series & Transforms": sample_unique_from(gen_fourier, 900, global_seen, "Fourier"),
        "Sampling & Reconstruction": sample_unique_from(gen_sampling, 500, global_seen, "Sampling"),
        "Multi-Sentence Lecture Derivations": sample_unique_from(gen_multi_sentence_derivation, 1500, global_seen, "Multi-Sentence"),
    }

    # Stratified 90/10 split per category (guaranteed 0% overlap since global_seen enforced uniqueness)
    train_pairs: List[Tuple[str, str]] = []
    val_pairs: List[Tuple[str, str]] = []

    for cat, pairs in pools.items():
        rng.shuffle(pairs)
        split = int(len(pairs) * 0.90)
        train_pairs.extend(pairs[:split])
        val_pairs.extend(pairs[split:])

    # 2. MathSpeech real + synthetic (respecting MathSpeech's own train/val split!)
    ms_train_raw = load_mathspeech_samples(MATHSPEECH_TRAIN, max_synthetic=2200)
    ms_val_raw = load_mathspeech_samples(MATHSPEECH_VAL, max_synthetic=250)

    ms_train_added = 0
    for spk, tgt in ms_train_raw:
        k = spk.lower()
        if k not in global_seen and has_balanced_latex(tgt):
            global_seen.add(k)
            train_pairs.append((spk, tgt))
            ms_train_added += 1

    ms_val_added = 0
    for spk, tgt in ms_val_raw:
        k = spk.lower()
        if k not in global_seen and has_balanced_latex(tgt):
            global_seen.add(k)
            val_pairs.append((spk, tgt))
            ms_val_added += 1
    print(f"  [MathSpeech] Added {ms_train_added} train + {ms_val_added} val unique spoken-math samples.")

    # 3. Clean non-math negative anchors (NPTEL real prose + synthetic classroom prose)
    neg_nptel = load_clean_nptel_prose_negatives(max_samples=500)
    neg_synth = gen_synthetic_conversational_negatives(count=500)
    neg_all = []
    for spk, tgt in neg_nptel + neg_synth:
        k = spk.lower()
        if k not in global_seen:
            global_seen.add(k)
            neg_all.append((spk, tgt))
    rng.shuffle(neg_all)
    neg_split = int(len(neg_all) * 0.90)
    train_pairs.extend(neg_all[:neg_split])
    val_pairs.extend(neg_all[neg_split:])
    print(f"  [Negative Anchors] Added {neg_split} train + {len(neg_all) - neg_split} val unique non-math anchors "
          f"(NPTEL clean prose: {len(neg_nptel)}, conversational: {len(neg_synth)}).")

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    # Strict integrity assertions
    train_keys = {s.lower() for s, _ in train_pairs}
    val_keys = {s.lower() for s, _ in val_pairs}
    assert len(train_keys) == len(train_pairs), "Duplicate detected in train split!"
    assert len(val_keys) == len(val_pairs), "Duplicate detected in val split!"
    assert len(train_keys & val_keys) == 0, f"Train/val leakage detected: {len(train_keys & val_keys)}"

    train_records = [format_chatml(s, t) for s, t in train_pairs]
    val_records = [format_chatml(s, t) for s, t in val_pairs]

    # Write uncompressed to datasets/ and compressed to training/data/
    train_gz = TRAINING_DATA_DIR / "train.jsonl.gz"
    val_gz = TRAINING_DATA_DIR / "val.jsonl.gz"
    write_jsonl_and_gz(train_records, TRAIN_FILE, train_gz)
    write_jsonl_and_gz(val_records, VAL_FILE, val_gz)

    # Also mirror .gz in DATA_DIR
    shutil.copy2(train_gz, DATA_DIR / "train.jsonl.gz")
    shutil.copy2(val_gz, DATA_DIR / "val.jsonl.gz")

    long_train = sum(1 for s, _ in train_pairs if len(s) >= 280)
    print("\n" + "=" * 72)
    print("  DATASET VERIFICATION SUMMARY")
    print("=" * 72)
    print(f"  Train samples:       {len(train_records):,} (100% unique)")
    print(f"  Val samples:         {len(val_records):,} (100% unique)")
    print(f"  Train/Val overlap:   0 (verified)")
    print(f"  Multi-sentence >280c:{long_train:,} train paragraphs")
    print(f"  Uncompressed Train:  {TRAIN_FILE.stat().st_size / (1024*1024):.2f} MB")
    print(f"  Uncompressed Val:    {VAL_FILE.stat().st_size / (1024*1024):.2f} MB")
    print(f"  Compressed Train:    {train_gz.stat().st_size / (1024*1024):.2f} MB -> {train_gz}")
    print(f"  Compressed Val:      {val_gz.stat().st_size / 1024:.1f} KB -> {val_gz}")
    print("=" * 72)


if __name__ == "__main__":
    generate_dataset()
