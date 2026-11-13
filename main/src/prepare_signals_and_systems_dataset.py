#!/usr/bin/env python3
"""
Comprehensive Signals & Systems Dataset Generator for Qwen2.5-1.5B LoRA Fine-Tuning.

Produces train.jsonl and val.jsonl in standard ChatML format:
  <|im_start|>system
  You are an expert Electrical Engineering and Signals & Systems lecture transcriber...
  <|im_end|>
  <|im_start|>user
  Spoken transcript:
  ...
  <|im_end|>
  <|im_start|>assistant
  ...
  <|im_end|>

Dataset Distribution:
  - 30% Continuous & Discrete Signals, Basic Operations, Disambiguation (x(t), x[n], x_2(t), exponentials, sinusoids)
  - 25% Laplace & Z-Transforms, ROC, Poles/Zeros, Initial & Final Value Theorems
  - 20% Fourier Series & Transforms (CTFS, DTFS, CTFT, DTFT, DFT, Parseval's)
  - 10% Sampling Theorem, Nyquist Rate, Reconstruction Filters, Aliasing
  - 10% Real NPTEL Spoken Text Anchors (Harvested from IIT Kanpur Course 108104100)
  -  5% Pure Conversational English & Negative Anchors (Must NEVER wrap in LaTeX)
"""

import json
from pathlib import Path
import random
import re
import sys
from typing import Dict, List, Tuple

HEADLESS_SRC = Path("/mnt/thesis/headless_agent/src")
if str(HEADLESS_SRC) not in sys.path:
    sys.path.insert(0, str(HEADLESS_SRC))

SEED = 42
TOTAL_SAMPLES = 15000
TRAIN_RATIO = 0.90
rng = random.Random(SEED)

DATA_DIR = Path("/mnt/thesis/datasets/Qwen_Signals_And_Systems_FineTune")
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"

HARVESTED_DIR = Path("/mnt/thesis/datasets/nptel_signals_and_systems/raw_transcripts")

SYSTEM_PROMPT = (
    "You are an expert Electrical Engineering lecture transcriber and mathematical typesetter specializing in "
    "Signals and Systems, Network Theory, Digital Signal Processing, and Control Systems.\n"
    "Your task is to take a spoken lecture transcript and convert it into a natural, academic transcript where spoken "
    "mathematical formulas, signals, variables, transforms, and equations are accurately converted into standard inline LaTeX ($...$).\n\n"
    "STRICT SIGNALS & SYSTEMS RULES:\n"
    "1. SIGNAL & FUNCTION NOTATION:\n"
    "   - Continuous-time coordinates and signals use parentheses: 'x of t' -> '$x(t)$', 'x 2 of t' or 'x2 (t)' -> '$x_2(t)$', "
    "'h of t' -> '$h(t)$', 'y of t' -> '$y(t)$', 'u of t' -> '$u(t)$', 'delta of t' -> '$\\delta(t)$'.\n"
    "   - Discrete-time indices and sequences use square brackets: 'x of n' -> '$x[n]$', 'y of n' -> '$y[n]$', 'h of n' -> '$h[n]$', "
    "'x of 2n' -> '$x[2n]$', 'x of n minus k' -> '$x[n-k]$', 'delta of n' -> '$\\delta[n]$', 'u of n' -> '$u[n]$'.\n"
    "2. EXPONENTIALS & HARMONICS:\n"
    "   - 'e to the power n minus 1' -> '$e^{n-1}$', 't to the power n minus 1' -> '$t^{n-1}$'.\n"
    "   - 'e power minus a t u of t' -> '$e^{-at} u(t)$', 'e power minus 3t u of t' -> '$e^{-3t} u(t)$'.\n"
    "   - 'e power j omega 0' or 'e power jw0' -> '$e^{j\\omega_0}$', 'e power j omega 0 t' -> '$e^{j\\omega_0 t}$', 'e power minus j omega 0 t' -> '$e^{-j\\omega_0 t}$'.\n"
    "   - 'cos omega 0 t' or 'cos omega 0' -> '$\\cos(\\omega_0 t)$', 'sin omega 0 t' -> '$\\sin(\\omega_0 t)$'.\n"
    "3. TRANSFORMS & REGIONS OF CONVERGENCE (ROC):\n"
    "   - Laplace: 'X of s' -> '$X(s)$', 'H of s' -> '$H(s)$', 's equals sigma plus j omega' -> '$s = \\sigma + j\\omega$'.\n"
    "   - Initial Value Theorem: 'x of 0 plus equals limit as s tends to infinity of s times X of s' -> '$x(0^+) = \\lim_{s \\to \\infty} s X(s)$'.\n"
    "   - Final Value Theorem: 'limit as t tends to infinity of x of t equals limit as s tends to 0 of s times X of s' -> '$\\lim_{t \\to \\infty} x(t) = \\lim_{s \\to 0} s X(s)$'.\n"
    "   - ROC: 'real part of s greater than minus 3' -> '$\\text{Re}(s) > -3$'.\n"
    "   - Z-Transform: 'X of z' -> '$X(z)$', 'sum from n equals minus infinity to infinity of x of n z to the minus n' -> '$\\sum_{n=-\\infty}^{\\infty} x[n] z^{-n}$', 'magnitude of z greater than r' -> '$|z| > r$'.\n"
    "   - Fourier: 'X of j omega' -> '$X(j\\omega)$', 'X of e to the j capital omega' -> '$X(e^{j\\Omega})$', 'Fourier series coefficient a k' -> '$a_k = \\frac{1}{T} \\int_{T} x(t) e^{-jk\\omega_0 t} \\, dt$'.\n"
    "4. CONVOLUTION & SAMPLING:\n"
    "   - 'x of t convolved with h of t' -> '$x(t) * h(t) = \\int_{-\\infty}^{\\infty} x(\\tau) h(t-\\tau) \\, d\\tau$'.\n"
    "   - 'discrete convolution sum x of k h of n minus k' -> '$y[n] = x[n] * h[n] = \\sum_{k=-\\infty}^{\\infty} x[k] h[n-k]$'.\n"
    "   - 'Nyquist rate omega s equals 2 omega m' -> '$\\omega_s \\ge 2\\omega_m$', 'sampling frequency f s' -> '$f_s \\ge 2f_{\\max}$'.\n"
    "5. PRESERVATION RULES:\n"
    "   - PRESERVE 100% of surrounding English lecture prose and academic explanations.\n"
    "   - NEVER wrap plain conversational phrases, chapter numbers, administrative talk, or countdowns in LaTeX.\n"
    "   - Clean speech disfluencies and false starts into correct mathematical form.\n"
    "   - Use ONLY inline math ($...$). NEVER output display math environments ($$...$$) in raw transduction."
)

# ---------------------------------------------------------------------------
# Generators across Disciplines & Topics
# ---------------------------------------------------------------------------

def gen_signal_basics_sample() -> Tuple[str, str]:
    """Signals, subscripts, operations, and basic exponential/sinusoidal expressions (30%)."""
    idx = rng.choice(["1", "2", "0", "k"])
    w0 = rng.choice(["omega 0", "omega zero", "omega_0", "ω0"])
    
    samples = [
        (
            f"Now let us look at the second signal which is x{idx} (t) equal to cos {w0}. Now this is not a causal signal anymore it is cos({w0} t) for all values of t.",
            f"Now let us look at the second signal which is $x_{idx}(t) = \\cos(\\omega_0 t)$. Now this is not a causal signal anymore, it is $\\cos(\\omega_0 t)$ for all values of $t$."
        ),
        (
            "In general say e to the power n minus 1.",
            "In general say $e^{n-1}$."
        ),
        (
            "In general say t to the power n minus 1 divided by n minus 1 factorial.",
            "In general say $\\frac{t^{n-1}}{(n-1)!}$."
        ),
        (
            f"So let us say I want to find Laplace transform of e power j{w0}",
            f"So let us say I want to find Laplace transform of $e^{{j\\omega_0}}$."
        ),
        (
            f"The continuous complex exponential signal x of t equals e to the power j {w0} t has fundamental period T 0 equals 2 pi divided by omega 0.",
            f"The continuous complex exponential signal $x(t) = e^{{j\\omega_0 t}}$ has fundamental period $T_0 = \\frac{{2\\pi}}{{\\omega_0}}$."
        ),
        (
            f"Consider two signals x 1 of t equals sin {w0} t and x 2 of t equals cos {w0} t.",
            f"Consider two signals $x_1(t) = \\sin(\\omega_0 t)$ and $x_2(t) = \\cos(\\omega_0 t)$."
        ),
        (
            "For discrete sequences we downsample by defining y of n equals x of 2n where odd indexed samples are discarded.",
            "For discrete sequences we downsample by defining $y[n] = x[2n]$ where odd-indexed samples are discarded."
        ),
        (
            "The time shifted sequence is given by x of n minus k where k represents the integer delay in samples.",
            "The time-shifted sequence is given by $x[n-k]$ where $k$ represents the integer delay in samples."
        ),
        (
            "The continuous unit impulse or Dirac delta function delta of t is zero everywhere except at t equals 0 where it has unit area.",
            "The continuous unit impulse or Dirac delta function $\\delta(t)$ is zero everywhere except at $t = 0$ where it has unit area."
        ),
        (
            "The discrete unit impulse delta of n equals 1 when n equals 0 and equals 0 when n is not equal to 0.",
            "The discrete unit impulse $\\delta[n] = 1$ when $n = 0$ and equals $0$ when $n \\neq 0$."
        ),
        (
            "The unit step signal u of t is related to the impulse by delta of t equals d u by d t in the distributional sense.",
            "The unit step signal $u(t)$ is related to the impulse by $\\delta(t) = \\frac{du}{dt}$ in the distributional sense."
        ),
        (
            "We can express the continuous unit ramp signal r of t as t times u of t which is the running integral of the step.",
            "We can express the continuous unit ramp signal $r(t) = t u(t)$ which is the running integral of the step."
        ),
        (
            "Consider the decaying exponential signal x of t equals e to the minus 3 t times u of t.",
            "Consider the decaying exponential signal $x(t) = e^{-3t} u(t)$."
        ),
        (
            "The energy of the continuous signal is the integral from minus infinity to infinity of absolute value of x of t squared dt.",
            "The energy of the continuous signal is $\\int_{-\\infty}^{\\infty} |x(t)|^2 \\, dt$."
        ),
        (
            "For a discrete sequence the total energy E x is the sum from n equals minus infinity to infinity of absolute x of n squared.",
            "For a discrete sequence the total energy $E_x = \\sum_{n=-\\infty}^{\\infty} |x[n]|^2$."
        )
    ]
    return rng.choice(samples)


def gen_laplace_ztransform_sample() -> Tuple[str, str]:
    """Laplace transforms, Z-transforms, ROC boundaries, initial & final value theorems (25%)."""
    a = rng.choice(["1", "2", "3", "a"])
    samples = [
        (
            f"The bilateral Laplace transform of continuous signal x of t is defined as X of s equals integral from minus infinity to infinity of x of t e to the minus s t dt.",
            f"The bilateral Laplace transform of continuous signal $x(t)$ is defined as $X(s) = \\int_{{-\\infty}}^{{\\infty}} x(t) e^{{-st}} \\, dt$."
        ),
        (
            f"Here the complex frequency variable s equals sigma plus j omega where sigma is the real attenuation and omega is the frequency.",
            f"Here the complex frequency variable $s = \\sigma + j\\omega$ where $\\sigma$ is the real attenuation and $\\omega$ is the frequency."
        ),
        (
            f"For the signal x of t equals e to the minus {a} t u of t the Laplace transform is 1 divided by s plus {a} with region of convergence real part of s greater than minus {a}.",
            f"For the signal $x(t) = e^{{-{a}t}} u(t)$ the Laplace transform is $\\frac{{1}}{{s+{a}}}$ with region of convergence $\\text{{Re}}(s) > -{a}$."
        ),
        (
            "The initial value theorem states that if x of t is causal and contains no impulses at origin then x of 0 plus equals limit as s tends to infinity of s times X of s.",
            "The initial value theorem states that if $x(t)$ is causal and contains no impulses at origin, then $x(0^+) = \\lim_{s \\to \\infty} s X(s)$."
        ),
        (
            "According to the final value theorem if all poles of s times X of s reside strictly in the left half plane then limit as t tends to infinity of x of t equals limit as s tends to 0 of s times X of s.",
            "According to the final value theorem, if all poles of $s X(s)$ reside strictly in the left-half plane, then $\\lim_{t \\to \\infty} x(t) = \\lim_{s \\to 0} s X(s)$."
        ),
        (
            "To evaluate the steady state output of an LTI system we compute the limit as s goes to 0 of s times X of s times H of s.",
            "To evaluate the steady-state output of an LTI system, we compute $\\lim_{s \\to 0} s X(s) H(s)$."
        ),
        (
            "The output Laplace transform is Y of s equals fraction 1 over s plus 1 minus fraction 1 over s plus 2.",
            "The output Laplace transform is $Y(s) = \\frac{1}{s+1} - \\frac{1}{s+2}$."
        ),
        (
            "The Laplace transform of cos omega 0 t u of t is s divided by s square plus omega 0 squared with ROC real part of s greater than 0.",
            "The Laplace transform of $\\cos(\\omega_0 t) u(t)$ is $\\frac{s}{s^2 + \\omega_0^2}$ with ROC $\\text{Re}(s) > 0$."
        ),
        (
            "The Laplace transform of sin omega 0 t u of t evaluates to omega 0 divided by s square plus omega 0 squared.",
            "The Laplace transform of $\\sin(\\omega_0 t) u(t)$ evaluates to $\\frac{\\omega_0}{s^2 + \\omega_0^2}$."
        ),
        (
            "The Z-transform of discrete sequence x of n is defined as X of z equals sum from n equals minus infinity to infinity of x of n times z to the minus n.",
            "The $Z$-transform of discrete sequence $x[n]$ is defined as $X(z) = \\sum_{n=-\\infty}^{\\infty} x[n] z^{-n}$."
        ),
        (
            "Here z is represented in polar form as r times e to the j capital omega where r is the radial distance and capital omega is discrete frequency.",
            "Here $z$ is represented in polar form as $r e^{j\\Omega}$ where $r$ is the radial distance and $\\Omega$ is discrete frequency."
        ),
        (
            "For a causal sequence x of n equals a to the n u of n the Z-transform is 1 divided by 1 minus a z to the minus 1 with ROC magnitude of z greater than absolute value of a.",
            "For a causal sequence $x[n] = a^n u[n]$ the $Z$-transform is $\\frac{1}{1 - a z^{-1}}$ with ROC $|z| > |a|$."
        ),
        (
            "The discrete initial value theorem states that x of 0 equals limit as z tends to infinity of X of z.",
            "The discrete initial value theorem states that $x[0] = \\lim_{z \\to \\infty} X(z)$."
        ),
        (
            "The discrete final value theorem yields limit as n tends to infinity of x of n equals limit as z tends to 1 of 1 minus z to the minus 1 times X of z.",
            "The discrete final value theorem yields $\\lim_{n \\to \\infty} x[n] = \\lim_{z \\to 1} (1 - z^{-1}) X(z)$."
        ),
        (
            "A discrete-time LTI system is BIBO stable if and only if the ROC of its rational system function H of z includes the unit circle magnitude of z equals 1.",
            "A discrete-time LTI system is BIBO stable if and only if the ROC of its rational system function $H(z)$ includes the unit circle $|z| = 1$."
        )
    ]
    return rng.choice(samples)


def gen_fourier_sample() -> Tuple[str, str]:
    """Fourier series, CTFT, DTFT, DFT, and Parseval energy relations (20%)."""
    samples = [
        (
            "The continuous-time Fourier series synthesis equation represents x of t as sum from k equals minus infinity to infinity of a k times e to the j k omega 0 t.",
            "The continuous-time Fourier series synthesis equation represents $x(t) = \\sum_{k=-\\infty}^{\\infty} a_k e^{jk\\omega_0 t}$."
        ),
        (
            "The analysis equation gives the harmonic coefficient a k equals 1 over T 0 integral over period T 0 of x of t e to the minus j k omega 0 t dt.",
            "The analysis equation gives the harmonic coefficient $a_k = \\frac{1}{T_0} \\int_{T_0} x(t) e^{-jk\\omega_0 t} \\, dt$."
        ),
        (
            "By Parseval's theorem for periodic signals the average power P x equals 1 over T 0 integral over T 0 of absolute x of t squared dt which equals sum of absolute a k squared.",
            "By Parseval's theorem for periodic signals, the average power $P_x = \\frac{1}{T_0} \\int_{T_0} |x(t)|^2 \\, dt = \\sum_{k=-\\infty}^{\\infty} |a_k|^2$."
        ),
        (
            "The continuous-time Fourier transform CTFT is given by X of j omega equals integral from minus infinity to infinity of x of t e to the minus j omega t dt.",
            "The continuous-time Fourier transform (CTFT) is given by $X(j\\omega) = \\int_{-\\infty}^{\\infty} x(t) e^{-j\\omega t} \\, dt$."
        ),
        (
            "The inverse CTFT reconstructs the time domain signal as x of t equals 1 over 2 pi integral from minus infinity to infinity of X of j omega e to the j omega t d omega.",
            "The inverse CTFT reconstructs the time-domain signal as $x(t) = \\frac{1}{2\\pi} \\int_{-\\infty}^{\\infty} X(j\\omega) e^{j\\omega t} \\, d\\omega$."
        ),
        (
            "Convolution in the time domain corresponds to multiplication in the frequency domain so Y of j omega equals X of j omega times H of j omega.",
            "Convolution in the time domain corresponds to multiplication in the frequency domain, so $Y(j\\omega) = X(j\\omega) H(j\\omega)$."
        ),
        (
            "The discrete-time Fourier transform DTFT of sequence x of n is defined as capital X of e to the j capital omega equals sum from n equals minus infinity to infinity of x of n e to the minus j capital omega n.",
            "The discrete-time Fourier transform (DTFT) of sequence $x[n]$ is defined as $X(e^{j\\Omega}) = \\sum_{n=-\\infty}^{\\infty} x[n] e^{-j\\Omega n}$."
        ),
        (
            "The DTFT is strictly periodic in discrete frequency with period 2 pi so X of e to the j open parenthesis capital omega plus 2 pi close parenthesis equals X of e to the j capital omega.",
            "The DTFT is strictly periodic in discrete frequency with period $2\\pi$, so $X(e^{j(\\Omega + 2\\pi)}) = X(e^{j\\Omega})$."
        ),
        (
            "The inverse DTFT is evaluated over one 2 pi period as x of n equals 1 over 2 pi integral from minus pi to pi of X of e to the j capital omega e to the j capital omega n d capital omega.",
            "The inverse DTFT is evaluated over one $2\\pi$ period as $x[n] = \\frac{1}{2\\pi} \\int_{-\\pi}^{\\pi} X(e^{j\\Omega}) e^{j\\Omega n} \\, d\\Omega$."
        ),
        (
            "For finite-length sequences of length N the discrete Fourier transform DFT is defined by X of k equals sum from n equals 0 to N minus 1 of x of n times W N to the k n where twiddle factor W N equals e to the minus j 2 pi over N.",
            "For finite-length sequences of length $N$, the discrete Fourier transform (DFT) is defined by $X[k] = \\sum_{n=0}^{N-1} x[n] W_N^{kn}$ where twiddle factor $W_N = e^{-j\\frac{2\\pi}{N}}$."
        ),
        (
            "Circular convolution of two N point sequences in time domain x 1 of n circular convolved with x 2 of n is equivalent to point wise multiplication of their DFTs X 1 of k times X 2 of k.",
            "Circular convolution of two $N$-point sequences in time domain $x_1[n] \\circledast x_2[n]$ is equivalent to pointwise multiplication of their DFTs $X_1[k] X_2[k]$."
        )
    ]
    return rng.choice(samples)


def gen_sampling_reconstruction_sample() -> Tuple[str, str]:
    """Sampling theorem, Nyquist criterion, reconstruction filters, and aliasing (10%)."""
    samples = [
        (
            "According to the Nyquist-Shannon sampling theorem a continuous band-limited signal with maximum frequency omega m can be perfectly reconstructed if sampling frequency omega s is greater than or equal to 2 omega m.",
            "According to the Nyquist-Shannon sampling theorem, a continuous band-limited signal with maximum frequency $\\omega_m$ can be perfectly reconstructed if sampling frequency $\\omega_s \\ge 2\\omega_m$."
        ),
        (
            "The minimum sampling rate required to avoid spectral overlap or aliasing is known as the Nyquist rate f s equals 2 times f max.",
            "The minimum sampling rate required to avoid spectral overlap or aliasing is known as the Nyquist rate $f_s = 2f_{\\max}$."
        ),
        (
            "The spectrum of the impulse-sampled signal X s of j omega equals 1 over T s sum from k equals minus infinity to infinity of X of j open parenthesis omega minus k omega s close parenthesis.",
            "The spectrum of the impulse-sampled signal $X_s(j\\omega) = \\frac{1}{T_s} \\sum_{k=-\\infty}^{\\infty} X(j(\\omega - k\\omega_s))$."
        ),
        (
            "To reconstruct the original signal we pass the sampled sequence through an ideal lowpass filter with cutoff frequency omega c equals omega s over 2 and gain T s.",
            "To reconstruct the original signal, we pass the sampled sequence through an ideal low-pass filter with cutoff frequency $\\omega_c = \\frac{\\omega_s}{2}$ and gain $T_s$."
        ),
        (
            "In the time domain ideal reconstruction corresponds to Whittaker-Shannon sinc interpolation where x of t equals sum from n equals minus infinity to infinity of x of n T s times sinc of open parenthesis t minus n T s close parenthesis over T s.",
            "In the time domain, ideal reconstruction corresponds to Whittaker-Shannon sinc interpolation where $x(t) = \\sum_{n=-\\infty}^{\\infty} x(nT_s) \\text{sinc}\\left(\\frac{t - nT_s}{T_s}\\right)$."
        ),
        (
            "When sampling below the Nyquist rate f s less than 2 f max high frequency components fold back into the baseband creating irreversible aliasing distortion.",
            "When sampling below the Nyquist rate $f_s < 2f_{\\max}$, high-frequency components fold back into the baseband creating irreversible aliasing distortion."
        )
    ]
    return rng.choice(samples)


def gen_nptel_harvested_sample(harvested_corpus: List[Dict]) -> Tuple[str, str]:
    """Real harvested NPTEL spoken sentences mapped to academic typeset forms (10%)."""
    if not harvested_corpus:
        return gen_signal_basics_sample()

    lec = rng.choice(harvested_corpus)
    transcripts = lec.get("transcripts", {})
    auto_track = transcripts.get("en_auto", [])
    if not auto_track or len(auto_track) < 5:
        return gen_signal_basics_sample()

    # Select a coherent window of 3-5 snippets
    start_idx = rng.randint(0, max(0, len(auto_track) - 4))
    window = auto_track[start_idx:start_idx + rng.randint(3, 5)]
    spoken_raw = " ".join([w["text"] for w in window])
    spoken_clean = re.sub(r'\s+', ' ', spoken_raw).strip()

    # If the window contains math phrases, apply deterministic transformation to build target
    from lecture_to_book import LectureToBookBuilder
    builder = LectureToBookBuilder()
    target_clean = builder.clean_speech_text(spoken_clean)

    # If no math was transformed, capitalize properly
    if target_clean == spoken_clean:
        target_clean = target_clean.capitalize() + "."

    return spoken_clean, target_clean


def gen_negative_sample() -> Tuple[str, str]:
    """Pure conversational English & administrative lecture speech (5%). MUST NEVER wrap in LaTeX."""
    negatives = [
        (
            "Good morning class. Today we will continue our discussion on frequency domain representation.",
            "Good morning class. Today we will continue our discussion on frequency domain representation."
        ),
        (
            "Please turn to lecture module four on the course website before Wednesday's session.",
            "Please turn to lecture module four on the course website before Wednesday's session."
        ),
        (
            "Are there any questions regarding the derivation we completed in the previous lecture?",
            "Are there any questions regarding the derivation we completed in the previous lecture?"
        ),
        (
            "Five, four, three, two, one, zero. Let us begin our analysis.",
            "Five, four, three, two, one, zero. Let us begin our analysis."
        ),
        (
            "The homework assignment will be posted on the portal by tonight and is due next Tuesday at five PM.",
            "The homework assignment will be posted on the portal by tonight and is due next Tuesday at 5 PM."
        ),
        (
            "Notice how the physical intuition matches the laboratory measurements we observed yesterday.",
            "Notice how the physical intuition matches the laboratory measurements we observed yesterday."
        ),
        (
            "In the next class we will examine practical filter design using analog circuit components.",
            "In the next class we will examine practical filter design using analog circuit components."
        ),
        (
            "Keep in mind that continuous-time and discrete-time methodologies share direct mathematical parallels.",
            "Keep in mind that continuous-time and discrete-time methodologies share direct mathematical parallels."
        )
    ]
    return rng.choice(negatives)


def load_all_harvested_lectures() -> List[Dict]:
    """Loads all harvested NPTEL JSON files."""
    corpus = []
    if not HARVESTED_DIR.exists():
        return corpus
    for p in HARVESTED_DIR.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                corpus.append(json.load(f))
        except Exception:
            pass
    return corpus


def format_chatml(spoken: str, target: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Spoken transcript:\n{spoken}"},
            {"role": "assistant", "content": target}
        ]
    }


def generate_dataset(total_samples: int = TOTAL_SAMPLES):
    print("=" * 65)
    print(f"Generating {total_samples}-Sample Signals & Systems LoRA Fine-Tuning Dataset")
    print(f"Output directory: {DATA_DIR}")
    print("=" * 65)

    harvested_corpus = load_all_harvested_lectures()
    print(f"[Dataset] Loaded {len(harvested_corpus)} harvested NPTEL lectures for real spoken anchors.")

    # Target counts based on distribution
    n_signals = int(total_samples * 0.30)
    n_transforms = int(total_samples * 0.25)
    n_fourier = int(total_samples * 0.20)
    n_sampling = int(total_samples * 0.10)
    n_nptel = int(total_samples * 0.10)
    n_neg = total_samples - (n_signals + n_transforms + n_fourier + n_sampling + n_nptel)

    samples = []

    print(f"[1/6] Synthesizing {n_signals} Signal Basics & Disambiguation pairs...")
    for _ in range(n_signals):
        spk, tgt = gen_signal_basics_sample()
        samples.append(format_chatml(spk, tgt))

    print(f"[2/6] Synthesizing {n_transforms} Laplace, Z-Transform & ROC pairs...")
    for _ in range(n_transforms):
        spk, tgt = gen_laplace_ztransform_sample()
        samples.append(format_chatml(spk, tgt))

    print(f"[3/6] Synthesizing {n_fourier} Fourier Series, Transforms & Parseval pairs...")
    for _ in range(n_fourier):
        spk, tgt = gen_fourier_sample()
        samples.append(format_chatml(spk, tgt))

    print(f"[4/6] Synthesizing {n_sampling} Sampling & Reconstruction pairs...")
    for _ in range(n_sampling):
        spk, tgt = gen_sampling_reconstruction_sample()
        samples.append(format_chatml(spk, tgt))

    print(f"[5/6] Mining {n_nptel} Real Harvested NPTEL Lecture pairs...")
    for _ in range(n_nptel):
        spk, tgt = gen_nptel_harvested_sample(harvested_corpus)
        samples.append(format_chatml(spk, tgt))

    print(f"[6/6] Generating {n_neg} Pure Conversational Negative Anchors...")
    for _ in range(n_neg):
        spk, tgt = gen_negative_sample()
        samples.append(format_chatml(spk, tgt))

    rng.shuffle(samples)

    split_idx = int(len(samples) * TRAIN_RATIO)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"\n[Packaging] Writing {len(train_samples)} training samples to {TRAIN_FILE}...")
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[Packaging] Writing {len(val_samples)} validation samples to {VAL_FILE}...")
    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n[Done] Signals & Systems Dataset successfully prepared!")
    print(f"  Train: {TRAIN_FILE} ({TRAIN_FILE.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Val:   {VAL_FILE} ({VAL_FILE.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    generate_dataset()
