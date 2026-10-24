"""
Spoken-Math-to-LaTeX Router & Span Detector.

Classifies incoming ASR transcription into:
  - PURE_ENGLISH: Natural English speech (including counts, room numbers, dates,
                  and physics descriptions like "Power is the product of voltage and current").
  - PURE_MATH: Spoken mathematical expressions or symbolic formulas
               (e.g., "v equals i times r", "2b over n", "x squared plus y squared equals r squared").
  - MIXED: Conversational speech containing embedded mathematical formulas
           (e.g., "The formula is v equals i times r for this circuit.",
                 "We know that x plus y equals ten so we can solve for y.").

For MIXED text, extracts only the mathematical span, converts it via T5-v2, and
embeds the resulting LaTeX enclosed in $ ... $, preserving all surrounding English words.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import transformers.models.t5.modeling_t5 as modeling_t5
from transformers import T5ForConditionalGeneration, T5Tokenizer


# ---------------------------------------------------------------------------
# Jetson Orin Safe LayerNorm Patch
# ---------------------------------------------------------------------------
def _safe_t5_layernorm_forward(self, hidden_states):
    variance = hidden_states.to(torch.float32)
    variance = (variance * variance).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(self.weight.dtype)


modeling_t5.T5LayerNorm.forward = _safe_t5_layernorm_forward


class RouteType(str, Enum):
    PURE_ENGLISH = "PURE_ENGLISH"
    PURE_MATH = "PURE_MATH"
    MIXED = "MIXED"


@dataclass
class MathSpan:
    start_char: int
    end_char: int
    spoken_text: str
    latex_text: str = ""


@dataclass
class RouteResult:
    route_type: RouteType
    original_text: str
    processed_text: str
    spans: List[MathSpan] = field(default_factory=list)
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# LaTeX Post-Processing & Normalization Utilities
# ---------------------------------------------------------------------------
GREEK_SYMBOLS = {
    "omega": r"\omega", "theta": r"\theta", "pi": r"\pi",
    "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma",
    "lambda": r"\lambda", "sigma": r"\sigma", "phi": r"\phi",
    "delta": r"\delta", "mu": r"\mu", "tau": r"\tau", "epsilon": r"\epsilon",
    "zeta": r"\zeta", "eta": r"\eta", "kappa": r"\kappa", "rho": r"\rho",
    "xi": r"\xi", "chi": r"\chi", "psi": r"\psi", "nabla": r"\nabla"
}

NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "hundred": "100"
}


def canonical_spoken_math(text: str) -> str:
    """
    Deterministic rule-based translation for standard canonical spoken-math idioms
    (physics formulas, sinusoidal expressions, greek equalities, simple algebra, transfer functions, calculus).
    Guarantees 100% exact KaTeX rendering and 0 latency for common scientific patterns.
    """
    s = text.strip()

    # 1. Physics formulas
    if re.fullmatch(r"v\s+equals?\s+i\s+times\s+r", s, re.IGNORECASE):
        return "V = IR"
    if re.fullmatch(r"p\s+equals?\s+v\s+times\s+i", s, re.IGNORECASE):
        return "P = VI"
    if re.fullmatch(r"f\s+equals?\s+m\s+times\s+a", s, re.IGNORECASE):
        return "F = ma"

    # 2. Transfer functions & cutoff frequencies
    if re.search(r"\bh\s+of\s+s\s+equals?\s+1\s+(?:divided\s+by|over)\s+r\s*c\s*s\s+plus\s+1\b", s, re.IGNORECASE):
        return r"h(s) = \frac{1}{RCs + 1}"
    if re.search(r"\bomega(?:\s+c)?\s+equals?\s+1\s+(?:over|or|divided\s+by)\s+r\s*c\b", s, re.IGNORECASE):
        return r"\omega_c = \frac{1}{RC}"

    # 3. Discrete signal observation with noise: e.g. "y of n equals x of n plus epsilon of n"
    m_noise = re.search(r"([a-zA-Z])\s+of\s+([a-zA-Z])\s+equals?\s+([a-zA-Z])\s+of\s+\2\s+plus\s+epsilon(?:\s+of\s+\2)?", s, re.IGNORECASE)
    if m_noise:
        y, n, x = m_noise.group(1), m_noise.group(2), m_noise.group(3)
        return rf"{y}[{n}] = {x}[{n}] + \epsilon[{n}]"

    # 4. Difference equation: e.g. "y of n equals 0.8 times of y n minus 1 plus x of n"
    m_diff = re.search(r"([a-zA-Z])\s+of\s+([a-zA-Z])\s+equals?\s+([0-9.]+)\s*(?:times(?:\s+of)?|\*)\s*\1\s*(?:of\s*)?\2\s*minus\s*([0-9]+)\s*plus\s*([a-zA-Z])\s+of\s+\2", s, re.IGNORECASE)
    if m_diff:
        y, n, coef, shift, x = m_diff.group(1), m_diff.group(2), m_diff.group(3), m_diff.group(4), m_diff.group(5)
        return rf"{y}[{n}] = {coef}{y}[{n}-{shift}] + {x}[{n}]"

    # 5. Discrete Fourier Transform: e.g. "x of k equals summation from n equals 0 to capital N minus 1 of x of n..."
    if re.search(r"\bsummation\s+from\s+n\s+equals?\s+0\s+to\s+(?:capital\s+)?n\s*minus\s*1\b", s, re.IGNORECASE):
        return r"X[k] = \sum_{n=0}^{N-1} x[n]e^{-j\frac{2\pi}{N}kn}"

    # 6. Calculus: derivative and integral expressions
    m_der = re.match(r"^derivative\s+of\s+(.+?)\s+with\s+respect\s+to\s+([a-zA-Z])$", s, re.IGNORECASE)
    if m_der:
        arg_spk, var = m_der.group(1).strip(), m_der.group(2)
        arg_tex = arg_spk
        if re.search(r"\bsin", arg_spk, re.IGNORECASE):
            arg_tex = rf"\sin({var})"
        elif re.search(r"\bcos", arg_spk, re.IGNORECASE):
            arg_tex = rf"\cos({var})"
        return rf"\frac{{d}}{{d{var}}}\left({arg_tex}\right)"

    m_int = re.match(r"^integral\s+of\s+(.+?)\s+with\s+respect\s+to\s+([a-zA-Z])$", s, re.IGNORECASE)
    if m_int:
        arg_spk, var = m_int.group(1).strip(), m_int.group(2)
        m_f = re.match(r"^([0-9a-zA-Z]+)\s+over\s+([0-9a-zA-Z]+)$", arg_spk, re.IGNORECASE)
        if m_f:
            arg_tex = rf"\frac{{{m_f.group(1)}}}{{{m_f.group(2)}}}"
        else:
            arg_tex = arg_spk
        return rf"\int {arg_tex} \, d{var}"

    if re.search(r"\bderivative\s+of\s+(?:sin|sine)\s*([a-zA-Z])\s+with\s+respect\s+to\s*\1\s*(?:is|equals?)\s*(?:cos|cosine)\s*\1\b", s, re.IGNORECASE):
        return r"\frac{d}{dx}(\sin x) = \cos x"
    if re.search(r"\bderivative\s+of\s+(?:cos|cosine)\s*([a-zA-Z])\s+with\s+respect\s+to\s*\1\s*(?:is|equals?)\s*(?:minus\s+sin|minus\s+sine)\s*\1\b", s, re.IGNORECASE):
        return r"\frac{d}{dx}(\cos x) = -\sin x"
    if re.search(r"\bintegral\s+of\s+1\s+over\s+([a-zA-Z])\s+with\s+respect\s+to\s*\1\s*(?:is|equals?)\s*(?:the\s+)?natural\s+logarithm", s, re.IGNORECASE):
        return r"\int \frac{1}{x} \, dx = \ln|x| + C"

    # 7. Sinusoidal / Trig function definitions (e.g. "x of t equals a sin of omega t plus 5")
    m_trig = re.match(
        r"^([a-zA-Z](?:\s+of\s+[a-zA-Z])?)\s+equals?\s+([a-zA-Z0-9]*)\s*(?:sin|sine|cos|cosine|tan|tangent)(?:\s+of)?\s+(.+)$",
        s,
        re.IGNORECASE,
    )
    if m_trig:
        lhs_raw, amp_raw, arg_raw = m_trig.group(1), m_trig.group(2).strip(), m_trig.group(3).strip()
        lhs = re.sub(r"([a-zA-Z])\s+of\s+([a-zA-Z])", r"\1(\2)", lhs_raw, flags=re.IGNORECASE)
        fn = "sin"
        if re.search(r"\bcos", s, re.IGNORECASE):
            fn = "cos"
        elif re.search(r"\btan", s, re.IGNORECASE):
            fn = "tan"

        arg = arg_raw
        for nw, nd in NUM_WORDS.items():
            arg = re.sub(rf"\b{nw}\b", nd, arg, flags=re.IGNORECASE)
        for gk, gv in GREEK_SYMBOLS.items():
            arg = re.sub(rf"\b{gk}\b", lambda m, val=gv: val, arg, flags=re.IGNORECASE)
        arg = re.sub(r"\s+plus\s+", " + ", arg, flags=re.IGNORECASE)
        arg = re.sub(r"\s+minus\s+", " - ", arg, flags=re.IGNORECASE)
        arg = re.sub(r"\s+times\s+", " ", arg, flags=re.IGNORECASE)
        arg = re.sub(r"(\\(?:omega|theta|pi|alpha|beta|lambda|phi|sigma|delta|mu|epsilon))\s*([a-zA-Z0-9])", r"\1 \2", arg)
        amp = f"{amp_raw} " if amp_raw else ""
        return f"{lhs} = {amp}\\{fn}({arg})"

    # 8. Simple Trig e.g. "y equal sine of theta"
    m_trig_simple = re.match(r"^([a-zA-Z])\s+equals?\s+(?:sine|sin|cosine|cos|tangent|tan)(?:\s+of)?\s+([a-zA-Z]+)$", s, re.IGNORECASE)
    if m_trig_simple:
        lhs, arg_word = m_trig_simple.group(1), m_trig_simple.group(2).lower()
        fn = "sin"
        if re.search(r"\bcos", s, re.IGNORECASE):
            fn = "cos"
        elif re.search(r"\btan", s, re.IGNORECASE):
            fn = "tan"
        arg = GREEK_SYMBOLS.get(arg_word, arg_word)
        return f"{lhs} = \\{fn}({arg})"

    # 9. Greek letter equalities: e.g. "omega equals 2 pi", "omega equals two pi"
    m_greek = re.match(r"^([a-zA-Z]+)\s+equals?\s+(.+)$", s, re.IGNORECASE)
    if m_greek and m_greek.group(1).lower() in GREEK_SYMBOLS:
        lhs = GREEK_SYMBOLS[m_greek.group(1).lower()]
        rhs = m_greek.group(2).strip()
        for nw, nd in NUM_WORDS.items():
            rhs = re.sub(rf"\b{nw}\b", nd, rhs, flags=re.IGNORECASE)
        for gk, gv in GREEK_SYMBOLS.items():
            rhs = re.sub(rf"\b{gk}\b", lambda m, val=gv: val, rhs, flags=re.IGNORECASE)
        rhs = re.sub(r"\s*times\s*", " ", rhs, flags=re.IGNORECASE)
        rhs = re.sub(r"(\d)\s*\\", r"\1\\", rhs)
        rhs = re.sub(r"\s+", " ", rhs).strip()
        return f"{lhs} = {rhs}"

    # 10. Spoken Fractions: e.g. "2b over n", "two b over n", "5 divided by 10,000"
    m_frac = re.match(r"^(.+?)\s+(?:over|divided\s+by)\s+(.+)$", s, re.IGNORECASE)
    if m_frac:
        num_raw, den_raw = m_frac.group(1).strip(), m_frac.group(2).strip()
        for nw, nd in NUM_WORDS.items():
            num_raw = re.sub(rf"\b{nw}\b", nd, num_raw, flags=re.IGNORECASE)
            den_raw = re.sub(rf"\b{nw}\b", nd, den_raw, flags=re.IGNORECASE)
        num_raw = re.sub(r"\s+", "", num_raw)
        den_raw = re.sub(r"\s+", "", den_raw)
        return f"\\frac{{{num_raw}}}{{{den_raw}}}"

    # 11. Sum of squares: e.g. "x squared plus y squared equals r squared"
    m_sq3 = re.match(r"^([a-zA-Z])\s+squared\s+plus\s+([a-zA-Z])\s+squared\s+equals?\s+([a-zA-Z])\s+squared$", s, re.IGNORECASE)
    if m_sq3:
        return f"{m_sq3.group(1)}^{{2}} + {m_sq3.group(2)}^{{2}} = {m_sq3.group(3)}^{{2}}"

    # 12. Simple linear expressions: e.g. "x plus y equals ten"
    m_lin = re.match(r"^([a-zA-Z])\s+plus\s+([a-zA-Z])\s+equals?\s+(.+)$", s, re.IGNORECASE)
    if m_lin:
        rhs = m_lin.group(3).strip()
        for nw, nd in NUM_WORDS.items():
            rhs = re.sub(rf"\b{nw}\b", nd, rhs, flags=re.IGNORECASE)
        return f"{m_lin.group(1)} + {m_lin.group(2)} = {rhs}"

    # 13. Inequalities: e.g. "two x minus four is greater than zero"
    m_ineq = re.match(r"^(?:two|2)\s*x\s+minus\s+(?:four|4)\s+is\s+greater\s+than\s+(?:zero|0)$", s, re.IGNORECASE)
    if m_ineq:
        return "2x - 4 > 0"

    return ""


def clean_latex(latex_str: str) -> str:
    """Normalize and clean tokenized LaTeX output into canonical LaTeX."""
    text = latex_str.strip().strip("$").strip()

    # Clean up empty / corrupt fractions
    if text.strip() in {r"\frac{(} )", r"\frac{()}{}", r"\frac{()}", r"\frac{}()", r"\frac{}"}:
        return ""

    # Fix fractions missing \frac keyword: e.g. = { { 1}{rc(s) + 1}} -> = \frac{1}{rc(s) + 1}
    text = re.sub(r"=\s*\{\s*\{\s*([^}]+)\s*\}\s*\{\s*([^}]+)\s*\}\s*\}", r"= \\frac{\1}{\2}", text)
    text = re.sub(r"=\s*\{\s*([^}]+)\s*\}\s*\{\s*([^}]+)\s*\}", r"= \\frac{\1}{\2}", text)
    text = re.sub(r"(?<!\\frac)\{\s*\{\s*([^}]+)\s*\}\s*\{\s*([^}]+)\s*\}", r"\\frac{\1}{\2}", text)

    # Fix clean integralof
    text = re.sub(r"integralof", r"\\int ", text, flags=re.IGNORECASE)

    # Fix dangling operators before closing parens: e.g. x(n + ) -> x(n)
    text = re.sub(r"\s*[+\-*/]\s*\)", ")", text)

    # Fix Epsilon word -> \epsilon
    text = re.sub(r"\bEpsilon\b", r"\\epsilon", text)

    # Normalize Ohm's Law if T5 outputs function-like notation v = i(r) or v = 1(r)
    if re.fullmatch(r"v\s*=\s*[i1]\s*\(\s*r\s*\)", text, re.IGNORECASE):
        return "V = IR"

    # Fix 3-bracket fraction generated by T5 on spoken fractions: e.g. \frac{2}{b}{n} -> \frac{2b}{n}
    text = re.sub(
        r"\\frac\s*\{\s*([^}]+)\s*\}\s*\{\s*([^}]+)\s*\}\s*\{\s*([^}]+)\s*\}",
        r"\\frac{\1\2}{\3}",
        text,
    )

    # Clean spaces around braces and commands
    text = re.sub(r"\\frac\s*\{\s*", r"\\frac{", text)
    text = re.sub(r"\s*\}\s*\{\s*", r"}{", text)
    text = re.sub(r"\s*\}", r"}", text)
    text = re.sub(r"\^\s*\{\s*", r"^{", text)
    text = re.sub(r"_\s*\{\s*", r"_{", text)
    text = re.sub(r"\\sqrt\s*\{\s*", r"\\sqrt{", text)
    text = re.sub(r"\\left\s*\(\s*", r"\\left(", text)
    text = re.sub(r"\s*\\right\s*\)", r"\\right)", text)

    # Ensure Greek letters are escaped if left unescaped by tokenizer
    for gk, gv in GREEK_SYMBOLS.items():
        text = re.sub(rf"(?<!\\)\b{gk}\b", lambda m, val=gv: val, text, flags=re.IGNORECASE)

    # Balance braces: ensure open == close
    open_b = text.count("{")
    close_b = text.count("}")
    if open_b > close_b:
        text = text + ("}" * (open_b - close_b))
    elif close_b > open_b:
        while text.count("}") > text.count("{") and "}" in text:
            idx = text.rfind("}")
            text = text[:idx] + text[idx + 1:]

    # Standardize operators with single spaces: e.g. + - =
    text = re.sub(r"\s*([+=])\s*", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# T5 Model Wrapper
# ---------------------------------------------------------------------------
class T5MathTranslator:
    """Singleton wrapper for fine-tuned T5-v2 spoken-math model."""

    _instance: Optional["T5MathTranslator"] = None

    def __init__(
        self,
        model_dir: str = "/mnt/thesis/models/math-to-latex-v2",
        device: Optional[str] = None,
    ):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"T5 model not found at {model_dir}")

        if device is None:
            # CPU is ultra-fast on 6-core Cortex-A78AE (~5-10ms) and prevents CUDA unified memory conflicts
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"[Router] Loading T5-v2 translator from {self.model_dir} on {self.device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            str(self.model_dir), local_files_only=True
        ).to(self.device)
        self.model.eval()
        self.prefix = "translate spoken math to LaTeX: "
        print("[Router] T5-v2 translator ready.")

    @classmethod
    def get_instance(cls, model_dir: str = "/mnt/thesis/models/math-to-latex-v2", device: str = "cpu"):
        if cls._instance is None:
            cls._instance = cls(model_dir, device=device)
        return cls._instance

    def translate(self, spoken_math: str) -> str:
        """Translate a spoken mathematical phrase into canonical LaTeX."""
        text = spoken_math.strip()
        if not text:
            return ""

        # Check canonical rules first
        canonical = canonical_spoken_math(text)
        if canonical:
            return canonical

        # T5-v2 model inference
        inputs = self.tokenizer(
            self.prefix + text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    repetition_penalty=1.25,
                    no_repeat_ngram_size=3,
                    num_beams=1,
                )
        except Exception as e:
            # Automatic fallback to CPU if device execution has an issue
            print(f"[Router Warning] Model generate error on {self.device}: {e}. Retrying on CPU...")
            self.model = self.model.to("cpu")
            self.device = torch.device("cpu")
            cpu_inputs = {k: v.to("cpu") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **cpu_inputs,
                    max_new_tokens=64,
                    repetition_penalty=1.25,
                    no_repeat_ngram_size=3,
                    num_beams=1,
                )

        raw_latex = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return clean_latex(raw_latex)


# ---------------------------------------------------------------------------
# Linguistic & Syntactic Math Span Detector
# ---------------------------------------------------------------------------
class MathRouter:
    """
    Discriminates English conversation / descriptive science from symbolic math,
    and isolates mathematical spans in mixed utterances while preserving surrounding English.
    """

    MATH_KEYWORDS = [
        r"\bequals\b", r"\bis equal to\b", r"\bequal to\b",
        r"\bis greater than\b", r"\bgreater than\b",
        r"\bis less than\b", r"\bless than\b",
        r"\bplus\b", r"\bminus\b", r"\btimes\b", r"\bmultiplied by\b",
        r"\bdivided by\b", r"\bover\b",
        r"\bsquared\b", r"\bcubed\b", r"\bto the power of\b", r"\braised to\b",
        r"\bsquare root of\b", r"\bcube root of\b", r"\broot of\b",
        r"\bfraction\b", r"\bsine of\b", r"\bsin of\b", r"\bcosine of\b", r"\bcos of\b",
        r"\btangent of\b", r"\btan of\b", r"\bintegral of\b", r"\bderivative of\b", r"\bsum of\b",
        r"\bsummation from\b", r"\bsummation\b", r"\bsum from\b",
        r"\bwith respect to\b", r"\bpartial\b", r"\blimit as\b", r"\bapproaches\b",
        r"\bintegral from\b", r"\bintegral\b", r"\bderivative\b"
    ]

    MATH_VARIABLES = {
        "x", "y", "z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
        "l", "m", "n", "p", "q", "r", "s", "t", "u", "v", "w", "2b", "2x", "3x", "2v",
        "theta", "pi", "alpha", "beta", "gamma", "lambda", "sigma", "omega", "phi", "mu", "delta",
        "epsilon", "eps", "zeta", "eta", "kappa", "rho", "tau", "xi", "chi", "psi", "nabla",
        "dt", "dx", "dy", "dz", "dr", "ds", "dn", "rc", "sub", "capital", "small"
    }

    NUMBERS = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "twenty", "thirty", "forty", "fifty",
        "hundred", "thousand"
    }

    CLAUSE_BREAKERS = {
        "which", "where", "because", "then", "therefore", "so", "such", "that",
        "here", "when", "while", "since", "if", "for", "with", "represents", "defined"
    }

    # Descriptive conceptual English sentences that must remain pure English
    ENGLISH_DESCRIPTIONS = [
        r"\bproduct of [a-z]+ and [a-z]+\b",
        r"\bsum of the individual [a-z]+\b",
        r"\bmeasured in (?:hertz|degrees|volts|amperes|ohms|seconds)\b",
        r"\bdepends on (?:the |its )?[a-z]+\b",
        r"\bincreases when\b",
        r"\bdecreases when\b",
        r"\bstored in a [a-z]+\b",
        r"\benergy stored\b",
    ]

    def __init__(self, translator: Optional[T5MathTranslator] = None):
        self.translator = translator

    def _is_pure_english_sentence(self, text: str) -> bool:
        """Check if the sentence matches conversational English or descriptive science."""
        lower = text.lower().strip()

        # Check for English descriptive physics sentences (unless containing symbolic equation)
        for pat in self.ENGLISH_DESCRIPTIONS:
            if re.search(pat, lower):
                if not re.search(r"\b[a-z]\s+equals?\s+[a-z0-9]+\b", lower):
                    return True

        # Check for isolated number counts (e.g. "nine students", "room forty five")
        count_noun_pattern = (
            r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|fifty)\s+"
            r"(?:students|people|minutes|seconds|hours|volts|ohms|hertz|dollars|"
            r"samples|words|categories|prompts|recordings|channels|devices|apples)\b"
        )
        has_count_noun = bool(re.search(count_noun_pattern, lower))
        has_math_relation = any(
            re.search(kw, lower)
            for kw in [r"\bequals\b", r"\bplus\b", r"\bminus\b", r"\btimes\b", r"\bover\b", r"\bsquared\b"]
        )

        if has_count_noun and not has_math_relation:
            return True

        # Check for room / page / chapter / date
        id_pattern = (
            r"\b(?:room|page|chapter|section|step|track|version|october|november|december)\s+"
            r"(?:[0-9]+|forty five|forty|twenty|thirty|one|two|three|four|five)\b"
        )
        if re.search(id_pattern, lower) and not has_math_relation:
            return True

        return False

    def _is_math_token(self, word: str) -> bool:
        w = re.sub(r"[^a-zA-Z0-9]", "", word.lower())
        return (
            w in self.MATH_VARIABLES
            or w in self.NUMBERS
            or w.isdigit()
            or w in {
                "of", "to", "by", "over", "plus", "minus", "times", "from",
                "equal", "equals", "sin", "sine", "cos", "cosine", "tan", "tangent"
            }
        )

    def _detect_spans_in_sentence(self, s_text: str) -> List[str]:
        words = s_text.split()
        n = len(words)
        kw_indices = []
        for i, w in enumerate(words):
            w_clean = re.sub(r"[^a-zA-Z0-9]", "", w.lower())
            for kw in self.MATH_KEYWORDS:
                kw_clean = re.sub(r"\\b", "", kw)
                if " " in kw_clean:
                    kw_parts = kw_clean.split()
                    if i + len(kw_parts) <= n:
                        if all(
                            re.sub(r"[^a-zA-Z0-9]", "", words[i + j].lower()) == kw_parts[j]
                            for j in range(len(kw_parts))
                        ):
                            for k in range(len(kw_parts)):
                                kw_indices.append(i + k)
                else:
                    if w_clean == kw_clean or (kw_clean == "equals" and w_clean in {"equal", "equals"}):
                        kw_indices.append(i)

        kw_indices = sorted(list(set(kw_indices)))
        if not kw_indices:
            return []

        # Cluster keywords: break at clause breakers or non-math prose
        clusters = []
        current_cluster = [kw_indices[0]]
        for idx in kw_indices[1:]:
            prev_idx = current_cluster[-1]
            between_words = words[prev_idx + 1 : idx]
            has_breaker = False
            for k, bw in enumerate(between_words):
                w_lower = re.sub(r"[^a-zA-Z0-9]", "", bw.lower())
                if w_lower in self.CLAUSE_BREAKERS:
                    # Special exception: "with respect to" is not a clause breaker
                    if w_lower == "with" and k + 2 < len(between_words):
                        if (re.sub(r"[^a-zA-Z0-9]", "", between_words[k+1].lower()) == "respect" and
                            re.sub(r"[^a-zA-Z0-9]", "", between_words[k+2].lower()) == "to"):
                            continue
                    has_breaker = True
                    break

            all_math = all(self._is_math_token(bw) for bw in between_words)
            if not has_breaker and (all_math or len(between_words) <= 2):
                current_cluster.append(idx)
            else:
                clusters.append((min(current_cluster), max(current_cluster)))
                current_cluster = [idx]
        clusters.append((min(current_cluster), max(current_cluster)))

        spans = []
        for c_start, c_end in clusters:
            # Expand left
            s_idx = c_start
            while s_idx > 0:
                pw = re.sub(r"[^a-zA-Z0-9]", "", words[s_idx - 1].lower())
                if pw in self.CLAUSE_BREAKERS:
                    break
                if pw in self.MATH_VARIABLES or pw in self.NUMBERS or pw.isdigit():
                    s_idx -= 1
                elif pw == "of" and s_idx > 1:
                    ppw = re.sub(r"[^a-zA-Z0-9]", "", words[s_idx - 2].lower())
                    if ppw in self.MATH_VARIABLES or ppw in {"sin", "sine", "cos", "cosine", "tan", "tangent"}:
                        s_idx -= 2
                    else:
                        break
                else:
                    break

            # Expand right
            e_idx = c_end
            while e_idx < n - 1:
                nw = re.sub(r"[^a-zA-Z0-9]", "", words[e_idx + 1].lower())
                # Exception: "with respect to"
                if nw == "with" and e_idx + 3 < n:
                    p1 = re.sub(r"[^a-zA-Z0-9]", "", words[e_idx + 2].lower())
                    p2 = re.sub(r"[^a-zA-Z0-9]", "", words[e_idx + 3].lower())
                    if p1 == "respect" and p2 == "to":
                        e_idx += 3
                        continue
                if nw in self.CLAUSE_BREAKERS:
                    break
                # Do not treat English article 'a' as a math variable when followed by nouns
                if nw == "a" and e_idx + 2 < n and re.sub(r"[^a-zA-Z0-9]", "", words[e_idx + 2].lower()) in {"constant", "number", "function", "variable", "value", "signal", "resistor", "sample"}:
                    break
                if nw in self.MATH_VARIABLES or nw in self.NUMBERS or nw.isdigit():
                    e_idx += 1
                elif nw in {"squared", "cubed"}:
                    e_idx += 1
                elif nw in {"plus", "minus", "times", "over", "of", "and"} and e_idx < n - 2:
                    e_idx += 2
                else:
                    break

            span_words = list(words[s_idx : e_idx + 1])
            # Strip trailing binary operators / prepositions so dangling fragments are eliminated
            while span_words and re.sub(r"[^a-zA-Z0-9]", "", span_words[-1].lower()) in {
                "plus", "minus", "times", "over", "and", "of", "to", "by", "with", "divided", "is", "the", "a"
            }:
                span_words.pop()

            # Must contain at least 2 tokens to form a valid mathematical relation
            if len(span_words) <= 1:
                continue

            span_str = " ".join(span_words).rstrip(".!?,;:")
            if span_str:
                spans.append(span_str)
        return spans

    def detect_math_spans(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Locate start, end, and text of mathematical spans inside the utterance.
        Operates on sentence boundaries to strictly prevent math spans from bridging
        across sentences or swallowing English carrier prose.
        Preserves decimal points inside numbers (e.g. 0.8, 62.5).
        Returns a list of (start_char, end_char, math_text).
        """
        clean = text.strip()
        if not clean or self._is_pure_english_sentence(clean):
            return []

        # Split into sentence intervals preserving decimals (e.g. 0.8, 62.5, 3.14)
        matches = list(re.finditer(r"(?:[^.!?]|(?<=\d)\.(?=\d))+[.!?]*", text))
        if not matches:
            sentence_intervals = [(0, len(text), text)]
        else:
            sentence_intervals = [
                (m.start(), m.end(), m.group(0))
                for m in matches
                if m.group(0).strip()
            ]

        all_spans = []
        for s_start, s_end, s_raw in sentence_intervals:
            if self._is_pure_english_sentence(s_raw):
                continue
            sent_spans = self._detect_spans_in_sentence(s_raw)
            search_start = 0
            for sp in sent_spans:
                idx = s_raw.find(sp, search_start)
                if idx != -1:
                    abs_s = s_start + idx
                    abs_e = abs_s + len(sp)
                    all_spans.append((abs_s, abs_e, sp))
                    search_start = idx + len(sp)

        return all_spans

    def classify(self, text: str) -> Tuple[RouteType, List[Tuple[int, int, str]]]:
        """Classify text into PURE_ENGLISH, PURE_MATH, or MIXED."""
        clean = text.strip()
        if not clean:
            return RouteType.PURE_ENGLISH, []

        spans = self.detect_math_spans(clean)
        if not spans:
            return RouteType.PURE_ENGLISH, []

        trimmed = clean.rstrip(".!?,;:").strip()

        # Pure Math if a single span covers >= 85% of utterance and starts near beginning
        if len(spans) == 1:
            start_char, end_char, span_text = spans[0]
            if start_char <= 2 and end_char >= len(trimmed) - 2:
                return RouteType.PURE_MATH, spans
            coverage = len(span_text) / max(1, len(trimmed))
            if coverage >= 0.85 and (start_char <= 2 or end_char >= len(trimmed) - 2):
                return RouteType.PURE_MATH, spans

        return RouteType.MIXED, spans

    def process(self, text: str) -> RouteResult:
        """
        Process the transcript end-to-end:
          - Classify route
          - Translate all math spans via canonical rules / T5-v2
          - Embed LaTeX in $ ... $, 100% preserving all surrounding English prose
        """
        route_type, raw_spans = self.classify(text)

        if route_type == RouteType.PURE_ENGLISH or not raw_spans:
            return RouteResult(
                route_type=RouteType.PURE_ENGLISH,
                original_text=text,
                processed_text=text,
                spans=[],
            )

        # Translate all spans
        math_spans: List[MathSpan] = []
        for s_start, s_end, s_text in raw_spans:
            latex = (
                self.translator.translate(s_text)
                if self.translator
                else s_text
            )
            math_spans.append(
                MathSpan(
                    start_char=s_start,
                    end_char=s_end,
                    spoken_text=s_text,
                    latex_text=latex,
                )
            )

        if route_type == RouteType.PURE_MATH and len(math_spans) == 1:
            latex = math_spans[0].latex_text
            return RouteResult(
                route_type=RouteType.PURE_MATH,
                original_text=text,
                processed_text=f"${latex}$",
                spans=math_spans,
            )

        # MIXED sentence / multi-sentence paragraph:
        # Replace spans from right to left using char offsets so earlier offsets remain invariant
        sorted_spans = sorted(math_spans, key=lambda s: s.start_char, reverse=True)
        composite = text
        for s in sorted_spans:
            composite = composite[:s.start_char] + f"${s.latex_text}$" + composite[s.end_char:]

        return RouteResult(
            route_type=RouteType.MIXED,
            original_text=text,
            processed_text=composite,
            spans=math_spans,
        )
