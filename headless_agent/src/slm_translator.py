#!/usr/bin/env python3
"""
SLM-Powered Mathematical Speech-to-LaTeX Translator.
Unified Small Language Model translator for mathematical speech.
Supports high-performance edge execution via:
  1. llama-server HTTP API (OpenAI-compatible /v1/chat/completions) - lowest latency
  2. llama-cli subprocess (standalone GGUF execution with CUDA offloading)
  3. Transformers AutoModelForCausalLM (native PyTorch fallback)
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Optional
import urllib.request

SYSTEM_PROMPT = (
    "You are an expert STEM and academic lecture transcriber and mathematical typesetter across all scientific and engineering disciplines.\n"
    "Your task is to take a spoken lecture or presentation transcript and convert mathematical formulas, symbols, physical laws, "
    "and equations into standard inline LaTeX ($...$).\n\n"
    "RULES:\n"
    "1. MATHEMATICAL & SCIENTIFIC NOTATION: Convert spoken variables, matrices, integrals, derivatives, limits, "
    "transforms, algorithms, physical constants, and equations from ANY discipline (mathematics, physics, computer science, "
    "engineering, chemistry, economics) into clean inline LaTeX ($...$).\n"
    "2. VERBATIM FIDELITY: Preserve EVERY surrounding spoken English word, explanation, presentation dialogue, "
    "and question VERBATIM. NEVER summarize, condense, delete, or rephrase non-mathematical speech. "
    "NEVER wrap normal conversational English words in LaTeX.\n"
    "3. DO NOT INVENT FORMULAS: Never fabricate or invent formulas, functions (such as g(x), s(t)), or symbols not present in the spoken transcript.\n"
    "4. CLEAN OUTPUT: Output ONLY the inline LaTeX-typeset transcript of the user message. NEVER output preamble, labels, answers, or commentary."
)


FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": "Spoken transcript:\nTaking the integral from zero to infinity of x squared times e to the power minus lambda x dx gives two divided by lambda cubed."
    },
    {
        "role": "assistant",
        "content": "Taking the integral $\\int_0^\\infty x^2 e^{-\\lambda x} \\, dx$ gives $\\frac{2}{\\lambda^3}$."
    },
    {
        "role": "user",
        "content": "Spoken transcript:\nA point x bar in R is called a minimum if f of x is greater than or equal to f of x bar for all x in R."
    },
    {
        "role": "assistant",
        "content": "A point $\\bar{x} \\in \\mathbb{R}$ is called a minimum if $f(x) \\ge f(\\bar{x})$ for all $x \\in \\mathbb{R}$."
    },
    {
        "role": "user",
        "content": "Spoken transcript:\nIn algorithms, if matrix a times vector x equals b, the Gaussian elimination runtime is big O of n cubed."
    },
    {
        "role": "assistant",
        "content": "In algorithms, if matrix $A\\mathbf{x} = \\mathbf{b}$, the Gaussian elimination runtime is $\\mathcal{O}(n^3)$."
    },
    {
        "role": "user",
        "content": "Spoken transcript:\nThe continuous-time signal x of t convolved with h of t yields y of t, which in frequency domain is X of j omega times H of j omega."
    },
    {
        "role": "assistant",
        "content": "The continuous-time signal $x(t) * h(t) = y(t)$, which in the frequency domain is $X(j\\omega) H(j\\omega)$."
    },
    {
        "role": "user",
        "content": "Spoken transcript:\nGood morning everyone. Welcome to today's presentation. Please turn to slide four where we discuss our quarterly results."
    },
    {
        "role": "assistant",
        "content": "Good morning everyone. Welcome to today's presentation. Please turn to slide four where we discuss our quarterly results."
    }
]

DEFAULT_GGUF_PATH = (
    "/home/ayush02/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    if Path("/home/ayush02/models/qwen2.5-1.5b-instruct-q4_k_m.gguf").exists()
    else "/mnt/thesis/models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"
)
DEFAULT_HF_PATH = "/mnt/thesis/models/Qwen2.5-1.5B-Instruct"
LLAMA_CLI_BIN = "/mnt/thesis/llama.cpp/build/bin/llama-cli"
LLAMA_SERVER_URL = "http://127.0.0.1:8080"


class LectureSymbolRegistry:
    """
    Lightweight, lecture-scoped symbol and domain registry.
    Tracks active continuous/discrete signal notations (e.g., x(t) vs x[n]) and
    transform domains (e.g., X(s), H(s), X(z)) across utterances, provides a
    compact KV-cache-safe hint appended after static few-shots, and applies
    conservative post-SLM domain consistency locking inside $...$ spans.
    """

    _SKIP_SYMBOLS = {
        "f", "g", "sin", "cos", "tan", "cot", "sec", "csc",
        "sinh", "cosh", "tanh", "exp", "log", "ln", "det",
        "lim", "adj", "max", "min", "sup", "inf", "arg", "Re", "Im",
    }
    _DISCRETE_ARG_RE = re.compile(
        r"^(?:[nkm]|-?\d+|[+-]?\d*[nkm](?:\s*[-+*/]\s*(?:\d+|[nkm]))?)$"
    )
    _CONTINUOUS_ARG_RE = re.compile(
        r"^(?:t|\\tau|-t|0\^?\+?|[+-]?\d*\.?\d*t(?:\s*[-+]\s*(?:\d+|\\tau|[a-zA-Z]_?\d*))?)$"
    )
    _MATH_SPAN_RE = re.compile(r"(?<!\\)\$[^$\n]+?(?<!\\)\$")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._signal_domains: dict[str, str] = {}   # symbol -> "discrete" | "continuous"
        self._symbol_counts: dict[str, int] = {}
        self._recent_symbols: list[str] = []         # ordered examples e.g. ["x[n]", "H(s)"]
        self._transforms: dict[str, str] = {}        # e.g. {"H": "s", "X": "s"}
        self.continuous_score: float = 0.0
        self.discrete_score: float = 0.0
        self.lock_repairs_count: int = 0
        self.last_repairs: list[tuple[str, str]] = []

    def _record_recent(self, notation: str) -> None:
        if notation in self._recent_symbols:
            self._recent_symbols.remove(notation)
        self._recent_symbols.append(notation)
        if len(self._recent_symbols) > 12:
            self._recent_symbols = self._recent_symbols[-12:]

    def observe(self, spoken_text: str, typeset_latex: str) -> None:
        """Update symbol and domain state from an accepted typeset utterance."""
        low = spoken_text.lower()
        explicit_disc = any(
            k in low for k in (
                "discrete-time", "discrete time", "z-transform", "z transform",
                "sequence", "up-sampling", "upsampling", "down-sampling", "downsampling",
            )
        )
        explicit_cont = any(
            k in low for k in (
                "continuous-time", "continuous time", "laplace transform",
                "differential equation", "fourier transform",
            )
        )

        # Smooth domain adaptation if lecturer explicitly transitions between domains
        if explicit_disc and not explicit_cont and self.continuous_score > self.discrete_score:
            self.continuous_score *= 0.4
            for sym, dom in list(self._signal_domains.items()):
                if dom == "continuous":
                    del self._signal_domains[sym]
                    self._symbol_counts.pop(sym, None)
        elif explicit_cont and not explicit_disc and self.discrete_score > self.continuous_score:
            self.discrete_score *= 0.4
            for sym, dom in list(self._signal_domains.items()):
                if dom == "discrete":
                    del self._signal_domains[sym]
                    self._symbol_counts.pop(sym, None)

        if explicit_disc:
            self.discrete_score += 1.0
        if explicit_cont:
            self.continuous_score += 1.0

        for span_m in self._MATH_SPAN_RE.finditer(typeset_latex or ""):
            span = span_m.group(0)[1:-1]

            # 1. Discrete signal notation: x[n], y_1[n], h[k]
            for m in re.finditer(r"\b([a-zA-Z](?:_[a-zA-Z0-9]+)?)\s*\[\s*([^\[\]\n]{1,18})\s*\]", span):
                sym, arg = m.group(1), m.group(2).strip()
                if sym in self._SKIP_SYMBOLS or sym.isupper():
                    continue
                if self._DISCRETE_ARG_RE.match(arg):
                    self._signal_domains[sym] = "discrete"
                    self._symbol_counts[sym] = self._symbol_counts.get(sym, 0) + 1
                    self.discrete_score += 1.0
                    self._record_recent(f"{sym}[{arg}]")

            # 2. Continuous signal notation: x(t), y(t), x(\tau)
            for m in re.finditer(r"\b([a-zA-Z](?:_[a-zA-Z0-9]+)?)\s*\(\s*([^()\n]{1,18})\s*\)", span):
                sym, arg = m.group(1), m.group(2).strip()
                if sym in self._SKIP_SYMBOLS:
                    continue
                if sym.isupper() and re.match(r"^(?:s|z|j\s*\\omega|\\omega)$", arg):
                    self._transforms[sym] = arg
                    self._record_recent(f"{sym}({arg})")
                elif not sym.isupper() and self._CONTINUOUS_ARG_RE.match(arg):
                    self._signal_domains[sym] = "continuous"
                    self._symbol_counts[sym] = self._symbol_counts.get(sym, 0) + 1
                    self.continuous_score += 1.0
                    self._record_recent(f"{sym}({arg})")

    def format_hint(self, spoken_text: str) -> str:
        """
        Return a compact (~20-35 token) active notation hint for the final user turn,
        leaving SYSTEM_PROMPT and FEW_SHOT_MESSAGES untouched for KV cache reuse.
        """
        if not self._recent_symbols:
            return ""

        low_words = set(re.findall(r"[a-zA-Z]+", spoken_text.lower()))
        prioritized: list[str] = []
        others: list[str] = []
        for entry in reversed(self._recent_symbols):
            base = re.split(r"[\[(]", entry, 1)[0].split("_", 1)[0].lower()
            if base in low_words:
                prioritized.append(entry)
            else:
                others.append(entry)

        selected = (prioritized + others)[:4]
        if not selected:
            return ""

        domain_note = ""
        if self.discrete_score > self.continuous_score + 0.5:
            domain_note = "discrete sequences [n]; "
        elif self.continuous_score > self.discrete_score + 0.5:
            domain_note = "continuous signals (t); "

        return f"[Active lecture notation: {domain_note}{', '.join(selected)}]"

    def apply_domain_lock(self, typeset_text: str, raw_text: str = "") -> str:
        """
        Conservatively repair cross-utterance bracket drift inside $...$ spans
        (e.g. y(1) -> y[1] when y is established as a discrete sequence y[n], or
        H[s] -> H(s) when H(s) is an active s/z-domain transform).
        Never alters t/tau continuous arguments or explicitly requested notations.
        """
        self.last_repairs = []
        if not typeset_text or "$" not in typeset_text:
            return typeset_text

        low_raw = (raw_text or "").lower()
        allow_disc_repair = "continuous" not in low_raw and "parenthes" not in low_raw

        def _rewrite_span(match: re.Match) -> str:
            inner = match.group(0)[1:-1]
            updated = inner

            # 1. Discrete signal variable accidentally emitted with (...) instead of [...]
            if allow_disc_repair:
                def _fix_disc(m: re.Match) -> str:
                    sym, arg = m.group(1), m.group(2).strip()
                    if sym in self._SKIP_SYMBOLS:
                        return m.group(0)
                    # Never rewrite t or tau continuous expressions
                    if self._CONTINUOUS_ARG_RE.match(arg):
                        return m.group(0)
                    target_sym = None
                    if self._signal_domains.get(sym) == "discrete":
                        target_sym = sym
                    elif (
                        len(sym) == 1
                        and sym.isupper()
                        and sym not in self._transforms
                        and self._signal_domains.get(sym.lower()) == "discrete"
                        and self.discrete_score > self.continuous_score
                    ):
                        target_sym = sym.lower()

                    if target_sym is not None and self._DISCRETE_ARG_RE.match(arg):
                        repl = f"{target_sym}[{arg}]"
                        self.last_repairs.append((m.group(0), repl))
                        self.lock_repairs_count += 1
                        return repl
                    return m.group(0)

                updated = re.sub(
                    r"\b([a-zA-Z](?:_[a-zA-Z0-9]+)?)\s*\(\s*([^()\n]{1,18})\s*\)",
                    _fix_disc,
                    updated,
                )

            # 2. Transform symbol accidentally emitted with [...] instead of (...)
            def _fix_tf(m: re.Match) -> str:
                sym, arg = m.group(1), m.group(2).strip()
                if sym in self._transforms and re.match(r"^(?:s|z|j\s*\\omega|\\omega)$", arg):
                    repl = f"{sym}({arg})"
                    self.last_repairs.append((m.group(0), repl))
                    self.lock_repairs_count += 1
                    return repl
                return m.group(0)

            updated = re.sub(
                r"\b([XYZHG])\s*\[\s*([^\[\]\n]{1,16})\s*\]",
                _fix_tf,
                updated,
            )
            return f"${updated}$"

        return self._MATH_SPAN_RE.sub(_rewrite_span, typeset_text)


class QwenMathTranslator:
    """Unified Small Language Model translator for mathematical speech."""

    _instance: Optional["QwenMathTranslator"] = None

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = "auto",
        device: str = "cuda",
        server_url: str = LLAMA_SERVER_URL,
        enable_cache_prompt: bool = True,
        enable_early_abort: bool = True,
        enable_symbol_lock: bool = True,
    ):
        self.device = device
        self.server_url = server_url
        self.backend = backend
        self.enable_cache_prompt = enable_cache_prompt
        self.enable_early_abort = enable_early_abort
        self.enable_symbol_lock = enable_symbol_lock
        self.last_abort_reason: Optional[str] = None
        self.last_stream_stats: dict = {}
        self.symbol_registry = LectureSymbolRegistry()
        self.model_path = None
        self.tokenizer = None
        self.model = None

        # Resolve model path
        if model_path:
            p = Path(model_path)
            if p.exists():
                self.model_path = p
        
        if self.model_path is None:
            # Check GGUF first (edge optimized)
            gguf_candidate = Path(DEFAULT_GGUF_PATH)
            if gguf_candidate.exists():
                self.model_path = gguf_candidate
            else:
                hf_candidate = Path(DEFAULT_HF_PATH)
                if hf_candidate.exists():
                    self.model_path = hf_candidate

        # Auto-detect backend
        if self.backend == "auto":
            # Check if llama-server is currently running
            if self._is_server_alive():
                self.backend = "server"
            elif self.model_path and (self.model_path.suffix == ".gguf" or self.model_path.is_file()):
                self.backend = "llama_cli"
            else:
                self.backend = "transformers"

        print(f"[SLM] Initializing QwenMathTranslator with backend='{self.backend}', model='{self.model_path}'")

        if self.backend == "transformers":
            self._init_transformers()
        elif self.backend == "server":
            print(f"[SLM] Connected to llama-server at {self.server_url}")
        elif self.backend == "llama_cli":
            if not Path(LLAMA_CLI_BIN).exists():
                print(f"[SLM Warning] llama-cli binary not yet at {LLAMA_CLI_BIN}. Will attempt when built.")

    def reset_symbol_registry(self) -> None:
        """Reset per-lecture notation/domain state at lecture boundaries."""
        self.symbol_registry.reset()

    def symbol_registry_snapshot(self) -> dict:
        """Return a serializable snapshot of active lecture notation bindings."""
        return {
            "signal_domains": dict(self.symbol_registry._signal_domains),
            "transforms": dict(self.symbol_registry._transforms),
            "recent_symbols": list(self.symbol_registry._recent_symbols),
            "continuous_score": round(self.symbol_registry.continuous_score, 2),
            "discrete_score": round(self.symbol_registry.discrete_score, 2),
            "lock_repairs_count": self.symbol_registry.lock_repairs_count,
            "last_repairs": list(self.symbol_registry.last_repairs),
        }

    def _finalize_cleaned_output(self, spoken_text: str, raw_output: str, context_text: str = "") -> str:
        """Clean model output and record valid symbol bindings without double-counting."""
        cleaned = self._clean_response(raw_output, raw_text=spoken_text, context_text=context_text)
        if self.enable_symbol_lock and self.last_abort_reason is None:
            self.symbol_registry.observe(spoken_text, cleaned)
        return cleaned

    def _is_server_alive(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.server_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _init_transformers(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.monotonic()
        dtype = torch.float16 if self.device == "cuda" and torch.cuda.is_available() else torch.float32
        print(f"[SLM] Loading HuggingFace model from {self.model_path} ({dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        print(f"[SLM] Transformers model ready in {(time.monotonic()-t0)*1000:.1f} ms.")

    @classmethod
    def get_instance(
        cls,
        model_path: Optional[str] = None,
        backend: str = "auto",
        device: str = "cuda",
    ) -> "QwenMathTranslator":
        if cls._instance is None:
            cls._instance = cls(model_path=model_path, backend=backend, device=device)
        return cls._instance

    def _build_messages(self, spoken_text: str, context: str = "", history: Optional[list] = None) -> list[dict]:
        """Construct multi-turn conversation messages ensuring zero context-echo hallucinations."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(FEW_SHOT_MESSAGES)
        if history:
            for item in history[-2:]:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    r, l = item
                    r_str, l_str = str(r).strip(), str(l).strip()
                    # Strip any corrupted tokens or repetition loops from history
                    l_str = re.sub(r"\bNative\b", "", l_str, flags=re.IGNORECASE).strip()
                    l_str = re.sub(r"(.)\1{4,}", "", l_str).strip()
                    if r_str and l_str:
                        messages.append({"role": "user", "content": f"Spoken transcript:\n{r_str}"})
                        messages.append({"role": "assistant", "content": l_str})
        elif context and str(context).strip():
            ctx_str = str(context).strip()
            messages.append({"role": "user", "content": "Spoken transcript:\n[Preceding lecture context]"})
            messages.append({"role": "assistant", "content": ctx_str})

        hint = self.symbol_registry.format_hint(spoken_text) if self.enable_symbol_lock else ""
        if hint:
            messages.append({"role": "user", "content": f"{hint}\nSpoken transcript:\n{spoken_text.strip()}"})
        else:
            messages.append({"role": "user", "content": f"Spoken transcript:\n{spoken_text.strip()}"})
        return messages

    def _build_chatml_prompt(self, spoken_text: str, context: str = "", history: Optional[list] = None) -> str:
        messages = self._build_messages(spoken_text, context=context, history=history)
        prompt = ""
        for msg in messages:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    @staticmethod
    def _strip_formatting_wrappers(text: str, context_text: str = "") -> str:
        """Strip markdown code fences, leaked prompt labels, delimiters, and context echo."""
        s = text.strip()
        s = re.sub(r"^```(?:latex|markdown)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = re.sub(r"\bNative\b", "", s, flags=re.IGNORECASE)
        s = re.sub(
            r"(?:^|\n)\s*\[?(?:Spoken transcript|Preceding context|Translation|Output|Text|Transcription)\]?:\s*",
            "\n",
            s,
            flags=re.IGNORECASE,
        )
        s = s.strip()
        s = re.sub(r"\$\$(?:\\text\{([^\}]+)\}\s*(?:\\quad\s*)?)+\$\$", r"\1", s)
        s = re.sub(r"\$(?:\\text\{([^\}]+)\}\s*(?:\\quad\s*)?)+\$", r"\1", s)
        s = re.sub(r"\$\$([^$]+)\$\$", r"$\1$", s)
        s = re.sub(r"\\\[(.*?)\\\]", r"$\1$", s, flags=re.DOTALL)
        s = re.sub(r"\\\((.*?)\\\)", r"$\1$", s, flags=re.DOTALL)
        s = re.sub(r"\$\s*\$", "", s)
        s = re.sub(r"(?<![a-zA-Z\$])\[([a-zA-Z0-9]+)\]\$", r"$x[\1]$", s)
        s = re.sub(r"\$0\s*\+\s*\\pm\$", r"$0, \\pm 1, \\pm 2, \\ldots$", s)
        s = re.sub(r"[ \t]+", " ", s).strip()

        if context_text:
            ctx_clean = context_text.strip()
            if s.startswith(ctx_clean):
                rem = s[len(ctx_clean):].strip()
                if rem:
                    s = rem
            for line in [l.strip() for l in ctx_clean.split("\n") if len(l.strip()) > 10]:
                if s.startswith(line):
                    rem = s[len(line):].strip()
                    if rem:
                        s = rem
        return s

    @staticmethod
    def _detect_guard_violation(
        candidate: str,
        raw_text: str,
        context_text: str = "",
        *,
        streaming: bool = False,
    ) -> Optional[str]:
        """
        Check whether a candidate (or partial streaming) response violates output guards.
        In streaming mode, only monotonic invariants are evaluated so early abort never
        discards a valid output that might recover once completed.
        """
        if not raw_text:
            return None

        # Avoid aborting mid-echo of context_text before full echo-stripping applies
        if streaming and context_text:
            ctx_clean = context_text.strip()
            if ctx_clean and len(candidate) <= len(ctx_clean) and ctx_clean.startswith(candidate):
                return None

        words_out = re.findall(r"\b[a-zA-Z0-9_]{3,}\b", candidate.lower())
        words_raw = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", raw_text.lower()))

        # 1. Monotonic length explosion check
        if len(words_out) > max(20, len(words_raw) * 3 + 10):
            return "length_explosion"

        # 2. Monotonic repetition loop check (ignore trailing partial token while streaming)
        eval_words = words_out[:-1] if (streaming and candidate and not candidate[-1].isspace()) else words_out
        if len(eval_words) >= 9:
            tri_grams = [" ".join(eval_words[i:i + 3]) for i in range(len(eval_words) - 2)]
            if any(cnt >= 3 for cnt in Counter(tri_grams).values()):
                return "repetition_loop"

        raw_lower = raw_text.lower()

        # 3. Monotonic prompt leakage / fabricated math checks
        physics_tokens = [r"\nabla", r"\mu \nabla", r"\partial \mathbf{v}", r"momentum equation"]
        if any(pt in candidate for pt in physics_tokens):
            if not any(pt.replace("\\", "").lower() in raw_lower for pt in physics_tokens):
                return "physics_leakage"

        if r"\begin{cases}" in candidate and not any(k in raw_lower for k in ["cases", "piecewise", "otherwise"]):
            return "fabricated_cases"

        if re.search(r"\\int(?![a-zA-Z])", candidate) and not any(k in raw_lower for k in ["integral", "integrate", "integration"]):
            return "fabricated_integral"

        if streaming:
            return None

        # Non-monotonic checks evaluated only on completed outputs
        if context_text and len(words_out) >= 5:
            words_ctx = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", context_text.lower()))
            spoken_overlap = len(set(words_out) & words_raw) / max(1, len(set(words_out)))
            context_overlap = len(set(words_out) & words_ctx) / max(1, len(set(words_out)))
            if context_overlap > 0.6 and spoken_overlap < 0.2:
                return "context_echo_hijack"

        if r"g(x)" in candidate and "g of x" not in raw_lower and "g(x)" not in raw_lower and r"\bar{f}(x)" in candidate:
            return "fabricated_g_of_x"

        return None

    def _clean_response(self, text: str, raw_text: str = "", context_text: str = "") -> str:
        s = self._strip_formatting_wrappers(text, context_text=context_text)
        violation = self._detect_guard_violation(s, raw_text, context_text=context_text, streaming=False)
        if violation is not None:
            return raw_text

        if raw_text and r"s(t)" in s and "s of t" not in raw_text.lower() and "s(t)" not in raw_text.lower():
            s = s.replace(r"s(t)", "such that")

        if self.enable_symbol_lock:
            s = self.symbol_registry.apply_domain_lock(s, raw_text=raw_text)

        return s

    def _consume_server_response(
        self,
        resp,
        spoken_text: str,
        ctx_str: str,
        t_start: float,
    ) -> str:
        """Read either an SSE streaming response with early abort or a plain JSON completion."""
        content_type = ""
        if getattr(resp, "headers", None) is not None:
            content_type = (resp.headers.get("Content-Type") or "").lower()

        if "text/event-stream" not in content_type and not self.enable_early_abort:
            resp_data = json.loads(resp.read().decode("utf-8"))
            content = resp_data["choices"][0]["message"]["content"]
            self.last_abort_reason = None
            self.last_stream_stats = {
                "aborted": False,
                "abort_reason": None,
                "chunks": 1,
                "chars_generated": len(content),
                "cache_prompt": self.enable_cache_prompt,
                "elapsed_ms": round((time.monotonic() - t_start) * 1000.0, 2),
            }
            return self._finalize_cleaned_output(spoken_text, content, context_text=ctx_str)

        chunks = []
        raw_lines_seen = 0
        for raw_line in resp:
            raw_lines_seen += 1
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            # Fallback if a mock/non-streaming server returns raw JSON without text/event-stream header
            if line.startswith("{"):
                resp_data = json.loads(line + resp.read().decode("utf-8", errors="replace"))
                content = resp_data["choices"][0]["message"]["content"]
                self.last_abort_reason = None
                self.last_stream_stats = {
                    "aborted": False,
                    "abort_reason": None,
                    "chunks": 1,
                    "chars_generated": len(content),
                    "cache_prompt": self.enable_cache_prompt,
                    "elapsed_ms": round((time.monotonic() - t_start) * 1000.0, 2),
                }
                return self._finalize_cleaned_output(spoken_text, content, context_text=ctx_str)

            if not line.startswith("data:"):
                continue
            payload_str = line[5:].strip()
            if payload_str == "[DONE]":
                break

            try:
                evt = json.loads(payload_str)
            except json.JSONDecodeError:
                continue

            choices = evt.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if piece is None and "message" in choices[0]:
                piece = (choices[0].get("message") or {}).get("content")
            if not piece:
                continue

            chunks.append(piece)

            if self.enable_early_abort and (len(chunks) % 3 == 0 or "\\" in piece or "}" in piece or piece[-1].isspace()):
                partial = self._strip_formatting_wrappers("".join(chunks), context_text=ctx_str)
                reason = self._detect_guard_violation(partial, spoken_text, context_text=ctx_str, streaming=True)
                if reason is not None:
                    self.last_abort_reason = reason
                    self.last_stream_stats = {
                        "aborted": True,
                        "abort_reason": reason,
                        "chunks": len(chunks),
                        "chars_generated": sum(len(c) for c in chunks),
                        "cache_prompt": self.enable_cache_prompt,
                        "elapsed_ms": round((time.monotonic() - t_start) * 1000.0, 2),
                    }
                    return spoken_text

        content = "".join(chunks)
        self.last_abort_reason = None
        self.last_stream_stats = {
            "aborted": False,
            "abort_reason": None,
            "chunks": len(chunks),
            "chars_generated": len(content),
            "cache_prompt": self.enable_cache_prompt,
            "elapsed_ms": round((time.monotonic() - t_start) * 1000.0, 2),
        }
        return self._finalize_cleaned_output(spoken_text, content, context_text=ctx_str)

    def _execute_server_request(
        self,
        data_bytes: bytes,
        spoken_text: str,
        ctx_str: str,
        timeout: float,
    ) -> str:
        t_start = time.monotonic()
        req = urllib.request.Request(
            f"{self.server_url}/v1/chat/completions",
            data=data_bytes,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return self._consume_server_response(resp, spoken_text, ctx_str, t_start)

    def translate_server(self, spoken_text: str, context: str = "", history: Optional[list] = None, max_new_tokens: int = 128) -> str:
        """Call llama-server via OpenAI-compatible REST API with KV prefix reuse, streaming early abort, and reconnect."""
        messages = self._build_messages(spoken_text, context=context, history=history)
        payload = {
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "repeat_penalty": 1.15,
            "stream": self.enable_early_abort,
            "stop": ["<|im_end|>", "<|endoftext|>", "\n\n"],
            "cache_prompt": self.enable_cache_prompt,
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        ctx_str = context if context else (" ".join(str(l) for _, l in history) if history else "")
        self.last_abort_reason = None
        self.last_stream_stats = {}

        try:
            return self._execute_server_request(data_bytes, spoken_text, ctx_str, timeout=60.0)
        except Exception as e:
            print(f"[SLM Server Warning] {e}. Checking server health...")
            if not self._is_server_alive():
                try:
                    main_dir = str(Path(__file__).resolve().parent.parent)
                    if main_dir not in sys.path:
                        sys.path.insert(0, main_dir)
                    try:
                        from headless_voice_agent import ensure_edge_daemons
                    except ImportError:
                        try:
                            from gui.server import ensure_edge_daemons
                        except ImportError:
                            ensure_edge_daemons = None
                    if ensure_edge_daemons:
                        ensure_edge_daemons()
                    time.sleep(3.0)
                except Exception as revive_err:
                    print(f"[SLM Revive Failed] {revive_err}")

            for _ in range(3):
                if self._is_server_alive():
                    try:
                        return self._execute_server_request(data_bytes, spoken_text, ctx_str, timeout=45.0)
                    except Exception:
                        time.sleep(1.5)
                else:
                    time.sleep(1.5)
            return spoken_text

    def translate_cli(self, spoken_text: str, context: str = "", history: Optional[list] = None, max_new_tokens: int = 128) -> str:
        """Execute translation directly through compiled llama-cli."""
        prompt = self._build_chatml_prompt(spoken_text, context=context, history=history)
        cmd = [
            LLAMA_CLI_BIN,
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", str(max_new_tokens),
            "--temp", "0.0",
            "--repeat-penalty", "1.15",
            "-ngl", "28" if self.device == "cuda" else "0",
            "-c", "4096",
            "--no-display-prompt",
            "-b", "512",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = res.stdout.strip()
        # Squeeze out any llama-cli banner output
        if "<|im_start|>assistant" in output:
            output = output.split("<|im_start|>assistant", 1)[1]
        if "<|im_end|>" in output:
            output = output.split("<|im_end|>", 1)[0]
        ctx_str = context if context else (" ".join(str(l) for _, l in history) if history else "")
        return self._finalize_cleaned_output(spoken_text, output, context_text=ctx_str)

    def translate_transformers(self, spoken_text: str, context: str = "", history: Optional[list] = None, max_new_tokens: int = 128) -> str:
        """Execute translation via HuggingFace PyTorch pipeline."""
        import torch

        messages = self._build_messages(spoken_text, context=context, history=history)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        ctx_str = context if context else (" ".join(str(l) for _, l in history) if history else "")
        return self._finalize_cleaned_output(spoken_text, response, context_text=ctx_str)

    def _has_math_cues(self, text: str) -> bool:
        """Check if spoken sentence contains mathematical terms, variables, or symbols."""
        # Strip English possessive/contraction suffixes like "today's", "it's" so "'s" is not misread as variable s
        cleaned = re.sub(r"([a-zA-Z])['’]s\b", r"\1", text)
        pattern = (
            r'\b(?:'
            r'[a-zA-Z]\s+of\s+[a-zA-Z0-9]+|'
            r'[xX]|[yY]|[zZ]|[nN]|[kK]|[sS]|[rR]|[fF]|[hH]|[uU]|[vV]|'
            r'omega|theta|phi|psi|alpha|beta|gamma|delta|lambda|pi|mu|rho|sigma|tau|'
            r'transform|transforms|transformation|fourier|laplace|z-transform|laplacian|'
            r'integral|integrals|integrate|integration|sum|summation|series|'
            r'derivative|derivatives|differential|partial|equation|equations|'
            r'matrix|matrices|vector|vectors|norm|magnitude|absolute|'
            r'pole|poles|zero|zeros|roc|convergence|converge|converges|diverge|diverges|'
            r'frequency|frequencies|domain|plane|axis|axes|circle|radius|unit|'
            r'continuous|discrete|causal|non-causal|anti-causal|impulse|step|exponential|'
            r'sin|cos|tan|log|ln|exp|sinh|cosh|root|sqrt|power|squared|cubed|'
            r'plus|minus|times|multiplied|divide|divided|over|equal|equals|greater|less|'
            r'sub|super|bracket|fraction|real|imaginary|complex'
            r')\b|\d+|[=+\-*/^<>~|]'
        )
        return bool(re.search(pattern, cleaned, re.IGNORECASE))

    def translate(self, spoken_text: str, context: str = "", history: Optional[list] = None, max_new_tokens: Optional[int] = None) -> str:
        """Translate spoken lecture transcript into clean prose + LaTeX notation with rolling context."""
        text = spoken_text.strip()
        if not text:
            return ""

        # Pre-clean spoken shorthands and meta-phrases before SLM evaluation
        text = re.sub(
            r'\b(?:I am writing a short hand s t for such that|short hand s t for such that|s t for such that|short hand s t)\b',
            'such that',
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(r'\bx\s+a\s+cube\b|\bx\s+cube\b', 'x cubed', text, flags=re.IGNORECASE)

        # If sentence has no mathematical cues or variables, return verbatim without invoking LLM
        if not self._has_math_cues(text):
            return text

        words = text.split()
        if len(words) > 45 and max_new_tokens is None:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            if len(sentences) > 1:
                chunks = []
                for i in range(0, len(sentences), 2):
                    chunks.append(" ".join(sentences[i:i+2]))
                results = []
                curr_ctx = context
                curr_hist = list(history) if history else []
                for ch in chunks:
                    ch_words = len(ch.split())
                    ch_tokens = min(512, max(180, ch_words * 4))
                    if self.backend == "server" or self._is_server_alive():
                        translated = self.translate_server(ch, context=curr_ctx, history=curr_hist, max_new_tokens=ch_tokens)
                    elif self.backend == "llama_cli" or (self.model_path and self.model_path.suffix == ".gguf"):
                        translated = self.translate_cli(ch, context=curr_ctx, history=curr_hist, max_new_tokens=ch_tokens)
                    else:
                        translated = self.translate_transformers(ch, context=curr_ctx, history=curr_hist, max_new_tokens=ch_tokens)
                    results.append(translated)
                    curr_ctx = translated
                    curr_hist = [(ch, translated)]
                    time.sleep(0.05)
                return " ".join(results)

        if max_new_tokens is None:
            max_new_tokens = min(1536, max(256, len(words) * 4))

        if self.backend == "server" or self._is_server_alive():
            try:
                return self.translate_server(text, context=context, history=history, max_new_tokens=max_new_tokens)
            except Exception as e:
                print(f"[SLM Warning] translate_server failed: {e}. Falling back to CLI/Transformers...")
                if self.model_path and (self.model_path.suffix == ".gguf" or self.model_path.is_file()):
                    return self.translate_cli(text, context=context, history=history, max_new_tokens=max_new_tokens)
                return self.translate_transformers(text, context=context, history=history, max_new_tokens=max_new_tokens)
        elif self.backend == "llama_cli" or (self.model_path and self.model_path.suffix == ".gguf"):
            return self.translate_cli(text, context=context, history=history, max_new_tokens=max_new_tokens)
        else:
            return self.translate_transformers(text, context=context, history=history, max_new_tokens=max_new_tokens)


def main():
    parser = argparse.ArgumentParser(description="Test SLM Spoken-Math-to-LaTeX Translator")
    parser.add_argument("text", type=str, nargs="?", default="ax plus by plus cz equals d", help="Spoken text to transcribe")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--backend", type=str, choices=["auto", "server", "llama_cli", "transformers"], default="auto")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    translator = QwenMathTranslator.get_instance(model_path=args.model_path, backend=args.backend, device=args.device)
    t0 = time.monotonic()
    result = translator.translate(args.text)
    latency_ms = (time.monotonic() - t0) * 1000.0

    print("\n" + "=" * 60)
    print(f"[Input]:   {args.text}")
    print(f"[Output]:  {result}")
    print(f"[Latency]: {latency_ms:.1f} ms")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
