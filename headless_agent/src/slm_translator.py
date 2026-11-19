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
    "3. CLEAN OUTPUT: Output ONLY the inline LaTeX-typeset transcript of the user message. NEVER output preamble, labels, answers, or commentary."
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


class QwenMathTranslator:
    """Unified Small Language Model translator for mathematical speech."""

    _instance: Optional["QwenMathTranslator"] = None

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = "auto",
        device: str = "cuda",
        server_url: str = LLAMA_SERVER_URL,
    ):
        self.device = device
        self.server_url = server_url
        self.backend = backend
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

        messages.append({"role": "user", "content": f"Spoken transcript:\n{spoken_text.strip()}"})
        return messages

    def _build_chatml_prompt(self, spoken_text: str, context: str = "", history: Optional[list] = None) -> str:
        messages = self._build_messages(spoken_text, context=context, history=history)
        prompt = ""
        for msg in messages:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def _clean_response(self, text: str, raw_text: str = "", context_text: str = "") -> str:
        s = text.strip()
        # Remove accidental markdown code fences
        s = re.sub(r"^```(?:latex|markdown)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        # Strip Native and repetitive hallucinations
        s = re.sub(r"\bNative\b", "", s, flags=re.IGNORECASE)
        # Remove conversational meta prefixes and leaked prompt markers
        s = re.sub(r"(?:^|\n)\s*\[?(?:Spoken transcript|Preceding context|Translation|Output|Text|Transcription)\]?:\s*", "\n", s, flags=re.IGNORECASE)
        s = s.strip()
        # Unpack accidental \text{...} wrappers around non-math speech
        s = re.sub(r"\$\$(?:\\text\{([^\}]+)\}\s*(?:\\quad\s*)?)+\$\$", r"\1", s)
        s = re.sub(r"\$(?:\\text\{([^\}]+)\}\s*(?:\\quad\s*)?)+\$", r"\1", s)
        # Convert any display math delimiters $$...$$ or \[...\] or \(...\) to inline $...$
        s = re.sub(r"\$\$([^$]+)\$\$", r"$\1$", s)
        s = re.sub(r"\\\[(.*?)\\\]", r"$\1$", s, flags=re.DOTALL)
        s = re.sub(r"\\\((.*?)\\\)", r"$\1$", s, flags=re.DOTALL)
        # Collapse empty $ $
        s = re.sub(r"\$\s*\$", "", s)
        # Fix unbracketed or dropped variable symbol before [n]$
        s = re.sub(r"(?<![a-zA-Z\$])\[([a-zA-Z0-9]+)\]\$", r"$x[\1]$", s)
        # Clean illegal math artifacts like $0 + \pm$ or $0 - 2 - 4 + ...$
        s = re.sub(r"\$0\s*\+\s*\\pm\$", r"$0, \\pm 1, \\pm 2, \\ldots$", s)
        # Collapse multiple spaces
        s = re.sub(r"[ \t]+", " ", s).strip()

        # Context echo stripping: remove verbatim context prefix echoed by model
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

        # Hallucination & fidelity checks if raw_text is provided
        if raw_text:
            words_out = re.findall(r"\b[a-zA-Z0-9_]{3,}\b", s.lower())
            words_raw = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", raw_text.lower()))
            # Length explosion check
            if len(words_out) > max(20, len(words_raw) * 3 + 10):
                return raw_text
            # Repetition loop check
            if len(words_out) >= 9:
                tri_grams = [" ".join(words_out[i:i+3]) for i in range(len(words_out)-2)]
                counts = Counter(tri_grams)
                if any(cnt >= 3 for cnt in counts.values()):
                    return raw_text
            # Context echo hijacking check
            if context_text and len(words_out) >= 5:
                words_ctx = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", context_text.lower()))
                spoken_overlap = len(set(words_out) & words_raw) / max(1, len(set(words_out)))
                context_overlap = len(set(words_out) & words_ctx) / max(1, len(set(words_out)))
                if context_overlap > 0.6 and spoken_overlap < 0.2:
                    return raw_text

            # Prompt leakage / cross-domain physics formula hallucination check
            physics_tokens = [r"\nabla", r"\mu \nabla", r"\partial \mathbf{v}", r"momentum equation"]
            if any(pt in s for pt in physics_tokens):
                raw_lower = raw_text.lower()
                if not any(pt.replace("\\", "").lower() in raw_lower for pt in physics_tokens):
                    return raw_text

        return s

    def translate_server(self, spoken_text: str, context: str = "", history: Optional[list] = None, max_new_tokens: int = 128) -> str:
        """Call llama-server via OpenAI-compatible REST API with automatic reconnect/revive."""
        messages = self._build_messages(spoken_text, context=context, history=history)
        payload = {
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "repeat_penalty": 1.15,
            "stream": False,
            "stop": ["<|im_end|>", "<|endoftext|>", "\n\n"],
            "cache_prompt": True,
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.server_url}/v1/chat/completions",
            data=data_bytes,
            headers={"Content-Type": "application/json"},
        )
        ctx_str = context if context else (" ".join(str(l) for _, l in history) if history else "")
        try:
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                content = resp_data["choices"][0]["message"]["content"]
                return self._clean_response(content, raw_text=spoken_text, context_text=ctx_str)
        except Exception as e:
            print(f"[SLM Server Warning] {e}. Checking server health...")
            # Auto-revive daemon if down
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
                    time.sleep(2.0)
                except Exception as revive_err:
                    print(f"[SLM Revive Failed] {revive_err}")
            # Retry loop with backoff
            for attempt in range(3):
                if self._is_server_alive():
                    try:
                        with urllib.request.urlopen(req, timeout=45.0) as resp:
                            resp_data = json.loads(resp.read().decode("utf-8"))
                            content = resp_data["choices"][0]["message"]["content"]
                            return self._clean_response(content, raw_text=spoken_text, context_text=ctx_str)
                    except Exception:
                        time.sleep(1.0)
                else:
                    time.sleep(1.0)
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
        return self._clean_response(output, raw_text=spoken_text, context_text=ctx_str)

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
        return self._clean_response(response, raw_text=spoken_text, context_text=ctx_str)

    def _has_math_cues(self, text: str) -> bool:
        """Check if spoken sentence contains mathematical terms, variables, or symbols."""
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
        return bool(re.search(pattern, text, re.IGNORECASE))

    def translate(self, spoken_text: str, context: str = "", history: Optional[list] = None, max_new_tokens: Optional[int] = None) -> str:
        """Translate spoken lecture transcript into clean prose + LaTeX notation with rolling context."""
        text = spoken_text.strip()
        if not text:
            return ""

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
