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
    "You are an expert STEM lecture transcriber and mathematical typesetter across all engineering disciplines "
    "(Mechanical, Civil, Electrical, Chemical, Aerospace, Computer Science, and Applied Mathematics).\n"
    "Your task is to take a spoken lecture transcript and convert it into a natural, continuous transcript where spoken mathematical formulas, symbols, physical laws, and equations are accurately converted into standard inline LaTeX ($...$).\n\n"
    "STRICT UNIVERSAL ENGINEERING RULES:\n"
    "1. MATHEMATICAL FORMULAS & SYMBOLS ($...$):\n"
    "   - Function & Signal Notation:\n"
    "     * Continuous coordinates/functions use parentheses: e.g. 'f of x' -> '$f(x)$', 'u of x and t' -> '$u(x, t)$', 't of r' -> '$T(r)$', 'psi of t' -> '$\\psi(t)$', 'h of s' -> '$H(s)$', 'g of j omega' -> '$G(j\\omega)$'.\n"
    "     * Discrete sequences/indices use brackets: e.g. 'x of n' -> '$x[n]$', 'y of n' -> '$y[n]$', 'y of 0' -> '$y[0]$', 'x of 0' -> '$x[0]$', 'x of 1' -> '$x[1]$', 'y of 1' -> '$y[1]$', 'h of k' -> '$h[k]$', 'x of 2n' -> '$x[2n]$'.\n"
    "     * Inner argument fractions: 'x of n over 2' -> '$x[n/2]$', 'x of n over capital N' -> '$x[n/N]$'. Do NOT divide the entire function by the denominator unless explicitly stated outside the argument.\n"
    "   - Derivatives, Partials & Calculus:\n"
    "     * 'd y by d t' or 'derivative of y with respect to t' -> '$\\frac{dy}{dt}$'.\n"
    "     * 'partial u with respect to x' -> '$\\frac{\\partial u}{\\partial x}$', 'partial squared t with respect to x squared' -> '$\\frac{\\partial^2 T}{\\partial x^2}$'.\n"
    "     * Vector calculus: 'grad p' -> '$\\nabla p$', 'del squared v' -> '$\\nabla^2 \\mathbf{v}$', 'curl of e' -> '$\\nabla \\times \\mathbf{E}$', 'divergence of b' -> '$\\nabla \\cdot \\mathbf{B}$'.\n"
    "     * Integrals & Summations: 'integral from a to b of f of x dx' -> '$\\int_{a}^{b} f(x) \\, dx$', 'sum from i equals 1 to n of x i' -> '$\\sum_{i=1}^{n} x_i$'.\n"
    "   - Physical Laws & Conservation Equations:\n"
    "     * Fluids: 'rho times partial v with respect to t plus v dot grad v equals minus grad p plus mu del squared v' -> '$\\rho \\left(\\frac{\\partial \\mathbf{v}}{\\partial t} + \\mathbf{v} \\cdot \\nabla \\mathbf{v}\\right) = -\\nabla p + \\mu \\nabla^2 \\mathbf{v}$'.\n"
    "     * Thermodynamics: 'd u equals delta q minus delta w' -> '$dU = \\delta Q - \\delta W$', 'p v equals n r t' -> '$PV = nRT$'.\n"
    "     * Mechanics & Materials: 'stress sigma equals young\\'s modulus e times strain epsilon' -> '$\\sigma = E \\epsilon$', 'f equals m a' -> '$\\mathbf{F} = m\\mathbf{a}$', 'tau equals mu du by dy' -> '$\\tau = \\mu \\frac{du}{dy}$'.\n"
    "     * Control & Circuits: 'v equals i times r' -> '$V = IR$', 'h of s equals omega n squared over s squared plus two zeta omega n s plus omega n squared' -> '$H(s) = \\frac{\\omega_n^2}{s^2 + 2\\zeta\\omega_n s + \\omega_n^2}$'.\n"
    "     * Algorithms & Probability: 't of n equals 2 t of n over 2 plus o of n' -> '$T(n) = 2T(n/2) + O(n)$', 'p of a given b equals p of b given a times p of a over p of b' -> '$P(A|B) = \\frac{P(B|A)P(A)}{P(B)}$'.\n"
    "   - Spoken Engineering Units: '50 mega pascals' -> '$50\\,\\text{MPa}$', '10 kilo hertz' -> '$10\\,\\text{kHz}$', '300 kelvin' -> '$300\\,\\text{K}$'.\n"
    "2. PRESERVE 100% of surrounding English explanations, conversational context, and engineering terminology ('boundary condition', 'steady state', 'control volume', 'eigenvalue', 'downsampling').\n"
    "3. NEVER wrap plain conversational phrases, numbering, or countdowns in LaTeX (e.g. 'Turn to page twenty. Step one, two, three.' must remain plain text).\n"
    "4. Clean verbal false starts, repetitions, and stammers into the intended mathematical form. NEVER produce broken or invalid LaTeX ($0 + \\pm$ or $\\frac{n}{n}$ from verbal hesitation).\n"
    "5. Use ONLY inline math ($...$). NEVER output display math ($$...$$ or \\( ... \\)). Output ONLY the resulting transcript."
)


FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": "Spoken transcript:\nConsider the Navier-Stokes momentum equation for incompressible flow: rho times open parenthesis partial v with respect to t plus v dot grad v close parenthesis equals minus grad p plus mu del squared v."
    },
    {
        "role": "assistant",
        "content": "Consider the Navier-Stokes momentum equation for incompressible flow: $\\rho \\left(\\frac{\\partial \\mathbf{v}}{\\partial t} + \\mathbf{v} \\cdot \\nabla \\mathbf{v}\\right) = -\\nabla p + \\mu \\nabla^2 \\mathbf{v}$."
    },
    {
        "role": "user",
        "content": "Spoken transcript:\nBy the first law of thermodynamics, the change in internal energy d u equals heat added delta q minus work done delta w."
    },
    {
        "role": "assistant",
        "content": "By the first law of thermodynamics, the change in internal energy $dU = \\delta Q - \\delta W$."
    },
    {
        "role": "user",
        "content": "Spoken transcript:\nFor discrete-time scaling, the sequence y of n equals x of 2n represents downsampling by two, while y of n equals x of n over capital N represents expansion."
    },
    {
        "role": "assistant",
        "content": "For discrete-time scaling, the sequence $y[n] = x[2n]$ represents downsampling by two, while $y[n] = x[n/N]$ represents expansion."
    },
    {
        "role": "user",
        "content": "Spoken transcript:\nIn structural mechanics, under elastic deformation, normal stress sigma equals Young's modulus e times strain epsilon, while shear stress tau equals mu times d u by d y."
    },
    {
        "role": "assistant",
        "content": "In structural mechanics, under elastic deformation, normal stress $\\sigma = E \\epsilon$, while shear stress $\\tau = \\mu \\frac{du}{dy}$."
    },
    {
        "role": "user",
        "content": "Spoken transcript:\nThe closed loop transfer function is h of s equals omega n squared over s squared plus two zeta omega n s plus omega n squared with poles at s equals minus two."
    },
    {
        "role": "assistant",
        "content": "The closed-loop transfer function is $H(s) = \\frac{\\omega_n^2}{s^2 + 2\\zeta\\omega_n s + \\omega_n^2}$ with poles at $s = -2$."
    },
    {
        "role": "user",
        "content": "Spoken transcript:\nThe definition of the one-sided Laplace transform of f of t is the integral from zero to infinity of f of t times e to the minus s t dt."
    },
    {
        "role": "assistant",
        "content": "The definition of the one-sided Laplace transform of $f(t)$ is $\\int_{0}^{\\infty} f(t) e^{-st} \\, dt$."
    },
    {
        "role": "user",
        "content": "Spoken transcript:\nBy Bayes' rule, the posterior probability p of a given b equals p of b given a times p of a divided by p of b."
    },
    {
        "role": "assistant",
        "content": "By Bayes' rule, the posterior probability $P(A|B) = \\frac{P(B|A)P(A)}{P(B)}$."
    },
    {
        "role": "user",
        "content": "Spoken transcript:\nGood morning everyone, welcome to lecture ten. Please turn to page forty five and make sure your homework is submitted by five pm."
    },
    {
        "role": "assistant",
        "content": "Good morning everyone, welcome to lecture ten. Please turn to page forty five and make sure your homework is submitted by five pm."
    }
]

DEFAULT_GGUF_PATH = "/mnt/thesis/models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"
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

    def _build_chatml_prompt(self, spoken_text: str, context: str = "") -> str:
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        for msg in FEW_SHOT_MESSAGES:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        if context:
            prompt += f"<|im_start|>user\n[Preceding context]: {context}\n[Spoken transcript]: {spoken_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt += f"<|im_start|>user\nSpoken transcript:\n{spoken_text}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def _clean_response(self, text: str) -> str:
        s = text.strip()
        # Remove accidental markdown code fences
        s = re.sub(r"^```(?:latex|markdown)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        # Remove conversational meta prefixes
        s = re.sub(r"^(?:Transcribed(?:\s+result)?|Output|Text|Transcription):\s*", "", s, flags=re.IGNORECASE)
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
        s = re.sub(r"[ \t]+", " ", s)
        return s.strip()

    def translate_server(self, spoken_text: str, context: str = "", max_new_tokens: int = 128) -> str:
        """Call llama-server via OpenAI-compatible REST API with automatic reconnect/revive."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(FEW_SHOT_MESSAGES)
        if context:
            user_msg = f"[Preceding context]: {context}\n[Spoken transcript]: {spoken_text}"
        else:
            user_msg = f"Spoken transcript:\n{spoken_text}"
        messages.append({"role": "user", "content": user_msg})
        payload = {
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "repeat_penalty": 1.1,
            "stream": False,
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.server_url}/v1/chat/completions",
            data=data_bytes,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                content = resp_data["choices"][0]["message"]["content"]
                return self._clean_response(content)
        except Exception as e:
            print(f"[SLM Server Warning] {e}. Checking server health...")
            # Auto-revive daemon if down
            if not self._is_server_alive():
                try:
                    from gui.server import ensure_edge_daemons
                    ensure_edge_daemons()
                    time.sleep(1.0)
                except Exception as revive_err:
                    print(f"[SLM Revive Failed] {revive_err}")
            # Retry once
            if self._is_server_alive():
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    content = resp_data["choices"][0]["message"]["content"]
                    return self._clean_response(content)
            raise

    def translate_cli(self, spoken_text: str, context: str = "", max_new_tokens: int = 128) -> str:
        """Execute translation directly through compiled llama-cli."""
        prompt = self._build_chatml_prompt(spoken_text, context=context)
        cmd = [
            LLAMA_CLI_BIN,
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", str(max_new_tokens),
            "--temp", "0.0",
            "--repeat-penalty", "1.1",
            "-ngl", "28" if self.device == "cuda" else "0",
            "-c", "2048",
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
        return self._clean_response(output)

    def translate_transformers(self, spoken_text: str, context: str = "", max_new_tokens: int = 128) -> str:
        """Execute translation via HuggingFace PyTorch pipeline."""
        import torch

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(FEW_SHOT_MESSAGES)
        if context:
            user_msg = f"[Preceding context]: {context}\n[Spoken transcript]: {spoken_text}"
        else:
            user_msg = f"Spoken transcript:\n{spoken_text}"
        messages.append({"role": "user", "content": user_msg})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        return self._clean_response(response)

    def translate(self, spoken_text: str, context: str = "", max_new_tokens: Optional[int] = None) -> str:
        """Translate spoken lecture transcript into clean prose + LaTeX notation with rolling context."""
        text = spoken_text.strip()
        if not text:
            return ""

        words = text.split()
        # For long continuous lecture transcripts (>45 words), chunk by sentence to:
        # 1. Prevent over-current throttling (micro-burst compute with power recovery)
        # 2. Maintain 100% mathematical precision without truncation
        # 3. Maintain exact sequential spoken order (text + latex + text...)
        if len(words) > 45 and max_new_tokens is None:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            if len(sentences) > 1:
                chunks = []
                for i in range(0, len(sentences), 2):
                    chunks.append(" ".join(sentences[i:i+2]))
                results = []
                curr_ctx = context
                for ch in chunks:
                    ch_words = len(ch.split())
                    ch_tokens = min(512, max(180, ch_words * 4))
                    if self.backend == "server" or self._is_server_alive():
                        translated = self.translate_server(ch, context=curr_ctx, max_new_tokens=ch_tokens)
                    elif self.backend == "llama_cli" or (self.model_path and self.model_path.suffix == ".gguf"):
                        translated = self.translate_cli(ch, context=curr_ctx, max_new_tokens=ch_tokens)
                    else:
                        translated = self.translate_transformers(ch, context=curr_ctx, max_new_tokens=ch_tokens)
                    results.append(translated)
                    curr_ctx = translated
                    time.sleep(0.05)  # brief power-rail recovery to prevent VDD_IN current spike
                return " ".join(results)

        if max_new_tokens is None:
            # Scaled token budget: minimum 256 tokens, up to 1536 tokens for long lectures
            max_new_tokens = min(1536, max(256, len(words) * 4))

        if self.backend == "server" or self._is_server_alive():
            try:
                return self.translate_server(text, context=context, max_new_tokens=max_new_tokens)
            except Exception as e:
                print(f"[SLM Warning] translate_server failed: {e}. Falling back to CLI/Transformers...")
                if self.model_path and (self.model_path.suffix == ".gguf" or self.model_path.is_file()):
                    return self.translate_cli(text, context=context, max_new_tokens=max_new_tokens)
                return self.translate_transformers(text, context=context, max_new_tokens=max_new_tokens)
        elif self.backend == "llama_cli" or (self.model_path and self.model_path.suffix == ".gguf"):
            return self.translate_cli(text, context=context, max_new_tokens=max_new_tokens)
        else:
            return self.translate_transformers(text, context=context, max_new_tokens=max_new_tokens)


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
