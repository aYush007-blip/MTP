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
    "You are a verbatim lecture typesetter. Your job is to output the EXACT spoken words in the EXACT order spoken, converting ONLY mathematical terms, variables, and formulas into inline LaTeX ($...$).\n\n"
    "STRICT RULES:\n"
    "1. NEVER delete, omit, summarize, or paraphrase any spoken words or explanations. Every single spoken word must appear in the output.\n"
    "2. NEVER invent, solve, derive, or add mathematical equations, proofs, or steps that were not spoken by the speaker.\n"
    "3. NEVER wrap non-math words, conversational phrases, numbers, or counting in LaTeX or \\text{}. For example, 'Four, two, one, zero. Okay, here is the short lecture.' must remain as plain text.\n"
    "4. Use ONLY inline math ($...$). NEVER use display math ($$...$$ or \\( ... \\)).\n"
    "5. Output ONLY the resulting transcript. Do not output preamble, notes, or explanations."
)

FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": "Spoken transcript:\nFour, three, two, one. Okay, here is the short lecture. Good evening."
    },
    {
        "role": "assistant",
        "content": "Four, three, two, one. Okay, here is the short lecture. Good evening."
    },
    {
        "role": "user",
        "content": "Spoken transcript:\nToday we are looking at the Laplace transform. It takes a time function, f of t, and turns it into a function of a complex variable, s. The definition is the integral from zero to infinity of f of t times e to the minus s t dt."
    },
    {
        "role": "assistant",
        "content": "Today we are looking at the Laplace transform. It takes a time function, $f(t)$, and turns it into a function of a complex variable, $s$. The definition is the integral from zero to infinity of $\\int_{0}^{\\infty} f(t) e^{-st} \\, dt$."
    },
    {
        "role": "user",
        "content": "Spoken transcript:\nFor instance, if dy by dt plus two y equals one with y of zero equals zero, applying the transform gives sy of s plus two y of s equals one over s."
    },
    {
        "role": "assistant",
        "content": "For instance, if $\\frac{dy}{dt} + 2y = 1$ with $y(0) = 0$, applying the transform gives $s y(s) + 2 y(s) = \\frac{1}{s}$."
    },
    {
        "role": "user",
        "content": "Spoken transcript:\nThat is the kind of result we use over and over in circuits, control, and signal processing."
    },
    {
        "role": "assistant",
        "content": "That is the kind of result we use over and over in circuits, control, and signal processing."
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

    def _build_chatml_prompt(self, spoken_text: str) -> str:
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        for msg in FEW_SHOT_MESSAGES:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
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
        # Collapse multiple spaces
        s = re.sub(r"[ \t]+", " ", s)
        return s.strip()

    def translate_server(self, spoken_text: str, max_new_tokens: int = 128) -> str:
        """Call llama-server via OpenAI-compatible REST API."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(FEW_SHOT_MESSAGES)
        messages.append({"role": "user", "content": f"Spoken transcript:\n{spoken_text}"})
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            content = resp_data["choices"][0]["message"]["content"]
            return self._clean_response(content)

    def translate_cli(self, spoken_text: str, max_new_tokens: int = 128) -> str:
        """Execute translation directly through compiled llama-cli."""
        prompt = self._build_chatml_prompt(spoken_text)
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

    def translate_transformers(self, spoken_text: str, max_new_tokens: int = 128) -> str:
        """Execute translation via HuggingFace PyTorch pipeline."""
        import torch

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(FEW_SHOT_MESSAGES)
        messages.append({"role": "user", "content": f"Spoken transcript:\n{spoken_text}"})
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

    def translate(self, spoken_text: str, max_new_tokens: Optional[int] = None) -> str:
        """Translate spoken lecture transcript into clean prose + LaTeX notation."""
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
                for ch in chunks:
                    ch_words = len(ch.split())
                    ch_tokens = min(512, max(180, ch_words * 4))
                    if self.backend == "server" or self._is_server_alive():
                        translated = self.translate_server(ch, max_new_tokens=ch_tokens)
                    elif self.backend == "llama_cli" or (self.model_path and self.model_path.suffix == ".gguf"):
                        translated = self.translate_cli(ch, max_new_tokens=ch_tokens)
                    else:
                        translated = self.translate_transformers(ch, max_new_tokens=ch_tokens)
                    results.append(translated)
                    time.sleep(0.05)  # brief power-rail recovery to prevent VDD_IN current spike
                return " ".join(results)

        if max_new_tokens is None:
            # Scaled token budget: minimum 256 tokens, up to 1536 tokens for long lectures
            max_new_tokens = min(1536, max(256, len(words) * 4))

        if self.backend == "server" or self._is_server_alive():
            return self.translate_server(text, max_new_tokens=max_new_tokens)
        elif self.backend == "llama_cli" or (self.model_path and self.model_path.suffix == ".gguf"):
            return self.translate_cli(text, max_new_tokens=max_new_tokens)
        else:
            return self.translate_transformers(text, max_new_tokens=max_new_tokens)


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
