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
    "You are an expert speech-to-LaTeX transcriber for university-level mathematics and engineering lectures.\n"
    "Your task: Given a raw spoken transcript from a microphone, produce the clean final text where all mathematical "
    "expressions, variables, formulas, equations, and calculus notation are formatted in standard LaTeX notation (enclosed in $...$).\n"
    "Rules:\n"
    "1. Convert spoken math into concise, correct LaTeX (e.g. 'v equals i times r' -> '$V = IR$', 'ax plus by plus cz equals d' -> '$ax + by + cz = d$').\n"
    "2. If an utterance is conversational or descriptive English without formulas (e.g. 'The voltage across a resistor depends on current'), keep it in plain English.\n"
    "3. If an utterance mixes English with math (e.g. 'We know that x plus y equals ten so we can solve for y'), keep the English words and enclose the math formula in $...$ (e.g. 'We know that $x + y = 10$ so we can solve for $y$.').\n"
    "4. Fix phonetic ASR misunderstandings (e.g. 'meter respect to' -> 'with respect to', 'quote' -> '=', 'side of x' -> '\\sin(x)').\n"
    "5. Output ONLY the transcribed result. Do not output any explanation, notes, or preamble."
)

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
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nSpoken transcript:\n{spoken_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _clean_response(self, text: str) -> str:
        s = text.strip()
        # Remove accidental markdown code fences
        s = re.sub(r"^```(?:latex|markdown)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        # Remove conversational meta prefixes
        s = re.sub(r"^(?:Transcribed(?:\s+result)?|Output|Text|Transcription):\s*", "", s, flags=re.IGNORECASE)
        return s.strip()

    def translate_server(self, spoken_text: str, max_new_tokens: int = 128) -> str:
        """Call llama-server via OpenAI-compatible REST API."""
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Spoken transcript:\n{spoken_text}"},
            ],
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

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Spoken transcript:\n{spoken_text}"},
        ]
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

        if max_new_tokens is None:
            # Scaled token budget: minimum 256 tokens, up to 1536 tokens for long lectures
            word_count = len(text.split())
            max_new_tokens = min(1536, max(256, word_count * 4))

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
