#!/usr/bin/env python3
"""
Hindi Technical Translation Engine for STEM Lectures & Academic Presentations.

Translates English lecture transcripts and textbook monographs into natural,
fluent Hindi (Devanagari script) to aid Indian students, while enforcing
100% strict preservation of all LaTeX formulas ($...$, $$...$$), symbols,
and mathematical variables.
"""

import json
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Tuple
import urllib.request

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")

HINDI_SYSTEM_PROMPT = (
    "You are a professional STEM Hindi translator. Translate English academic lectures and presentations into "
    "natural, fluent Hindi (Devanagari script) to aid Indian college students.\n"
    "CRITICAL RULES:\n"
    "1. Keep all LaTeX equations ($...$, $$...$$), variables (e.g. $x(t)$, $X(s)$, $h(t)$, $A$, $\\omega$), "
    "and numbers 100% verbatim and unchanged.\n"
    "2. Translate the surrounding English prose into natural Hindi. Keep standard English technical terms in parentheses for pedagogical clarity.\n"
    "3. Output ONLY the translated Hindi text. Do NOT add meta commentary, explanations, or quotes."
)

HINDI_FEW_SHOTS = [
    {
        "role": "system",
        "content": HINDI_SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": "Taking the integral from zero to infinity of $x^2 e^{-\\lambda x} \\, dx$ gives $\\frac{2}{\\lambda^3}$."
    },
    {
        "role": "assistant",
        "content": "शून्य से अनंत तक $x^2 e^{-\\lambda x} \\, dx$ का समाकलन (integral) लेने पर $\\frac{2}{\\lambda^3}$ प्राप्त होता है।"
    },
    {
        "role": "user",
        "content": "A point $x^* \\in \\mathbb{R}$ is called a minimum if $f(x) \\ge f(x^*)$ for all $x \\in \\mathbb{R}$."
    },
    {
        "role": "assistant",
        "content": "एक बिंदु $x^* \\in \\mathbb{R}$ को न्यूनतम (minimum) कहा जाता है यदि सभी $x \\in \\mathbb{R}$ के लिए $f(x) \\ge f(x^*)$ हो।"
    },
    {
        "role": "user",
        "content": "Good morning everyone. Welcome to today's presentation on algorithms."
    },
    {
        "role": "assistant",
        "content": "शुभ प्रभात सभी को। एल्गोरिदम पर आज की प्रस्तुति में आपका स्वागत है।"
    }
]


class HindiLectureTranslator:
    """
    Cross-lingual technical translation coordinator.
    Interfaces with the local Qwen2.5-1.5B edge model on llama-server.
    """

    _instance: Optional["HindiLectureTranslator"] = None

    def __init__(self, server_url: str = LLAMA_SERVER_URL):
        self.server_url = server_url.rstrip("/")

    @classmethod
    def get_instance(cls, server_url: str = LLAMA_SERVER_URL) -> "HindiLectureTranslator":
        if cls._instance is None:
            cls._instance = cls(server_url=server_url)
        return cls._instance

    def is_server_alive(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.server_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def translate_text(self, text: str, max_tokens: int = 512) -> str:
        """
        Translates an English technical paragraph into Hindi, preserving LaTeX math.
        """
        clean_in = text.strip()
        if not clean_in or len(clean_in) < 3:
            return clean_in

        prompt = ""
        for m in HINDI_FEW_SHOTS:
            prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{clean_in}<|im_end|>\n<|im_start|>assistant\n"

        token_budget = min(max_tokens, max(96, len(clean_in.split()) * 4 + 64))
        payload = {
            "prompt": prompt,
            "temperature": 0.15,
            "repeat_penalty": 1.15,
            "n_predict": token_budget,
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "cache_prompt": False,
        }

        for attempt in range(2):
            try:
                req = urllib.request.Request(
                    f"{self.server_url}/completion",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=120.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    raw_translation = data.get("content", "").strip()
                    if raw_translation:
                        # Clean any leading meta prefixes
                        clean_res = re.sub(
                            r"^(?:यहाँ अनुवाद है|अनुवाद|हिंदी अनुवाद|Here is the translation|Translation):\s*",
                            "",
                            raw_translation,
                            flags=re.IGNORECASE
                        ).strip()
                        # Strip degenerate repetitive phrase loops (repeated 15+ char clauses)
                        clean_res = re.sub(r'(.{15,}?)(?:\s+\1){2,}', r'\1', clean_res)
                        return clean_res
            except Exception as e:
                err_str = str(e)
                if ("Connection refused" in err_str or "Remote end closed" in err_str) and attempt == 0:
                    print(f"[HindiTranslator] Connection dropped ({err_str}). Reviving daemon...")
                    try:
                        from test_system_wav import ensure_llama_server
                        ensure_llama_server()
                        time.sleep(2.0)
                        continue
                    except Exception:
                        pass
                print(f"[HindiTranslator] Translation request failed: {e}. Falling back to original text.")
                break

        return clean_in

    def translate_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Translates a list of cohesive English lecture paragraphs into Hindi."""
        print(f"[HindiTranslator] Translating {len(paragraphs)} monograph sections into Hindi...")
        translated = []
        for i, p in enumerate(paragraphs, 1):
            if not p.strip():
                continue
            t0 = time.monotonic()
            trans = self.translate_text(p)
            dt = time.monotonic() - t0
            print(f"[HindiTranslator] Section {i}/{len(paragraphs)} translated in {dt:.1f}s")
            translated.append(trans)
        return translated

    def translate_subtitles_batch(self, lines: List[str]) -> List[str]:
        """
        Translates a batch of short subtitle segments into natural Hindi in a single LLM call.
        Preserves all LaTeX mathematical expressions ($...$) verbatim.
        """
        if not lines:
            return []

        numbered = "\n".join(f"{i+1}. {line.strip()}" for i, line in enumerate(lines))
        prompt = (
            "<|im_start|>system\n"
            "You are a professional Hindi translator for academic STEM lectures. "
            "Translate each numbered line into natural Hindi (Devanagari script) preserving line numbers "
            "and keeping all LaTeX math ($...$) verbatim. Output ONLY the numbered lines.<|im_end|>\n"
            "<|im_start|>user\n"
            "Translate each numbered subtitle line into natural Hindi for college students.\n"
            "Keep all LaTeX equations ($...$, $$...$$), variables, and numbers 100% verbatim.\n"
            "Maintain exact line numbering (1., 2., etc.). Output ONLY the numbered translations:\n\n"
            f"{numbered}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        token_budget = min(1024, max(128, sum(len(l.split()) for l in lines) * 4 + 64))
        payload = {
            "prompt": prompt,
            "temperature": 0.15,
            "repeat_penalty": 1.15,
            "n_predict": token_budget,
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "cache_prompt": False
        }

        for attempt in range(2):
            try:
                req = urllib.request.Request(
                    f"{self.server_url}/completion",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=120.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    raw_out = data.get("content", "").strip()

                    result_map = {}
                    for line in raw_out.split("\n"):
                        m = re.match(r"^\s*(\d+)[\.\)]\s*(.*)", line.strip())
                        if m:
                            idx = int(m.group(1)) - 1
                            result_map[idx] = m.group(2).strip()

                    translated_lines = []
                    for i, orig in enumerate(lines):
                        if i in result_map and result_map[i]:
                            translated_lines.append(result_map[i])
                        else:
                            translated_lines.append(orig)
                    return translated_lines
            except Exception as e:
                err_str = str(e)
                if ("Connection refused" in err_str or "Remote end closed" in err_str) and attempt == 0:
                    print(f"[HindiTranslator] Connection dropped during subtitle translation ({err_str}). Reviving daemon...")
                    try:
                        from test_system_wav import ensure_llama_server
                        ensure_llama_server()
                        time.sleep(2.0)
                        continue
                    except Exception:
                        pass
                print(f"[HindiTranslator] Subtitle batch translation fallback: {e}")
                break
        return lines

    def translate_title(self, title: str) -> str:
        """Translates the inferred monograph title to Hindi."""
        try:
            prompt = (
                "<|im_start|>system\n"
                "You are a professional translator. Translate this academic title into formal Hindi (Devanagari script). Keep acronyms intact. Output ONLY the translated title.<|im_end|>\n"
                f"<|im_start|>user\n{title}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            payload = {
                "prompt": prompt,
                "temperature": 0.1,
                "n_predict": 32,
                "stop": ["<|im_end|>", "\n"],
                "cache_prompt": False
            }
            req = urllib.request.Request(
                f"{self.server_url}/completion",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                ans = data.get("content", "").strip()
                if ans and len(ans) > 2:
                    return re.sub(r'^(?:शीर्षक|Title):\s*', '', ans, flags=re.IGNORECASE).strip()
        except Exception:
            pass
        return title
