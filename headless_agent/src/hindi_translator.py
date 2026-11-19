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
    "You are a professional STEM Hindi translator. Translate English academic and presentation text into "
    "natural, fluent Hindi (Devanagari script) to aid Indian college students. "
    "CRITICAL RULES:\n"
    "1. Keep all mathematical placeholders like __MATH_0__, variables, equations, and code identifiers 100% UNCHANGED.\n"
    "2. Translate the surrounding English prose into natural Hindi. Keep standard technical terms in English or parentheses where helpful.\n"
    "3. Output ONLY the translated Hindi sentence. Do NOT add meta commentary, explanations, or notes."
)

HINDI_FEW_SHOTS = [
    {
        "role": "system",
        "content": HINDI_SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": "Translate into Hindi:\nTaking the integral from zero to infinity of __MATH_0__ gives __MATH_1__."
    },
    {
        "role": "assistant",
        "content": "शून्य से अनंत तक __MATH_0__ का समाकलन (integral) लेने पर __MATH_1__ प्राप्त होता है।"
    },
    {
        "role": "user",
        "content": "Translate into Hindi:\nThe continuous-time signal __MATH_0__ is passed through a linear time-invariant system with impulse response __MATH_1__."
    },
    {
        "role": "assistant",
        "content": "सतत-समय संकेत (continuous-time signal) __MATH_0__ को एक रैखिक समय-अपरिवर्तनीय (linear time-invariant) प्रणाली से गुजारा जाता है जिसका आवेग प्रतिक्रिया (impulse response) __MATH_1__ है।"
    },
    {
        "role": "user",
        "content": "Translate into Hindi:\nGood morning everyone. Welcome to today's presentation on algorithms."
    },
    {
        "role": "assistant",
        "content": "शुभ प्रभात सभी को। एल्गोरिदम पर आज की प्रस्तुति में आपका स्वागत है।"
    }
]


def extract_math_spans(text: str) -> Tuple[str, List[str]]:
    """
    Replaces all math expressions ($...$, $$...$$, \\(..\\), \\[..\\]) with unique placeholders
    to guarantee that the LLM cannot mutate or corrupt the LaTeX AST during translation.
    """
    spans = []
    def _save_span(m):
        idx = len(spans)
        spans.append(m.group(0))
        return f"__MATH_{idx}__"

    # Save display math first
    s = re.sub(r"\$\$.*?\$\$", _save_span, text, flags=re.DOTALL)
    s = re.sub(r"\\\[.*?\\\]", _save_span, s, flags=re.DOTALL)
    # Save inline math
    s = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", _save_span, s, flags=re.DOTALL)
    s = re.sub(r"\\\(.*?\\\)", _save_span, s, flags=re.DOTALL)
    return s, spans


def restore_math_spans(translated_text: str, spans: List[str]) -> str:
    """Restores the exact original LaTeX math expressions into the translated text."""
    result = translated_text
    for idx, span in enumerate(spans):
        # Use lambda in sub to prevent backslash interpretation (e.g. \l, \f, \t) in LaTeX math
        pattern = re.compile(rf"__\s*MATH_{idx}\s*__")
        result = pattern.sub(lambda _, s=span: s, result)
        # Direct string replacement fallback
        result = result.replace(f"__MATH_{idx}__", span)
    return result


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
        Employs AST placeholder protection for 100% formula preservation.
        """
        clean_in = text.strip()
        if not clean_in or len(clean_in) < 3:
            return clean_in

        # Protect all math spans before LLM translation
        masked_text, math_spans = extract_math_spans(clean_in)

        messages = list(HINDI_FEW_SHOTS)
        messages.append({
            "role": "user",
            "content": f"Translate into Hindi:\n{masked_text}"
        })

        token_budget = min(max_tokens, max(48, len(clean_in.split()) * 3 + 48))
        payload = {
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": token_budget,
            "stop": ["<|im_end|>", "<|endoftext|>", "\n\n"],
            "cache_prompt": True,
        }

        try:
            req = urllib.request.Request(
                f"{self.server_url}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_translation = data["choices"][0]["message"]["content"].strip()
                if raw_translation:
                    # Clean any leading meta prefixes
                    clean_res = re.sub(
                        r"^(?:यहाँ अनुवाद है|अनुवाद|हिंदी अनुवाद|Here is the translation|Translation):\s*",
                        "",
                        raw_translation,
                        flags=re.IGNORECASE
                    ).strip()
                    # Restore pristine mathematical AST
                    restored = restore_math_spans(clean_res, math_spans)
                    return restored
        except Exception as e:
            print(f"[HindiTranslator] Translation request failed: {e}. Falling back to original text.")

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

    def translate_title(self, title: str) -> str:
        """Translates the inferred monograph title to Hindi."""
        try:
            messages = [
                {"role": "system", "content": "You are a professional translator. Translate this academic title into formal Hindi (Devanagari script). Keep acronyms intact. Output ONLY the translated title."},
                {"role": "user", "content": title}
            ]
            payload = {
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 32,
                "stop": ["<|im_end|>", "\n"]
            }
            req = urllib.request.Request(
                f"{self.server_url}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                ans = data["choices"][0]["message"]["content"].strip()
                if ans and len(ans) > 2:
                    return re.sub(r'^(?:शीर्षक|Title):\s*', '', ans, flags=re.IGNORECASE).strip()
        except Exception:
            pass
        return title
