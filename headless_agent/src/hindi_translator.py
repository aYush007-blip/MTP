#!/usr/bin/env python3
"""
Hindi Technical Translation Engine for STEM Lectures & Academic Presentations.

Translates English lecture transcripts and textbook monographs into natural,
fluent Hindi (Devanagari script) to aid Indian students, while enforcing
100% strict preservation of all LaTeX formulas ($...$, $$...$$), symbols,
and mathematical variables.
"""

import json
from collections import Counter
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Tuple
import urllib.request

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")

MATH_SPAN_PATTERN = re.compile(
    r"\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\}[\s\S]*?\\end\{(?:equation|align|gather|multline|eqnarray)\*?\}"
    r"|\$\$[\s\S]*?\$\$"
    r"|\\\[[\s\S]*?\\\]"
    r"|\\\([\s\S]*?\\\)"
    r"|(?<!\\)\$[^$\n]+?(?<!\\)\$"
    r"|\\(?:label|ref|eqref)\{[^}]+\}"
)

PLACEHOLDER_PATTERN = re.compile(r"\[\[MATH_(\d+)\]\]")
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def mask_math_spans(text: str) -> Tuple[str, List[str]]:
    """
    Extract LaTeX math spans and replace each with a unique ordered placeholder [[MATH_i]].
    Returns (masked_text, extracted_spans).
    """
    spans: List[str] = []

    def _replace(match: re.Match) -> str:
        idx = len(spans)
        spans.append(match.group(0))
        return f"[[MATH_{idx}]]"

    masked = MATH_SPAN_PATTERN.sub(_replace, text)
    return masked, spans


def extract_outside_numbers(masked_text: str) -> List[str]:
    """
    Extract numerical values outside protected math placeholders, normalizing
    Devanagari numerals and thousands commas so numerical integrity can be verified.
    """
    without_placeholders = PLACEHOLDER_PATTERN.sub(" ", masked_text)
    normalized = without_placeholders.translate(DEVANAGARI_DIGITS)
    raw_nums = re.findall(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?![\w.])", normalized)
    return [n.replace(",", "") for n in raw_nums]


def restore_and_verify_math_spans(
    source_masked: str,
    translated_masked: str,
    spans: List[str],
) -> Tuple[bool, str, str]:
    """
    Verify placeholder preservation (exact count=1 and exact ascending order) and
    numerical preservation outside math spans, then restore original LaTeX spans.
    Returns (is_valid, restored_text, status_reason).
    """
    if not translated_masked or not translated_masked.strip():
        return False, "", "empty_translation"

    found_indices = [int(m.group(1)) for m in PLACEHOLDER_PATTERN.finditer(translated_masked)]
    expected_indices = list(range(len(spans)))
    if found_indices != expected_indices:
        return False, "", f"placeholder_mismatch: expected={expected_indices}, got={found_indices}"

    # Ensure no partial/malformed placeholder marker slipped through
    stripped_valid = PLACEHOLDER_PATTERN.sub("", translated_masked)
    if "MATH_" in stripped_valid or "[[" in stripped_valid or "]]" in stripped_valid:
        return False, "", "malformed_placeholder_residue"

    # Ensure LLM did not introduce stray unmasked $ delimiters in natural-language text
    src_dollar_cnt = len(re.findall(r"(?<!\\)\$", PLACEHOLDER_PATTERN.sub("", source_masked)))
    dst_dollar_cnt = len(re.findall(r"(?<!\\)\$", stripped_valid))
    if dst_dollar_cnt != src_dollar_cnt:
        return False, "", f"stray_math_delimiter: src={src_dollar_cnt}, dst={dst_dollar_cnt}"

    # Verify numerical content outside math spans was neither lost nor altered
    src_nums = extract_outside_numbers(source_masked)
    dst_nums = extract_outside_numbers(translated_masked)
    if Counter(src_nums) != Counter(dst_nums):
        return False, "", f"numeric_mismatch: expected={src_nums}, got={dst_nums}"

    # Restore original math spans verbatim
    restored = PLACEHOLDER_PATTERN.sub(lambda m: spans[int(m.group(1))], translated_masked)
    if PLACEHOLDER_PATTERN.search(restored):
        return False, "", "unrestored_placeholder"

    return True, restored, "ok"


HINDI_SYSTEM_PROMPT = (
    "You are a professional STEM Hindi translator. Translate English academic lectures and presentations into "
    "natural, fluent Hindi (Devanagari script) to aid Indian college students.\n"
    "CRITICAL RULES:\n"
    "1. Keep every placeholder token like [[MATH_0]], [[MATH_1]], etc. 100% VERBATIM, in exact order, and never modify, drop, or translate them.\n"
    "2. Keep all numbers (e.g. 50, 0.85, 100) 100% verbatim in ASCII digits.\n"
    "3. Translate the surrounding English prose into natural Hindi. Keep standard English technical terms in parentheses for pedagogical clarity.\n"
    "4. Output ONLY the translated Hindi text. Do NOT add meta commentary, explanations, or quotes."
)

HINDI_FEW_SHOTS = [
    {
        "role": "system",
        "content": HINDI_SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": "Taking the integral from 0 to infinity of [[MATH_0]] gives [[MATH_1]]."
    },
    {
        "role": "assistant",
        "content": "0 से अनंत तक [[MATH_0]] का समाकलन (integral) लेने पर [[MATH_1]] प्राप्त होता है।"
    },
    {
        "role": "user",
        "content": "A point [[MATH_0]] is called a minimum if [[MATH_1]] for all [[MATH_2]]."
    },
    {
        "role": "assistant",
        "content": "एक बिंदु [[MATH_0]] को न्यूनतम (minimum) कहा जाता है यदि सभी [[MATH_2]] के लिए [[MATH_1]] हो।"
    },
    {
        "role": "user",
        "content": "At 50 Hz frequency, the efficiency is 0.85 across 3 phases."
    },
    {
        "role": "assistant",
        "content": "50 Hz आवृत्ति (frequency) पर, 3 कलाओं (phases) में दक्षता (efficiency) 0.85 है।"
    }
]


class HindiLectureTranslator:
    """
    Cross-lingual technical translation coordinator.
    Interfaces with the local Qwen2.5-1.5B edge model on llama-server and enforces
    span-level mathematical and numerical placeholder verification.
    """

    _instance: Optional["HindiLectureTranslator"] = None

    def __init__(self, server_url: str = LLAMA_SERVER_URL):
        self.server_url = server_url.rstrip("/")
        self.last_validation_log: List[str] = []
        self.reset_audit_stats()

    def reset_audit_stats(self) -> None:
        """Reset structured span-preservation audit counters for a new lecture."""
        self.last_validation_log = []
        self.audit_stats = {
            "paragraphs_total": 0,
            "paragraphs_unchanged_no_prose": 0,
            "spans_protected_total": 0,
            "passed_attempt_1": 0,
            "recovered_attempt_2": 0,
            "english_fallbacks": 0,
            "subtitle_lines_total": 0,
            "subtitle_lines_preserved": 0,
            "subtitle_lines_fallback": 0,
            "events": [],
        }

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

    def _request_completion(self, prompt: str, max_tokens: int, temperature: float = 0.15) -> str:
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "repeat_penalty": 1.15,
            "n_predict": max_tokens,
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "cache_prompt": True,
        }
        req = urllib.request.Request(
            f"{self.server_url}/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("content", "").strip()

    @staticmethod
    def _clean_translation_output(raw_translation: str) -> str:
        clean_res = re.sub(
            r"^(?:यहाँ अनुवाद है|अनुवाद|हिंदी अनुवाद|Here is the translation|Translation):\s*",
            "",
            raw_translation,
            flags=re.IGNORECASE
        ).strip()
        return re.sub(r'(.{15,}?)(?:\s+\1){2,}', r'\1', clean_res)

    def _build_translation_prompt(self, masked_in: str, strict_retry: bool = False) -> str:
        prompt = ""
        for m in HINDI_FEW_SHOTS:
            prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        user_msg = masked_in
        if strict_retry:
            placeholders = [m.group(0) for m in PLACEHOLDER_PATTERN.finditer(masked_in)]
            if placeholders:
                user_msg = (
                    f"[CRITICAL: Keep every placeholder {', '.join(placeholders)} exactly once in exact order, "
                    "and keep all numbers unchanged.]\n" + masked_in
                )
            else:
                user_msg = "[CRITICAL: Keep all numbers unchanged in ASCII digits.]\n" + masked_in
        prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def translate_text(self, text: str, max_tokens: int = 512) -> str:
        """
        Translates an English technical paragraph into Hindi using placeholder masking
        and span-level verification. Retries once on validation failure and falls back
        to the original English paragraph rather than emitting corrupted mathematics.
        """
        clean_in = text.strip()
        if not clean_in or len(clean_in) < 3:
            return clean_in

        masked_in, spans = mask_math_spans(clean_in)
        self.audit_stats["paragraphs_total"] += 1
        self.audit_stats["spans_protected_total"] += len(spans)
        prose_only = PLACEHOLDER_PATTERN.sub(" ", masked_in).strip()
        if not re.search(r"[a-zA-Z]", prose_only):
            self.audit_stats["paragraphs_unchanged_no_prose"] += 1
            return clean_in

        token_budget = min(max_tokens, max(96, len(masked_in.split()) * 4 + 64))

        for attempt in range(2):
            prompt = self._build_translation_prompt(masked_in, strict_retry=(attempt == 1))
            try:
                raw_translation = self._request_completion(
                    prompt,
                    max_tokens=token_budget,
                    temperature=0.15 if attempt == 0 else 0.05,
                )
                if raw_translation:
                    cleaned = self._clean_translation_output(raw_translation)
                    ok, restored, reason = restore_and_verify_math_spans(masked_in, cleaned, spans)
                    if ok:
                        self.last_validation_log.append(f"ok(attempt={attempt+1})")
                        if attempt == 0:
                            self.audit_stats["passed_attempt_1"] += 1
                        else:
                            self.audit_stats["recovered_attempt_2"] += 1
                        self.audit_stats["events"].append(
                            {"stage": "paragraph", "spans": len(spans), "status": f"ok_attempt_{attempt+1}", "detail": "ok"}
                        )
                        return restored
                    print(f"[HindiTranslator] Verification failed on attempt {attempt+1}: {reason}")
                    self.last_validation_log.append(f"fail({reason})")
                    if attempt == 1:
                        self.audit_stats["events"].append(
                            {"stage": "paragraph", "spans": len(spans), "status": "fallback", "detail": reason}
                        )
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
                self.last_validation_log.append(f"error({e})")
                self.audit_stats["events"].append(
                    {"stage": "paragraph", "spans": len(spans), "status": "fallback_error", "detail": str(e)}
                )
                break

        self.audit_stats["english_fallbacks"] += 1
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
        Masks math spans with placeholders and validates span & number preservation per line,
        retaining the original English line if validation fails.
        """
        if not lines:
            return []

        masked_lines: List[str] = []
        line_spans: List[List[str]] = []
        for line in lines:
            m_line, sp = mask_math_spans(line.strip())
            masked_lines.append(m_line)
            line_spans.append(sp)
            self.audit_stats["subtitle_lines_total"] += 1
            self.audit_stats["spans_protected_total"] += len(sp)

        numbered = "\n".join(f"{i+1}. {ml}" for i, ml in enumerate(masked_lines))
        prompt = (
            "<|im_start|>system\n"
            "You are a professional Hindi translator for academic STEM lectures. "
            "Translate each numbered line into natural Hindi (Devanagari script) preserving line numbers, "
            "all [[MATH_i]] placeholders verbatim in exact order, and all numbers verbatim. Output ONLY the numbered lines.<|im_end|>\n"
            "<|im_start|>user\n"
            "Translate each numbered subtitle line into natural Hindi for college students.\n"
            "Keep all [[MATH_0]], [[MATH_1]], etc. placeholders and numbers 100% verbatim.\n"
            "Maintain exact line numbering (1., 2., etc.). Output ONLY the numbered translations:\n\n"
            f"{numbered}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        token_budget = min(1024, max(128, sum(len(l.split()) for l in masked_lines) * 4 + 64))

        for attempt in range(2):
            try:
                raw_out = self._request_completion(prompt, max_tokens=token_budget, temperature=0.15)

                result_map = {}
                for line in raw_out.split("\n"):
                    m = re.match(r"^\s*(\d+)[\.\)]\s*(.*)", line.strip())
                    if m:
                        idx = int(m.group(1)) - 1
                        result_map[idx] = m.group(2).strip()

                translated_lines = []
                for i, orig in enumerate(lines):
                    cand = result_map.get(i, "")
                    if cand:
                        ok, restored, _ = restore_and_verify_math_spans(
                            masked_lines[i], cand, line_spans[i]
                        )
                        if ok:
                            self.audit_stats["subtitle_lines_preserved"] += 1
                            translated_lines.append(restored)
                            continue
                    self.audit_stats["subtitle_lines_fallback"] += 1
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
        self.audit_stats["subtitle_lines_fallback"] += len(lines)
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
                "cache_prompt": True
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
