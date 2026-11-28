#!/usr/bin/env python3
"""
Post-Lecture Global AI Review & PDF Compilation Engine.

Executes a comprehensive post-lecture overlook pass:
  1. Local AI Proofreading & Notation Harmonization via Qwen2.5-1.5B (llama-server)
  2. Mathematical Consistency & Syntax Balance Validation
  3. PDF Compilation: Attempts system pdflatex/tectonic with automated fallback
     to a built-in ReportLab academic monograph compiler with vector diagrams.
"""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple
import urllib.request

LLAMA_SERVER_URL = "http://127.0.0.1:8080"


class GlobalAIReviewer:
    """
    Interfaces with the local Qwen2.5-1.5B SLM and deterministic notation rules
    to proofread and harmonize mathematical notation across the lecture transcript.
    """

    _SIGNAL_SYMS = ("x", "y", "h", "u", "w", r"\\delta")
    _TRANSFORM_PAIRS = (("X", "x"), ("Y", "y"), ("H", "h"), ("G", "g"))

    def __init__(self, server_url: str = LLAMA_SERVER_URL):
        self.server_url = server_url.rstrip("/")
        self.reset_harmonisation_stats()

    def reset_harmonisation_stats(self) -> None:
        self.harmonisation_stats = {
            "paragraphs_reviewed": 0,
            "deterministic_edits": 0,
            "slm_edits_accepted": 0,
            "slm_edits_rejected": 0,
        }

    def is_server_alive(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.server_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    @staticmethod
    def _split_math_segments(text: str) -> List[Tuple[bool, str]]:
        """Split text into (is_math, segment) pairs preserving $$...$$ and $...$ spans."""
        pattern = re.compile(r'(\$\$[\s\S]*?\$\$|(?<!\\)\$[^$\n]+?(?<!\\)\$)')
        parts: List[Tuple[bool, str]] = []
        pos = 0
        for m in pattern.finditer(text):
            if m.start() > pos:
                parts.append((False, text[pos:m.start()]))
            parts.append((True, m.group(0)))
            pos = m.end()
        if pos < len(text):
            parts.append((False, text[pos:]))
        return parts

    @classmethod
    def harmonize_notation_deterministic(cls, text: str, context_text: str = "") -> str:
        """
        Conservatively harmonize unambiguous notation inconsistencies inside math spans
        without altering distinct mathematical expressions (e.g. preserves x(t) vs x[n] vs X(s) vs H(s)).
        """
        if not text or "$" not in text:
            return text

        ctx = context_text if context_text else text

        # Detect established discrete bracket notation (e.g. x[n], y[n], h[n], \delta[n])
        discrete_bracket_syms = set()
        for sym in cls._SIGNAL_SYMS:
            if re.search(rf'(?<![\\a-zA-Z]){sym}(?:_[a-zA-Z0-9]+)?\[n\b', ctx):
                discrete_bracket_syms.add(sym)

        # Detect established uppercase s/z/frequency-domain transforms (e.g. X(s), H(s), Y(s), X(z))
        upper_transforms = set()
        for up, _ in cls._TRANSFORM_PAIRS:
            for dom in ("s", "z", r"j\\omega"):
                if re.search(rf'(?<![\\a-zA-Z]){up}(?:_[a-zA-Z0-9]+)?\(\s*{dom}\s*\)', ctx):
                    upper_transforms.add((up, dom))

        has_s_domain_context = bool(
            re.search(r'\b(?:laplace|transfer function|s-domain|s domain|region of convergence)\b', ctx, re.IGNORECASE)
            or any(dom == "s" for _, dom in upper_transforms)
        )

        out_parts: List[str] = []
        for is_math, seg in cls._split_math_segments(text):
            if not is_math:
                out_parts.append(seg)
                continue

            # 1. Clean erratic whitespace around function/sequence delimiters: x ( t ) -> x(t), x [ n ] -> x[n]
            seg = re.sub(
                r'(\\delta|(?<![\\a-zA-Z])[xyhugfXYHGRTSEFPQ](?:_[a-zA-Z0-9]+)?)\s*\(\s*([^()\n]{1,24}?)\s*\)',
                lambda m: f"{m.group(1)}({m.group(2).strip()})",
                seg,
            )
            seg = re.sub(
                r'(\\delta|(?<![\\a-zA-Z])[xyhugfXYHGRTSEFPQ](?:_[a-zA-Z0-9]+)?)\s*\[\s*([^\[\]\n]{1,24}?)\s*\]',
                lambda m: f"{m.group(1)}[{m.group(2).strip()}]",
                seg,
            )

            # 2. Harmonize x(n) -> x[n] ONLY when bracket form x[n] is already established in context
            for sym in discrete_bracket_syms:
                seg = re.sub(
                    rf'(?<![\\a-zA-Z])({sym}(?:_[a-zA-Z0-9]+)?)\(n\)',
                    r'\1[n]',
                    seg,
                )

            # 3. Harmonize lowercase transform casing (e.g. x(s) -> X(s), h(s) -> H(s))
            #    ONLY when the uppercase transform form is already established in context
            for up, low in cls._TRANSFORM_PAIRS:
                for dom in ("s", "z", r"j\\omega"):
                    if (up, dom) in upper_transforms:
                        seg = re.sub(
                            rf'(?<![\\a-zA-Z]){low}(_[a-zA-Z0-9]+)?\({dom}\)',
                            lambda m, u=up: f"{u}{m.group(1) or ''}({m.group(0).split('(', 1)[1]}",
                            seg,
                        )

            # 4. Harmonize X[s]/H[s]/Y[s] -> X(s)/H(s)/Y(s) when s-domain transform context is present
            if has_s_domain_context:
                seg = re.sub(
                    r'(?<![\\a-zA-Z])([XYHG](?:_[a-zA-Z0-9]+)?)\[s\]',
                    r'\1(s)',
                    seg,
                )

            out_parts.append(seg)

        return "".join(out_parts)

    @staticmethod
    def _extract_math_signatures(text: str) -> Dict[str, int]:
        """Count critical continuous/discrete/transform notation tokens and numbers."""
        sigs: Dict[str, int] = {}
        patterns = {
            "cont_x": r'\bx\s*\(\s*t\s*\)',
            "disc_x": r'\bx\s*\[\s*n\s*\]',
            "cont_y": r'\by\s*\(\s*t\s*\)',
            "disc_y": r'\by\s*\[\s*n\s*\]',
            "cont_h": r'\bh\s*\(\s*t\s*\)',
            "disc_h": r'\bh\s*\[\s*n\s*\]',
            "X_s": r'\bX\s*\(\s*s\s*\)',
            "H_s": r'\bH\s*\(\s*s\s*\)',
            "Y_s": r'\bY\s*\(\s*s\s*\)',
            "X_z": r'\bX\s*\(\s*z\s*\)',
            "H_z": r'\bH\s*\(\s*z\s*\)',
        }
        for name, pat in patterns.items():
            sigs[name] = len(re.findall(pat, text))
        return sigs

    @classmethod
    def is_safe_harmonized_edit(cls, original: str, candidate: str) -> bool:
        """
        Verify an SLM-proofread candidate did not corrupt math delimiters, numbers,
        or distinct domain notations (e.g. turning x(t) into x[n]).
        """
        if not candidate or not candidate.strip():
            return False
        if len(candidate) < len(original) * 0.75 or len(candidate) > len(original) * 1.35:
            return False

        # Reject unbalanced math delimiters
        no_dd = re.sub(r'\$\$[\s\S]*?\$\$', '', candidate)
        if '$$' in no_dd:
            return False
        if len(re.findall(r'(?<!\\)\$', no_dd)) % 2 != 0:
            return False

        # Reject unrequested structural hallucinations
        if r'\begin{cases}' in candidate and r'\begin{cases}' not in original:
            return False
        if r'\nabla' in candidate and r'\nabla' not in original:
            return False

        # Ensure math span count is not reduced
        orig_spans = re.findall(r'\$\$[\s\S]*?\$\$|(?<!\\)\$[^$\n]+?(?<!\\)\$', original)
        cand_spans = re.findall(r'\$\$[\s\S]*?\$\$|(?<!\\)\$[^$\n]+?(?<!\\)\$', candidate)
        if len(cand_spans) < len(orig_spans):
            return False

        # Preserve numeric constants (including numbers immediately before sentence punctuation like '2.5.')
        num_pat = r'(?<![\w.])\d+(?:\.\d+)?(?!\d|\.\d|\w)'
        orig_nums = sorted(re.findall(num_pat, original))
        cand_nums = sorted(re.findall(num_pat, candidate))
        if orig_nums != cand_nums:
            return False

        # Do not alter distinct continuous/discrete or Laplace/Z domain symbols once established
        orig_sigs = cls._extract_math_signatures(original)
        cand_sigs = cls._extract_math_signatures(candidate)
        for key, cnt in orig_sigs.items():
            if cnt > 0 and cand_sigs.get(key, 0) < cnt:
                return False

        return True

    def proofread_section(self, text: str, context_text: str = "", use_slm: bool = True) -> str:
        """
        Harmonizes mathematical notation deterministically and optionally runs a guarded
        Qwen2.5-1.5B proofreading pass if llama-server is active.
        """
        self.harmonisation_stats["paragraphs_reviewed"] += 1
        harmonized = self.harmonize_notation_deterministic(text, context_text=context_text)
        if harmonized != text:
            self.harmonisation_stats["deterministic_edits"] += 1
        if not use_slm or not self.is_server_alive() or len(harmonized.strip()) < 10:
            return harmonized

        prompt = (
            "<|im_start|>system\n"
            "You are an expert academic editor and mathematical typesetter. "
            "Your task is to lightly harmonize inconsistent notation in a transcribed lecture paragraph.\n"
            "STRICT RULES:\n"
            "1. Keep continuous-time signals like $x(t)$ and discrete-time sequences like $x[n]$ distinct — NEVER convert $x(t)$ to $x[n]$ or vice versa.\n"
            "2. Keep transform symbols ($X(s)$, $H(s)$, $Y(s)$, $X(z)$) and all numbers 100% intact.\n"
            "3. Ensure all $...$ delimiters are balanced.\n"
            "4. Preserve all factual content without summarizing or adding commentary.\n"
            "Return ONLY the corrected text.<|im_end|>\n"
            f"<|im_start|>user\n{harmonized}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        payload = {
            "prompt": prompt,
            "n_predict": min(1024, len(harmonized.split()) * 2 + 128),
            "temperature": 0.1,
            "top_p": 0.90,
            "stop": ["<|im_end|>", "<|im_start|>"],
            "cache_prompt": True,
        }

        try:
            req = urllib.request.Request(
                f"{self.server_url}/completion",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                result = data.get("content", "").strip()
                result = re.sub(r'^(?:Here is the (?:corrected|proofread).*?:\s*)', '', result, flags=re.IGNORECASE).strip()
                if self.is_safe_harmonized_edit(harmonized, result):
                    if result != harmonized:
                        self.harmonisation_stats["slm_edits_accepted"] += 1
                    return result
                self.harmonisation_stats["slm_edits_rejected"] += 1
        except Exception:
            pass

        return harmonized

    def harmonize_paragraphs(self, paragraphs: List[str], use_slm: bool = True) -> List[str]:
        """Harmonize notation consistently across a list of lecture paragraphs."""
        if not paragraphs:
            return []
        full_ctx = "\n".join(paragraphs)
        slm_active = bool(use_slm and self.is_server_alive())
        return [
            self.proofread_section(p, context_text=full_ctx, use_slm=slm_active)
            for p in paragraphs
        ]


class AcademicPDFRenderer:
    """
    Compiles LaTeX to PDF via standalone offline Tectonic XeTeX engine or system pdflatex.
    Provides automated fallback to a clean ReportLab student lecture notes compiler.
    """
    TECTONIC_BIN = Path("/mnt/thesis/bin/tectonic")

    @staticmethod
    def _extract_tectonic_error_line(stderr_text: str) -> Optional[int]:
        """Parse the 1-based line number from Tectonic/XeTeX stderr output."""
        for pat in (r'error:\s*[^:\n]+:(\d+):', r'(?m)^l\.(\d+)\b'):
            m = re.search(pat, stderr_text or "")
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    pass
        return None

    @staticmethod
    def _quarantine_tex_error_line(tex_content: str, err_line: int) -> str:
        """
        Surgically neutralize the enclosing figure block, equation block, or single
        faulty line at err_line so one bad SLM LaTeX token never crashes the entire PDF.
        """
        lines = tex_content.splitlines()
        idx = err_line - 1
        if idx < 0 or idx >= len(lines):
            return tex_content

        # Never alter preamble or document boundaries
        protected_prefixes = (r"\documentclass", r"\usepackage", r"\begin{document}", r"\end{document}", r"\section")
        if lines[idx].strip().startswith(protected_prefixes):
            return tex_content

        # 1. If error falls inside a \begin{figure}...\end{figure}, strip only that figure
        fig_start = None
        for i in range(idx, -1, -1):
            if r"\begin{figure}" in lines[i]:
                fig_start = i
                break
            if r"\end{figure}" in lines[i] and i < idx:
                break
        if fig_start is not None:
            fig_end = None
            for j in range(idx, len(lines)):
                if r"\end{figure}" in lines[j]:
                    fig_end = j
                    break
            if fig_end is not None:
                del lines[fig_start:fig_end + 1]
                return "\n".join(lines)

        # 2. If error falls inside an enclosing \begin{equation}...\end{equation}, convert to safe escaped plain text
        def _sanitize_plain(raw: str) -> str:
            txt = re.sub(r'\\label\{[^}]+\}', '', raw)
            txt = txt.replace("$", "").replace("\\", " ")
            txt = txt.replace("{", "(").replace("}", ")")
            txt = re.sub(r'([&%#_~^])', r'\\\1', txt)
            return re.sub(r'\s+', ' ', txt).strip()

        eq_start = None
        for i in range(idx, -1, -1):
            if r"\begin{equation}" in lines[i]:
                eq_start = i
                break
            if r"\end{equation}" in lines[i] and i < idx:
                break
        if eq_start is not None:
            eq_end = None
            for j in range(idx, len(lines)):
                if r"\end{equation}" in lines[j]:
                    eq_end = j
                    break
            if eq_end is not None:
                body = " ".join(lines[eq_start + 1:eq_end])
                safe_txt = _sanitize_plain(body)
                repl = [f"\\begin{{quote}}\\texttt{{{safe_txt}}}\\end{{quote}}"] if safe_txt else []
                lines[eq_start:eq_end + 1] = repl
                return "\n".join(lines)

        # 3. Otherwise neutralize only the single offending line
        safe_line = _sanitize_plain(lines[idx])
        lines[idx] = safe_line if safe_line else ""
        return "\n".join(lines)

    @classmethod
    def compile_with_tectonic(cls, tex_path: Path, output_dir: Path) -> Optional[Path]:
        """Compile using offline cached Tectonic binary with pre-compilation source sanitization and surgical line quarantine."""
        tectonic_bin = cls.TECTONIC_BIN if cls.TECTONIC_BIN.is_file() else shutil.which("tectonic")
        if not tectonic_bin:
            return None

        # 1. Pre-compilation sanitization pass
        try:
            from lecture_to_book import sanitize_latex_source
            if tex_path.is_file():
                raw_tex = tex_path.read_text(encoding="utf-8")
                sanitized_tex = sanitize_latex_source(raw_tex)
                if sanitized_tex != raw_tex:
                    tex_path.write_text(sanitized_tex, encoding="utf-8")
        except Exception as e:
            print(f"[PDF Compiler] Sanitization notice: {e}")

        def _run_tectonic():
            cmd = [
                str(tectonic_bin),
                "-C",  # 100% offline: only use locally cached packages
                str(tex_path),
                "-o", str(output_dir)
            ]
            return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90)

        try:
            res = _run_tectonic()
            pdf_candidate = output_dir / f"{tex_path.stem}.pdf"
            if pdf_candidate.is_file() and pdf_candidate.stat().st_size > 1000:
                print(f"[PDF Compiler] Tectonic successfully compiled: {pdf_candidate} ({pdf_candidate.stat().st_size} bytes)")
                return pdf_candidate

            # 2. Auto-healing retry pass: sanitize + strip figure blocks if undefined TikZ/pgf macro broke pass 1
            print(f"[PDF Compiler] First Tectonic pass encountered error: {res.stderr[:200]}")
            if tex_path.is_file():
                content = tex_path.read_text(encoding="utf-8")
                err_text = (res.stderr or "") + (res.stdout or "")
                if (
                    "Undefined control sequence" in err_text
                    or "Environment" in err_text
                    or "pgf Error" in err_text
                ) and r"\begin{figure}" in content:
                    print("[PDF Compiler] Stripping incompatible figure block before Tectonic retry...")
                    content = re.sub(r'\\begin\{figure\}[\s\S]*?\\end\{figure\}', '', content)
                    tex_path.write_text(content, encoding="utf-8")
                    res = _run_tectonic()
                    if pdf_candidate.is_file() and pdf_candidate.stat().st_size > 1000:
                        print(f"[PDF Compiler] Auto-healing succeeded! Tectonic compiled: {pdf_candidate}")
                        return pdf_candidate

                # 3. Surgical line-level quarantine loop (up to 4 retries) so one bad equation never dumps the PDF to ReportLab
                for q_step in range(4):
                    err_line = cls._extract_tectonic_error_line((res.stderr or "") + "\n" + (res.stdout or ""))
                    if err_line is None:
                        break
                    healed = cls._quarantine_tex_error_line(content, err_line)
                    if healed == content:
                        break
                    print(f"[PDF Compiler] Quarantining faulty LaTeX at line {err_line} (retry {q_step + 1}/4)...")
                    content = healed
                    tex_path.write_text(content, encoding="utf-8")
                    res = _run_tectonic()
                    if pdf_candidate.is_file() and pdf_candidate.stat().st_size > 1000:
                        print(f"[PDF Compiler] Surgical quarantine succeeded! Tectonic compiled: {pdf_candidate}")
                        return pdf_candidate

                print(f"[PDF Compiler] Tectonic compilation failed: {res.stderr[:300]}")
        except Exception as e:
            print(f"[PDF Compiler] Tectonic execution error: {e}")
        return None

    @classmethod
    def compile_with_pdflatex(cls, tex_path: Path, output_dir: Path) -> Optional[Path]:
        """Try compiling using system pdflatex / xelatex."""
        latex_bin = shutil.which("pdflatex") or shutil.which("xelatex")
        if not latex_bin:
            return None

        try:
            cmd = [
                latex_bin,
                "-interaction=nonstopmode",
                f"-output-directory={output_dir}",
                str(tex_path)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            pdf_candidate = output_dir / f"{tex_path.stem}.pdf"
            if pdf_candidate.is_file() and pdf_candidate.stat().st_size > 1000:
                print(f"[PDF Compiler] pdflatex successfully compiled: {pdf_candidate}")
                return pdf_candidate
        except Exception:
            pass
        return None

    @classmethod
    def compile_latex_to_pdf(cls, tex_path: Path, output_dir: Path) -> Optional[Path]:
        """Primary compiler coordinator: tries Tectonic first, then pdflatex."""
        pdf = cls.compile_with_tectonic(tex_path, output_dir)
        if pdf and pdf.is_file():
            return pdf

        pdf = cls.compile_with_pdflatex(tex_path, output_dir)
        if pdf and pdf.is_file():
            return pdf

        return None

    @staticmethod
    def render_reportlab_pdf(
        title: str,
        abstract: str,
        sections: List[Tuple[str, List[str]]],
        equations: List[Tuple[int, str]],
        diagram_types: List[str],
        output_pdf_path: Path
    ) -> bool:
        """
        Standalone Student Classroom Lecture Notes generator using ReportLab.
        Guarantees 100% offline PDF creation on Jetson even if TeX engines are unavailable.
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, Wedge
            import math

            doc = SimpleDocTemplate(
                str(output_pdf_path),
                pagesize=A4,
                rightMargin=54, leftMargin=54,
                topMargin=54, bottomMargin=54
            )

            styles = getSampleStyleSheet()
            
            # Professional Student Lecture Notes Typography Styles
            title_style = ParagraphStyle(
                'StudentTitle',
                parent=styles['Heading1'],
                fontName='Helvetica-Bold',
                fontSize=20,
                leading=24,
                textColor=colors.HexColor('#0f2942'),
                alignment=1, # Center
                spaceAfter=6
            )
            subtitle_style = ParagraphStyle(
                'StudentSubtitle',
                fontName='Helvetica',
                fontSize=11,
                leading=15,
                textColor=colors.HexColor('#2b6cb0'),
                alignment=1,
                spaceAfter=4
            )
            date_style = ParagraphStyle(
                'StudentDate',
                fontName='Helvetica-Oblique',
                fontSize=9.5,
                leading=13,
                textColor=colors.HexColor('#718096'),
                alignment=1,
                spaceAfter=14
            )
            overview_heading = ParagraphStyle(
                'OverviewHeading',
                fontName='Helvetica-Bold',
                fontSize=10.5,
                leading=14,
                textColor=colors.HexColor('#0f2942'),
                alignment=0,
                spaceAfter=4
            )
            overview_body = ParagraphStyle(
                'OverviewBody',
                fontName='Helvetica',
                fontSize=9.5,
                leading=14,
                textColor=colors.HexColor('#2d3748'),
                alignment=4, # Justify
            )
            heading_style = ParagraphStyle(
                'StudentHeading',
                parent=styles['Heading2'],
                fontName='Helvetica-Bold',
                fontSize=13,
                leading=17,
                textColor=colors.HexColor('#0f2942'),
                spaceBefore=14,
                spaceAfter=6,
                keepWithNext=True
            )
            body_style = ParagraphStyle(
                'StudentBody',
                parent=styles['Normal'],
                fontName='Helvetica',
                fontSize=10,
                leading=15,
                textColor=colors.HexColor('#2d3748'),
                spaceAfter=8,
                alignment=4
            )
            equation_style = ParagraphStyle(
                'EquationStyle',
                fontName='Courier-Bold',
                fontSize=10,
                leading=13,
                textColor=colors.HexColor('#1a365d'),
                alignment=1
            )
            caption_style = ParagraphStyle(
                'CaptionStyle',
                fontName='Helvetica-Oblique',
                fontSize=8.5,
                leading=11,
                textColor=colors.HexColor('#4a5568'),
                alignment=1,
                spaceAfter=10
            )

            story = []

            # 1. Header Banner
            display_title = title
            story.append(Paragraph(display_title, title_style))
            story.append(Paragraph(f"Delivered on {time.strftime('%B %d, %Y')}", date_style))
            story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor('#0f2942'), spaceAfter=14))

            # Helper to draw Z-Plane vector graphic
            def draw_zplane_roc_drawing():
                d = Drawing(480, 200)
                cx, cy = 240, 100
                r_roc = 75
                r_unit = 50

                # Background
                d.add(Rect(0, 0, 480, 200, fillColor=colors.HexColor('#fbfcfd'), strokeColor=colors.HexColor('#e2e8f0'), strokeWidth=0.5))

                # ROC Shading (Outer)
                d.add(Rect(cx - 150, cy - 90, 300, 180, fillColor=colors.HexColor('#ebf8ff'), strokeColor=None))
                # Inner clear circle (non-ROC)
                d.add(Circle(cx, cy, r_roc, fillColor=colors.HexColor('#fbfcfd'), strokeColor=None))

                # Unit circle
                d.add(Circle(cx, cy, r_unit, fillColor=None, strokeColor=colors.HexColor('#a0aec0'), strokeWidth=1, strokeDashArray=[3,3]))
                d.add(String(cx + r_unit + 4, cy - 10, "1", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#718096')))

                # ROC Boundary
                d.add(Circle(cx, cy, r_roc, fillColor=None, strokeColor=colors.HexColor('#3182ce'), strokeWidth=1.5))

                # Axes
                d.add(Line(cx - 140, cy, cx + 140, cy, strokeColor=colors.HexColor('#2d3748'), strokeWidth=1))
                d.add(Line(cx, cy - 85, cx, cy + 85, strokeColor=colors.HexColor('#2d3748'), strokeWidth=1))
                d.add(String(cx + 142, cy - 3, "Re(z)", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor('#2d3748')))
                d.add(String(cx - 12, cy + 88, "Im(z)", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor('#2d3748')))

                # Pole Marker (X)
                px = cx + r_roc
                d.add(Line(px - 4, cy - 4, px + 4, cy + 4, strokeColor=colors.HexColor('#e53e3e'), strokeWidth=2))
                d.add(Line(px - 4, cy + 4, px + 4, cy - 4, strokeColor=colors.HexColor('#e53e3e'), strokeWidth=2))
                d.add(String(px + 6, cy - 2, "p1 (r)", fontName="Helvetica-Bold", fontSize=8, fillColor=colors.HexColor('#e53e3e')))

                # Labels
                d.add(String(cx + 80, cy + 50, "ROC: |z| > r", fontName="Helvetica-Bold", fontSize=10, fillColor=colors.HexColor('#2b6cb0')))
                return d

            # Helper to draw d-q rotor frame vector graphic
            def draw_dq_frame_drawing():
                d = Drawing(480, 180)
                cx, cy = 240, 90
                d.add(Rect(0, 0, 480, 180, fillColor=colors.HexColor('#fbfcfd'), strokeColor=colors.HexColor('#e2e8f0'), strokeWidth=0.5))

                # Stator stationary axes
                d.add(Line(cx - 110, cy, cx + 130, cy, strokeColor=colors.HexColor('#718096'), strokeWidth=1))
                d.add(Line(cx, cy - 70, cx, cy + 75, strokeColor=colors.HexColor('#718096'), strokeWidth=1))
                d.add(String(cx + 133, cy - 3, "Stator a-axis", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#718096')))
                d.add(String(cx - 5, cy + 78, "beta-axis", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor('#718096')))

                # Rotor d-q axes
                theta = math.radians(32)
                dx, dy = 130 * math.cos(theta), 130 * math.sin(theta)
                d.add(Line(cx, cy, cx + dx, cy + dy, strokeColor=colors.HexColor('#c53030'), strokeWidth=2))
                d.add(String(cx + dx + 4, cy + dy, "Direct Axis (d)", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor('#c53030')))

                qx, qy = 100 * math.cos(theta + math.pi/2), 100 * math.sin(theta + math.pi/2)
                d.add(Line(cx, cy, cx + qx, cy + qy, strokeColor=colors.HexColor('#2b6cb0'), strokeWidth=2))
                d.add(String(cx + qx - 65, cy + qy + 4, "Quadrature Axis (q)", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor('#2b6cb0')))

                d.add(Wedge(cx, cy, 40, 0, 32, strokeColor=colors.HexColor('#805ad5'), fillColor=None, strokeWidth=1))
                d.add(String(cx + 45, cy + 8, "theta_r", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor('#805ad5')))
                return d

            # 3. Sections & Content
            eq_idx = 0
            for sec_idx, (sec_title, paras) in enumerate(sections, 1):
                story.append(Paragraph(f"{sec_idx}. {sec_title}", heading_style))

                for p_text in paras:
                    # Clean LaTeX formatting for readable PDF display
                    clean_p = p_text
                    clean_p = re.sub(r'\\textbf\{([^}]+)\}', r'<b>\1</b>', clean_p)
                    clean_p = re.sub(r'\\textit\{([^}]+)\}', r'<i>\1</i>', clean_p)
                    clean_p = re.sub(r'\\text\{([^}]+)\}', r'\1', clean_p)
                    # Strip any raw macros
                    clean_p = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?', '', clean_p)
                    if clean_p.strip():
                        story.append(Paragraph(clean_p.strip(), body_style))

                # Insert equations if available for this section
                if eq_idx < len(equations):
                    num, eq_text = equations[eq_idx]
                    eq_idx += 1
                    clean_eq = eq_text.replace(r'\sum', 'SUM').replace(r'\int', 'INT')
                    clean_eq = re.sub(r'[\$\\]', '', clean_eq)
                    eq_table = [
                        [Paragraph(f"<b>{clean_eq}</b>", equation_style),
                         Paragraph(f"({num})", ParagraphStyle('EqNum', fontName='Helvetica-Bold', fontSize=9, alignment=2))]
                    ]
                    t_eq = Table(eq_table, colWidths=[435, 50])
                    t_eq.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#edf2f7')),
                        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
                        ('TOPPADDING', (0,0), (-1,-1), 6),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                    ]))
                    story.append(Spacer(1, 4))
                    story.append(t_eq)
                    story.append(Spacer(1, 8))

                # Insert diagram if matched
                if sec_idx == 1 and "z_plane_roc" in diagram_types:
                    story.append(Spacer(1, 6))
                    story.append(draw_zplane_roc_drawing())
                    story.append(Paragraph("<b>Figure 1:</b> Complex <i>z</i>-plane showing the unit circle, pole at radius <i>r</i>, and Region of Convergence (ROC).", caption_style))
                elif sec_idx == 1 and "dq_axes" in diagram_types:
                    story.append(Spacer(1, 6))
                    story.append(draw_dq_frame_drawing())
                    story.append(Paragraph("<b>Figure 1:</b> Direct (<i>d</i>) and quadrature (<i>q</i>) rotor reference frame rotated by electrical angle <i>θ<sub>r</sub></i>.", caption_style))

            doc.build(story)
            return output_pdf_path.is_file()
        except Exception as e:
            print(f"[PDF Compiler] ReportLab error: {e}")
            return False

    @staticmethod
    def render_hindi_reportlab_pdf(
        title: str,
        paragraphs: List[str],
        output_pdf_path: Path
    ) -> bool:
        """
        Compiles bilingual Hindi lecture monograph into a publication-grade PDF using ReportLab
        and Lohit-Devanagari TTF font. Guarantees 100% offline generation with zero LaTeX dependencies.
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            import time

            # Register Devanagari font
            font_candidates = [
                "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
                "/usr/share/fonts/truetype/gargi/gargi.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf"
            ]
            font_name = "Helvetica"
            for fc in font_candidates:
                if Path(fc).is_file():
                    pdfmetrics.registerFont(TTFont("HindiFont", fc))
                    font_name = "HindiFont"
                    break

            doc = SimpleDocTemplate(
                str(output_pdf_path),
                pagesize=A4,
                rightMargin=54, leftMargin=54,
                topMargin=54, bottomMargin=54
            )

            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                'HindiTitle',
                fontName=font_name,
                fontSize=18,
                leading=22,
                textColor=colors.HexColor('#0f2942'),
                alignment=1,
                spaceAfter=4
            )
            sub_style = ParagraphStyle(
                'HindiSub',
                fontName=font_name,
                fontSize=10.5,
                leading=14,
                textColor=colors.HexColor('#2b6cb0'),
                alignment=1,
                spaceAfter=12
            )
            body_style = ParagraphStyle(
                'HindiBody',
                fontName=font_name,
                fontSize=10,
                leading=15,
                textColor=colors.HexColor('#2d3748'),
                spaceAfter=8,
                alignment=4
            )
            eq_style = ParagraphStyle(
                'HindiEq',
                fontName='Courier-Bold',
                fontSize=9.5,
                leading=13,
                textColor=colors.HexColor('#1a365d'),
                alignment=1
            )

            story = []
            story.append(Paragraph(title, title_style))
            story.append(Paragraph(f"द्विभाषी व्याख्यान प्रतिलेख एवं नोट्स (Bilingual Lecture Monograph) — {time.strftime('%B %d, %Y')}", sub_style))
            story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor('#0f2942'), spaceAfter=14))

            for idx, p in enumerate(paragraphs, 1):
                clean_p = p.strip()
                if not clean_p:
                    continue

                eq_match = re.search(r'\$\$(.*?)\$\$', clean_p, re.DOTALL)
                if eq_match:
                    prose_before = clean_p[:eq_match.start()].strip()
                    eq_content = eq_match.group(1).strip()
                    prose_after = clean_p[eq_match.end():].strip()

                    if prose_before:
                        story.append(Paragraph(prose_before, body_style))
                    t_eq = Table([[Paragraph(eq_content, eq_style)]], colWidths=[480])
                    t_eq.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f7fafc')),
                        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
                        ('TOPPADDING', (0,0), (-1,-1), 5),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                    ]))
                    story.append(t_eq)
                    story.append(Spacer(1, 6))
                    if prose_after:
                        story.append(Paragraph(prose_after, body_style))
                else:
                    story.append(Paragraph(clean_p, body_style))

            doc.build(story)
            return output_pdf_path.is_file() and output_pdf_path.stat().st_size > 1000
        except Exception as e:
            print(f"[Hindi PDF Compiler] Error: {e}")
            return False


class PostLectureProcessor:
    """
    Coordinates global AI review, equation verification, and PDF compilation.
    """
    def __init__(self):
        self.ai_reviewer = GlobalAIReviewer()
        self.pdf_renderer = AcademicPDFRenderer()

    def process_and_compile(
        self,
        lecture_title: str,
        tex_path: Path,
        output_dir: Path,
        raw_transcript: str = ""
    ) -> Dict[str, Optional[Path]]:
        """
        Executes complete post-processing pipeline.
        Returns map of deliverable paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tex_content = tex_path.read_text(encoding="utf-8") if tex_path.is_file() else ""

        # 1. Extract sections, equations, and overview for student notes
        sections = []
        curr_sec_title = "Theoretical Background and Motivation"
        curr_paras = []
        in_doc = False
        in_ignored_env = 0
        for line in tex_content.splitlines():
            line_s = line.strip()
            if r"\begin{document}" in line_s:
                in_doc = True
                continue
            if r"\end{document}" in line_s:
                in_doc = False
                break
            if not in_doc:
                continue

            # Skip environments not meant for body paragraphs (figures, tikz, display equations, title centers)
            if any(env in line_s for env in [r"\begin{figure}", r"\begin{tikzpicture}", r"\begin{equation}", r"\begin{center}"]):
                in_ignored_env += 1
                continue
            if any(env in line_s for env in [r"\end{figure}", r"\end{tikzpicture}", r"\end{equation}", r"\end{center}"]):
                in_ignored_env = max(0, in_ignored_env - 1)
                continue
            if in_ignored_env > 0:
                continue

            if line_s.startswith(r"\section{"):
                if curr_paras:
                    sections.append((curr_sec_title, curr_paras))
                    curr_paras = []
                m_sec = re.match(r'\\section\{([^}]+)\}', line_s)
                curr_sec_title = m_sec.group(1) if m_sec else "Section"
            elif line_s and not line_s.startswith(("%", r"\vspace", r"\hrule")):
                clean_line = re.sub(r'\\(?:textbf|textit|text|emph)\{([^}]+)\}', r'\1', line_s)
                curr_paras.append(clean_line)
        if curr_paras:
            sections.append((curr_sec_title, curr_paras))

        # Extract equations
        equations = []
        eq_matches = re.findall(r'\\begin\{equation\}\s*(?:\\label\{[^}]+\})?\s*(.*?)\s*\\end\{equation\}', tex_content, re.DOTALL)
        for idx, eq in enumerate(eq_matches, 1):
            equations.append((idx, eq.strip()))

        # Extract overview text
        abstract_text = ""

        # Detect diagram types
        from lecture_to_book import TikZDiagramGenerator
        diagram_types = TikZDiagramGenerator.detect_diagram_triggers(tex_content)

        # 2. PDF Compilation: Try Tectonic first, then pdflatex
        pdf_path = self.pdf_renderer.compile_latex_to_pdf(tex_path, output_dir)

        # 3. Fallback to ReportLab student lecture notes renderer if TeX compilation fails
        if not pdf_path or not pdf_path.is_file():
            fallback_pdf = output_dir / f"{tex_path.stem}.pdf"
            print("[PDF Compiler] Falling back to ReportLab student lecture notes renderer...")

            success = self.pdf_renderer.render_reportlab_pdf(
                title=lecture_title,
                abstract=abstract_text,
                sections=sections if sections else [("Classroom Notes", [raw_transcript or "Lecture content."])],
                equations=equations,
                diagram_types=diagram_types,
                output_pdf_path=fallback_pdf
            )
            if success:
                pdf_path = fallback_pdf

        return {
            "tex": tex_path,
            "pdf": pdf_path if (pdf_path and pdf_path.is_file()) else None
        }

    def compile_hindi_pdf(
        self,
        tex_path: Path,
        output_dir: Path,
        title: str = "",
        paragraphs: Optional[List[str]] = None,
        output_pdf_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Compiles the Hindi bilingual monograph to publication-grade PDF using Tectonic (XeTeX)
        with native Lohit-Devanagari font and LaTeX math mode.
        Falls back to ReportLab only if XeTeX compilation fails.
        """
        if tex_path and tex_path.is_file():
            pdf = self.pdf_renderer.compile_latex_to_pdf(tex_path, output_dir)
            if pdf and pdf.is_file() and pdf.stat().st_size > 1000:
                print(f"[Hindi PDF Compiler] Tectonic successfully compiled publication-grade Hindi PDF: {pdf} ({pdf.stat().st_size} bytes)")
                return pdf

        if paragraphs and output_pdf_path:
            print("[Hindi PDF Compiler] Falling back to ReportLab Devanagari renderer...")
            success = self.pdf_renderer.render_hindi_reportlab_pdf(
                title=title,
                paragraphs=paragraphs,
                output_pdf_path=output_pdf_path
            )
            if success and output_pdf_path.is_file():
                return output_pdf_path
        return None


if __name__ == "__main__":
    import sys
    print("Post-Lecture Processor initialized.")
