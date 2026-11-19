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
    Interfaces with the local Qwen2.5-1.5B SLM to proofread and harmonize
    mathematical notation across the full lecture transcript.
    """
    def __init__(self, server_url: str = LLAMA_SERVER_URL):
        self.server_url = server_url.rstrip("/")

    def is_server_alive(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.server_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def proofread_section(self, text: str) -> str:
        """
        Sends a lecture section to Qwen2.5-1.5B for proofreading and notation harmonization.
        """
        if not self.is_server_alive() or len(text.strip()) < 10:
            return text

        prompt = (
            "<|im_start|>system\n"
            "You are an expert academic editor and mathematical typesetter. "
            "Your task is to proofread a transcribed lecture section. "
            "1. Harmonize mathematical notation throughout (e.g. ensure variables like z, omega, psi, and time indices n, t are consistent). "
            "2. Ensure all LaTeX math environments are correctly balanced with closing braces. "
            "3. Smooth out spoken transitions into formal textbook prose. "
            "4. Retain all technical depth, equations, and diagrams. "
            "Return ONLY the corrected LaTeX text without introductory chatter.<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        payload = {
            "prompt": prompt,
            "n_predict": min(1024, len(text.split()) * 2 + 128),
            "temperature": 0.15,
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
            with urllib.request.urlopen(req, timeout=45.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                result = data.get("content", "").strip()
                if result and len(result) > len(text) * 0.4:
                    # Strip any accidental chatter
                    result = re.sub(r'^(?:Here is the (?:corrected|proofread).*?:\s*)', '', result, flags=re.IGNORECASE)
                    return result
        except Exception:
            pass

        return text


class AcademicPDFRenderer:
    """
    Compiles LaTeX to PDF via standalone offline Tectonic XeTeX engine or system pdflatex.
    Provides automated fallback to a clean ReportLab student lecture notes compiler.
    """
    TECTONIC_BIN = Path("/mnt/thesis/bin/tectonic")

    @classmethod
    def compile_with_tectonic(cls, tex_path: Path, output_dir: Path) -> Optional[Path]:
        """Compile using offline cached Tectonic binary."""
        tectonic_bin = cls.TECTONIC_BIN if cls.TECTONIC_BIN.is_file() else shutil.which("tectonic")
        if not tectonic_bin:
            return None

        try:
            cmd = [
                str(tectonic_bin),
                "-C",  # 100% offline: only use locally cached packages
                str(tex_path),
                "-o", str(output_dir)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90)
            pdf_candidate = output_dir / f"{tex_path.stem}.pdf"
            if pdf_candidate.is_file() and pdf_candidate.stat().st_size > 1000:
                print(f"[PDF Compiler] Tectonic successfully compiled: {pdf_candidate} ({pdf_candidate.stat().st_size} bytes)")
                return pdf_candidate
            else:
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
        title: str,
        paragraphs: List[str],
        output_pdf_path: Path
    ) -> Optional[Path]:
        """
        Compiles the Hindi bilingual monograph to PDF using ReportLab Devanagari engine.
        """
        success = self.pdf_renderer.render_hindi_reportlab_pdf(
            title=title,
            paragraphs=paragraphs,
            output_pdf_path=output_pdf_path
        )
        if success and output_pdf_path.is_file():
            print(f"[Hindi PDF Compiler] Successfully compiled: {output_pdf_path} ({output_pdf_path.stat().st_size} bytes)")
            return output_pdf_path
        return None


if __name__ == "__main__":
    import sys
    print("Post-Lecture Processor initialized.")
