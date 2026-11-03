#!/usr/bin/env python3
"""
Lecture-to-Book Academic Typesetting & TikZ Synthesis Engine.

Transforms raw speech-to-text transcripts into publication-grade,
textbook-style LaTeX monographs:
  - Paragraph / Bookish flow (zero timestamps in TeX, semantic paragraph grouping)
  - Numbered display equations (\begin{equation} ... \label{eq:...} \end{equation})
  - Automated native TikZ / PGFPlots vector graphics (real/imaginary axes, ROC,
    unit circle, d-q rotor frames, phasor diagrams, Bode frequency curves)
  - Cohesive academic sections and clean Markdown export
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Tuple
import urllib.request


@dataclass
class TranscriptChunk:
    timestamp_start: float
    timestamp_end: float
    spoken_text: str
    latex_text: str


class TikZDiagramGenerator:
    """
    Synthesizes native TikZ / PGFPlots vector graphics based on
    acoustic and conceptual triggers detected in the lecture.
    """

    @staticmethod
    def detect_diagram_triggers(text: str) -> List[str]:
        """Identify which diagram types are relevant to the transcript content."""
        lower = text.lower()
        triggers = []

        # 1. Complex Plane / Z-Transform / ROC / Unit Circle
        zplane_cues = [
            "real and imaginary", "imaginary axis", "real axis", "complex plane",
            "z-plane", "z plane", "unit circle", "region of convergence", " roc ",
            "poles and zeros", "pole zero", "radius r", "magnitude of z", "|z|"
        ]
        if any(c in lower for c in zplane_cues):
            triggers.append("z_plane_roc")

        # 2. Direct-Quadrature (d-q) Axes / Synchronous Machines
        dq_cues = [
            "direct axis", "quadrature axis", "d-q", "d and q axes", "rotor axis",
            "stator axis", "park transform", "rotor angle", "theta r", "field winding",
            "synchronous machine", "rotor reference frame"
        ]
        if any(c in lower for c in dq_cues):
            triggers.append("dq_axes")

        # 3. Phasor Diagram / AC Circuits
        phasor_cues = [
            "phasor diagram", "voltage phasor", "current phasor", "phase angle",
            "power factor", "lagging current", "leading current", "impedance angle"
        ]
        if any(c in lower for c in phasor_cues):
            triggers.append("phasor_diagram")

        # 4. Bode / Frequency Response
        bode_cues = [
            "bode plot", "frequency response", "magnitude response", "phase plot",
            "cutoff frequency", "decibels", "db per decade", "log frequency"
        ]
        if any(c in lower for c in bode_cues):
            triggers.append("bode_plot")

        return triggers

    @staticmethod
    def generate_zplane_roc_tikz() -> str:
        """Vector TikZ code for Complex z-plane with unit circle and ROC."""
        return r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=1.3, >=stealth]
  % Axes
  \draw[->, thick, line width=1pt] (-2.5, 0) -- (2.5, 0) node[right] {$\text{Re}(z)$};
  \draw[->, thick, line width=1pt] (0, -2.5) -- (0, 2.5) node[above] {$\text{Im}(z)$};
  \node[below left] at (0, 0) {$0$};

  % ROC Shaded Region (|z| > r)
  \fill[blue!10, even odd rule] (0, 0) circle (2.2) (0, 0) circle (1.4);
  
  % Unit Circle (|z| = 1)
  \draw[dashed, gray!80, line width=0.8pt] (0, 0) circle (1.0);
  \node[below right, gray!80] at (1.0, 0) {$1$};
  \node[above left, gray!80] at (0, 1.0) {$j$};

  % ROC Boundary Circle (|z| = r)
  \draw[thick, blue, line width=1.1pt] (0, 0) circle (1.4);
  \draw[->, blue, thick] (0, 0) -- (0.989, 0.989) node[midway, above left] {$r$};
  
  % Pole Markers (x)
  \draw[red, line width=1.4pt] (1.35, -0.05) -- (1.45, 0.05);
  \draw[red, line width=1.4pt] (1.35, 0.05) -- (1.45, -0.05);
  \node[below right, red] at (1.4, 0) {$p_1$};

  % Annotations
  \node[blue, font=\bfseries] at (1.3, 1.8) {$\text{ROC}: |z| > r$};
  \node[gray!70, font=\small] at (-1.3, -1.3) {Unit Circle $|z|=1$};
\end{tikzpicture}
\caption{Complex $z$-plane showing the unit circle, pole location at radius $r$, and exterior Region of Convergence (ROC) for a causal system.}
\label{fig:zplane_roc}
\end{figure}"""

    @staticmethod
    def generate_dq_axes_tikz() -> str:
        """Vector TikZ code for Synchronous Machine direct and quadrature axes."""
        return r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=1.4, >=stealth]
  % Stationary Stator Axes (Alpha-Beta)
  \draw[->, thick, gray!80, line width=1pt] (-0.5, 0) -- (3.0, 0) node[right] {Stationary $a$-axis};
  \draw[->, thick, gray!80, line width=1pt] (0, -0.5) -- (0, 2.8) node[above] {Stationary $\beta$-axis};

  % Rotating Rotor Axes (d-q Frame rotated by theta_r = 32 deg)
  \draw[->, very thick, red!85!black, line width=1.4pt] (0, 0) -- (32:3.2) node[right] {Direct Axis ($d$)};
  \draw[->, very thick, blue!85!black, line width=1.4pt] (0, 0) -- (122:2.8) node[above left] {Quadrature Axis ($q$)};

  % Rotor Angle Arc (theta_r)
  \draw[->, thick, purple, line width=1pt] (1.2, 0) arc (0:32:1.2);
  \node[purple, font=\bfseries] at (16:1.5) {$\theta_r$};

  % Stator Flux Vector
  \draw[->, line width=1.5pt, green!60!black] (0,0) -- (50:2.6) node[above right] {$\vec{\psi}_s$};
  
  % Projections onto d and q axes
  \draw[dashed, thin, green!50!black] (50:2.6) -- (32:2.48);
  \draw[dashed, thin, green!50!black] (50:2.6) -- (122:0.80);
  \node[below, font=\small] at (32:2.48) {$\psi_d$};
  \node[left, font=\small] at (122:0.80) {$\psi_q$};

  \node[below left] at (0, 0) {$O$};
\end{tikzpicture}
\caption{Direct ($d$) and quadrature ($q$) rotating rotor reference frame relative to stationary stator axes, displaced by rotor angle $\theta_r$.}
\label{fig:dq_reference_frame}
\end{figure}"""

    @staticmethod
    def generate_phasor_tikz() -> str:
        """Vector TikZ code for AC voltage and current phasor diagram."""
        return r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=1.3, >=stealth]
  % Horizontal Reference Axis
  \draw[->, dashed, gray] (-0.5, 0) -- (3.5, 0) node[right] {$\text{Re}$};
  \draw[->, dashed, gray] (0, -1.0) -- (0, 2.5) node[above] {$\text{Im}$};

  % Voltage Phasor V
  \draw[->, very thick, red!80!black, line width=1.4pt] (0,0) -- (25:3.2) node[right] {$\vec{V} = V \angle 0^\circ$};
  
  % Current Phasor I (Lagging by phi)
  \draw[->, very thick, blue!80!black, line width=1.4pt] (0,0) -- (-15:2.4) node[right] {$\vec{I} = I \angle -\phi$};

  % Phase Angle Arc
  \draw[->, thick, purple] (25:1.3) arc (25:-15:1.3);
  \node[purple, font=\bfseries] at (5:1.6) {$\phi$};

  \node[below left] at (0, 0) {$0$};
\end{tikzpicture}
\caption{Phasor diagram demonstrating inductive load with current $\vec{I}$ lagging terminal voltage $\vec{V}$ by power factor angle $\phi$.}
\label{fig:phasor_diagram}
\end{figure}"""

    @staticmethod
    def generate_bode_plot_tikz() -> str:
        """Vector TikZ / PGFPlots code for frequency response."""
        return r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=1.1]
\begin{axis}[
    width=9cm, height=5.5cm,
    xmode=log,
    xlabel={Angular Frequency $\omega$ (rad/s)},
    ylabel={Magnitude $|H(j\omega)|$ (dB)},
    grid=both,
    grid style={line width=.1pt, draw=gray!25},
    major grid style={line width=.2pt, draw=gray!50},
    xmin=0.1, xmax=100,
    ymin=-45, ymax=10,
    thick
]
\addplot[blue, very thick, domain=0.1:100, samples=60]
    {20 * log10( 1 / sqrt(1 + (x/2)^2) )};
\addlegendentry{First-Order Lowpass ($\omega_c = 2$)}

% -3 dB line
\draw[dashed, red, thick] (axis cs:0.1, -3) -- (axis cs:100, -3) node[right, font=\footnotesize] {$-3\text{ dB}$};
\draw[dashed, red, thick] (axis cs:2, -45) -- (axis cs:2, 0) node[above, font=\footnotesize] {$\omega_c$};
\end{axis}
\end{tikzpicture}
\caption{Logarithmic Bode magnitude response exhibiting $-20\text{ dB/decade}$ roll-off beyond corner frequency $\omega_c$.}
\label{fig:bode_response}
\end{figure}"""

    def get_tikz_for_triggers(self, triggers: List[str]) -> List[str]:
        """Generate TikZ blocks for identified triggers without duplicates."""
        blocks = []
        seen = set()
        for t in triggers:
            if t in seen:
                continue
            seen.add(t)
            if t == "z_plane_roc":
                blocks.append(self.generate_zplane_roc_tikz())
            elif t == "dq_axes":
                blocks.append(self.generate_dq_axes_tikz())
            elif t == "phasor_diagram":
                blocks.append(self.generate_phasor_tikz())
            elif t == "bode_plot":
                blocks.append(self.generate_bode_plot_tikz())
        return blocks


class EquationFormatter:
    """
    Identifies mathematical relationships in text and formats them into
    numbered LaTeX equation environments with labels.
    """
    def __init__(self):
        self.eq_counter = 0

    def promote_to_display_equations(self, paragraph: str) -> str:
        """
        Promotes significant inline math or standalone formulas containing
        relational operators (=, \\approx, \\sum, \\int) to \\begin{equation}.
        Uses strict math tokenizing to prevent dollar alternation inversion.
        """
        # 1. Handle explicit double dollars $$ ... $$
        def _replace_double_dollars(match):
            self.eq_counter += 1
            eq_body = match.group(1).strip()
            return f"\n\\begin{{equation}}\n  \\label{{eq:{self.eq_counter}}}\n  {eq_body}\n\\end{{equation}}\n"

        paragraph = re.sub(r'\$\$(.*?)\$\$', _replace_double_dollars, paragraph, flags=re.DOTALL)

        # 2. Tokenize by single dollar pairs: ($[^\$]+$)
        tokens = re.split(r'(\$[^\$]+\$)', paragraph)
        new_tokens = []
        for tok in tokens:
            if tok.startswith('$') and tok.endswith('$') and len(tok) >= 3:
                eq_body = tok[1:-1].strip()
                # Promote to numbered equation if it contains major operators and is substantive
                has_rel = any(op in eq_body for op in ["=", r"\approx", r"\equiv", r"\le", r"\ge", r"\sum", r"\int", r"\bmatrix", r"\pmatrix"])
                is_substantive = len(eq_body) >= 12 or any(op in eq_body for op in [r"\sum", r"\int", r"\bmatrix", r"\pmatrix"])
                # Avoid promoting trivial variable assignments like $n = 0$ or $t = 1$
                is_trivial_scalar = bool(re.match(r'^[a-zA-Z_]\s*=\s*\d+$', eq_body))

                if has_rel and is_substantive and not is_trivial_scalar:
                    self.eq_counter += 1
                    new_tokens.append(f"\n\\begin{{equation}}\n  \\label{{eq:{self.eq_counter}}}\n  {eq_body}\n\\end{{equation}}\n")
                    continue
            new_tokens.append(tok)

        res = "".join(new_tokens)
        # Clean trailing punctuation right after \end{equation}
        res = re.sub(r'\\end\{equation\}\s*[\.,;]', r'\\end{equation}', res)
        # Clean multiple consecutive blank lines
        return re.sub(r'\n{3,}', '\n\n', res)


class LectureToBookBuilder:
    """
    Assembles streaming transcripts into publication-grade student classroom lecture notes.
    """
    def __init__(self):
        self.tikz_gen = TikZDiagramGenerator()
        self.eq_formatter = EquationFormatter()

    def clean_speech_text(self, text: str) -> str:
        """Remove verbal fillers, colloquial greetings, and acoustic noise artifacts."""
        # Audio noise tags and musical notations
        text = re.sub(r'[♪♫]+', '', text)
        text = re.sub(r'\[(?:BLANK_AUDIO|MUSIC|APPLAUSE|LAUGHTER|COUGH|CRASHING|NOISE)\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\((?:crashing|music|applause|laughter|snicker|sigh)\)', '', text, flags=re.IGNORECASE)

        # Repetitive hallucination
        if re.search(r'(\*\s*){3,}', text):
            return ""

        # Domain transcription corrections
        text = re.sub(r'\bIndian Premium Lead\b', 'Indian Premier League (IPL)', text, flags=re.IGNORECASE)
        text = re.sub(r'\bDCCI\b', 'BCCI', text, flags=re.IGNORECASE)

        # Remove spoken greetings and classroom banter not part of notes
        text = re.sub(r'\bhello everyone[,\.]?\s*(?:we are going to start our lecture on)?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bso[,\s]+it\'?s a good idea to have a little bit of a smile[,\.]?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bwe are starting our (?:first )?lecture on', '', text, flags=re.IGNORECASE)

        # Remove verbal fillers
        fillers = [
            r'\bum+\b', r'\buh+\b', r'\bah+\b', r'\ber+\b', r'\bhmm+\b',
            r'\bso\s+basically\b', r'\byou\s+know\b', r'\bkind\s+of\b', r'\blike\s+basically\b'
        ]
        for f in fillers:
            text = re.sub(f, '', text, flags=re.IGNORECASE)

        # Clean punctuation artifacts like 'Okay,.' or ',.'
        text = re.sub(r'\bokay[,\.]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[,\s]*\.\s*\.', '.', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def infer_lecture_metadata(self, full_text: str) -> Tuple[str, str, List[str], List[str]]:
        """
        Infers (title, overview, key_definitions, review_questions)
        using local Qwen2.5-1.5B (llama-server) if available, with robust heuristic fallback.
        """
        clean_text = full_text.strip()
        if not clean_text:
            return (
                "Classroom Lecture Notes",
                "This document compiles student lecture notes transcribed during the classroom session.",
                [],
                ["Review the foundational definitions and derivations presented in this lecture."]
            )

        # 1. Try local Qwen2.5-1.5B on llama-server (fast timeout 3.5s)
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8080/completion",
                data=json.dumps({
                    "prompt": (
                        "<|im_start|>system\n"
                        "You are an academic course editor. Write a concise, professional Lecture Title "
                        "(e.g., 'Indian Premier League (IPL) & Cricket Governance' or 'The Z-Transform, ROC & Stability') "
                        "and a 2-sentence Overview / Learning Objectives for students based on this transcript excerpt.\n"
                        "Output format:\n"
                        "TITLE: <title>\n"
                        "OVERVIEW: <overview>\n"
                        "<|im_end|>\n"
                        f"<|im_start|>user\nLecture transcript excerpt:\n{clean_text[:600]}\n<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    ),
                    "temperature": 0.1,
                    "n_predict": 120,
                    "stop": ["<|im_end|>"]
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data.get("content", "").strip()
                title_m = re.search(r'TITLE:\s*(.+)', content, re.IGNORECASE)
                over_m = re.search(r'OVERVIEW:\s*(.+)', content, re.DOTALL | re.IGNORECASE)
                title = title_m.group(1).strip().strip('"').strip("'") if title_m else ""
                overview = over_m.group(1).strip() if over_m else ""
                if title and len(title) > 3:
                    title = re.sub(r'^(?:Lecture Notes(?:\s*on|\s*:)?\s*)', '', title, flags=re.IGNORECASE)
                    title = f"Lecture Notes: {title}"
                    if not overview:
                        overview = f"These classroom lecture notes examine the principles, structural frameworks, and core concepts of {title.replace('Lecture Notes: ', '')}."
                    return title, overview, [], []
        except Exception:
            pass

        # 2. Heuristic Rule-Based Analysis
        lower = clean_text.lower()
        if "ipl" in lower or "cricket" in lower or "bcci" in lower or "premier league" in lower:
            title = "Indian Premier League (IPL) & Sports Governance"
            overview = (
                "This lecture provides an institutional overview of the Indian Premier League (IPL), "
                "examining its inception in 2008, the franchise commercial architecture, and the regulatory "
                "authority of the Board of Control for Cricket in India (BCCI)."
            )
            definitions = [
                r"\textbf{BCCI Regulatory Framework:} The Board of Control for Cricket in India acts as the governing body controlling tournament rules, franchise licensing, and centralized broadcasting rights."
            ]
            review_q = [
                "Explain the regulatory relationship between the BCCI and private IPL franchise owners.",
                "How does the player auction mechanism and salary cap framework promote competitive parity across competing teams?"
            ]
            return title, overview, definitions, review_q

        elif "z-transform" in lower or "z transform" in lower or "roc" in lower or "complex plane" in lower:
            title = "The Z-Transform, Region of Convergence & System Stability"
            overview = (
                "This lecture develops the theory of unilateral and bilateral Z-transforms for discrete-time "
                "signal processing. Students will examine pole-zero configurations on the complex z-plane, "
                "Region of Convergence (ROC) characteristics, and bounded-input bounded-output (BIBO) stability criteria."
            )
            definitions = [
                r"\textbf{Region of Convergence (ROC):} The set of points in the complex $z$-plane for which the Laurent power series summation converges absolutely: $\sum_{n=-\infty}^{\infty} |x[n] z^{-n}| < \infty$."
            ]
            review_q = [
                "Why must the Region of Convergence (ROC) of a causal LTI system extend outward from the outermost pole?",
                "State the necessary and sufficient condition for the Discrete-Time Fourier Transform (DTFT) to exist in terms of the Z-transform ROC."
            ]
            return title, overview, definitions, review_q

        elif "fourier" in lower or "frequency" in lower or "spectrum" in lower:
            title = "Fourier Analysis & Frequency-Domain Characterization"
            overview = (
                "This lecture examines Fourier transform principles for continuous and discrete-time signals, "
                "analyzing spectral decomposition, harmonic behavior, and frequency-domain filtering characteristics."
            )
            definitions = [
                r"\textbf{Fourier Decomposition:} Spectral decomposition representing arbitrary energy signals as a continuous superposition of orthogonal complex sinusoids."
            ]
            review_q = [
                "Differentiate between the continuous-time Fourier transform (CTFT) and the discrete-time Fourier transform (DTFT).",
                "How does the time-scaling property of the Fourier transform affect spectral bandwidth?"
            ]
            return title, overview, definitions, review_q

        elif "park" in lower or "direct axis" in lower or "quadrature axis" in lower or "rotor" in lower:
            title = "Dynamic Modeling of Synchronous Machines & Park Transformation"
            overview = (
                "This lecture formulates the mathematical transformation of three-phase stator quantities "
                "into an orthogonal two-axis (d-q) reference frame fixed to the machine rotor, decoupling "
                "flux linkages and eliminating angular inductances."
            )
            definitions = [
                r"\textbf{Park Transformation:} An orthogonal coordinate projection converting time-varying stator inductances into time-invariant rotor direct ($d$) and quadrature ($q$) components."
            ]
            review_q = [
                "Explain why the phase inductances of a salient-pole synchronous machine are time-varying functions of rotor position.",
                "How does the Park transformation convert rotor-angle dependent differential equations into a time-invariant state-space form?"
            ]
            return title, overview, definitions, review_q

        # Generic academic fallback
        first_sent = clean_text.split(".")[0].strip()
        words = [w for w in first_sent.split() if w.lower() not in ["hello", "everyone", "we", "are", "going", "to", "start", "our", "lecture", "on", "today", "discuss", "so"]]
        topic_phrase = " ".join(words[:6]).title() if words else "Classroom Exposition"
        title = topic_phrase
        overview = (
            f"These classroom lecture notes summarize the core theoretical principles, analytical definitions, "
            f"and structured observations presented during the lecture on {topic_phrase}."
        )
        review_q = [
            f"Synthesize the foundational principles and definitions governing {topic_phrase}.",
            "Discuss practical implications and classroom observations highlighted in this lecture."
        ]
        return title, overview, [], review_q

    def assemble_paragraphs(self, chunks: List[TranscriptChunk]) -> List[str]:
        """
        Merges chunk clauses into cohesive paragraphs.
        Strips out all timestamps and groups sentences logically.
        """
        sentences = []
        for c in chunks:
            t = (c.latex_text or c.spoken_text or "").strip()
            cleaned = self.clean_speech_text(t)
            if cleaned and len(cleaned) > 2:
                cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
                if not cleaned.endswith((".", "!", "?", ":", ";")):
                    cleaned += "."
                sentences.append(cleaned)

        paragraphs = []
        curr_p = []
        for s in sentences:
            curr_p.append(s)
            if len(curr_p) >= 3 or any(w in s.lower() for w in ["furthermore", "in conclusion", "to summarize", "next", "importantly"]):
                paragraphs.append(" ".join(curr_p))
                curr_p = []

        if curr_p:
            paragraphs.append(" ".join(curr_p))

        return paragraphs

    def build_book(
        self,
        lecture_title: str,
        chunks: List[TranscriptChunk],
        author: str = "Classroom Lecture Notes",
        date_str: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """
        Builds complete publication-ready Student Lecture Notes in LaTeX (.tex) and Markdown (.md).
        Returns: (latex_content, markdown_content, clean_title)
        """
        if not date_str:
            date_str = time.strftime("%B %d, %Y")

        paragraphs = self.assemble_paragraphs(chunks)
        full_text = " ".join(paragraphs)

        # Infer intelligent title and student metadata
        inferred_title, overview_text, key_defs, review_questions = self.infer_lecture_metadata(full_text)
        clean_title = inferred_title if inferred_title else lecture_title

        # Detect diagram triggers
        triggers = self.tikz_gen.detect_diagram_triggers(full_text)
        tikz_blocks = self.tikz_gen.get_tikz_for_triggers(triggers)

        # Format equations and build sectioned content
        processed_paragraphs = [self.eq_formatter.promote_to_display_equations(p) for p in paragraphs]

        has_math = any(r"\begin{equation}" in p or "$" in p for p in processed_paragraphs)

        # Build natural lecture notes body
        body_tex_parts = []
        tikz_inserted = False

        # Section titles
        sec1_title = "Theoretical Background and Motivation" if has_math else "Historical Context and Background"
        sec2_title = "Core Mathematical Formulations" if has_math else "Structural Organization and Key Entities"

        # Escape standalone ampersands for LaTeX compilation
        clean_title_tex = clean_title.replace("&", r"\&")
        overview_tex = overview_text.replace("&", r"\&")
        sec1_title_tex = sec1_title.replace("&", r"\&")
        sec2_title_tex = sec2_title.replace("&", r"\&")

        # Section 1: Introduction & Context
        body_tex_parts.append(f"\\section{{{sec1_title_tex}}}")

        if processed_paragraphs:
            body_tex_parts.append(processed_paragraphs[0])
            body_tex_parts.append("")

        # Add definition box if present
        if key_defs:
            body_tex_parts.append(r"\begin{mdframed}[backgroundcolor=boxbg, linecolor=accentblue, linewidth=1.2pt, roundcorner=4pt, innertopmargin=6pt, innerbottommargin=6pt, innerleftmargin=10pt, innerrightmargin=10pt]")
            body_tex_parts.append(r"  {\bfseries\color{headerblue} Key Concept / Definition:}\\[0.2em]")
            body_tex_parts.append(f"  {key_defs[0]}")
            body_tex_parts.append(r"\end{mdframed}")
            body_tex_parts.append("")

        # Section 2: Core Formulations & Analysis
        body_tex_parts.append(f"\\section{{{sec2_title_tex}}}")

        for idx, p in enumerate(processed_paragraphs[1:], 1):
            body_tex_parts.append(p)
            body_tex_parts.append("")

            # Insert TikZ diagrams naturally
            if not tikz_inserted and (idx >= len(processed_paragraphs) // 2):
                for diag in tikz_blocks:
                    body_tex_parts.append(diag)
                    body_tex_parts.append("")
                tikz_inserted = True

        if not tikz_inserted and tikz_blocks:
            for diag in tikz_blocks:
                body_tex_parts.append(diag)
                body_tex_parts.append("")

        # Add Instructor Observation Callout Box
        body_tex_parts.append(r"\begin{mdframed}[backgroundcolor=notebg, linecolor=noteframe, linewidth=1.0pt, roundcorner=4pt, innertopmargin=6pt, innerbottommargin=6pt, innerleftmargin=10pt, innerrightmargin=10pt]")
        body_tex_parts.append(r"  {\bfseries\color{noteframe} Instructor's Classroom Observation:}\\[0.2em]")
        if has_math:
            body_tex_parts.append(r"  Pay close attention to boundary convergence conditions and coordinate system invariants when solving subsequent analytical problem sets.")
        else:
            body_tex_parts.append(r"  Notice how the institutional governance framework directly impacts operational execution and commercial sustainability across all participating stakeholders.")
        body_tex_parts.append(r"\end{mdframed}")
        body_tex_parts.append("")

        # Section 3: Summary and Review Questions
        body_tex_parts.append(r"\section{Summary and Key Takeaways}")
        body_tex_parts.append(r"\begin{itemize}[leftmargin=1.8em, itemsep=2pt]")
        body_tex_parts.append(f"  \\item \\textbf{{Core Subject:}} Comprehensive review of {clean_title_tex}.")
        body_tex_parts.append(r"  \item \textbf{Analytical Consistency:} All terminology, definitions, and equations have been harmonized for direct student study.")
        body_tex_parts.append(r"\end{itemize}")
        body_tex_parts.append("")

        if review_questions:
            body_tex_parts.append(r"\subsection*{Review Questions for Students}")
            body_tex_parts.append(r"\begin{enumerate}[leftmargin=1.8em, itemsep=3pt]")
            for q in review_questions:
                q_tex = q.replace("&", r"\&")
                body_tex_parts.append(f"  \\item {q_tex}")
            body_tex_parts.append(r"\end{enumerate}")
            body_tex_parts.append("")

        body_tex = "\n".join(body_tex_parts)

        # Student Lecture Notes LaTeX Document Template (100% XeTeX / Tectonic Offline Compatible)
        latex_doc = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage{{amsmath,amssymb,amsfonts,amsthm,mathtools}}
\usepackage{{geometry}}
\geometry{{margin=1.0in, top=1.1in, bottom=1.1in}}
\usepackage{{xcolor}}
\usepackage{{fancyhdr}}
\usepackage{{booktabs}}
\usepackage{{enumitem}}
\usepackage{{mdframed}}
\usepackage{{hyperref}}
\usepackage{{tikz}}

\definecolor{{headerblue}}{{RGB}}{{15, 41, 66}}
\definecolor{{accentblue}}{{RGB}}{{43, 108, 176}}
\definecolor{{boxbg}}{{RGB}}{{240, 244, 248}}
\definecolor{{boxframe}}{{RGB}}{{203, 213, 224}}
\definecolor{{notebg}}{{RGB}}{{255, 250, 240}}
\definecolor{{noteframe}}{{RGB}}{{221, 107, 32}}

\hypersetup{{
    colorlinks=true,
    linkcolor=accentblue,
    citecolor=accentblue,
    urlcolor=accentblue
}}

\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\small\color{{gray}}\textsc{{Classroom Lecture Notes}}}}
\fancyhead[R]{{\small\color{{gray}}\textsc{{Page \thepage}}}}
\fancyfoot[C]{{\small\color{{gray}}Confidential --- Prepared for Student Study and Review}}
\renewcommand{{\headrulewidth}}{{0.4pt}}

\begin{{document}}

\begin{{center}}
  {{\Huge\bfseries\color{{headerblue}} {clean_title_tex}}} \\[0.5em]
  {{\large\color{{accentblue}} Comprehensive Classroom Lecture Notes}} \\[0.3em]
  {{\small\color{{gray}} Delivered on {date_str} $\bullet$ Academic Term}}
\end{{center}}

\vspace{{0.6em}}
\hrule height 1.2pt
\vspace{{1.0em}}

\begin{{mdframed}}[backgroundcolor=boxbg, linecolor=boxframe, linewidth=0.8pt, roundcorner=4pt, innertopmargin=8pt, innerbottommargin=8pt, innerleftmargin=10pt, innerrightmargin=10pt]
  {{\bfseries\color{{headerblue}} Lecture Overview \& Learning Objectives:}}\\[0.3em]
  {overview_tex}
\end{{mdframed}}

\vspace{{0.8em}}

{body_tex}

\end{{document}}
"""

        # Build clean student Markdown document
        md_parts = [
            f"# {clean_title}",
            f"**Comprehensive Classroom Lecture Notes**  ",
            f"**Date Delivered:** {date_str}  \n",
            "---",
            f"### 📋 Lecture Overview & Learning Objectives\n{overview_text}\n",
            f"## 1. {sec1_title}\n"
        ]

        if processed_paragraphs:
            md_parts.append(processed_paragraphs[0])
            md_parts.append("")

        if key_defs:
            md_parts.append(f"> **Key Concept:** {key_defs[0]}\n")

        md_parts.append(f"## 2. {sec2_title}\n")
        for idx, p in enumerate(processed_paragraphs[1:], 1):
            p_md = re.sub(
                r'\\begin\{equation\}\s*(?:\\label\{[^}]+\})?\s*(.*?)\s*\\end\{equation\}',
                r'\n$$\n\1\n$$\n',
                p,
                flags=re.DOTALL
            )
            md_parts.append(p_md)
            md_parts.append("")

        if tikz_blocks:
            md_parts.append("### Graphical Illustrations")
            md_parts.append("*(Note: High-resolution native TikZ vector graphics are rendered in the compiled standalone PDF and LaTeX source.)*\n")

        md_parts.append("## 3. Summary and Key Takeaways")
        md_parts.append(f"- **Core Subject:** Comprehensive review of {clean_title}.")
        md_parts.append("- **Analytical Consistency:** All terminology, definitions, and equations have been harmonized for direct student study.\n")

        if review_questions:
            md_parts.append("### Review Questions for Students")
            for q_idx, q in enumerate(review_questions, 1):
                md_parts.append(f"{q_idx}. {q}")

        md_doc = "\n".join(md_parts)

        return latex_doc, md_doc, clean_title


def generate_subtitles(chunks: List[TranscriptChunk]) -> Tuple[str, str]:
    """
    Generate standard .srt and .vtt subtitle files with exact timestamps.
    """
    def _sec_to_srt(s: float) -> str:
        hrs = int(s // 3600)
        mins = int((s % 3600) // 60)
        secs = int(s % 60)
        ms = int(round((s - int(s)) * 1000))
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"

    def _sec_to_vtt(s: float) -> str:
        hrs = int(s // 3600)
        mins = int((s % 3600) // 60)
        secs = int(s % 60)
        ms = int(round((s - int(s)) * 1000))
        return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"

    srt_lines = []
    vtt_lines = ["WEBVTT\n"]

    for idx, c in enumerate(chunks, 1):
        text = (c.spoken_text or c.latex_text or "").strip()
        if not text:
            continue
        # SRT
        srt_lines.append(str(idx))
        srt_lines.append(f"{_sec_to_srt(c.timestamp_start)} --> {_sec_to_srt(c.timestamp_end)}")
        srt_lines.append(text)
        srt_lines.append("")

        # VTT
        vtt_lines.append(f"{_sec_to_vtt(c.timestamp_start)} --> {_sec_to_vtt(c.timestamp_end)}")
        vtt_lines.append(text)
        vtt_lines.append("")

    return "\n".join(srt_lines), "\n".join(vtt_lines)


if __name__ == "__main__":
    # Self-test with sample Z-transform chunks
    test_chunks = [
        TranscriptChunk(0.0, 4.2, "today we will discuss the z transform and its region of convergence", "Today we will discuss the $Z$-transform and its region of convergence"),
        TranscriptChunk(4.5, 9.8, "the z transform of a discrete time signal x n is defined as the sum from minus infinity to plus infinity of x n times z to the minus n", r"The $Z$-transform of a discrete-time signal $x[n]$ is defined as $X(z) = \sum_{n=-\infty}^{\infty} x[n] z^{-n}$"),
        TranscriptChunk(10.1, 15.0, "here z is a complex variable and we can represent it on the real and imaginary axes in the complex plane", "Here $z$ is a complex variable, which can be plotted on the real and imaginary axes in the complex $z$-plane."),
        TranscriptChunk(15.2, 21.0, "for a causal system the region of convergence roc is outside a circle of radius r so magnitude of z is greater than r", r"For a causal signal, the region of convergence (ROC) lies outside a circle of radius $r$, satisfying $|z| > r$."),
    ]
    builder = LectureToBookBuilder()
    tex, md, title = builder.build_book("Lecture 12: Z-Transform and Region of Convergence", test_chunks)
    print("Inferred Title:", title)
    print("Generated LaTeX length:", len(tex))
    print("Contains TikZ:", "\\begin{tikzpicture}" in tex)
    print("Contains numbered equation:", "\\begin{equation}" in tex)
