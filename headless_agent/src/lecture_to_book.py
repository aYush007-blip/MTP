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


def balance_math_delimiters(text: str) -> str:
    """Ensure all inline math delimiters ($...$) in text are strictly balanced."""
    text = re.sub(r"\$\s*\$", "", text)
    placeholders = []
    def _save_dd(m):
        placeholders.append(m.group(0))
        return f"__DD_{len(placeholders)-1}__"
    text = re.sub(r"\$\$.*?\$\$", _save_dd, text, flags=re.DOTALL)

    indices = [m.start() for m in re.finditer(r"(?<!\\)\$", text)]
    if len(indices) % 2 != 0:
        last_idx = indices[-1]
        rem = text[last_idx+1:]
        m_word = re.match(r"^([a-zA-Z0-9\^_\+\-\{\}\\]+)", rem)
        if m_word:
            end_pos = last_idx + 1 + len(m_word.group(1))
            text = text[:end_pos] + "$" + text[end_pos:]
        else:
            text = text[:last_idx] + text[last_idx+1:]

    for idx, ph in enumerate(placeholders):
        text = text.replace(f"__DD_{idx}__", ph)
    return text


def escape_latex_text(paragraph: str) -> str:
    """
    Sanitizes LaTeX text and math:
    - Balances math delimiters ($...$)
    - Replaces unicode Greek and math symbols with proper LaTeX commands
    - Wraps stray math expressions (e.g. e^(j\\omega t), x^n, R^n) into inline math
    - Escapes special LaTeX characters (&, %, #, _, ^) outside math mode
    """
    paragraph = balance_math_delimiters(paragraph)
    unicode_map = {
        'ω': r'\omega', 'Ω': r'\Omega', 'θ': r'\theta', 'π': r'\pi',
        'σ': r'\sigma', 'λ': r'\lambda', 'μ': r'\mu', 'α': r'\alpha',
        'β': r'\beta', 'γ': r'\gamma', 'δ': r'\delta', 'Δ': r'\Delta',
        '∞': r'\infty', '±': r'\pm', '≤': r'\le', '≥': r'\ge',
        '≠': r'\ne', '×': r'\times', '•': r'\cdot', '−': '-'
    }

    tokens = re.split(r'(\${1,2}[^\$]+\${1,2})', paragraph)
    result = []
    for tok in tokens:
        if tok.startswith('$') and tok.endswith('$'):
            s = tok
            for uc, lc in unicode_map.items():
                s = s.replace(uc, " " + lc + " ")
            result.append(s)
        else:
            s = tok
            for uc, lc in unicode_map.items():
                s = s.replace(uc, f" ${lc}$ ")
            # Wrap stray math commands (\frac{...}{...}, \int, \sum, \lim, \partial) if found outside math mode
            s = re.sub(r'(\\frac\{[^{}]+\}\{[^{}]+\})', r'$\1$', s)
            s = re.sub(r'(\\int\b)', r'$\1$', s)
            s = re.sub(r'(\\sum\b)', r'$\1$', s)
            s = re.sub(r'(\\lim\b[^\s]+)', r'$\1$', s)
            s = re.sub(r'\b([a-zA-Z])\^([a-zA-Z0-9\-\+\*]+)\b', r'$\1^{\2}$', s)
            s = re.sub(r'([a-zA-Z])\^\(([^\)]+)\)', r'$\1^{\2}$', s)
            s = re.sub(r'([a-zA-Z])\^\{([^}]+)\}', r'$\1^{\2}$', s)
            s = re.sub(r'(?<!\\)&', r'\&', s)
            s = re.sub(r'(?<!\\)%', r'\%', s)
            s = re.sub(r'(?<!\\)#', r'\#', s)
            s = re.sub(r'(?<!\\)_', r'\_', s)
            s = re.sub(r'(?<!\\)\^', r'\\textasciicircum{}', s)
            result.append(s)
    out = "".join(result)
    return balance_math_delimiters(out)


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

        # 5. Feedback Control Systems / Block Diagrams
        control_cues = [
            "closed loop", "closed-loop", "open loop", "open-loop", "feedback",
            "plant", "controller", "disturbance", "steady state error", "sensitivity",
            "loop gain", "tracking", "regulation"
        ]
        if any(c in lower for c in control_cues):
            triggers.append("control_feedback_loop")

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

    @staticmethod
    def generate_control_system_tikz() -> str:
        """Vector TikZ code for canonical closed-loop feedback architecture."""
        return r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[auto, node distance=1.8cm, >=Stealth]
    \node [coordinate] (input) {};
    \node [draw, fill=blue!5, thick, circle, right=1.0cm of input] (sum_ref) {$+$};
    \node [draw, fill=blue!5, thick, rectangle, minimum height=0.9cm, minimum width=1.5cm, right=1.2cm of sum_ref] (controller) {$G_c(s)$};
    \node [draw, fill=blue!5, thick, circle, right=1.5cm of controller] (sum_dist) {$+$};
    \node [coordinate, above=1.2cm of sum_dist] (dist_in) {};
    \node [draw, fill=blue!5, thick, rectangle, minimum height=0.9cm, minimum width=1.5cm, right=1.5cm of sum_dist] (plant) {$G(s)$};
    \node [coordinate, right=1.2cm of plant] (branch) {};
    \node [coordinate, right=1.5cm of branch] (output) {};
    \node [draw, fill=blue!5, thick, rectangle, minimum height=0.9cm, minimum width=1.5cm, below=1.2cm of controller] (sensor) {$H(s)$};

    \draw [->, thick] (input) -- node[above] {$R(s)$} (sum_ref);
    \draw [->, thick] (sum_ref) -- node[above] {$E(s)$} (controller);
    \draw [->, thick] (controller) -- node[above] {$U(s)$} (sum_dist);
    \draw [->, thick] (dist_in) -- node[right] {$D(s)$} (sum_dist);
    \draw [->, thick] (sum_dist) -- (plant);
    \draw [->, thick] (plant) -- (branch) -- node[above] {$Y(s)$} (output);
    \draw [->, thick] (branch) |- (sensor);
    \draw [->, thick] (sensor) -| node[pos=0.95, left] {$-$} (sum_ref);
\end{tikzpicture}
\caption{Canonical closed-loop feedback architecture with reference $R(s)$, disturbance $D(s)$, controller $G_c(s)$, plant $G(s)$, and sensor feedback $H(s)$.}
\label{fig:feedback_block_diagram}
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
            elif t == "control_feedback_loop":
                blocks.append(self.generate_control_system_tikz())
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
                eq_body = tok[1:-1].replace('$', '').strip()
                # Clean unclosed cases or matrix environments
                eq_body = re.sub(r'\\begin\{cases\}\.?\s*$', '', eq_body).strip()
                for env in ["cases", "matrix", "pmatrix", "bmatrix", "aligned"]:
                    beg = eq_body.count(rf"\begin{{{env}}}")
                    end = eq_body.count(rf"\end{{{env}}}")
                    if beg > end:
                        eq_body += rf"\end{{{env}}}" * (beg - end)

                if not eq_body:
                    continue

                # Promote to numbered equation if it contains major operators and is substantive
                has_rel = any(op in eq_body for op in ["=", r"\approx", r"\equiv", r"\le", r"\ge", r"\sum", r"\int", r"\bmatrix", r"\pmatrix"])
                is_substantive = len(eq_body) >= 12 or any(op in eq_body for op in [r"\sum", r"\int", r"\bmatrix", r"\pmatrix"])
                # Avoid promoting trivial variable assignments like $n = 0$ or $t = 1$
                is_trivial_scalar = bool(re.match(r'^[a-zA-Z_]\s*=\s*\d+$', eq_body))

                if has_rel and is_substantive and not is_trivial_scalar:
                    self.eq_counter += 1
                    new_tokens.append(f"\n\\begin{{equation}}\n  \\label{{eq:{self.eq_counter}}}\n  {eq_body}\n\\end{{equation}}\n")
                    continue
                else:
                    new_tokens.append(f"${eq_body}$")
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
        """Clean acoustic noise tags and artifacts while strictly preserving 100% of spoken words."""
        # Audio noise tags and musical notations
        text = re.sub(r'[♪♫]+', '', text)
        text = re.sub(r'\[(?:BLANK_AUDIO|MUSIC|APPLAUSE|LAUGHTER|COUGH|CRASHING|NOISE)\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\((?:crashing|music|applause|laughter|snicker|sigh)\)', '', text, flags=re.IGNORECASE)

        # Repetitive hallucination (e.g. * * * * or repetition loops)
        if re.search(r'(\*\s*){3,}', text):
            return ""

        # Remove hesitation fillers (um, uh, er) while keeping all spoken content
        fillers = [r'\bum+\b', r'\buh+\b', r'\bah+\b', r'\ber+\b', r'\bhmm+\b']
        for f in fillers:
            text = re.sub(f, '', text, flags=re.IGNORECASE)

        # Clean stray dots and whitespace
        text = re.sub(r'[,\s]*\.\s*\.', '.', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def infer_lecture_title(self, full_text: str) -> str:
        """
        Infers a concise, clean academic title from the transcript.
        Never includes 'Lecture Notes', 'Summary', or 'Comprehensive'.
        """
        clean_text = full_text.strip()
        if not clean_text:
            return "Classroom Exposition"

        # 1. Try local Qwen2.5-1.5B on llama-server (fast timeout 3.0s)
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8080/completion",
                data=json.dumps({
                    "prompt": (
                        "<|im_start|>system\n"
                        "You are an academic course editor. Write a concise, 3 to 6 word academic title for this lecture topic. "
                        "Do NOT include the words 'Lecture Notes', 'Summary', 'Comprehensive', or any introductory chatter. "
                        "Return ONLY the clean topic title.<|im_end|>\n"
                        f"<|im_start|>user\nLecture transcript excerpt:\n{clean_text[:600]}\n<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    ),
                    "temperature": 0.1,
                    "n_predict": 30,
                    "stop": ["<|im_end|>", "\n"]
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data.get("content", "").strip().strip('"').strip("'")
                content = re.sub(r'^(?:Title|Topic|Lecture(?:\s+Notes)?(?:\s*on|\s*:)?)\s*', '', content, flags=re.IGNORECASE).strip()
                if content and len(content) > 3 and "notes" not in content.lower():
                    return content
        except Exception:
            pass

        # 2. Heuristic Rule-Based Fallbacks (domain STEM topics prioritized with word boundaries)
        lower = clean_text.lower()
        if re.search(r'\b(?:z[- ]transform|roc|region of convergence|complex plane|unit circle)\b', lower):
            return "The Z-Transform, Region of Convergence & System Stability"
        elif re.search(r'\b(?:fourier|frequency domain|spectrum)\b', lower):
            return "Fourier Analysis & Frequency-Domain Characterization"
        elif re.search(r'\b(?:park|direct axis|quadrature axis|rotor reference)\b', lower):
            return "Dynamic Modeling of Synchronous Machines & Park Transformation"
        elif re.search(r'\b(?:ipl|cricket|bcci|premier league)\b', lower):
            return "Indian Premier League (IPL) & Sports Governance"

        # Generic academic fallback: take topic keywords from opening sentence
        first_sent = clean_text.split(".")[0].strip()
        words = [w for w in first_sent.split() if w.lower() not in ["hello", "everyone", "we", "are", "going", "to", "start", "our", "lecture", "on", "today", "discuss", "so", "talk", "about"]]
        if words:
            return " ".join(words[:5]).title()
        return "Classroom Exposition"

    def assemble_paragraphs(self, chunks: List[TranscriptChunk]) -> List[str]:
        """
        Merges chunk clauses into cohesive paragraphs.
        Preserves 100% of spoken content from the lecturer in exact chronological order.
        """
        paragraphs = []
        curr_sentences = []

        for c in chunks:
            t = (c.latex_text or c.spoken_text or "").strip()
            cleaned = self.clean_speech_text(t)
            if not cleaned:
                continue

            # Ensure sentence has appropriate initial capitalization
            if len(cleaned) > 1:
                cleaned = cleaned[0].upper() + cleaned[1:]
            else:
                cleaned = cleaned.upper()

            # Ensure terminal punctuation unless it ends with math equation
            if not cleaned.endswith((".", "!", "?", ":", ";", "$")):
                cleaned += "."

            curr_sentences.append(cleaned)
            # Group naturally into paragraphs of 3-4 sentences
            if len(curr_sentences) >= 4:
                paragraphs.append(" ".join(curr_sentences))
                curr_sentences = []

        if curr_sentences:
            paragraphs.append(" ".join(curr_sentences))

        return paragraphs

    def build_book(
        self,
        lecture_title: str,
        chunks: List[TranscriptChunk],
        author: str = "",
        date_str: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """
        Builds the complete typeset lecture transcript in LaTeX (.tex) and Markdown (.md).
        Preserves 100% of the lecturer's words without artificial summaries or omitted content.
        Returns: (latex_content, markdown_content, clean_title)
        """
        if not date_str:
            date_str = time.strftime("%B %d, %Y")

        paragraphs = self.assemble_paragraphs(chunks)
        full_text = " ".join(paragraphs)

        # Infer intelligent title based on transcript content
        inferred_title = self.infer_lecture_title(full_text)
        clean_title = inferred_title if inferred_title else lecture_title

        # Detect diagram triggers
        triggers = self.tikz_gen.detect_diagram_triggers(full_text)
        tikz_blocks = self.tikz_gen.get_tikz_for_triggers(triggers)

        # Sanitize text and inline math first, then promote major formulas to numbered display equations
        processed_paragraphs = [self.eq_formatter.promote_to_display_equations(escape_latex_text(p)) for p in paragraphs]

        # Escape ampersands and strip math delimiters for clean document header
        clean_title_tex = clean_title.replace("$", "").replace("&", r"\&").strip()

        # Build complete transcript body with optional TikZ diagrams
        body_tex_parts = []
        tikz_inserted = False
        mid_idx = max(1, len(processed_paragraphs) // 2)

        for idx, p in enumerate(processed_paragraphs):
            body_tex_parts.append(p)
            body_tex_parts.append("")

            # Insert TikZ vector diagram at appropriate midpoint if triggered
            if not tikz_inserted and idx >= mid_idx and tikz_blocks:
                for diag in tikz_blocks:
                    body_tex_parts.append(diag)
                    body_tex_parts.append("")
                tikz_inserted = True

        if not tikz_inserted and tikz_blocks:
            for diag in tikz_blocks:
                body_tex_parts.append(diag)
                body_tex_parts.append("")

        body_tex = "\n".join(body_tex_parts)

        # LaTeX Document Template (100% Complete Transcript, Typeset with Vector Math)
        latex_doc = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage{{amsmath,amssymb,amsfonts,amsthm,mathtools}}
\usepackage{{geometry}}
\geometry{{margin=1.0in, top=1.1in, bottom=1.1in}}
\usepackage{{xcolor}}
\usepackage{{fancyhdr}}
\usepackage{{booktabs}}
\usepackage{{enumitem}}
\usepackage{{hyperref}}
\usepackage{{tikz}}

\definecolor{{primarydark}}{{RGB}}{{20, 30, 45}}
\definecolor{{accentblue}}{{RGB}}{{43, 108, 176}}

\hypersetup{{
    colorlinks=true,
    linkcolor=accentblue,
    citecolor=accentblue,
    urlcolor=accentblue
}}

\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[R]{{\small\color{{gray}} Page \thepage}}
\renewcommand{{\headrulewidth}}{{0pt}}

\setlength{{\parskip}}{{0.8em}}
\setlength{{\parindent}}{{0pt}}

\begin{{document}}

\begin{{center}}
  {{\LARGE\bfseries\color{{primarydark}} {clean_title_tex}}} \\[0.4em]
  {{\small\color{{gray}} {date_str}}}
\end{{center}}

\vspace{{0.4em}}
\hrule
\vspace{{1.2em}}

{body_tex}

\end{{document}}
"""

        # Build clean complete Markdown document
        md_parts = [
            f"# {clean_title}",
            f"*{date_str}*\n",
            "---",
            ""
        ]
        for p in processed_paragraphs:
            p_md = re.sub(
                r'\\begin\{equation\}\s*(?:\\label\{[^}]+\})?\s*(.*?)\s*\\end\{equation\}',
                r'\n$$\n\1\n$$\n',
                p,
                flags=re.DOTALL
            )
            md_parts.append(p_md)
            md_parts.append("")

        if tikz_blocks:
            md_parts.append("*(Vector graphics rendered in compiled PDF)*\n")

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
