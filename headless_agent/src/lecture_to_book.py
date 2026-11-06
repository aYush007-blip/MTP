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

        # 1. Complex s-Plane / Continuous-Time Laplace ROC
        splane_cues = [
            "s-plane", "s plane", "s-domain", "s domain", "laplace transform",
            "inverse laplace", "right-half plane", "left-half plane", "j omega", "jomega"
        ]
        has_roc_cues = any(c in lower for c in ["region of convergence", " roc ", "poles and zeros", "pole zero", "complex plane", "real and imaginary", "right-hand half-plane"])
        if any(c in lower for c in splane_cues) and has_roc_cues:
            triggers.append("s_plane_roc")

        # 2. Complex z-Plane / Discrete-Time Z-Transform ROC / Unit Circle
        zplane_cues = [
            "z-plane", "z plane", "unit circle", "z-transform", "z transform",
            "magnitude of z", "|z|", "radius r"
        ]
        if any(c in lower for c in zplane_cues) and has_roc_cues:
            triggers.append("z_plane_roc")

        # 3. Direct-Quadrature (d-q) Axes / Synchronous Machines
        dq_cues = [
            "direct axis", "quadrature axis", "d-q", "d and q axes", "rotor axis",
            "stator axis", "park transform", "rotor angle", "theta r", "field winding",
            "synchronous machine", "rotor reference frame"
        ]
        if any(c in lower for c in dq_cues):
            triggers.append("dq_axes")

        # 4. Phasor Diagram / AC Circuits
        phasor_cues = [
            "phasor diagram", "voltage phasor", "current phasor", "phase angle",
            "power factor", "lagging current", "leading current", "impedance angle"
        ]
        if any(c in lower for c in phasor_cues):
            triggers.append("phasor_diagram")

        # 5. Bode / Frequency Response
        bode_cues = [
            "bode plot", "bode diagram", "db per decade", "log frequency",
            "corner frequency", "gain margin", "phase margin"
        ]
        if any(c in lower for c in bode_cues):
            triggers.append("bode_plot")

        # 6. Feedback Control Systems / Block Diagrams
        control_cues = [
            "closed loop", "closed-loop", "open loop", "open-loop", "feedback loop",
            "disturbance rejection", "steady state error", "loop gain"
        ]
        if any(c in lower for c in control_cues):
            triggers.append("control_feedback_loop")

        # 7. Discrete-Time Sequences / Signal Energy / Power
        seq_cues = [
            "energy of a sequence", "energy of the sequence", "discrete-time sequence",
            "discrete sequence", "partial energy", "power of a sequence", "finite duration signal",
            "energy signal", "periodic sequence", "x[n]", "sequence energy"
        ]
        if any(c in lower for c in seq_cues):
            triggers.append("discrete_sequence_energy")

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
    def generate_splane_roc_tikz() -> str:
        """Vector pure TikZ code for Complex s-plane with imaginary axis and ROC."""
        return r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=1.2, >=stealth]
  % Shaded ROC: Re(s) > sigma_0
  \fill[blue!10] (1.0, -2.5) rectangle (3.2, 2.5);
  % Axes
  \draw[->, thick, line width=1pt] (-2.5, 0) -- (3.5, 0) node[right] {$\text{Re}(s)$};
  \draw[->, thick, line width=1pt] (0, -2.5) -- (0, 2.5) node[above] {$\text{Im}(s) = j\omega$};
  \node[below left] at (0, 0) {$0$};
  % ROC Boundary: Re(s) = sigma_0
  \draw[dashed, blue!80!black, line width=1.1pt] (1.0, -2.5) -- (1.0, 2.5);
  \node[below, blue!80!black] at (1.0, -2.5) {$\sigma_0$};
  % Poles (x markers)
  \draw[red, line width=1.4pt] (0.9, -0.1) -- (1.1, 0.1);
  \draw[red, line width=1.4pt] (0.9, 0.1) -- (1.1, -0.1);
  \node[above left, red] at (1.0, 0.1) {$p_1$};
  % Annotations
  \node[blue, font=\bfseries] at (2.2, 1.5) {$\text{ROC}: \text{Re}(s) > \sigma_0$};
  \node[gray!70, font=\small] at (-1.3, -1.8) {Left-Half Plane};
  \node[gray!70, font=\small] at (2.2, -1.8) {Right-Half Plane};
\end{tikzpicture}
\caption{Complex $s$-plane showing imaginary axis $j\omega$, pole location $p_1 = \sigma_0$, and right-sided Region of Convergence (ROC) for a causal continuous-time system.}
\label{fig:splane_roc}
\end{figure}"""

    @staticmethod
    def generate_bode_plot_tikz() -> str:
        """Vector pure TikZ code for logarithmic frequency response (no pgfplots dependency)."""
        return r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=1.1, >=stealth]
  % Grid lines
  \draw[dashed, gray!30] (0, -1) -- (6, -1);
  \draw[dashed, gray!30] (0, -2) -- (6, -2);
  \draw[dashed, gray!30] (0, -3) -- (6, -3);
  \draw[dashed, gray!30] (2.5, -3.5) -- (2.5, 0.8);

  % Axes
  \draw[->, thick, gray!80] (0, 0) -- (6.5, 0) node[right, font=\small] {$\log_{10}\omega$};
  \draw[->, thick, gray!80] (0, -3.5) -- (0, 1.0) node[above, font=\small] {$|H(j\omega)|_{\text{dB}}$};

  % Magnitude asymptotes
  \draw[very thick, accentblue] (0, 0) -- (2.5, 0);
  \draw[very thick, accentblue] (2.5, 0) -- (5.5, -3.0) node[right, font=\small] {$-20\text{ dB/decade}$};

  % Actual smooth curve
  \draw[thick, red!80!black, dashed] (0, -0.1) to[out=0, in=145] (2.5, -0.6) to[out=-35, in=135] (5.5, -3.0);

  % Annotations
  \node[below, font=\small] at (2.5, 0) {$\omega_c$};
  \node[left, font=\small, red!80!black] at (0, -0.6) {$-3\text{ dB}$};
  \draw[dotted, red!80!black] (0, -0.6) -- (2.5, -0.6);
  \filldraw[red!80!black] (2.5, -0.6) circle (2pt);
\end{tikzpicture}
\caption{Logarithmic Bode magnitude response exhibiting $-3\text{ dB}$ attenuation at corner frequency $\omega_c$ and $-20\text{ dB/decade}$ roll-off.}
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

    @staticmethod
    def generate_discrete_sequence_tikz() -> str:
        """Vector TikZ code for Discrete-Time Sequence Energy & Windowing."""
        return r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}[scale=1.1, >=stealth]
  % Axes
  \draw[->, thick, gray!80] (-4.5, 0) -- (4.5, 0) node[right] {$n$};
  \draw[->, thick, gray!80] (0, -0.4) -- (0, 2.5) node[above] {$|x[n]|^2$};
  % Stems
  \foreach \n/\val in {-4/0.15, -3/0.35, -2/0.8, -1/1.5, 0/2.1, 1/1.5, 2/0.8, 3/0.35, 4/0.15} {
    \draw[thick, accentblue] (\n, 0) -- (\n, \val);
    \filldraw[accentblue] (\n, \val) circle (2pt);
    \node[below, font=\small, text=mutedgray] at (\n, 0) {$\n$};
  }
  % Partial Energy Window [-K, K]
  \draw[dashed, red!75!black, thick] (-2.5, -0.2) -- (-2.5, 2.1);
  \draw[dashed, red!75!black, thick] (2.5, -0.2) -- (2.5, 2.1);
  \draw[<->, red!75!black, thick] (-2.5, 1.8) -- (2.5, 1.8) node[midway, fill=white, font=\small] {Partial Energy Window ($2K+1$ samples)};
  \node[red!75!black, font=\small] at (-2.5, -0.4) {$-K$};
  \node[red!75!black, font=\small] at (2.5, -0.4) {$+K$};
\end{tikzpicture}
\caption{Discrete-time sequence energy summation $\mathcal{E}_K = \sum_{n=-K}^K |x[n]|^2$ evaluated across symmetric observation window $2K+1$.}
\label{fig:discrete_sequence_energy}
\end{figure}"""

    def get_tikz_for_triggers(self, triggers: List[str]) -> List[str]:
        """Generate TikZ blocks for identified triggers without duplicates."""
        blocks = []
        seen = set()
        for t in triggers:
            if t in seen:
                continue
            seen.add(t)
            if t == "s_plane_roc":
                blocks.append(self.generate_splane_roc_tikz())
            elif t == "z_plane_roc":
                blocks.append(self.generate_zplane_roc_tikz())
            elif t == "dq_axes":
                blocks.append(self.generate_dq_axes_tikz())
            elif t == "phasor_diagram":
                blocks.append(self.generate_phasor_tikz())
            elif t == "bode_plot":
                blocks.append(self.generate_bode_plot_tikz())
            elif t == "control_feedback_loop":
                blocks.append(self.generate_control_system_tikz())
            elif t == "discrete_sequence_energy":
                blocks.append(self.generate_discrete_sequence_tikz())
        return blocks


class EquationFormatter:
    """
    Identifies mathematical relationships in text and formats them into
    numbered LaTeX equation environments with labels, separated from paragraph text.
    """
    def __init__(self):
        self.eq_counter = 0

    def synthesize_spoken_equations(self, text: str) -> str:
        """
        Synthesizes spoken relational phrases (e.g. '$x(t)$ equal to $e^{-3t} u(t)$')
        into unified mathematical formulas before display promotion.
        """
        subs = [
            # 1. Initial & Final value theorem formulas
            (r'(?:that\s+is\s+)?(\$x\(0\^\+\)\$)\s+(?:and\s+this\s+)?will\s+be\s+equal\s+to\s+(\$\\lim_\{s\s*\\to\s*\\infty\}[^\$]+\$)',
             r':\n$$\1 = \2$$\n'),
            (r'(\$x\(0\^[-+]\)\$)\s+equal\s+to\s+(\$[^\$]+\$)', r':\n$$\1 = \2$$\n'),
            (r'evaluate\s+(\$s\s+X\(s\)\$)\s+and\s+find\s+the\s+(\$\\lim_\{s\s*\\to\s*0\}\$)',
             r'evaluate the limit:\n$$\\lim_{s \\to 0} s X(s)$$\n'),
            (r'what\s+is\s+the\s+(\$\\lim_\{s\s*\\to\s*0\}\s*s\s*X\(s\)\s*H\(s\)\$)',
             r'evaluate:\n$$\1$$\n'),

            # 2. Input/Output signal assignments: $x(t)$ equal to $e^{-3t} u(t)$
            (r'(\$[a-zA-Z0-9_\(\)]+\$)\s+equal\s+to\s+(\$[^\$]+\$)', r':\n$$\1 = \2$$\n'),
            (r'(\$[a-zA-Z0-9_\(\)]+\$)\s+is\s+equal\s+to\s+(\$[^\$]+\$)', r':\n$$\1 = \2$$\n'),
            (r'(\$[a-zA-Z0-9_\(\)]+\$)\s+equals\s+(\$[^\$]+\$)', r':\n$$\1 = \2$$\n'),
            (r'(\$x_?2?\(t\)\$)\s+equal\s+to\s+(\$\\cos\([^\$]+\$)', r':\n$$\1 = \2$$\n'),
            (r'(\$x_?1?\(t\)\$)\s+is\s+(\$\\cos\([^\$]+\$)', r':\n$$\1 = \2$$\n'),
            (r'(\$x_?2?\(t\)\$)\s+(\$\\cos\([^\$]+\$)', r':\n$$\1 = \2$$\n'),

            # 3. Transform formulas & general properties
            (r'(?:In\s+general\s+(?:say\s+)?)?\$t\^\{n-1\}\$,?\s*\$\(n-1\)!?\$\s*\$e\^\{-at\}\s*u\(t\)\$\s+is\s+going\s+to\s+be\s+(\$\\frac\{1\}\{\(s\+a\)\^n\}\$)',
             r'In general, we have:\n$$\\mathcal{L}\\left\\{\\frac{t^{n-1}}{(n-1)!} e^{-at} u(t)\\right\\} = \\frac{1}{(s+a)^n}$$\n'),
            (r'Laplace\s+transform\s+of\s+\$x\(t\)\$\.?\s*(?:So,?\s*)?(?:this\s+is\s+)?equal\s+to\s+(\$\\frac\{1\}\{s\+3\}\$)',
             r'Laplace transform of $x(t)$ is given by:\n$$X(s) = \1$$\n'),
            (r'Laplace\s+transform\s+of\s+\$y\(t\)\$\.?\s*What\s+will\s+be\s+the\s+answer\s+here\??\s*(\$\\frac\{1\}\{s\+1\}\s*-\s*\\frac\{1\}\{s\+2\}\$)',
             r'Laplace transform of $y(t)$ is given by:\n$$Y(s) = \1$$\n'),
            (r'Laplace\s+transform\s+of\s+the\s+impulse\s+response\s+is\s+equal\s+to\s+(\$\\frac\{1\}\{X\(s\)\}\$)',
             r'Laplace transform of the impulse response is:\n$$H(s) = \1$$\n'),
            (r'this\s+will\s+be\s+y\s+(\$s\s*Y\(s\)\$)\s+plus\s+the\s+(\$Y\(s\)\$)\s+equal\s+to\s+(\$X\(s\)\$)',
             r'we obtain:\n$$s Y(s) + Y(s) = X(s)$$\n'),
            (r'So,?\s*this\s+is\s+(\$\\frac\{s\}\{s\^2\s*\+\s*\\omega_0\^2\}\$)',
             r'So, the Laplace transform is:\n$$X(s) = \1$$\n'),
            (r'this\s+is\s+minus\s+okay\s+and\s+this\s+is\s+equal\s+to\s+(\$\\frac\{1\}\{\(s\+a\)\^2\}\$)',
             r'this is equal to:\n$$\\frac{1}{(s+a)^2}$$\n'),

            # 4. Complex plane & ROC inequalities
            (r'(?:Set\s+of\s+all\s+complex\s+numbers\s+such\s+that\s+)?(\$\\text\{Re\}\(s\)\s*>\s*-\d+\$)',
             r':\n$$\\text{ROC}: \\quad \1$$\n'),
            (r's\s+equal\s+to\s+some\s+\$j\\omega\$|s\s+equal\s+to\s+\$j\\omega\$',
             r'$$s = j\\omega$$'),
        ]

        for pat, rep in subs:
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)

        text = re.sub(r'\${3,}', '$$', text)
        return text

    def should_promote_to_display(self, eq_body: str) -> bool:
        """Determines if a math expression is a formula that should be on a separate display line."""
        eq = eq_body.strip()

        # 1. Any equation or relational operator (=, >, <, \le, \ge, \approx, \equiv)
        if any(op in eq for op in ['=', '>', '<', r'\ge', r'\le', r'\approx', r'\equiv']):
            if not re.match(r'^[a-zA-Z_]\s*=\s*\d+$', eq):
                return True

        # 2. Fractions, limits, calculus, sums, integrals, ROC
        if any(st in eq for st in [r'\frac', r'\lim', r'\sum', r'\int', r'\partial', r'\text{Re}', r'\text{Im}', r'\text{ROC}']):
            return True

        # 3. Compound signals: e.g. e^{-3t} u(t), (e^{-t} - e^{-2t}) u(t), t e^{-at} u(t), \cos(\omega_0 t) u(t)
        if re.search(r'(?:e\^\{|\bcos\(|\bsin\().*?u\(t\)', eq):
            return True

        # 4. Multi-term compound expressions (e.g. s X(s) H(s), s^2 + \omega_0^2)
        if any(op in eq for op in [' - ', ' + ', r'\pm', r'\times']) and len(eq) >= 8:
            return True

        return False

    def promote_to_display_equations(self, paragraph: str) -> str:
        """
        Promotes significant inline math, equations, formulas, fractions, limits,
        and ROC relations to numbered \\begin{equation} blocks on separate display lines.
        """
        paragraph = self.synthesize_spoken_equations(paragraph)

        # 1. Handle explicit double dollars $$ ... $$
        def _replace_double_dollars(match):
            self.eq_counter += 1
            eq_body = match.group(1).replace('$', '').strip()
            return f"\n\\begin{{equation}}\n  \\label{{eq:{self.eq_counter}}}\n  {eq_body}\n\\end{{equation}}\n"

        paragraph = re.sub(r'\$\$(.*?)\$\$', _replace_double_dollars, paragraph, flags=re.DOTALL)

        # 2. Tokenize by single dollar pairs: ($[^\$]+$)
        tokens = re.split(r'(\$[^\$]+\$)', paragraph)
        new_tokens = []
        for tok in tokens:
            if tok.startswith('$') and tok.endswith('$') and len(tok) >= 3:
                eq_body = tok[1:-1].replace('$', '').strip()
                if not eq_body:
                    continue

                # Clean unclosed cases or matrix environments
                eq_body = re.sub(r'\\begin\{cases\}\.?\s*$', '', eq_body).strip()
                for env in ["cases", "matrix", "pmatrix", "bmatrix", "aligned"]:
                    beg = eq_body.count(rf"\begin{{{env}}}")
                    end = eq_body.count(rf"\end{{{env}}}")
                    if beg > end:
                        eq_body += rf"\end{{{env}}}" * (beg - end)

                if not eq_body:
                    continue

                if self.should_promote_to_display(eq_body):
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


class AcademicSectionizer:
    """
    Partitions raw stream of paragraphs into cohesive thematic sections
    with formal LaTeX section headers, definitions, and theorems across ALL engineering disciplines.
    """
    SECTION_RULES = [
        # Continuous-Time Signals & LTI Systems
        (r"\b(initial[- ]value theorem|final[- ]value theorem|initial value theorem|final value theorem|steady[- ]state (?:value|output|response))\b",
         "Initial and Final Value Theorems for Continuous-Time Systems"),
        (r"\b(laplace transform|differential equations?|s[- ]domain|transfer function|inverse laplace)\b",
         "S-Domain Formulation, Differential Equations, and Transfer Functions"),
        (r"\b(impulse response|system causality|bibo stability|stability condition|stable and causal)\b",
         "System Impulse Response, Causality, and Stability Criteria"),
        (r"\b(region of convergence|roc|right[- ]half plane|left[- ]half plane|strip of convergence|pole[- ]zero dynamics)\b",
         "Region of Convergence (ROC) and Complex Plane Dynamics"),

        # Signals and Systems / DSP
        (r"\b(notion of (?:the )?energy|energy of (?:a|the) sequence|discrete[- ]time sequence energy)\b",
         "Introduction and Energy of Discrete-Time Sequences"),
        (r"\b(periodic (?:signal|sequence)|infinite energy of|divergence of energy|average power of (?:a|the) sequence|p subscript x|2k\s*\+\s*1)\b",
         "Energy Divergence in Periodic Signals and Definition of Power"),
        (r"\b(partial energy|finite observation window|epsilon.*k|over a finite window)\b",
         "Partial Energy Formulation over Finite Observation Windows"),
        (r"\b(energy signals? and power signals?|classification of signals|finite[- ]duration signals?)\b",
         "Signal Classification: Energy Signals vs. Power Signals"),
        (r"\b(definition of z[- ]transform|z[- ]transform formulation|bilateral z[- ]transform|unilateral z[- ]transform)\b",
         "Mathematical Formulation of the Z-Transform"),
        (r"\b(unit circle|discrete[- ]time stability|annular region|roc in z[- ]plane)\b",
         "Region of Convergence (ROC) and Discrete-Time Stability"),
        (r"\b(fourier transform|dtft|dft|fft|harmonic analysis|nyquist rate)\b",
         "Frequency Spectrum and Discrete-Time Fourier Representations"),
        (r"\b(discrete convolution|discrete filter|lti system convolution)\b",
         "Linear Time-Invariant Systems and Discrete Convolution"),

        # Control Systems & Dynamic Systems
        (r"\b(in earlier lectures|to begin with|first, let us|introduction|background|motivation)\b",
         "Introduction and Dynamic Foundations"),
        (r"\b(open[- ]loop|feedforward|plant is|transfer function from|nominal tracking)\b",
         "Open-Loop Formulation and Nominal Tracking"),
        (r"\b(disturbance|superposition|total output|shurupan|external load|perturbation)\b",
         "System Response under External Load Disturbances"),
        (r"\b(unstable|s square minus 1|pole[- ]zero cancellation|cancellation|right[- ]half plane)\b",
         "The Pole-Zero Cancellation Dilemma in Unstable Plants"),
        (r"\b(tracking|regulation|thermal control|300 degrees|cruise control|speed)\b",
         "Control Objectives: Trajectory Tracking vs. Disturbance Regulation"),
        (r"\b(closed[- ]loop|feedback loop|negative feedback|error signal|sensor)\b",
         "Closed-Loop Feedback Architecture and Error Dynamics"),
        (r"\b(type[- ]0|type[- ]1|integral control|proportional control|integral action)\b",
         "Disturbance Rejection via Integral Action and System Type"),
        (r"\b(sensitivity|delta g|parameter variation|desensitization|bode sensitivity)\b",
         "Plant Parameter Sensitivity and Robustness Analysis"),
        (r"\b(pole placement|characteristic polynomial|solutions to this polynomial|dominant poles)\b",
         "Closed-Loop Stability and Arbitrary Pole Placement"),
        (r"\b(frequency|bode|complementary sensitivity|waterbed|low frequency|high frequency|trade[- ]off)\b",
         "Frequency-Domain Design Trade-Offs: Sensitivity vs. Noise Attenuation"),

        # Electrical Machines & Power Systems
        (r"\b(park transform|direct axis|quadrature axis|d[- ]q frame|rotor reference)\b",
         "Park Transformation and d-q Rotor Reference Frame"),
        (r"\b(synchronous machine|stator flux|rotor winding|field current|salient pole)\b",
         "Synchronous Machine Dynamics and Flux Linkages"),
        (r"\b(induction motor|slip|torque[- ]speed|equivalent circuit|rotor resistance)\b",
         "Induction Machine Characteristics and Electromechanical Torque"),
        (r"\b(phasor|reactive power|apparent power|power factor|lagging|leading)\b",
         "AC Power Analysis, Phasors, and Power Factor Correction"),

        # Electronics & Circuits
        (r"\b(kirchhoff|kvl|kcl|node voltage|mesh current|thevenin|norton)\b",
         "Network Analysis and Canonical Circuit Theorems"),
        (r"\b(op[- ]amp|operational amplifier|virtual ground|inverting amplifier|non[- ]inverting amplifier|differential amplifier)\b",
         "Operational Amplifier Configurations and Closed-Loop Gain"),
        (r"\b(transistor|bjt|mosfet|cmos|small[- ]signal|biasing|saturation)\b",
         "Semiconductor Physics, Device Biasing, and Small-Signal Modeling"),

        # Universal Engineering Discourse Cues
        (r"\b(now straight away|observation|consider the case|on the other hand|let us analyze|behavior)\b",
         "Analytical Behavior and Characteristic Dynamics"),
        (r"\b(now we will define|new term|second step|furthermore|extension|generalized)\b",
         "Formal Analytical Definitions and Metric Formulations"),
        (r"\b(two categories|classification|distinction between|comparison of)\b",
         "Systematic Classification and Property Comparison"),
        (r"\b(example|for instance|to illustrate|consider a specific|numerical|practical)\b",
         "Illustrative Analysis and Applied Case Studies"),
        (r"\b(along the same sequence|in conclusion|to summarize|summary|takeaway|finally)\b",
         "Theoretical Synthesis and Concluding Summary")
    ]

    @classmethod
    def structure_monograph(
        cls,
        paragraphs: List[str],
        tikz_blocks: List[str],
        full_text: str = ""
    ) -> List[str]:
        """
        Structures paragraphs into formal numbered LaTeX sections.
        Inserts definitions, theorems, and TikZ figures contextually.
        Guarantees 3-5 structured sections across any engineering lecture.
        """
        if not paragraphs:
            return []

        sections = []
        current_sec = "Introduction and Theoretical Foundations"
        current_paras = []
        used_headings = set([current_sec])

        total_p = len(paragraphs)
        min_p = 3 if total_p >= 8 else (2 if total_p >= 4 else 1)
        max_sec = 6

        for idx, p in enumerate(paragraphs):
            matched_heading = None
            if idx > 0 and len(current_paras) >= min_p and len(sections) < max_sec - 1:
                for pattern, heading in cls.SECTION_RULES:
                    if heading not in used_headings and re.search(pattern, p, re.IGNORECASE):
                        matched_heading = heading
                        break

            if matched_heading:
                sections.append((current_sec, current_paras))
                used_headings.add(matched_heading)
                current_sec = matched_heading
                current_paras = [p]
            else:
                current_paras.append(p)

        if current_paras:
            sections.append((current_sec, current_paras))

        # Adaptive Fallback: If rule matching produced fewer than 3 sections for a multi-paragraph lecture,
        # evenly partition paragraphs into 3-4 structured sections with meaningful academic headings.
        if len(sections) < 3 and len(paragraphs) >= 4:
            total_p = len(paragraphs)
            num_sec = min(4, max(3, total_p // 2))
            chunk_size = max(1, total_p // num_sec)
            fallback_headings = [
                "Introduction and Fundamental Principles",
                "Mathematical Formulation and Analytical Dynamics",
                "Signal Properties and Metric Evaluation",
                "Theoretical Synthesis and Concluding Observations"
            ]
            sections = []
            for i in range(num_sec):
                start_p = i * chunk_size
                end_p = total_p if i == num_sec - 1 else (i + 1) * chunk_size
                sec_title = fallback_headings[i] if i < len(fallback_headings) else f"Advanced Theoretical Analysis {i+1}"
                sections.append((sec_title, paragraphs[start_p:end_p]))

        # Build output body with LaTeX sections and contextual TikZ insertion
        body_parts = []
        tikz_remaining = list(tikz_blocks)

        for sec_idx, (sec_title, sec_paras) in enumerate(sections, 1):
            body_parts.append(f"\\section{{{sec_title}}}\n")
            for p in sec_paras:
                # Wrap definition cues
                if re.search(r'\b(we define|definition:)\b', p, re.IGNORECASE) and not p.startswith(r"\begin{"):
                    p = f"\\begin{{definition}}\n{p}\n\\end{{definition}}"
                # Wrap theorem cues
                elif re.search(r'\b(a feedback control system achieves|principle of|fundamental theorem)\b', p, re.IGNORECASE) and not p.startswith(r"\begin{"):
                    p = f"\\begin{{theorem}}\n{p}\n\\end{{theorem}}"

                body_parts.append(p)
                body_parts.append("")

            # Insert TikZ diagrams contextually
            if tikz_remaining:
                lower_sec = sec_title.lower()
                diag_to_insert = None
                for d in tikz_remaining:
                    if ("closed-loop" in lower_sec or "feedback architecture" in lower_sec) and "feedback_block_diagram" in d:
                        diag_to_insert = d
                        break
                    elif ("laplace" in lower_sec or "convergence" in lower_sec or "s-domain" in lower_sec) and "splane" in d:
                        diag_to_insert = d
                        break
                    elif ("z-transform" in lower_sec or "discrete-time stability" in lower_sec) and "zplane" in d:
                        diag_to_insert = d
                        break
                    elif ("sequence" in lower_sec or "energy" in lower_sec or "power" in lower_sec) and "discrete_sequence" in d:
                        diag_to_insert = d
                        break
                    elif "bode" in lower_sec and "bode" in d:
                        diag_to_insert = d
                        break

                if not diag_to_insert and sec_idx == max(1, len(sections) // 2):
                    diag_to_insert = tikz_remaining[0]

                if diag_to_insert:
                    body_parts.append(diag_to_insert)
                    body_parts.append("")
                    tikz_remaining.remove(diag_to_insert)

        # Append any remaining diagrams
        for d in tikz_remaining:
            body_parts.append(d)
            body_parts.append("")

        return body_parts


class LectureToBookBuilder:
    """
    Assembles streaming transcripts into publication-grade student classroom lecture notes.
    """
    def __init__(self):
        self.tikz_gen = TikZDiagramGenerator()
        self.eq_formatter = EquationFormatter()
        self.sectionizer = AcademicSectionizer()

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

        # Spoken phonetic speech normalization for math & technical symbols across STEM disciplines
        speech_subs = [
            # Acoustic corrections & discrete signals
            (r'\bpartic signals?\b', lambda m: 'periodic signals'),
            (r'\bmod x of n squared\b', lambda m: r'$|x[n]|^2$'),
            (r'\bmod x of n magnitude squared\b', lambda m: r'$|x[n]|^2$'),
            (r'\bmod x of n\b', lambda m: r'$|x[n]|$'),
            (r'\bp subscript x\b', lambda m: r'$P_x$'),
            (r'\b2k plus 1\b', lambda m: r'$2K+1$'),
            (r'\be of x\b', lambda m: r'$E_x$'),

            # --- 1. Longest compound expressions & limit formulas ---
            (r'\blimit\s+s\s+(?:going|goes|tends)\s+to\s+infinity\s*,\s*s\s+times\s+x\s+of\s+s\b', lambda m: r'$\lim_{s \to \infty} s X(s)$'),
            (r'\blimit\s+s\s+(?:going|goes|tends)\s+to\s+infinity\s+s\s+times\s+x\s+of\s+s\b', lambda m: r'$\lim_{s \to \infty} s X(s)$'),
            (r'\blimit\s+(?:as\s+)?s\s+(?:going|goes|tends)\s+to\s+0\s*,\s*s\s+times\s+x\s+of\s+s\s+h\s+of\s+s\b', lambda m: r'$\lim_{s \to 0} s X(s) H(s)$'),
            (r'\blimit\s+(?:as\s+)?s\s+(?:going|goes|tends)\s+to\s+0\s*,\s*s\s+times\s+x\s+of\s+s\b', lambda m: r'$\lim_{s \to 0} s X(s)$'),
            (r'\blimit\s+(?:as\s+)?s\s+(?:going|goes|tends)\s+to\s+0\b', lambda m: r'$\lim_{s \to 0}$'),
            (r'\blimit\s+s\s+(?:going|goes|tends)\s+to\s+infinity\b', lambda m: r'$\lim_{s \to \infty}$'),
            (r'\blimit\s+t\s+(?:going|goes|tends)\s+to\s+infinity\b', lambda m: r'$\lim_{t \to \infty}$'),
            (r'\blimit\s+t\s+plus\s+infinity\b', lambda m: r'$\lim_{t \to +\infty}$'),
            (r'\bx0\s*plus\b|\bx\s+zero\s+plus\b', lambda m: r'$x(0^+)$'),
            (r'\bx0\s*minus\b|\bx\s+zero\s+minus\b', lambda m: r'$x(0^-)$'),
            (r'\bx\s+infinity\b', lambda m: r'$x(\infty)$'),

            # --- 2. Compound expressions: fractions and differences ---
            (r'\b1\s+by\s+s\s+plus\s+(\d+)\s+minus\s+1\s+by\s+s\s+plus\s+(\d+)\b', lambda m: rf'$\frac{{1}}{{s+{m.group(1)}}} - \frac{{1}}{{s+{m.group(2)}}}$'),
            (r'\be\s+(?:to\s+the\s+)?power\s+minus\s+t\s+minus\s+e\s+(?:to\s+the\s+)?power\s+minus\s+(\d+)t\s+u\s+of\s+t\b', lambda m: rf'$(e^{{-t}} - e^{{-{m.group(1)}t}}) u(t)$'),
            (r'\bs\s+divided\s+by\s+s\s+square\s+plus\s+omega\s*0?\s*(?:square|s)?\b', lambda m: r'$\frac{s}{s^2 + \omega_0^2}$'),
            (r'\bs\s+plus\s+j\s+omega\s*0?\s*s\s+minus\s+j\s+omega\s*0?\b', lambda m: r'$(s + j\omega_0)(s - j\omega_0)$'),
            (r'\bs\s+minus\s+j\s+omega\s*0?\s*plus\s+s\s+plus\s+j\s+omega\s*0?\b', lambda m: r'$(s - j\omega_0) + (s + j\omega_0)$'),
            (r'\bs\s+minus\s+j\s+omega\s*0\b', lambda m: r'$s - j\omega_0$'),
            (r'\bs\s+plus\s+j\s+omega\s*0\b', lambda m: r'$s + j\omega_0$'),
            (r'\b1\s+by\s+j\s+omega\s*0?\s*minus\s+a\b', lambda m: r'$\frac{1}{j\omega - a}$'),
            (r'\b1\s+by\s+j\s+omega\s*0?\s*plus\s+a\b', lambda m: r'$\frac{1}{j\omega + a}$'),
            (r'\b1\s+by\s+s\s+plus\s+(\d+)\b', lambda m: rf'$\frac{{1}}{{s+{m.group(1)}}}$'),
            (r'\b1\s+by\s+s\s+minus\s+(\d+)\b', lambda m: rf'$\frac{{1}}{{s-{m.group(1)}}}$'),
            (r'\b1\s+by\s+s\s+plus\s+a\s+square\b', lambda m: r'$\frac{1}{(s+a)^2}$'),
            (r'\b1\s+by\s+s\s+plus\s+a\s+(?:to\s+the\s+power\s*n?|to\s+the\s+n|to\s+the\s+power)\b', lambda m: r'$\frac{1}{(s+a)^n}$'),
            (r'\b1\s+by\s+s\s+plus\s+a\b', lambda m: r'$\frac{1}{s+a}$'),
            (r'\b1\s+by\s+s\s+minus\s+a\b', lambda m: r'$\frac{1}{s-a}$'),
            (r'\b1\s+by\s+x\s+of\s+s\b|\b1\s+by\s+xs\b', lambda m: r'$\frac{1}{X(s)}$'),
            (r'\b1\s+by\s+h\s+of\s+s\b|\b1\s+by\s+hs\b', lambda m: r'$\frac{1}{H(s)}$'),

            # --- 3. Compound multiplications ---
            (r'\bs\s+times\s+x\s+of\s+s\s+h\s+of\s+s\b', lambda m: r'$s X(s) H(s)$'),
            (r'\bs\s+times\s+x\s+of\s+s\b|\bs\s+times\s+xs\b', lambda m: r'$s X(s)$'),
            (r'\bs\s+times\s+y\s+of\s+s\b|\bs\s+times\s+ys\b', lambda m: r'$s Y(s)$'),
            (r'\bs\s+times\s+h\s+of\s+s\b|\bs\s+times\s+hs\b', lambda m: r'$s H(s)$'),
            (r'\by\s+s\s+times\s+y\s+of\s+s\b', lambda m: r'$s Y(s)$'),
            (r'\bxs\s+times\s+xs\b', lambda m: r'$X(s) X(s)$'),
            (r'\bxs\s+times\s+hs\b', lambda m: r'$X(s) H(s)$'),

            # --- 4. Exponentials and sinusoids ---
            (r'\be\s+(?:to\s+the\s+)?power\s+minus\s+(\d+)t\s+u\s+of\s+t\b', lambda m: rf'$e^{{-{m.group(1)}t}} u(t)$'),
            (r'\be\s+(?:to\s+the\s+)?power\s+minus\s+t\s+u\s+of\s+t\b', lambda m: r'$e^{-t} u(t)$'),
            (r'\bt\s+(?:times\s+)?e\s+(?:to\s+the\s+)?power\s+minus\s+a\s+t\s+u\s+of\s+t\b', lambda m: r'$t e^{-at} u(t)$'),
            (r'\be\s+(?:to\s+the\s+)?power\s+minus\s+a\s+t\s+u\s+of\s+t\b', lambda m: r'$e^{-at} u(t)$'),
            (r'\be\s+(?:to\s+the\s+)?power\s+(?:plus\s+)?j\s+omega\s*0\s*(?:t)?\b', lambda m: r'$e^{j \omega_0 t}$'),
            (r'\be\s+(?:to\s+the\s+)?power\s+minus\s+j\s+omega\s*0\s*(?:t)?\b', lambda m: r'$e^{-j \omega_0 t}$'),
            (r'\be\s+(?:to\s+the\s+)?power\s+(?:plus\s+)?j\s+omega\s*(?:t)?\b', lambda m: r'$e^{j \omega t}$'),
            (r'\be\s+(?:to\s+the\s+)?power\s+minus\s+j\s+omega\s*(?:t)?\b', lambda m: r'$e^{-j \omega t}$'),
            (r'\be\s+power\s+minus\s+0\s*t\b', lambda m: r'$e^{-0 t}$'),
            (r'\be\s+power\s+minus\s+s\s*t\b', lambda m: r'$e^{-st}$'),
            (r'\bcos\s+omega\s*0\s*(?:t)?\s+u\s+of\s+t\b', lambda m: r'$\cos(\omega_0 t) u(t)$'),
            (r'\bcos\s+omega\s*0\s*(?:t)?\b', lambda m: r'$\cos(\omega_0 t)$'),
            (r'\bcos\s+omega\s*(?:t)?\s+u\s+of\s+t\b', lambda m: r'$\cos(\omega t) u(t)$'),
            (r'\bcos\s+omega\s*(?:t)?\b', lambda m: r'$\cos(\omega t)$'),
            (r'\bsin\s+omega\s*0\s*(?:t)?\s+u\s+of\s+t\b', lambda m: r'$\sin(\omega_0 t) u(t)$'),
            (r'\bsin\s+omega\s*0\s*(?:t)?\b', lambda m: r'$\sin(\omega_0 t)$'),
            (r'\bsin\s+omega\s*(?:t)?\s+u\s+of\s+t\b', lambda m: r'$\sin(\omega t) u(t)$'),
            (r'\bsin\s+omega\s*(?:t)?\b', lambda m: r'$\sin(\omega t)$'),
            (r'\b[et]\s+(?:to\s+the\s+)?power\s+n\s+minus\s+1\b', lambda m: r'$t^{n-1}$'),
            (r'\bn\s+minus\s+1\s+factorial\b', lambda m: r'$(n-1)!$'),

            # --- 5. ROC & Complex Plane ---
            (r'\breal\s+part\s+of\s+s\s+is\s+greater\s+than\s+minus\s+(\d+)\b', lambda m: rf'$\text{{Re}}(s) > -{m.group(1)}$'),
            (r'\breal\s+part\s+of\s+s\s+greater\s+than\s+minus\s+(\d+)\b', lambda m: rf'$\text{{Re}}(s) > -{m.group(1)}$'),
            (r'\breal\s+part\s+of\s+s\s+greater\s+than\s+(\d+)\b', lambda m: rf'$\text{{Re}}(s) > {m.group(1)}$'),
            (r'\breal\s+part\s+of\s+s\b', lambda m: r'$\text{Re}(s)$'),
            (r'\bimaginary\s+part\s+of\s+s\b', lambda m: r'$\text{Im}(s)$'),
            (r'\bj\s+omega\s*0\s*t\b', lambda m: r'$j\omega_0 t$'),
            (r'\bj\s+omega\s*t\b', lambda m: r'$j\omega t$'),
            (r'\bj\s+omega\s*0\b', lambda m: r'$j\omega_0$'),
            (r'\bj\s+omega\b', lambda m: r'$j\omega$'),
            (r'\bs\s+equal\s+to\s+(?:some\s+)?j\s+omega\s*1\b', lambda m: r'$s = j\omega_1$'),
            (r'\bs\s+equal\s+to\s+(?:some\s+)?j\s+omega\s*0\b', lambda m: r'$s = j\omega_0$'),
            (r'\bs\s+equal\s+to\s+(?:some\s+)?j\s+omega\b', lambda m: r'$s = j\omega$'),
            (r'\bs\s+equal\s+to\s+sigma\s+plus\s+j\s+omega\b', lambda m: r'$s = \sigma + j\omega$'),
            (r'\bsigma\s*0\b', lambda m: r'$\sigma_0$'),

            # --- 6. Derivatives & Differentials ---
            (r'\bd2y\s+by\s+dt2\b|\bd\s+square\s+y\s+by\s+d\s+t\s+square\b', lambda m: r'$\frac{d^2y}{dt^2}$'),
            (r'\bdy\s+by\s+dt\b|\bd\s+y\s+by\s+d\s+t\b', lambda m: r'$\frac{dy}{dt}$'),
            (r'\bdx\s+by\s+dt\b|\bd\s+x\s+by\s+d\s+t\b', lambda m: r'$\frac{dx}{dt}$'),

            # --- 7. Base continuous-time signals & transforms ---
            (r'\bx\s+of\s+t\b', lambda m: r'$x(t)$'),
            (r'\bx\s+of\s+s\b|\bx\s+of\s+p\b', lambda m: r'$X(s)$'),
            (r'\by\s+of\s+t\b', lambda m: r'$y(t)$'),
            (r'\by\s+of\s+s\b', lambda m: r'$Y(s)$'),
            (r'\bh\s+of\s+t\b', lambda m: r'$h(t)$'),
            (r'\bh\s+of\s+s\b', lambda m: r'$H(s)$'),
            (r'\bu\s+of\s+t\b', lambda m: r'$u(t)$'),
            (r'\bx2\s+of\s+t\b', lambda m: r'$x_2(t)$'),
            (r'\bx1\s+of\s+t\b', lambda m: r'$x_1(t)$'),
            (r'\bx\s+of\s+n\b', lambda m: r'$x[n]$'),
            (r'\by\s+of\s+n\b', lambda m: r'$y[n]$'),
            (r'\bh\s+of\s+n\b', lambda m: r'$h[n]$'),
            (r'\bxt\b', lambda m: r'$x(t)$'),
            (r'\byt\b', lambda m: r'$y(t)$'),
            (r'\bht\b', lambda m: r'$h(t)$'),
            (r'\bxs\b', lambda m: r'$X(s)$'),
            (r'\bys\b', lambda m: r'$Y(s)$'),
            (r'\bhs\b', lambda m: r'$H(s)$'),
        ]
        for pat, rep in speech_subs:
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)

        # Clean stray dots, comma-dot artifacts, and whitespace
        text = re.sub(r'[,;]\s*\.', '.', text)
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

        # 2. Comprehensive Multi-Domain STEM Classifier
        lower = clean_text.lower()
        domain_rules = [
            # Continuous-Time Signals & LTI Systems
            (r'\b(?:initial[- ]value theorem|final[- ]value theorem|initial value theorem|final value theorem|continuous[- ]time lti|impulse response)\b',
             "Continuous-Time LTI Systems: Initial & Final Value Theorems"),
            # Control Systems
            (r'\b(?:steady[- ]state error|disturbance rejection|open[- ]loop|closed[- ]loop|feedback control|loop gain|sensitivity function|complementary sensitivity)\b',
             "Steady-State Error Analysis, Disturbance Rejection & Feedback Control"),
            (r'\b(?:root locus|nyquist|bode plot|gain margin|phase margin|state space|pole placement)\b',
             "Frequency-Domain Analysis, Stability Margins & State Feedback Design"),
            # Signal Processing
            (r'\b(?:energy of (?:a|the) sequence|partial energy|discrete[- ]time sequence|periodic signals?|finite[- ]duration signal)\b',
             "Energy of a Sequence, Average Power & Discrete Signal Classification"),
            (r'\b(?:z[- ]transform|roc|region of convergence|complex plane|unit circle)\b',
             "The Z-Transform, Region of Convergence & Discrete-Time System Stability"),
            (r'\b(?:fourier transform|dft|fft|spectrum|frequency response|sampling theorem|nyquist rate)\b',
             "Fourier Analysis, Discrete Spectrum & Sampling Dynamics"),
            (r'\b(?:fir filter|iir filter|filter design|butterworth|chebyshev|convolution)\b',
             "Digital Filter Design, Linear Convolution & System Realization"),
            # Power & Electrical Machines
            (r'\b(?:park transform|direct axis|quadrature axis|rotor reference|synchronous machine|d-q)\b',
             "Dynamic Modeling of Synchronous Machines & Park Transformation"),
            (r'\b(?:induction motor|transformer|torque-speed|phasor|reactive power|apparent power)\b',
             "Electrical Machines, Phasor Dynamics & Electromechanical Conversion"),
            # Electronics & Circuits
            (r'\b(?:operational amplifier|op[- ]amp|differential pair|bjt|mosfet|cmos|biasing)\b',
             "Active Circuit Analysis, Amplifiers & Semiconductor Device Dynamics"),
            # Thermodynamics & Fluid Dynamics
            (r'\b(?:heat transfer|conduction|convection|radiation|navier-stokes|reynolds number)\b',
             "Thermal Transport Phenomena, Fluid Dynamics & Energy Conservation"),
            # Computer Science & Algorithms
            (r'\b(?:dynamic programming|graph algorithm|shortest path|computational complexity|big o)\b',
             "Algorithmic Analysis, Discrete Optimization & Computational Complexity"),
        ]
        for pat, title in domain_rules:
            if re.search(pat, lower):
                return title

        # Fallback: identify top recurrent non-stopword technical nouns
        tokens = [w.strip(".,;:?!()[]{}\"'") for w in lower.split()]
        stopwords = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by", "from",
                     "is", "are", "was", "were", "be", "been", "have", "has", "had", "this", "that", "these", "those",
                     "it", "its", "we", "you", "they", "i", "so", "as", "if", "then", "also", "just", "now", "what",
                     "how", "why", "which", "will", "can", "could", "would", "should", "some", "all", "our", "about"}
        keywords = [t for t in tokens if len(t) > 3 and t not in stopwords and not t.isdigit()]
        if keywords:
            from collections import Counter
            common = [w.title() for w, _ in Counter(keywords).most_common(4)]
            return f"Foundational Analysis of {' & '.join(common[:3])}"

        return "Advanced Engineering Exposition"

    def assemble_paragraphs(self, chunks: List[TranscriptChunk]) -> List[str]:
        """
        Merges chunk clauses into cohesive paragraphs.
        Preserves 100% of spoken content from the lecturer in exact chronological order.
        Eliminates artificial mid-sentence chunk splitting.
        """
        if not chunks:
            return []

        # Step 1: Join chunk clauses seamlessly preserving sentence boundaries
        full_text_parts = []
        for c in chunks:
            t = (c.latex_text or c.spoken_text or "").strip()
            if not t:
                continue
            full_text_parts.append(t)

        full_raw = " ".join(full_text_parts)
        # Step 2: Clean speech and normalize mathematical speech across the full stream
        cleaned_stream = self.clean_speech_text(full_raw)

        # Step 3: Segment into genuine sentences using punctuation boundaries
        sentence_delims = re.split(r'(?<=[.!?])\s+(?=[A-Z$])', cleaned_stream)

        paragraphs = []
        curr_group = []
        for s in sentence_delims:
            s = s.strip()
            if not s:
                continue
            if len(s) > 1 and s[0].islower():
                s = s[0].upper() + s[1:]
            if not s.endswith(('.', '!', '?', '$')):
                s += '.'
            curr_group.append(s)
            if len(curr_group) >= 4:
                paragraphs.append(" ".join(curr_group))
                curr_group = []

        if curr_group:
            paragraphs.append(" ".join(curr_group))

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

        # Build structured monograph with formal sections, theorems, definitions, and contextual TikZ diagrams
        body_tex_parts = self.sectionizer.structure_monograph(processed_paragraphs, tikz_blocks, full_text)
        body_tex = "\n".join(body_tex_parts)

        # LaTeX Document Template (100% Complete Transcript, Typeset with Vector Math)
        latex_doc = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage{{amsmath,amssymb,amsfonts,amsthm,mathtools}}
\usepackage{{geometry}}
\geometry{{margin=0.9in, top=1.0in, bottom=1.0in}}
\usepackage{{xcolor}}
\usepackage{{fancyhdr}}
\usepackage{{booktabs}}
\usepackage{{enumitem}}
\usepackage{{hyperref}}
\usepackage{{tikz}}
\usetikzlibrary{{shapes,arrows.meta,positioning,calc}}

\definecolor{{primarydark}}{{RGB}}{{20, 35, 60}}
\definecolor{{accentblue}}{{RGB}}{{30, 90, 160}}
\definecolor{{mutedgray}}{{RGB}}{{90, 100, 115}}

\hypersetup{{
    colorlinks=true,
    linkcolor=accentblue,
    citecolor=accentblue,
    urlcolor=accentblue
}}

\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\small\color{{mutedgray}} {clean_title_tex}}}
\fancyhead[R]{{\small\color{{mutedgray}} Page \thepage}}
\renewcommand{{\headrulewidth}}{{0.4pt}}

\setlength{{\parskip}}{{0.6em}}
\setlength{{\parindent}}{{0pt}}

\newtheorem{{theorem}}{{Theorem}}
\newtheorem{{definition}}{{Definition}}

\begin{{document}}

\begin{{center}}
  {{\LARGE\bfseries\color{{primarydark}} {clean_title_tex}}} \\[0.5em]
  {{\small\color{{mutedgray}} Academic Monograph \quad $\cdot$ \quad {date_str}}}
\end{{center}}

\vspace{{0.3em}}
\hrule height 1pt
\vspace{{1.0em}}

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
