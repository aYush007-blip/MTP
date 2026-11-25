#!/usr/bin/env python3
"""
Regression Test Suite for Live Lecture Transcription and Bilingual Study Notes Pipeline.

Covers:
  - Audio preprocessing (48 kHz stereo -> 16 kHz mono) and HighPassHumFilter
  - AdaptiveEnergyVAD hysteresis and pre-roll buffer
  - Mathematical-cue detection and SLM output validation guards
  - Notation consistency integration (Priority 3)
  - Hindi translation math-span placeholder & numerical preservation (Priority 4)
  - Note generation and offline Tectonic PDF compilation
  - Headless workflow state transitions and graceful server/USB failure handling
  - USB export, ZIP bundling, and SHA-256 integrity checks
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

HEADLESS_SRC = Path("/mnt/thesis/headless_agent/src")
if str(HEADLESS_SRC) not in sys.path:
    sys.path.insert(0, str(HEADLESS_SRC))

from audio import stereo_48k_to_mono_16k
from energy_vad import AdaptiveEnergyVAD
from headless_voice_agent import (
    AgentState,
    EdgeBackpressureScheduler,
    HighPassHumFilter,
    ProcessingStage,
    ProcessingTracker,
    clean_trigger_text,
    is_service_alive,
    query_whisper_stream,
)
from hindi_translator import (
    HindiLectureTranslator,
    extract_outside_numbers,
    mask_math_spans,
    restore_and_verify_math_spans,
)
from lecture_to_book import (
    LectureToBookBuilder,
    TranscriptChunk,
    balance_math_delimiters,
    escape_latex_text,
    generate_subtitles,
    sanitize_latex_source,
)
from postprocess_lecture_ai import GlobalAIReviewer, PostLectureProcessor
from slm_translator import QwenMathTranslator
from usb_exporter import LectureExporter, compute_sha256


class TestAudioAndVAD(unittest.TestCase):
    def test_stereo_48k_to_mono_16k_decimation_and_range(self):
        # 4800 stereo frames (9600 int16 samples) -> 1600 mono 16kHz samples
        t = np.arange(4800, dtype=np.float32) / 48000.0
        tone = (0.5 * 32767.0 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.int16)
        stereo = np.column_stack([tone, tone]).reshape(-1)
        out = stereo_48k_to_mono_16k(stereo.tobytes())
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(len(out), 1600)
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)
        self.assertGreater(float(np.sqrt(np.mean(out ** 2))), 0.25)

    def test_stereo_48k_to_mono_16k_empty(self):
        out = stereo_48k_to_mono_16k(b"")
        self.assertEqual(len(out), 0)
        self.assertEqual(out.dtype, np.float32)

    def test_high_pass_hum_filter_attenuates_50hz(self):
        hpf = HighPassHumFilter()
        t = np.arange(16000, dtype=np.float32) / 16000.0
        hum_50hz = 0.5 * np.sin(2 * np.pi * 50.0 * t).astype(np.float32)
        speech_1khz = 0.5 * np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)

        hum_out = hpf.process(hum_50hz)
        hpf.reset()
        speech_out = hpf.process(speech_1khz)

        rms_in = float(np.sqrt(np.mean(hum_50hz[1600:] ** 2)))
        rms_hum = float(np.sqrt(np.mean(hum_out[1600:] ** 2)))
        rms_speech = float(np.sqrt(np.mean(speech_out[1600:] ** 2)))
        self.assertLess(rms_hum, rms_in * 0.20)
        self.assertGreater(rms_speech, 0.30)

    def test_adaptive_energy_vad_hysteresis_and_preroll(self):
        vad = AdaptiveEnergyVAD(
            speech_threshold=0.05,
            silence_threshold=0.02,
            min_speech_chunks=2,
            min_silence_chunks=3,
            pre_roll_chunks=4,
        )
        quiet = np.full(1600, 0.005, dtype=np.float32)
        speech = np.full(1600, 0.10, dtype=np.float32)
        hysteresis_hold = np.full(1600, 0.03, dtype=np.float32)

        for _ in range(3):
            ev, pre = vad.process(quiet)
            self.assertIsNone(ev)
            self.assertEqual(pre, [])

        # Single impulse chunk must not trigger speech onset
        ev, _ = vad.process(speech)
        self.assertIsNone(ev)
        ev, _ = vad.process(quiet)
        self.assertIsNone(ev)

        # Two consecutive speech chunks trigger onset with pre-roll history
        ev, _ = vad.process(speech)
        self.assertIsNone(ev)
        ev, pre = vad.process(speech)
        self.assertEqual(ev, "start")
        self.assertEqual(len(pre), 4)

        # Hysteresis: 0.03 is below onset (0.05) but above silence (0.02), so speech stays active
        ev, _ = vad.process(hysteresis_hold)
        self.assertIsNone(ev)
        self.assertTrue(vad.in_speech)

        # Three silent chunks trigger end
        self.assertIsNone(vad.process(quiet)[0])
        self.assertIsNone(vad.process(quiet)[0])
        self.assertEqual(vad.process(quiet)[0], "end")


class TestMathCueAndOutputGuards(unittest.TestCase):
    def setUp(self):
        self.slm = QwenMathTranslator(backend="server", server_url="http://127.0.0.1:19999")

    def test_math_cue_gate(self):
        self.assertFalse(
            self.slm._has_math_cues("Good morning everyone. Welcome to today's presentation.")
        )
        self.assertTrue(
            self.slm._has_math_cues("The Laplace transform of x of t is X of s.")
        )
        self.assertTrue(
            self.slm._has_math_cues("Taking the integral from zero to infinity of x squared.")
        )

    def test_clean_response_guards(self):
        raw = "The signal x of t has Laplace transform X of s."
        # Normalizes $$...$$ and \[...\] to inline $...$
        cleaned = self.slm._clean_response(
            "```latex\nThe signal $$x(t)$$ has Laplace transform \\[X(s)\\].\n```",
            raw_text=raw,
        )
        self.assertEqual(cleaned, "The signal $x(t)$ has Laplace transform $X(s)$.")

        # Rejects repetition loop
        loop_out = " ".join(["the value equals"] * 6)
        self.assertEqual(self.slm._clean_response(loop_out, raw_text=raw), raw)

        # Rejects fabricated integral when absent in raw_text
        self.assertEqual(
            self.slm._clean_response("We compute $\\int_0^1 x(t) dt$.", raw_text=raw),
            raw,
        )

        # Rejects fabricated cases environment
        self.assertEqual(
            self.slm._clean_response(r"We get $\begin{cases} 1 \\ 0 \end{cases}$.", raw_text=raw),
            raw,
        )

    def test_latex_delimiters_and_sanitization(self):
        self.assertEqual(balance_math_delimiters("The value is $x_1"), "The value is $x_1$")
        escaped = escape_latex_text("50% of $x_1(t)$ & $H(s)$")
        self.assertIn(r"50\%", escaped)
        self.assertIn(r"\&", escaped)
        self.assertIn("$x_1(t)$", escaped)

        deduped = sanitize_latex_source(
            r"\begin{equation}\label{eq:1} X(s) = \frac{1}{s}\end{equation}" + "\n"
            + r"\begin{equation}\label{eq:1} Y(s) = H(s)X(s)\end{equation}"
        )
        self.assertEqual(deduped.count(r"\label{eq:1}"), 1)
        self.assertIn(r"\label{eq:1_1}", deduped)

    def test_streaming_early_abort_and_cache_prompt_payload(self):
        class _FakeSSEResp:
            def __init__(self, tokens, tracker):
                self._tokens = tokens
                self._tracker = tracker
                self.headers = {"Content-Type": "text/event-stream"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __iter__(self):
                for tok in self._tokens:
                    self._tracker["consumed"] += 1
                    evt = json.dumps({"choices": [{"delta": {"content": tok}}]})
                    yield f"data: {evt}\n".encode("utf-8")
                yield b"data: [DONE]\n"

        captured_payloads = []
        raw = "The signal x of t is causal."

        # 1. Repetition runaway aborts mid-stream before consuming all chunks
        rep_tokens = ["the ", "value ", "equals "] * 15  # 45 tokens
        rep_tracker = {"consumed": 0}

        def _mock_rep(req, timeout=60.0):
            captured_payloads.append(json.loads(req.data.decode("utf-8")))
            return _FakeSSEResp(rep_tokens, rep_tracker)

        with patch("slm_translator.urllib.request.urlopen", side_effect=_mock_rep):
            out = self.slm.translate_server(raw, max_new_tokens=256)

        self.assertEqual(out, raw)
        self.assertEqual(self.slm.last_abort_reason, "repetition_loop")
        self.assertTrue(self.slm.last_stream_stats.get("aborted"))
        self.assertLess(rep_tracker["consumed"], len(rep_tokens))
        self.assertTrue(captured_payloads[0]["cache_prompt"])
        self.assertTrue(captured_payloads[0]["stream"])

        # 2. Fabricated integral aborts early mid-stream
        int_tokens = ["We ", "compute ", r"\int", "_0^1 ", "x(t) ", "dt ", "now."]
        int_tracker = {"consumed": 0}

        def _mock_int(req, timeout=60.0):
            return _FakeSSEResp(int_tokens, int_tracker)

        with patch("slm_translator.urllib.request.urlopen", side_effect=_mock_int):
            out_int = self.slm.translate_server(raw, max_new_tokens=256)

        self.assertEqual(out_int, raw)
        self.assertEqual(self.slm.last_abort_reason, "fabricated_integral")
        self.assertLess(int_tracker["consumed"], len(int_tokens))

        # 3. Valid stream completes normally without false abort
        ok_tokens = ["The ", "signal ", "$x(t)$ ", "is ", "causal."]
        ok_tracker = {"consumed": 0}

        def _mock_ok(req, timeout=60.0):
            return _FakeSSEResp(ok_tokens, ok_tracker)

        with patch("slm_translator.urllib.request.urlopen", side_effect=_mock_ok):
            out_ok = self.slm.translate_server(raw, max_new_tokens=256)

        self.assertEqual(out_ok, "The signal $x(t)$ is causal.")
        self.assertIsNone(self.slm.last_abort_reason)
        self.assertFalse(self.slm.last_stream_stats.get("aborted"))
        self.assertEqual(ok_tracker["consumed"], len(ok_tokens))


class TestNotationHarmonisation(unittest.TestCase):
    def test_preserves_distinct_meaningful_expressions(self):
        unchanged_cases = [
            "For continuous-time signals $x(t)$ and discrete sequences $x[n]$, sampling gives $x[n] = x(nT_s)$.",
            "The system $H(s) = \\frac{Y(s)}{X(s)}$ maps input $x(t)$ to output $y(t)$.",
            "Consider the integer-indexed function $x(n)$ without any bracket convention.",
            "Let $h(s)$ denote an auxiliary scalar function in this derivation.",
        ]
        for text in unchanged_cases:
            out = GlobalAIReviewer.harmonize_notation_deterministic(text, context_text=text)
            self.assertEqual(out, text)

    def test_harmonizes_inconsistent_spacing_and_cross_chunk_notation(self):
        # Spacing normalization
        spaced = "We write $x ( t )$, $x [ n ]$, $X ( s )$, and $H ( s )$."
        self.assertEqual(
            GlobalAIReviewer.harmonize_notation_deterministic(spaced),
            "We write $x(t)$, $x[n]$, $X(s)$, and $H(s)$.",
        )

        # Cross-chunk discrete & s-domain consistency without altering x(t)
        reviewer = GlobalAIReviewer(server_url="http://127.0.0.1:19999")
        paras = [
            "Consider a continuous signal $x(t)$ with Laplace transform $X(s)$ and transfer function $H(s)$.",
            "After sampling, the discrete sequence $x[n]$ is related to earlier samples $x(n)$ and transform $x(s)$ with $h(s)$ and $H[s]$.",
        ]
        harmonized = reviewer.harmonize_paragraphs(paras, use_slm=False)
        self.assertIn("$x(t)$", harmonized[0])
        self.assertIn("$X(s)$", harmonized[0])
        self.assertIn("$H(s)$", harmonized[0])
        self.assertIn("$x[n]$", harmonized[1])
        self.assertNotIn("$x(n)$", harmonized[1])
        self.assertIn("$X(s)$", harmonized[1])
        self.assertIn("$H(s)$", harmonized[1])
        self.assertNotIn("$H[s]$", harmonized[1])

    def test_slm_rewrite_guard_rejects_corrupted_domains(self):
        orig = "The continuous input $x(t)$ produces $X(s)$ and discrete $x[n]$ with gain 2.5."
        self.assertFalse(
            GlobalAIReviewer.is_safe_harmonized_edit(
                orig, "The discrete input $x[n]$ produces $X(s)$ and discrete $x[n]$ with gain 2.5."
            )
        )
        self.assertFalse(
            GlobalAIReviewer.is_safe_harmonized_edit(
                orig, "The continuous input $x(t)$ produces $X(s)$ and discrete $x[n]$ with gain 3.5."
            )
        )
        self.assertTrue(
            GlobalAIReviewer.is_safe_harmonized_edit(
                orig, "The continuous-time input $x(t)$ produces $X(s)$ and discrete sequence $x[n]$ with gain 2.5."
            )
        )

    def test_live_symbol_registry_domain_lock_and_topic_shift(self):
        slm = QwenMathTranslator(backend="server", server_url="http://127.0.0.1:19999", enable_symbol_lock=True)
        # 1. Establish discrete sequence y[n], x[n] and transform H(s)
        slm.symbol_registry.observe(
            "For discrete-time sequences y1 of n equals x of 3n and transfer function H of s.",
            "$y_1[n] = x[3n]$ and transfer function $H(s)$.",
        )
        slm.symbol_registry.observe(
            "Now compute y of n from x of n.",
            "Now compute $y[n]$ from $x[n]$.",
        )
        snap = slm.symbol_registry_snapshot()
        self.assertEqual(snap["signal_domains"].get("x"), "discrete")
        self.assertEqual(snap["signal_domains"].get("y"), "discrete")
        self.assertEqual(snap["transforms"].get("H"), "s")

        # 2. Verify KV cache prefix (system + few-shot) stays unchanged when hint is added
        msgs_plain = QwenMathTranslator(backend="server", server_url="http://127.0.0.1:19999", enable_symbol_lock=False)._build_messages("Y of 1 equals x of 2")
        msgs_hinted = slm._build_messages("Y of 1 equals x of 2")
        self.assertEqual(msgs_plain[:-1], msgs_hinted[:-1])
        self.assertIn("[Active lecture notation:", msgs_hinted[-1]["content"])

        # 3. Domain lock repairs paren & sentence-start case drift (y(1)/Y(1)->y[1], X(2)->x[2], H[s]->H(s)), leaving x(t) untouched
        repaired = slm._clean_response("$Y(1)$ will be $X(2)$ with $H[s]$ and continuous $x(t)$.", raw_text="Y of 1 will be x of 2")
        self.assertIn("$y[1]$", repaired)
        self.assertIn("$x[2]$", repaired)
        self.assertIn("$H(s)$", repaired)
        self.assertIn("$x(t)$", repaired)

        # 4. Explicit continuous-time transition clears conflicting discrete lock and reset clears all state
        slm.symbol_registry.observe(
            "Moving to continuous-time systems with Laplace transform, input x of t.",
            "Moving to continuous-time systems with Laplace transform, input $x(t)$.",
        )
        self.assertEqual(slm.symbol_registry_snapshot()["signal_domains"].get("x"), "continuous")

        slm.reset_symbol_registry()
        self.assertEqual(slm.symbol_registry_snapshot()["recent_symbols"], [])
        self.assertEqual(slm.symbol_registry_snapshot()["signal_domains"], {})


class TestHindiMathPreservation(unittest.TestCase):
    def test_mask_and_restore_all_math_constructs(self):
        src = (
            "At 50 Hz and gain 0.85, the continuous signal $x_1(t)$ with Greek $\\alpha + \\omega_0^2$ "
            "satisfies $\\frac{\\omega_n^2}{s^2 + 2\\zeta\\omega_n s + \\omega_n^2}$ and display block "
            "$$X(s) = \\int_0^\\infty x(t) e^{-st} \\, dt$$ "
            "and \\begin{equation}\\label{eq:1} y[n] = x[n]^2 \\end{equation} over 3 steps."
        )
        masked, spans = mask_math_spans(src)
        self.assertEqual(len(spans), 5)
        self.assertNotIn("$", masked)
        self.assertEqual(extract_outside_numbers(masked), ["50", "0.85", "3"])

        # Simulate valid Hindi translation preserving placeholders in exact order & numbers
        simulated_hi = (
            "50 Hz और लाभ 0.85 पर, सतत संकेत [[MATH_0]] तथा ग्रीक [[MATH_1]] "
            "समीकरण [[MATH_2]] और प्रदर्शन ब्लॉक [[MATH_3]] तथा 3 चरणों में [[MATH_4]] को संतुष्ट करता है।"
        )
        ok, restored, reason = restore_and_verify_math_spans(masked, simulated_hi, spans)
        self.assertTrue(ok, reason)
        for sp in spans:
            self.assertIn(sp, restored)
        self.assertIn("50", restored)
        self.assertIn("0.85", restored)
        self.assertIn("3", restored)

    def test_validation_rejects_swapped_missing_and_modified_numbers(self):
        src = "At 10 V, $x(t)$ yields $H(s)$."
        masked, spans = mask_math_spans(src)

        # Swapped placeholder order
        ok, _, _ = restore_and_verify_math_spans(masked, "10 V पर [[MATH_1]] से [[MATH_0]]।", spans)
        self.assertFalse(ok)

        # Duplicate placeholder
        ok, _, _ = restore_and_verify_math_spans(masked, "10 V पर [[MATH_0]] और [[MATH_0]]।", spans)
        self.assertFalse(ok)

        # Missing placeholder
        ok, _, _ = restore_and_verify_math_spans(masked, "10 V पर [[MATH_0]]।", spans)
        self.assertFalse(ok)

        # Modified outside number (10 -> 100)
        ok, _, _ = restore_and_verify_math_spans(masked, "100 V पर [[MATH_0]] से [[MATH_1]]।", spans)
        self.assertFalse(ok)

    def test_translator_retries_once_and_falls_back_to_english(self):
        translator = HindiLectureTranslator(server_url="http://127.0.0.1:19999")
        src = "For 2 systems, $x(t)$ gives $X(s)$."

        # Attempt 1 drops [[MATH_1]], Attempt 2 succeeds
        with patch.object(
            translator,
            "_request_completion",
            side_effect=[
                "2 प्रणालियों के लिए [[MATH_0]] देता है।",
                "2 प्रणालियों के लिए [[MATH_0]], [[MATH_1]] देता है।",
            ],
        ):
            out = translator.translate_text(src)
        self.assertIn("$x(t)$", out)
        self.assertIn("$X(s)$", out)
        self.assertIn("प्रणालियों", out)

        # Both attempts fail -> safe fallback to original English
        with patch.object(
            translator,
            "_request_completion",
            side_effect=[
                "corrupted output without placeholders",
                "2 प्रणालियों के लिए [[MATH_1]] और [[MATH_0]]।",
            ],
        ):
            out_fb = translator.translate_text(src)
        self.assertEqual(out_fb, src)

    def test_subtitle_batch_fallback_per_line(self):
        translator = HindiLectureTranslator(server_url="http://127.0.0.1:19999")
        lines = ["Signal $x(t)$ is causal.", "At 5 Hz, $H(s) = 1$."]
        with patch.object(
            translator,
            "_request_completion",
            return_value="1. संकेत [[MATH_0]] कार्य-कारण है।\n2. 50 Hz पर [[MATH_0]]।",
        ):
            out = translator.translate_subtitles_batch(lines)
        self.assertEqual(out[0], "संकेत $x(t)$ कार्य-कारण है।")
        self.assertEqual(out[1], lines[1])  # line 2 had 5 -> 50 mismatch, falls back


class TestNoteGenerationPDFAndExport(unittest.TestCase):
    def test_book_generation_and_offline_tectonic_pdf(self):
        chunks = [
            TranscriptChunk(
                0.0,
                4.0,
                "We define a causal continuous-time LTI system with input x(t) and impulse response h(t).",
                "We define a causal continuous-time LTI system with input $x(t)$ and impulse response $h(t)$.",
            ),
            TranscriptChunk(
                4.0,
                9.0,
                "The Laplace transform is X(s) and the transfer function is H(s) equals 1 over s plus 3.",
                r"The Laplace transform is $X(s)$ and the transfer function is $H(s) = \frac{1}{s+3}$.",
            ),
        ]
        builder = LectureToBookBuilder()
        tex_doc, md_doc, title = builder.build_book("LTI Systems Test", chunks)
        self.assertIn(r"\begin{equation}", tex_doc)
        self.assertIn(r"\section{", tex_doc)
        self.assertIn("$x(t)$", tex_doc)
        self.assertIn("$$", md_doc)

        srt, vtt = generate_subtitles(chunks)
        self.assertIn("00:00:00,000 --> 00:00:04,000", srt)
        self.assertTrue(vtt.startswith("WEBVTT"))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            tex_path = tmp_dir / "smoke_test.tex"
            tex_path.write_text(tex_doc, encoding="utf-8")

            processor = PostLectureProcessor()
            res = processor.process_and_compile(title, tex_path, tmp_dir)
            pdf_path = res.get("pdf")
            self.assertIsNotNone(pdf_path)
            self.assertTrue(pdf_path.is_file())
            self.assertGreater(pdf_path.stat().st_size, 1000)
            self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF-"))

            # Also verify Hindi bilingual monograph generation + offline Tectonic XeTeX compilation
            hi_translator = HindiLectureTranslator.get_instance()
            with patch.object(hi_translator, "translate_title", return_value="सतत-समय LTI प्रणालियाँ"), \
                 patch.object(
                     hi_translator,
                     "translate_paragraphs",
                     return_value=[
                         "हम एक कार्य-कारण सतत-समय LTI प्रणाली को इनपुट $x(t)$ और आवेग अनुक्रिया $h(t)$ के साथ परिभाषित करते हैं। "
                         "लाप्लास रूपांतरण $X(s)$ है तथा अंतरण फलन $H(s) = \\frac{1}{s+3}$ है।"
                     ],
                 ), \
                 patch.object(
                     hi_translator,
                     "translate_subtitles_batch",
                     side_effect=lambda batch: [f"हिंदी: {b}" for b in batch],
                 ):
                hi_tex, hi_md, hi_title, hi_chunks, hi_paras = builder.build_hindi_book(title, chunks)

            self.assertIn(r"\setmainfont{Lohit Devanagari}", hi_tex)
            self.assertIn("$x(t)$", hi_tex)
            self.assertIn(r"\begin{equation}", hi_tex)
            self.assertEqual(len(hi_chunks), len(chunks))

            hi_tex_path = tmp_dir / "smoke_test_hindi.tex"
            hi_tex_path.write_text(hi_tex, encoding="utf-8")
            hi_pdf = processor.compile_hindi_pdf(
                tex_path=hi_tex_path,
                output_dir=tmp_dir,
                title=hi_title,
                paragraphs=hi_paras,
                output_pdf_path=tmp_dir / "smoke_test_hindi.pdf",
            )
            self.assertIsNotNone(hi_pdf)
            self.assertTrue(hi_pdf.is_file())
            self.assertGreater(hi_pdf.stat().st_size, 1000)
            self.assertTrue(hi_pdf.read_bytes().startswith(b"%PDF-"))

    def test_headless_state_transitions_and_unavailable_servers(self):
        tracker = ProcessingTracker()
        self.assertEqual(tracker.stage, AgentState.STANDBY if False else ProcessingStage.IDLE)
        tracker.start_pipeline(estimated_total=30.0)
        self.assertEqual(tracker.stage, ProcessingStage.STARTING)
        for st in (
            ProcessingStage.SAVING_AUDIO,
            ProcessingStage.TRANSCRIBING,
            ProcessingStage.STRUCTURING_NOTES,
            ProcessingStage.TRANSLATING_HINDI,
            ProcessingStage.COMPILING_PDF,
            ProcessingStage.EXPORTING_USB,
        ):
            tracker.set_stage(st, additional_est=5.0)
            pct = tracker.get_progress_percentage()
            self.assertGreaterEqual(pct, 1)
            self.assertLessEqual(pct, 99)
        tracker.set_stage(ProcessingStage.COMPLETE)
        self.assertEqual(tracker.get_progress_percentage(), 100)
        self.assertEqual(tracker.get_remaining_seconds(), 0)
        self.assertEqual(clean_trigger_text("Hey Jetson, start recording!"), "hey jetson start recording")

        # Unavailable server handling must degrade gracefully without crashing
        self.assertFalse(is_service_alive("http://127.0.0.1:19999"))
        pcm = np.full(8000, 0.05, dtype=np.float32)
        with patch("headless_voice_agent.WHISPER_SERVER_URL", "http://127.0.0.1:19999"):
            self.assertEqual(query_whisper_stream(pcm), "")

    def test_usb_export_missing_and_checksum_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            local_base = tmp_dir / "local_exports"
            usb_root = tmp_dir / "fake_usb"
            usb_root.mkdir()

            sample_tex = tmp_dir / "sample.tex"
            sample_tex.write_text("hello $x(t)$", encoding="utf-8")
            sample_pdf = tmp_dir / "sample.pdf"
            sample_pdf.write_bytes(b"%PDF-1.4\n" + b"0" * 1200)

            exporter = LectureExporter(local_base=str(local_base))

            # 1. Missing USB drive -> saves locally and returns usb_success=False
            with patch("usb_exporter.find_usb_mount", return_value=None), patch("usb_exporter.time.sleep"):
                res_missing = exporter.export(
                    "Smoke_Missing_USB",
                    {"tex": sample_tex, "pdf": sample_pdf},
                    timestamp="20261125_000001",
                )
            self.assertFalse(res_missing["usb_success"])
            self.assertIsNone(res_missing["usb_path"])
            local_dir = Path(res_missing["local_path"])
            self.assertTrue((local_dir / "SHA256SUMS.txt").is_file())
            self.assertTrue((local_dir / "Smoke_Missing_USB_bundle.zip").is_file())

            # 2. Present USB drive -> exports, creates valid ZIP, and verifies SHA256SUMS.txt
            with patch("usb_exporter.find_usb_mount", return_value=usb_root):
                res_ok = exporter.export(
                    "Smoke_Valid_USB",
                    {"tex": sample_tex, "pdf": sample_pdf},
                    timestamp="20261125_000002",
                )
            self.assertTrue(res_ok["usb_success"])
            usb_dir = Path(res_ok["usb_path"])
            checksum_lines = (usb_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(checksum_lines), 3)
            for line in checksum_lines:
                expected_hash, fname = line.split("  ", 1)
                self.assertEqual(compute_sha256(usb_dir / fname), expected_hash)

            zip_path = usb_dir / "Smoke_Valid_USB_bundle.zip"
            with zipfile.ZipFile(zip_path, "r") as zf:
                self.assertIn("sample.tex", zf.namelist())
                self.assertIn("sample.pdf", zf.namelist())

    def test_backpressure_scheduler_regimes_and_telemetry(self):
        # 1. Verify live sysfs/meminfo reader returns non-negative Jetson telemetry
        live_sched = EdgeBackpressureScheduler()
        tel = live_sched.read_system_telemetry()
        self.assertGreater(tel["mem_total_mb"], 0.0)
        self.assertGreater(tel["mem_available_mb"], 0.0)
        self.assertGreaterEqual(tel["max_temp_c"], 0.0)

        # 2. Verify NORMAL -> PRESSURE -> CRITICAL transitions from synthetic sysfs/queue inputs
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            mem_f = tdir / "meminfo"
            z0 = tdir / "tz0"
            z0.mkdir()
            (z0 / "type").write_text("cpu-thermal\n", encoding="utf-8")
            temp_f = z0 / "temp"

            sched = EdgeBackpressureScheduler(
                meminfo_path=str(mem_f),
                thermal_glob=str(tdir / "tz*/temp"),
            )

            # NORMAL: ample unified RAM, cool SoC, low queue depth
            mem_f.write_text("MemTotal:        8000000 kB\nMemAvailable:    3600000 kB\n", encoding="utf-8")
            temp_f.write_text("52000\n", encoding="utf-8")
            d1 = sched.evaluate(audio_q_depth=2, typeset_q_depth=1)
            self.assertEqual(d1["regime"], "NORMAL")
            self.assertFalse(d1["defer_live_slm"])
            self.assertEqual(d1["throttle_ms"], 0)

            # PRESSURE: thermal rise (80 C) triggers pacing without dropping SLM
            temp_f.write_text("80000\n", encoding="utf-8")
            d2 = sched.evaluate(audio_q_depth=4, typeset_q_depth=4)
            self.assertEqual(d2["regime"], "PRESSURE")
            self.assertFalse(d2["defer_live_slm"])
            self.assertEqual(d2["throttle_ms"], 120)

            # CRITICAL: low free RAM (<950 MB) or thermal spike defers live SLM to protect capture (R4)
            mem_f.write_text("MemTotal:        8000000 kB\nMemAvailable:     750000 kB\n", encoding="utf-8")
            temp_f.write_text("85500\n", encoding="utf-8")
            d3 = sched.evaluate(audio_q_depth=10, typeset_q_depth=12)
            self.assertEqual(d3["regime"], "CRITICAL")
            self.assertTrue(d3["defer_live_slm"])
            self.assertTrue(d3["defer_heavy_rewrite"])

            summ = sched.summary()
            self.assertEqual(summ["samples"], 3)
            self.assertEqual(summ["deferred_chunks"], 1)
            self.assertEqual(summ["peak_typeset_queue"], 12)
            self.assertAlmostEqual(summ["peak_temp_c"], 85.5, places=1)
            self.assertEqual(summ["regimes_seen"], ["NORMAL", "PRESSURE", "CRITICAL"])

    def test_audit_telemetry_counters_and_bundle_binding(self):
        # 1. Verify Hindi translator structured audit counters (attempt 1, retry 2, fallback)
        tr = HindiLectureTranslator(server_url="http://127.0.0.1:19999")
        with patch.object(
            tr,
            "_request_completion",
            side_effect=[
                "सिग्नल [[MATH_0]] कारण-निष्ठ है।",              # pass attempt 1
                "यह केवल [[MATH_0]] है।",                       # fail attempt 1 (drops MATH_1)
                "यह [[MATH_0]] तथा [[MATH_1]] है।",              # recover attempt 2
                "यहाँ संख्या बदल गई।",                          # fail attempt 1 (drops 5)
                "यहाँ भी बदली।",                                # fail attempt 2 -> English fallback
            ],
        ):
            _ = tr.translate_text("Signal $x(t)$ is causal.")
            _ = tr.translate_text("From $x(t)$ we get $X(s)$.")
            _ = tr.translate_text("We need 5 samples of $x[n]$.")

        st = tr.audit_stats
        self.assertEqual(st["paragraphs_total"], 3)
        self.assertEqual(st["spans_protected_total"], 4)
        self.assertEqual(st["passed_attempt_1"], 1)
        self.assertEqual(st["recovered_attempt_2"], 1)
        self.assertEqual(st["english_fallbacks"], 1)

        # 2. Verify GlobalAIReviewer harmonisation counters
        reviewer = GlobalAIReviewer(server_url="http://127.0.0.1:19999")
        reviewer.harmonize_paragraphs(["We write $x ( t )$ and $X ( s )$."], use_slm=False)
        self.assertEqual(reviewer.harmonisation_stats["paragraphs_reviewed"], 1)
        self.assertEqual(reviewer.harmonisation_stats["deterministic_edits"], 1)

        # 3. Verify invariance_audit.json binds artifact hashes inside LectureExporter ZIP + SHA256SUMS.txt
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            eng_pdf = tdir / "lec.pdf"
            hi_pdf = tdir / "lec_hindi.pdf"
            eng_pdf.write_bytes(b"%PDF-1.4\n" + b"A" * 1200)
            hi_pdf.write_bytes(b"%PDF-1.4\n" + b"B" * 1200)
            audit_json = tdir / "lec_invariance_audit.json"
            audit_payload = {
                "schema": "jetson-math-notes-invariance-audit-v1",
                "hindi_math_preservation": st,
                "notation_harmonisation": reviewer.harmonisation_stats,
                "artifact_hashes_sha256": {
                    eng_pdf.name: compute_sha256(eng_pdf),
                    hi_pdf.name: compute_sha256(hi_pdf),
                },
            }
            audit_json.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")

            exporter = LectureExporter(local_base=str(tdir / "exports"))
            with patch("usb_exporter.find_usb_mount", return_value=None), patch("usb_exporter.time.sleep"):
                res = exporter.export(
                    "Audit_Test",
                    {"pdf": eng_pdf, "hindi_pdf": hi_pdf, "audit": audit_json},
                    timestamp="20261126_000003",
                )
            local_dir = Path(res["local_path"])
            sums = (local_dir / "SHA256SUMS.txt").read_text(encoding="utf-8")
            self.assertIn("lec_invariance_audit.json", sums)
            zip_path = local_dir / "Audit_Test_bundle.zip"
            with zipfile.ZipFile(zip_path, "r") as zf:
                self.assertIn("lec_invariance_audit.json", zf.namelist())
                loaded = json.loads(zf.read("lec_invariance_audit.json").decode("utf-8"))
                self.assertEqual(loaded["artifact_hashes_sha256"]["lec.pdf"], compute_sha256(eng_pdf))


if __name__ == "__main__":
    unittest.main()
