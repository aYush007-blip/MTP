"""
Voice Activity Detection (VAD) module for NVIDIA Jetson Orin Nano.
Uses AdaptiveEnergyVAD with hysteresis (Schmitt Trigger) and pre-roll buffers.
"""

import numpy as np
from energy_vad import AdaptiveEnergyVAD


class SileroVAD:
    """
    Robust edge Voice Activity Detector.
    Matches the Silero VAD event interface ([{"start": idx}], [{"end": idx}])
    while executing with pure NumPy / SciPy without PyTorch/torchaudio glibc crashes.
    """

    def __init__(self, threshold: float = 0.060):
        self.energy_vad = AdaptiveEnergyVAD(
            speech_threshold=threshold,
            silence_threshold=max(threshold * 0.58, 0.025),
            min_speech_chunks=2,
            min_silence_chunks=7,
            pre_roll_chunks=4,
        )
        self._total_samples = 0

    def process(self, audio: np.ndarray) -> list[dict]:
        """
        Process float32 audio (16kHz mono).
        Returns list of events, e.g. [{"start": sample_idx}] or [{"end": sample_idx}].
        """
        event, _ = self.energy_vad.process(audio)
        self._total_samples += len(audio)
        results = []
        if event == "start":
            results.append({"start": self._total_samples})
        elif event == "end":
            results.append({"end": self._total_samples})
        return results

    def reset(self):
        self.energy_vad.reset()
        self._total_samples = 0
