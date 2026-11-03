import collections
import numpy as np


class EnergyVAD:
    """Legacy static-threshold Energy VAD."""
    def __init__(
        self,
        threshold=0.01,
        min_speech_chunks=2,
        min_silence_chunks=10,
    ):
        self.threshold = threshold
        self.min_speech_chunks = min_speech_chunks
        self.min_silence_chunks = min_silence_chunks

        self.speech_count = 0
        self.silence_count = 0
        self.in_speech = False

    def process(self, audio):
        """
        audio: float32 mono audio, normalized to [-1, 1]
        Returns:
            "start"  -> speech started
            "end"    -> speech ended
            None     -> no state change
        """
        rms = np.sqrt(np.mean(audio ** 2))

        if rms > self.threshold:
            self.speech_count += 1
            self.silence_count = 0
        else:
            self.silence_count += 1
            self.speech_count = 0

        # Speech begins after consecutive speech chunks
        if not self.in_speech and self.speech_count >= self.min_speech_chunks:
            self.in_speech = True
            self.speech_count = 0
            return "start"

        # Speech ends after consecutive silent chunks
        if self.in_speech and self.silence_count >= self.min_silence_chunks:
            self.in_speech = False
            self.silence_count = 0
            return "end"

        return None

    def process_chunk(self, audio_chunk: np.ndarray) -> tuple[bool, float]:
        """
        Evaluate if current chunk exceeds speech threshold, applying hysteresis.
        Returns:
            (is_speech: bool, rms: float)
        """
        rms = float(np.sqrt(np.mean(audio_chunk ** 2))) if len(audio_chunk) > 0 else 0.0
        thold = self.silence_threshold if self.in_speech else self.speech_threshold
        if rms >= thold:
            self.in_speech = True
            return True, rms
        else:
            self.in_speech = False
            return False, rms

    def reset(self):
        self.speech_count = 0
        self.silence_count = 0
        self.in_speech = False


class AdaptiveEnergyVAD:
    """
    Robust Adaptive Energy Voice Activity Detector with Hysteresis (Schmitt Trigger).
    
    Features:
      - Separate speech_threshold (onset) and silence_threshold (offset) to prevent jitter.
      - Integrated circular pre-roll buffer to prevent clipping initial syllables/consonants.
      - Rejection of short impulse noises (clicks, keyboard taps, breaths).
    """

    def __init__(
        self,
        speech_threshold: float = 0.060,
        silence_threshold: float = 0.035,
        min_speech_chunks: int = 2,
        min_silence_chunks: int = 7,
        pre_roll_chunks: int = 4,
    ):
        self.speech_threshold = speech_threshold
        self.silence_threshold = silence_threshold
        self.min_speech_chunks = min_speech_chunks
        self.min_silence_chunks = min_silence_chunks

        self.speech_count = 0
        self.silence_count = 0
        self.in_speech = False

        self.pre_roll = collections.deque(maxlen=pre_roll_chunks)

    def process(self, audio_chunk: np.ndarray) -> tuple[str | None, list[np.ndarray]]:
        """
        Processes an audio chunk.
        
        Returns:
            (event, initial_audio_chunks)
            event: "start", "end", or None
            initial_audio_chunks: On "start", returns pre-roll buffer + triggering chunks.
                                  Otherwise returns empty list.
        """
        rms = float(np.sqrt(np.mean(audio_chunk ** 2))) if len(audio_chunk) > 0 else 0.0

        if not self.in_speech:
            self.pre_roll.append(audio_chunk)
            if rms >= self.speech_threshold:
                self.speech_count += 1
            else:
                self.speech_count = 0

            if self.speech_count >= self.min_speech_chunks:
                self.in_speech = True
                self.speech_count = 0
                self.silence_count = 0
                # Retrieve pre-roll history to prevent clipping opening phonemes
                pre_audio = list(self.pre_roll)
                return "start", pre_audio

            return None, []

        else:
            # Inside speech state: use lower silence_threshold for hysteresis
            if rms < self.silence_threshold:
                self.silence_count += 1
            else:
                self.silence_count = 0

            if self.silence_count >= self.min_silence_chunks:
                self.in_speech = False
                self.speech_count = 0
                self.silence_count = 0
                return "end", []

            return None, []

    def process_chunk(self, audio_chunk: np.ndarray) -> tuple[bool, float]:
        """
        Evaluate if current chunk exceeds speech threshold, applying hysteresis.
        Returns:
            (is_speech: bool, rms: float)
        """
        rms = float(np.sqrt(np.mean(audio_chunk ** 2))) if len(audio_chunk) > 0 else 0.0
        thold = self.silence_threshold if self.in_speech else self.speech_threshold
        if rms >= thold:
            self.in_speech = True
            return True, rms
        else:
            self.in_speech = False
            return False, rms

    def reset(self):
        self.speech_count = 0
        self.silence_count = 0
        self.in_speech = False
        self.pre_roll.clear()