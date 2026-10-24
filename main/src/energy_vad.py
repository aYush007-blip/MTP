import numpy as np


class EnergyVAD:
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

    def reset(self):
        self.speech_count = 0
        self.silence_count = 0
        self.in_speech = False