import torch
from silero_vad import load_silero_vad, VADIterator


class SileroVAD:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_silero_vad()
        self.model = self.model.to(self.device)

        self.vad = VADIterator(
            self.model,
            sampling_rate=16000,
            threshold=0.5,
            min_silence_duration_ms=500,
            speech_pad_ms=100,
        )

        self.buffer = torch.empty(0, device=self.device)

    def process(self, audio):
        audio_tensor = torch.from_numpy(audio).to(self.device)
        self.buffer = torch.cat((self.buffer, audio_tensor))

        results = []

        while self.buffer.numel() >= 512:
            chunk = self.buffer[:512]
            self.buffer = self.buffer[512:]

            result = self.vad(chunk)

            if result is not None:
                results.append(result)

        return results

    def reset(self):
        self.vad.reset_states()
        self.buffer = torch.empty(0, device=self.device)
