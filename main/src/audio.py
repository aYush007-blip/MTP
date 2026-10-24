import numpy as np


def stereo_48k_to_mono_16k(audio_bytes: bytes) -> np.ndarray:
    """
    Convert 48 kHz, stereo, signed 16-bit PCM audio to
    16 kHz, mono, float32 audio in the range [-1, 1].
    Uses vectorized 3-tap averaging and 3:1 decimation (pure NumPy).
    """
    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    if len(audio) == 0:
        return np.empty(0, dtype=np.float32)

    # Stereo: L R L R ...
    audio = audio.reshape(-1, 2)

    # Stereo -> mono
    mono = audio.mean(axis=1)

    # 48 kHz -> 16 kHz (exact 3:1 integer decimation)
    n = (len(mono) // 3) * 3
    if n > 0:
        resampled = mono[:n].reshape(-1, 3).mean(axis=1)
    else:
        resampled = mono

    # Convert to float32 [-1, 1]
    return np.clip(resampled / 32768.0, -1.0, 1.0).astype(np.float32)
