#!/usr/bin/env python3
"""
Edge TTS Engine for NVIDIA Jetson Orin Nano.

Provides 100% offline neural text-to-speech feedback and zero-latency
playback for cached voice prompts (countdown, recording status, export notices).
Uses Piper TTS with ONNX Runtime, outputting to PulseAudio / ALSA.
"""

import os
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Optional

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "voice_prompts"
MODELS_DIR = Path("/mnt/thesis/models/piper")
PIPER_MODEL = MODELS_DIR / "en_US-lessac-medium.onnx"
PIPER_BIN = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "piper"
PAPLAY_BIN = shutil.which("paplay")
APLAY_BIN = shutil.which("aplay")


def _get_player_cmd(wav_path: str) -> list:
    """Determine best available audio playback command."""
    if PAPLAY_BIN:
        return [PAPLAY_BIN, str(wav_path)]
    if APLAY_BIN:
        return [APLAY_BIN, "-D", "default", str(wav_path)]
    return []


def play_wav(wav_path: str, non_blocking: bool = False) -> bool:
    """Play a WAV audio file through system speakers/earphones."""
    cmd = _get_player_cmd(wav_path)
    if not cmd:
        return False

    def _run():
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
        except Exception:
            pass

    if non_blocking:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return True
    else:
        _run()
        return True


def play_prompt(prompt_name: str, non_blocking: bool = False) -> bool:
    """
    Play a pre-rendered system voice prompt.
    Prompt names: 'system_ready', 'countdown_start', 'recording_stopped',
                  'transcription_saved', 'usb_warning'.
    """
    if not prompt_name.endswith(".wav"):
        prompt_name += ".wav"
    wav_path = ASSETS_DIR / prompt_name
    if wav_path.is_file():
        return play_wav(str(wav_path), non_blocking=non_blocking)
    return False


def speak(text: str, non_blocking: bool = False, cache_dir: Optional[str] = None) -> bool:
    """
    Synthesize and speak arbitrary text using offline Piper TTS.
    """
    if not PIPER_BIN.is_file() or not PIPER_MODEL.is_file():
        # Fallback to speech-dispatcher if piper unavailable
        spd = shutil.which("spd-say")
        if spd:
            subprocess.run([spd, text], check=False)
            return True
        return False

    target_dir = Path(cache_dir) if cache_dir else Path("/tmp")
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_wav = target_dir / f"piper_speech_{os.getpid()}_{hash(text) & 0xffffffff}.wav"

    def _synthesize_and_play():
        try:
            cmd = [
                str(PIPER_BIN),
                "-m", str(PIPER_MODEL),
                "-f", str(temp_wav),
            ]
            p = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            p.communicate(input=text.encode("utf-8"), timeout=15)
            if temp_wav.is_file():
                play_wav(str(temp_wav), non_blocking=False)
                temp_wav.unlink(missing_ok=True)
        except Exception:
            pass

    if non_blocking:
        t = threading.Thread(target=_synthesize_and_play, daemon=True)
        t.start()
        return True
    else:
        _synthesize_and_play()
        return True


if __name__ == "__main__":
    import sys
    test_text = sys.argv[1] if len(sys.argv) > 1 else "Testing Jetson offline speech synthesis."
    print(f"Speaking: '{test_text}'")
    speak(test_text)
