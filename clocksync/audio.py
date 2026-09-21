"""Audio I/O helpers. Everything is resampled to 16 kHz mono float64."""
from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from .core import SegmentReader

SAMPLE_RATE = 16000


def read_wav_segment(path: str | os.PathLike, start_s: float = 0.0, duration_s: float | None = None) -> np.ndarray:
    """Read a slice of a 16-bit PCM WAV without loading the whole file."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        total = wf.getnframes()
        start = max(0, int(start_s * rate))
        n = total - start if duration_s is None else min(int(duration_s * rate), total - start)
        wf.setpos(start)
        raw = wf.readframes(n)
        channels = wf.getnchannels()
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


def wav_duration_s(path: str | os.PathLike) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def wav_segment_reader(path: str | os.PathLike) -> SegmentReader:
    return lambda start_s, duration_s: read_wav_segment(path, start_s, duration_s)


def extract_audio(media_path: str | os.PathLike, start_s: float, duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Pull a mono 16 kHz slice out of any container ffmpeg can read (e.g. a GoPro MP4)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "slice.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "quiet", "-ss", str(start_s), "-i", str(media_path),
             "-t", str(duration_s), "-ac", "1", "-ar", str(sample_rate), "-acodec", "pcm_s16le", "-f", "wav", out],
            check=True, capture_output=True,
        )
        return read_wav_segment(out)
