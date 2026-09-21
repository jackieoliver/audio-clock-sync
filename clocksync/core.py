"""Core algorithm: normalized FFT cross-correlation with an SNR confidence metric.

Terminology
-----------
probe      audio from the device whose clock we don't trust (e.g. a GoPro)
reference  audio from the device whose clock we do trust (e.g. a dedicated recorder)
lag        where the probe starts inside the reference, in seconds of true time
offset     probe_clock_time - true_time, i.e. how far ahead the untrusted clock runs
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Sequence

import numpy as np
from scipy.signal import fftconvolve

# Below this standard deviation (int16 scale) a signal is treated as silence:
# there is no structure to correlate and any "peak" would be noise.
SILENCE_STD = 1.0

# Correlation values within this many seconds of the peak are excluded when
# estimating the noise floor, so the peak's own skirt doesn't inflate it.
PEAK_EXCLUSION_S = 2.0

# Empirical threshold from real GoPro/recorder pairs: true matches sat well
# above 10, unrelated windows below ~6.
CONFIDENT_SNR = 10.0


@dataclass(frozen=True)
class Match:
    lag_s: float
    peak: float
    snr: float

    @property
    def confident(self) -> bool:
        return self.snr >= CONFIDENT_SNR


@dataclass(frozen=True)
class Offset:
    offset_ms: int
    match: Match


@dataclass(frozen=True)
class Hypothesis:
    candidate_offset_s: float
    refined_offset_ms: int | None
    match: Match


def _normalize(x: np.ndarray) -> np.ndarray | None:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    std = x.std()
    if std < SILENCE_STD:
        return None
    return x / std


def cross_correlate(probe: np.ndarray, reference: np.ndarray, sample_rate: int) -> Match:
    """Locate `probe` inside `reference`.

    Both signals are demeaned and unit-scaled so loudness differences between
    devices don't matter. Correlation is computed as a convolution with the
    time-reversed probe. The confidence metric is the peak height divided by
    the standard deviation of the correlation *away* from the peak: a real
    match is a spike over a flat floor, a spurious one is barely above it.
    """
    p = _normalize(probe)
    r = _normalize(reference)
    if p is None or r is None:
        return Match(lag_s=0.0, peak=0.0, snr=0.0)

    corr = fftconvolve(r, p[::-1], mode="full")
    i = int(np.argmax(corr))
    lag_s = (i - len(p) + 1) / sample_rate

    w = int(sample_rate * PEAK_EXCLUSION_S)
    mask = np.ones(len(corr), dtype=bool)
    mask[max(0, i - w) : i + w] = False
    noise = float(corr[mask].std()) if mask.sum() > 100 else 1.0
    peak = float(corr[i])
    return Match(lag_s=lag_s, peak=peak, snr=peak / noise if noise > 0 else 0.0)


def measure_offset(
    probe: np.ndarray,
    probe_clock_start: datetime,
    reference: np.ndarray,
    reference_start: datetime,
    sample_rate: int,
) -> Offset:
    """How far ahead is the probe device's clock?

    `probe_clock_start` is when the probe device *claims* its audio began.
    `reference_start` is when the reference audio truly began.
    """
    m = cross_correlate(probe, reference, sample_rate)
    true_start = reference_start + timedelta(seconds=m.lag_s)
    offset = (probe_clock_start - true_start).total_seconds()
    return Offset(offset_ms=round(offset * 1000), match=m)


SegmentReader = Callable[[float, float], np.ndarray]
"""(start_s, duration_s) -> samples. Lets long reference recordings be read lazily."""


def rank_hypotheses(
    probe: np.ndarray,
    probe_clock_start: datetime,
    read_reference: SegmentReader,
    reference_start: datetime,
    reference_duration_s: float,
    candidate_offsets_s: Sequence[float],
    sample_rate: int,
    search_window_s: float = 15.0,
) -> list[Hypothesis]:
    """Test several candidate offsets and rank them by confidence.

    For each candidate, cut the slice of the reference where the probe *would*
    be if that candidate were right (plus or minus `search_window_s`), then
    correlate only inside that slice. Narrow windows are what make this
    precise: a wide search over an hour of ambient audio finds spurious peaks
    from repeating background noise, a 15 s window around a good guess does not.

    This is how the two-regime clock jump was confirmed: every clip was tested
    against both candidate offsets, and exactly one produced a confident match.
    """
    probe_dur_s = len(probe) / sample_rate
    results: list[Hypothesis] = []
    for cand in candidate_offsets_s:
        true_start = probe_clock_start - timedelta(seconds=cand)
        expected_pos = (true_start - reference_start).total_seconds()
        seg_start = max(0.0, expected_pos - search_window_s)
        seg_dur = min(probe_dur_s + 2 * search_window_s, reference_duration_s - seg_start)
        if seg_dur <= probe_dur_s:
            results.append(Hypothesis(cand, None, Match(0.0, 0.0, 0.0)))
            continue
        segment = read_reference(seg_start, seg_dur)
        m = cross_correlate(probe, segment, sample_rate)
        matched_start = reference_start + timedelta(seconds=seg_start + m.lag_s)
        refined = (probe_clock_start - matched_start).total_seconds()
        results.append(Hypothesis(cand, round(refined * 1000), m))
    results.sort(key=lambda h: h.match.snr, reverse=True)
    return results
