from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from clocksync import cross_correlate, measure_offset, rank_hypotheses

SR = 16000
UTC = timezone.utc


def ambient(seconds: float, seed: int) -> np.ndarray:
    """Room-tone stand-in: low-passed noise, int16-scaled like real capture."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * SR))
    sos = butter(4, 3000, btype="low", fs=SR, output="sos")
    return sosfilt(sos, x) * 2000


def test_recovers_known_lag_to_one_sample():
    ref = ambient(60, seed=1)
    lag_s = 17.3
    start = int(lag_s * SR)
    probe = ref[start : start + 20 * SR] + ambient(20, seed=2) * 0.3  # a second mic hears the same room, plus its own noise
    m = cross_correlate(probe, ref, SR)
    assert abs(m.lag_s - lag_s) <= 1 / SR
    assert m.confident


def test_offset_recovered_when_probe_clock_runs_ahead():
    ref = ambient(60, seed=3)
    ref_start = datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)
    true_probe_start = ref_start + timedelta(seconds=12.0)
    device_clock_ahead_s = 306.1
    probe = ref[12 * SR : 32 * SR]
    claimed_start = true_probe_start + timedelta(seconds=device_clock_ahead_s)
    off = measure_offset(probe, claimed_start, ref, ref_start, SR)
    assert abs(off.offset_ms - 306100) <= 1
    assert off.match.confident


def test_silence_yields_zero_confidence():
    m = cross_correlate(np.zeros(SR * 5), np.zeros(SR * 30), SR)
    assert m.snr == 0.0 and not m.confident


def test_unrelated_audio_is_not_confident():
    m = cross_correlate(ambient(20, seed=10), ambient(60, seed=11), SR)
    assert m.snr < 8.0
    assert not m.confident


def test_rank_hypotheses_picks_the_true_offset_from_a_long_reference():
    # 1 h of reference audio; the probe device's clock is 3608 s ahead, not 306.1 s.
    ref = ambient(3600, seed=20)
    ref_start = datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)
    true_pos_s = 1500.0
    probe = ref[int(true_pos_s * SR) : int((true_pos_s + 120) * SR)]
    claimed_start = ref_start + timedelta(seconds=true_pos_s + 3608.0)

    ranked = rank_hypotheses(
        probe, claimed_start, lambda s, d: ref[int(s * SR) : int((s + d) * SR)],
        ref_start, 3600.0, candidate_offsets_s=[306.1, 3608.0, 50.0], sample_rate=SR, search_window_s=15.0,
    )
    best = ranked[0]
    assert best.candidate_offset_s == 3608.0
    assert best.match.confident
    assert abs(best.refined_offset_ms - 3608000) <= 1
    assert all(not h.match.confident for h in ranked[1:])


def test_candidate_outside_reference_reports_no_overlap():
    ref = ambient(120, seed=30)
    ref_start = datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)
    probe = ref[10 * SR : 30 * SR]
    claimed = ref_start + timedelta(seconds=10 + 99999)
    ranked = rank_hypotheses(probe, claimed, lambda s, d: ref[int(s * SR) : int((s + d) * SR)], ref_start, 120.0, [1.0], SR)
    assert ranked[0].refined_offset_ms is None
