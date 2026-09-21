"""clocksync: test candidate clock offsets between a probe recording and a trusted reference.

Example — is the camera clock 306.1 s or 3608 s ahead of the recorder?

    clocksync camera.MP4 2000-01-01T13:00:00Z recorder.WAV 2000-01-01T12:00:00Z \
        --candidates 306.1 3608 --probe-pos 300 --duration 120
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

from .audio import SAMPLE_RATE, extract_audio, wav_duration_s, wav_segment_reader
from .core import rank_hypotheses


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clocksync", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe", help="media file from the untrusted-clock device, e.g. a camera MP4 (anything ffmpeg reads)")
    ap.add_argument("probe_start", type=_dt, help="what the probe device claims its recording started, ISO-8601")
    ap.add_argument("reference", help="16-bit WAV from the trusted-clock device")
    ap.add_argument("reference_start", type=_dt, help="true start time of the reference WAV, ISO-8601")
    ap.add_argument("--candidates", type=float, nargs="+", required=True, help="candidate offsets in seconds (probe clock minus true time)")
    ap.add_argument("--probe-pos", type=float, default=60.0, help="seconds into the probe to sample from")
    ap.add_argument("--duration", type=float, default=120.0, help="seconds of probe audio to correlate")
    ap.add_argument("--search-window", type=float, default=15.0, help="seconds of slack around each candidate")
    a = ap.parse_args(argv)

    probe = extract_audio(a.probe, a.probe_pos, a.duration)
    probe_clock_start = a.probe_start + timedelta(seconds=a.probe_pos)
    ranked = rank_hypotheses(
        probe, probe_clock_start, wav_segment_reader(a.reference), a.reference_start,
        wav_duration_s(a.reference), a.candidates, SAMPLE_RATE, a.search_window,
    )
    for h in ranked:
        flag = "***" if h.match.confident else ("ok " if h.match.snr > 5 else "-- ")
        refined = f"{h.refined_offset_ms} ms" if h.refined_offset_ms is not None else "no overlap"
        print(f"{flag} candidate {h.candidate_offset_s:8.1f} s  snr {h.match.snr:6.1f}  refined {refined}")
    return 0 if ranked and ranked[0].match.confident else 1


if __name__ == "__main__":
    raise SystemExit(main())
