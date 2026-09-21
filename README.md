# audio-clock-sync

Recover the clock offset between two recording devices from the ambient audio they both heard.

Consumer capture devices lie about time. In a multi-device recording setup — a camera, a dedicated audio recorder, a phone — every file carries a timestamp, and none of them agree. If you want to line up what one device saw with what another heard, you need the true offset between their clocks, and the devices won't tell you.

But if two devices were in the same room, they recorded the same sound. Cross-correlating that ambient audio finds exactly where one recording sits inside the other, which gives you the offset directly — to within a sample.

## What it does

```python
from clocksync import measure_offset

off = measure_offset(probe_audio, probe_clock_start, reference_audio, reference_start, 16000)
off.offset_ms       # how far ahead the probe device's clock runs
off.match.snr       # confidence: peak height over the correlation noise floor
off.match.confident # snr >= 10, an empirical threshold from real device pairs
```

- Both signals are demeaned and unit-scaled, so loudness differences between devices don't matter.
- Correlation via `scipy.signal.fftconvolve` against the time-reversed probe.
- **Confidence is the peak divided by the standard deviation of the correlation away from the peak.** A real match is a spike over a flat floor; a spurious one is barely above it. Silence returns zero confidence rather than a made-up lag.
- `rank_hypotheses()` tests several candidate offsets against a long reference by correlating only in a narrow window where each candidate predicts the probe should be. This is what makes the method precise — see below.

## The finding it was built for

Over roughly three weeks of multi-device field recordings, a consumer camera's clock was compared against a dedicated audio recorder. The result was not a single drift value:

| Period | Camera clock ahead by | Cause |
|---|---|---|
| First regime | **306.1 s** | Clock set slightly wrong, plus it stayed on standard time after a daylight-saving change |
| Second regime | **3608 s** | Camera re-synced to a phone during a 13-hour recording gap — and the phone was an hour off |

Two distinct regimes with a sudden jump, one of them a missed DST transition. Nothing in the file metadata hinted at either. Cross-correlation measurements in the second regime clustered at 3608–3613 s (median 3609 s, σ ≈ 8.7 s across seven windows); the full numbers are in [`examples/two-regime-findings.json`](examples/two-regime-findings.json).

The regime boundary was confirmed by testing every clip around the gap against *both* candidate offsets. Exactly one produced a confident match per clip. That is `rank_hypotheses()`.

Downstream, the pipeline stored `clock_offset_ms` and `clock_confidence` on every artifact rather than correcting timestamps in place — so nothing consuming the data ever mistook a device timestamp for the truth.

## Precision, honestly

- A wide search over an hour of ambient audio finds **spurious peaks** from repeating background noise. Precision in that mode was only ~10–15 s.
- A **narrow window (3–15 s) around an approximate offset** resolves to the sample. Get a rough offset first (file metadata, a known event), then refine.
- It needs the devices to have actually overlapped in the same acoustic space. Two rooms, no result.
- Regime changes are found by hypothesis testing, not detected automatically.

## Usage

```bash
pip install -e ".[test]"
pytest                    # synthetic room-tone tests, ~2 s

# Which is right — is the camera 306.1 s or 3608 s ahead?
clocksync camera.MP4 2000-01-01T13:00:00Z recorder.WAV 2000-01-01T12:00:00Z \
    --candidates 306.1 3608 --probe-pos 300 --duration 120
```

`ffmpeg` is required for reading anything other than 16-bit WAV.

## Origin

Extracted from a multi-device field-recording pipeline, where four devices with four disagreeing clocks had to be fused into one timeline.

MIT.
