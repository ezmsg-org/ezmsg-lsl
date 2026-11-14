"""Finding breaks in a stream's timestamps and removing the jitter between them."""

import numpy as np
import pytest

from ezmsg.lsl.timing import StreamTiming

SRATE = 1000.0
JITTER = 0.002  # s, per-sample timestamp noise
CHUNK = 20


def _stream(n: int, pauses: dict[int, float] | None = None, seed: int = 0):
    """``(truth, raw)`` timestamps: ``pauses`` maps a sample index to seconds lost just before it."""
    truth = 5000.0 + np.arange(n) / SRATE
    for idx, dur in (pauses or {}).items():
        truth[idx:] += dur
    return truth, truth + np.random.default_rng(seed).normal(scale=JITTER, size=n)


def _feed(timing: StreamTiming, raw: np.ndarray, chunk: int = CHUNK):
    """Run ``raw`` through in chunks; return ``(global start, time, is_break)`` for every run."""
    out = []
    for first in range(0, raw.size, chunk):
        for start, stop, t, is_break in timing.split(raw[first : first + chunk]):
            out.append((first + start, t, is_break))
    return out


def test_a_continuous_stream_is_one_run_per_chunk_and_loses_its_jitter():
    truth, raw = _stream(20_000)
    runs = _feed(StreamTiming(SRATE), raw)
    assert len(runs) == 20_000 // CHUNK
    assert not any(is_break for _, _, is_break in runs)
    # Once it has settled, far closer to the truth than any one raw timestamp is.
    err = np.array([t - truth[i] for i, t, _ in runs[200:]])
    assert np.abs(err).max() < JITTER / 10


def test_a_pause_cuts_the_chunk_and_restarts_the_fit():
    at, pause = 10_007, 2.0  # mid-chunk
    truth, raw = _stream(20_000, {at: pause})
    runs = _feed(StreamTiming(SRATE), raw)

    breaks = [(i, t) for i, t, is_break in runs if is_break]
    assert [i for i, _ in breaks] == [at]
    # The run after the pause is timed from its own samples, not from where the old fit pointed.
    assert breaks[0][1] == pytest.approx(truth[at], abs=5 * JITTER)
    # And a second later the fit is good again. (liblsl's own smoothing is still ~0.5 s out here.)
    later = [t - truth[i] for i, t, _ in runs if at + 1000 <= i < at + 2000]
    assert np.abs(later).max() < JITTER


def test_a_break_must_exceed_both_thresholds():
    truth, raw = _stream(4000, {2000: 2.0})
    # 2 s is over 1 s but under 5000 samples (5 s at 1 kHz): not a break.
    assert not any(b for _, _, b in _feed(StreamTiming(SRATE, break_seconds=1.0, break_samples=5000), raw))
    assert any(b for _, _, b in _feed(StreamTiming(SRATE, break_seconds=1.0, break_samples=500), raw))


def test_a_break_between_chunks_is_found():
    truth, raw = _stream(4000, {2000: 2.0})  # 2000 is a chunk boundary
    runs = _feed(StreamTiming(SRATE), raw)
    assert [i for i, _, is_break in runs if is_break] == [2000]
    assert len(runs) == 4000 // CHUNK  # no extra cut


def test_a_step_backward_is_a_break():
    truth, raw = _stream(4000, {2000: -3.0})
    assert [i for i, _, b in _feed(StreamTiming(SRATE), raw) if b] == [2000]


def test_zero_thresholds_never_break():
    truth, raw = _stream(4000, {2000: 30.0})
    assert not any(b for _, _, b in _feed(StreamTiming(SRATE, break_seconds=0, break_samples=0), raw))


def test_without_dejitter_each_run_keeps_its_first_raw_timestamp():
    truth, raw = _stream(400, {200: 2.0})
    runs = _feed(StreamTiming(SRATE, dejitter=False), raw)
    assert all(t == raw[i] for i, t, _ in runs)
    assert [i for i, _, b in runs if b] == [200]


def test_the_fit_follows_a_rate_that_is_not_the_nominal_one():
    """A device running 0.1% fast: the fitted line must take the real slope, not the advertised one."""
    n = 60_000
    truth = 5000.0 + np.arange(n) / (SRATE * 1.001)
    raw = truth + np.random.default_rng(1).normal(scale=JITTER, size=n)
    runs = _feed(StreamTiming(SRATE), raw)
    err = np.array([t - truth[i] for i, t, _ in runs[1000:]])
    assert np.abs(err).max() < JITTER / 5
