"""When the inlet starts a new ``AxisArray.segment``.

On a break in the stream's timestamps, and on a reconnect that lands on a different stream. A
reconnect to the same stream with nothing missed continues the segment.
"""

import time

import numpy as np
import pylsl
import pytest
from test_inlet import _FakeInlet, _FakeStreamInfo

from ezmsg.lsl.inlet import LSLInfo, LSLInletProducer, LSLInletSettings

SRATE = 1000.0


class _PullableInlet(_FakeInlet):
    """A fake connected inlet that hands out the timestamps it is given."""

    def __init__(self, info):
        super().__init__(info)
        self.next_stamps: np.ndarray = np.array([])

    def pull_chunk(self, dest_obj=None, **kwargs):
        n = len(self.next_stamps)
        dest_obj[:n] = np.arange(n, dtype=np.float32)[:, None]
        return dest_obj[:n], self.next_stamps


class _NoClockSync:
    """The LSL and system clocks agree."""

    async def arun_once(self):
        pass

    def lsl2system(self, t):
        return t

    def system2lsl(self, t):
        return t


def _producer(info=None, **settings):
    producer = LSLInletProducer(settings=LSLInletSettings(use_lsl_clock=True, **settings))
    producer._state.clock_sync = _NoClockSync()
    _reconnect(producer, info)
    return producer


def _reconnect(producer, info=None):
    info = info if info is not None else _FakeStreamInfo(srate=SRATE)
    producer._state.inlet = _PullableInlet(info)
    assert producer._setup_after_open() is True


def _pull(producer, stamps):
    """One pull returning ``stamps``; every message it produced, in order."""
    producer._state.inlet.next_stamps = np.asarray(stamps, dtype=float)
    first = producer._pull(producer._snapshot_pull_state())
    rest = list(producer._pending)
    producer._pending.clear()
    return [first, *rest]


def _stamps(start: float, n: int) -> np.ndarray:
    return start + np.arange(n) / SRATE


def test_a_break_inside_a_pull_cuts_it_and_starts_a_segment():
    producer = _producer()
    (before,) = _pull(producer, _stamps(100.0, 50))
    stamps = np.concatenate([_stamps(100.05, 30), _stamps(103.0, 20)])  # ~3 s lost after 30 samples
    first, second = _pull(producer, stamps)

    assert [m.data.shape[0] for m in (first, second)] == [30, 20]
    assert [before.segment, first.segment, second.segment] == [0, 0, 1]
    assert second.axes["time"].offset == pytest.approx(103.0, abs=1e-6)
    np.testing.assert_array_equal(second.data[:, 0], np.arange(30, 50))


def test_the_rest_of_a_cut_pull_is_produced_next():
    import asyncio

    producer = _producer()
    producer._warmed_up = True
    producer._state.inlet.next_stamps = np.concatenate([_stamps(100.0, 30), _stamps(103.0, 20)])

    async def two():
        return await producer._produce(), await producer._produce()

    first, second = asyncio.run(two())
    assert (first.segment, second.segment) == (0, 1)
    assert second.data.shape[0] == 20


def test_reconnecting_to_the_same_stream_with_nothing_missed_continues_the_segment():
    producer = _producer()
    _pull(producer, _stamps(100.0, 50))
    _reconnect(producer, _FakeStreamInfo(srate=SRATE, uid="uid-2"))  # a restarted outlet, same source_id
    (msg,) = _pull(producer, _stamps(100.05, 50))
    assert msg.segment == 0


def test_reconnecting_to_the_same_stream_after_a_break_starts_a_segment():
    producer = _producer()
    _pull(producer, _stamps(100.0, 50))
    _reconnect(producer)
    (msg,) = _pull(producer, _stamps(130.0, 50))
    assert msg.segment == 1


@pytest.mark.parametrize(
    "changed",
    [
        _FakeStreamInfo(srate=SRATE, source_id="src-2"),
        _FakeStreamInfo(srate=SRATE, n_ch=3),
        _FakeStreamInfo(srate=SRATE / 2),
    ],
    ids=["source_id", "channel_count", "nominal_srate"],
)
def test_reconnecting_to_a_different_stream_starts_a_segment(changed):
    producer = _producer()
    _pull(producer, _stamps(100.0, 50))
    _reconnect(producer, changed)
    (msg,) = _pull(producer, _stamps(100.05, 50))  # the timestamps alone would say it continues
    assert msg.segment == 1


def test_a_stream_without_a_source_id_is_never_assumed_to_be_the_same_one():
    anonymous = _FakeStreamInfo(srate=SRATE, source_id="")
    producer = _producer(anonymous)
    _pull(producer, _stamps(100.0, 50))
    _reconnect(producer, anonymous)
    (msg,) = _pull(producer, _stamps(100.05, 50))
    assert msg.segment == 1


def test_an_irregular_stream_keeps_its_timestamps_and_only_breaks_on_a_new_stream():
    irregular = _FakeStreamInfo(srate=0.0)
    producer = _producer(irregular)
    stamps = np.array([100.0, 100.4, 160.0])
    (msg,) = _pull(producer, stamps)
    np.testing.assert_array_equal(msg.axes["time"].data, stamps)
    assert msg.segment == 0
    _reconnect(producer, _FakeStreamInfo(srate=0.0, source_id="src-2"))
    assert _pull(producer, stamps + 1)[0].segment == 1


def test_liblsl_is_not_asked_to_smooth_a_regular_stream():
    producer = LSLInletProducer(settings=LSLInletSettings())
    regular = producer._inlet_flags(SRATE)
    assert regular & pylsl.proc_clocksync
    assert not regular & (pylsl.proc_dejitter | pylsl.proc_monotonize)
    assert producer._inlet_flags(0.0) == pylsl.proc_ALL


def test_without_proc_dejitter_runs_keep_their_raw_first_timestamp():
    producer = _producer(processing_flags=pylsl.proc_clocksync)
    rng = np.random.default_rng(0)
    stamps = _stamps(100.0, 50) + rng.normal(scale=0.002, size=50)
    (msg,) = _pull(producer, stamps)
    assert msg.axes["time"].offset == stamps[0]


def test_arrival_time_stamps_the_newest_sample_now():
    producer = _producer(use_arrival_time=True)
    before = time.monotonic()
    first, second = _pull(producer, np.concatenate([_stamps(100.0, 30), _stamps(103.0, 20)]))
    after = time.monotonic()
    newest = second.axes["time"].offset + 19 / SRATE
    assert before <= newest <= after
    # The break keeps its length.
    assert second.axes["time"].offset - first.axes["time"].offset == pytest.approx(3.0, abs=1e-6)


def test_distinct_key_per_connection_is_deprecated():
    with pytest.warns(DeprecationWarning, match="segment"):
        LSLInletSettings(distinct_key_per_connection=True)


def test_a_paused_outlet_starts_a_segment():
    """End to end, through liblsl: an outlet that stops for longer than the thresholds."""
    name = f"pause-{time.monotonic_ns()}"
    outlet = pylsl.StreamOutlet(pylsl.StreamInfo(name, "test", 2, SRATE, pylsl.cf_float32, name))
    producer = LSLInletProducer(info=LSLInfo(name=name, type="test"), pull_timeout=0.01, use_lsl_clock=True)

    def push(seconds: float, collected: list):
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            outlet.push_chunk(np.zeros((10, 2), dtype=np.float32))
            time.sleep(0.01)
            msg = producer()
            if msg is not None and msg.data.shape[0]:
                collected.append(msg)

    msgs: list = []
    deadline = time.monotonic() + 15.0  # resolving the stream takes a moment
    while not msgs and time.monotonic() < deadline:
        push(0.2, msgs)
    assert msgs, "inlet never connected"
    push(1.0, msgs)
    time.sleep(1.5)  # over both thresholds at 1 kHz: 1 s and 500 samples
    push(1.0, msgs)
    producer.shutdown()

    segments = [m.segment for m in msgs]
    assert segments == sorted(segments) and set(segments) == {0, 1}
    first_after = next(m for m in msgs if m.segment == 1)
    last_before = [m for m in msgs if m.segment == 0][-1]
    gap = first_after.axes["time"].offset - (last_before.axes["time"].offset + last_before.data.shape[0] / SRATE)
    assert gap == pytest.approx(1.5, abs=0.2)
