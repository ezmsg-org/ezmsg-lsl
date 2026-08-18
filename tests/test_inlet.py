"""
These unit tests aren't really testable in a runner without a complicated setup with inlets and outlets.
This code exists mostly to use during development and debugging.
"""

import asyncio
import tempfile
import threading
import typing
from pathlib import Path

import ezmsg.core as ez
import numpy as np
import pylsl
import pytest
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTotalSettings

from ezmsg.lsl.inlet import LSLInfo, LSLInletProducer, LSLInletSettings, LSLInletUnit


def test_inlet_init_defaults():
    settings = LSLInletSettings(info=LSLInfo(name="", type=""))
    _ = LSLInletUnit(settings)
    assert True


@pytest.mark.parametrize(
    "buffer_size,cap,expected_max_samples",
    [
        (8, None, 8),
        (8, 3, 3),
        (None, 3, 3),
        (None, None, None),
    ],
)
def test_inlet_pull_waits_for_one_then_drains(
    buffer_size: typing.Optional[int],
    cap: typing.Optional[int],
    expected_max_samples: typing.Optional[int],
):
    class RecordingInlet:
        def __init__(self):
            self.calls: list[dict] = []

        def pull_chunk(self, **kwargs):
            self.calls.append(kwargs)
            return (None if "dest_obj" in kwargs else []), []

    producer = LSLInletProducer(settings=LSLInletSettings(max_pull_samples=cap))
    inlet = RecordingInlet()
    producer._state.inlet = inlet
    producer._state.clock_sync = object()
    producer._state.msg_template = object()
    if buffer_size is not None:
        producer._state.fetch_buffer = np.zeros((buffer_size, 2), dtype=np.float32)

    assert asyncio.run(producer._apull(timeout=0.25)) is None

    assert len(inlet.calls) == 1
    call = inlet.calls[0]
    assert call["timeout"] == 0.25
    assert call["min_samples"] == 1
    if expected_max_samples is None:
        assert "max_samples" not in call
    else:
        assert call["max_samples"] == expected_max_samples
    if buffer_size is None:
        assert "dest_obj" not in call
    else:
        assert call["dest_obj"] is producer._state.fetch_buffer


def test_inlet_pull_snapshot_survives_shutdown_and_discards_stale_result(monkeypatch):
    producer = LSLInletProducer(settings=LSLInletSettings())
    inlet = object()
    fetch_buffer = np.zeros((8, 2), dtype=np.float32)
    msg_template = object()
    clock_sync = object()
    producer._state.inlet = inlet
    producer._state.fetch_buffer = fetch_buffer
    producer._state.msg_template = msg_template
    producer._state.clock_sync = clock_sync

    pull_started = threading.Event()
    release_pull = threading.Event()
    stale_result = object()

    def blocking_pull(snapshot, timeout):
        assert snapshot.inlet is inlet
        assert snapshot.fetch_buffer is fetch_buffer
        assert snapshot.msg_template is msg_template
        assert snapshot.clock_sync is clock_sync
        pull_started.set()
        assert release_pull.wait(timeout=1.0)
        return stale_result

    monkeypatch.setattr(producer, "_pull", blocking_pull)

    async def run_test():
        pull_task = asyncio.create_task(producer._apull(timeout=0.1))
        assert await asyncio.to_thread(pull_started.wait, 1.0)
        producer.shutdown()
        release_pull.set()
        assert await pull_task is None

    asyncio.run(run_test())


def test_inlet_producer():
    """
    Test the inlet producer object without invoking ezmsg.
    """
    rate = 32.0
    nch = 8
    dummy_out_info = pylsl.StreamInfo(
        name="dummy",
        type="dummy",
        channel_count=nch,
        nominal_srate=rate,
        channel_format=pylsl.cf_float32,
    )
    outlet = pylsl.StreamOutlet(dummy_out_info)
    state = {"pushed": 0}

    def step_outlet(n_interval: int = 10):
        dummy_data = np.arange(state["pushed"], state["pushed"] + n_interval)[:, None] / rate + np.zeros((1, nch))
        outlet.push_chunk(dummy_data.astype(np.float32))
        state["pushed"] += n_interval

    producer = LSLInletProducer(info=LSLInfo(name="dummy", type="dummy"))
    counter = 0
    for msg in producer:
        step_outlet()
        if msg is None or np.prod(msg.data.shape) == 0:
            continue
        assert msg.data.shape[1] == nch
        assert not np.any(msg.data - msg.data[:, :1])
        counter += 1
        if counter > 10:
            break


def test_inlet_reconnect_on_settings_change():
    """Pushing new settings that target a different stream must drop the old
    inlet and reconnect to the new one.

    Regression test: ``_reset_state`` previously recreated the resolver but
    left ``_state.inlet`` pointing at the old connection, so ``_produce`` kept
    pulling the original stream and a settings change appeared to do nothing.
    Streams A and B differ in channel count, so the channel dimension tells us
    which one the producer is actually pulling.
    """
    rate = 32.0
    outlet_a = pylsl.StreamOutlet(
        pylsl.StreamInfo(
            name="dummyA", type="dummy", channel_count=8, nominal_srate=rate, channel_format=pylsl.cf_float32
        )
    )
    outlet_b = pylsl.StreamOutlet(
        pylsl.StreamInfo(
            name="dummyB", type="dummy", channel_count=4, nominal_srate=rate, channel_format=pylsl.cf_float32
        )
    )

    def push():
        outlet_a.push_chunk(np.zeros((10, 8), dtype=np.float32))
        outlet_b.push_chunk(np.ones((10, 4), dtype=np.float32))

    producer = LSLInletProducer(info=LSLInfo(name="dummyA", type="dummy"))

    phase = "A"  # currently expecting the 8-channel stream
    reconnected = False
    for count, msg in enumerate(producer):
        push()
        if count > 2000:
            break
        if msg is None or np.prod(msg.data.shape) == 0:
            continue
        if phase == "A":
            # Once we've confirmed we're on A (8ch), retarget to B (4ch).
            if msg.data.shape[1] == 8:
                producer.update_settings(LSLInletSettings(info=LSLInfo(name="dummyB", type="dummy")))
                phase = "B"
        elif msg.data.shape[1] == 4:  # phase B: reconnected to the 4-channel stream
            reconnected = True
            break

    assert phase == "B", "never connected to the initial stream A"
    assert reconnected, "did not reconnect to stream B after the settings change"


class DummyOutletSettings(ez.Settings):
    rate: float = 100.0
    n_chans: int = 8
    running: bool = True


class DummyOutlet(ez.Unit):
    SETTINGS = DummyOutletSettings

    @ez.task
    async def run_dummy(self) -> None:
        info = pylsl.StreamInfo(
            name="dummy",
            type="dummy",
            channel_count=self.SETTINGS.n_chans,
            nominal_srate=self.SETTINGS.rate,
            channel_format=pylsl.cf_float32,
        )
        outlet = pylsl.StreamOutlet(info)
        eff_rate = self.SETTINGS.rate or 100.0
        n_interval = int(eff_rate / 10)
        n_pushed = 0
        t0 = pylsl.local_clock()
        while self.SETTINGS.running:
            t_next = t0 + (n_pushed + n_interval) / eff_rate
            t_now = pylsl.local_clock()
            await asyncio.sleep(t_next - t_now)
            data_offset = n_pushed / eff_rate
            data = np.arange(n_interval)[:, None] / eff_rate + data_offset
            data = data + np.zeros((1, self.SETTINGS.n_chans))  # Expand channels dim
            outlet.push_chunk(data.astype(np.float32))
            n_pushed += n_interval


def test_inlet_collection():
    """The primary purpose of this test is to verify that LSLInletUnit can be included in a collection."""
    file_path = Path(tempfile.gettempdir())
    file_path = file_path / Path("test_inlet_collection.txt")
    file_path.unlink(missing_ok=True)

    class LSLTestSystemSettings(ez.Settings):
        stream_name: str = "dummy"
        stream_type: str = "dummy"

    class LSLTestSystem(ez.Collection):
        SETTINGS = LSLTestSystemSettings

        DUMMY = DummyOutlet()
        INLET = LSLInletUnit()
        LOGGER = MessageLogger()
        TERM = TerminateOnTotal()

        def configure(self) -> None:
            self.DUMMY.apply_settings(DummyOutletSettings(rate=100.0, n_chans=8))
            self.INLET.apply_settings(
                LSLInletSettings(LSLInfo(name=self.SETTINGS.stream_name, type=self.SETTINGS.stream_type))
            )
            self.LOGGER.apply_settings(MessageLoggerSettings(output=file_path))
            self.TERM.apply_settings(TerminateOnTotalSettings(total=10))

        def network(self) -> ez.NetworkDefinition:
            return (
                (self.INLET.OUTPUT_SIGNAL, self.LOGGER.INPUT_MESSAGE),
                (self.LOGGER.OUTPUT_MESSAGE, self.TERM.INPUT_MESSAGE),
            )

    # This next line raises an error if the ClockSync object runs its own thread.
    system = LSLTestSystem()
    ez.run(SYSTEM=system)
    messages: typing.List[AxisArray] = [_ for _ in message_log(file_path)]
    file_path.unlink(missing_ok=True)
    assert len(messages) >= 10
    cat_messages = AxisArray.concatenate(*messages, dim="time")
    # Data are repeated across channels. Subtracting ch0 from all chans should yield an array of zeros.
    assert not np.any(cat_messages.data - cat_messages.data[:, :1])
    # Data are incrementing by 1/100.0. Check we aren't missing any.
    samp_steps = np.diff(cat_messages.data[:, 0])
    assert np.allclose(samp_steps, np.ones_like(samp_steps) / 100)


@pytest.mark.parametrize("rate", [100.0, 0.0])
def test_inlet_comps_conns(rate: float):
    n_messages = 20
    file_path = Path(tempfile.gettempdir())
    file_path = file_path / Path("test_inlet_system.txt")

    comps = {
        "DUMMY": DummyOutlet(rate=rate, n_chans=8),
        "SRC": LSLInletUnit(info=LSLInfo(name="dummy", type="dummy")),
        "LOGGER": MessageLogger(output=file_path),
        "TERM": TerminateOnTotal(total=n_messages),
    }
    conns = (
        (comps["SRC"].OUTPUT_SIGNAL, comps["LOGGER"].INPUT_MESSAGE),
        (comps["LOGGER"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)

    messages: typing.List[AxisArray] = [_ for _ in message_log(file_path)]
    file_path.unlink(missing_ok=True)

    # We merely verify that the messages are being sent to the logger.
    assert len(messages) >= n_messages


class _FakeStreamInfo:
    """Just enough of pylsl.StreamInfo for _setup_after_open."""

    def __init__(self, name="CURSOR_PLAN", stype="BCIPlan", host="rpi5", n_ch=2, srate=50.0):
        self._name, self._type, self._host = name, stype, host
        self._n_ch, self._srate = n_ch, srate

    def name(self):
        return self._name

    def type(self):
        return self._type

    def hostname(self):
        return self._host

    def channel_count(self):
        return self._n_ch

    def nominal_srate(self):
        return self._srate

    def channel_format(self):
        return pylsl.cf_float32

    def desc(self):
        return _FakeXML()


class _FakeXML:
    def child(self, _name):
        return self

    def empty(self):
        return True


def test_inlet_logs_the_stream_it_resolved(caplog):
    producer = LSLInletProducer(settings=LSLInletSettings(info=LSLInfo(name="CURSOR_PLAN", type="BCIPlan")))
    producer._state.inlet = type("I", (), {"info": lambda _self: _FakeStreamInfo()})()

    with caplog.at_level("INFO"):
        producer._setup_after_open()

    assert "CURSOR_PLAN" in caplog.text
    assert "BCIPlan" in caplog.text
    assert "rpi5" in caplog.text


def test_inlet_logs_once_when_no_stream_matches(caplog):
    producer = LSLInletProducer(settings=LSLInletSettings(info=LSLInfo(name="CURSOR_PLAN", type="BCIPlan")))
    producer._state.clock_sync = None

    async def run_test():
        for _ in range(3):
            assert await producer._produce() is None

    with caplog.at_level("INFO"):
        asyncio.run(run_test())

    searching = [r for r in caplog.records if "still looking" in r.message]
    assert len(searching) == 1
    # Names what it wanted, so a typo'd stream name is legible from the log.
    assert "CURSOR_PLAN" in searching[0].getMessage()


def test_a_reconnect_can_log_again(caplog):
    """The log-once flags describe one connection, not the process lifetime."""
    producer = LSLInletProducer(settings=LSLInletSettings(info=LSLInfo(name="CURSOR_PLAN", type="BCIPlan")))
    producer._logged_searching = True
    producer._logged_lost = True

    producer._state.inlet = type("I", (), {"info": lambda _self: _FakeStreamInfo()})()
    producer._setup_after_open()
    assert producer._logged_searching is False
    assert producer._logged_lost is False

    producer._state.clock_sync = None
    producer._state.inlet = None

    async def run_test():
        assert await producer._produce() is None

    with caplog.at_level("INFO"):
        asyncio.run(run_test())

    assert any("still looking" in r.message for r in caplog.records)


def test_inlet_logs_a_lost_stream_once(caplog):
    class RaisingInlet:
        def pull_chunk(self, **_kwargs):
            raise RuntimeError("stream lost")

    producer = LSLInletProducer(settings=LSLInletSettings())
    inlet = RaisingInlet()
    producer._state.inlet = inlet
    producer._state.clock_sync = object()
    producer._state.msg_template = AxisArray(
        data=np.empty((0, 2)),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=50.0)},
        key="CURSOR_PLAN",
    )

    with caplog.at_level("INFO"):
        for _ in range(3):
            assert asyncio.run(producer._apull(timeout=0.0)) is None

    lost = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(lost) == 1
    assert "CURSOR_PLAN" in lost[0].getMessage()


def test_a_pull_racing_shutdown_is_not_reported_as_a_lost_stream(caplog):
    """A pull raising on a handle shutdown() already dropped is teardown, not loss."""

    class RaisingInlet:
        def pull_chunk(self, **_kwargs):
            raise RuntimeError("inlet closed")

    producer = LSLInletProducer(settings=LSLInletSettings())
    snapshot_inlet = RaisingInlet()
    producer._state.inlet = snapshot_inlet
    producer._state.clock_sync = object()
    producer._state.msg_template = AxisArray(
        data=np.empty((0, 2)),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=50.0)},
        key="CURSOR_PLAN",
    )
    snapshot = producer._snapshot_pull_state()
    producer.shutdown()

    with caplog.at_level("INFO"):
        assert producer._pull(snapshot, timeout=0.0) is None

    assert [r for r in caplog.records if r.levelname == "WARNING"] == []
