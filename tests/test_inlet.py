"""
These unit tests aren't really testable in a runner without a complicated setup with inlets and outlets.
This code exists mostly to use during development and debugging.
"""

import asyncio
import os
import tempfile
import threading
import time
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

    def __init__(
        self,
        name="CURSOR_PLAN",
        stype="BCIPlan",
        host="rpi5",
        n_ch=2,
        srate=50.0,
        uid="uid-1",
        source_id="src-1",
    ):
        self._name, self._type, self._host = name, stype, host
        self._n_ch, self._srate = n_ch, srate
        self._uid, self._source_id = uid, source_id

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

    def uid(self):
        return self._uid

    def source_id(self):
        return self._source_id

    def desc(self):
        return _FakeXML()


class _FakeXML:
    def child(self, _name):
        return self

    def empty(self):
        return True


class _FakeInlet:
    """Stands in for a connected pylsl.StreamInlet during _setup_after_open."""

    def __init__(self, info=None):
        self._info = info if info is not None else _FakeStreamInfo()

    def info(self, timeout=None):
        return self._info


def _connect(producer, info=None):
    """Drive _setup_after_open as though a stream had just been opened."""
    producer._state.inlet = _FakeInlet(info)
    assert producer._setup_after_open() is True


def test_inlet_logs_the_stream_it_resolved(caplog):
    producer = LSLInletProducer(settings=LSLInletSettings(info=LSLInfo(name="CURSOR_PLAN", type="BCIPlan")))

    with caplog.at_level("INFO"):
        _connect(producer)

    assert "CURSOR_PLAN" in caplog.text
    assert "BCIPlan" in caplog.text
    assert "rpi5" in caplog.text


def test_inlet_stamps_stream_identity_onto_every_message():
    """`key` alone can't answer which outlet instance produced a segment."""
    producer = LSLInletProducer(settings=LSLInletSettings())
    _connect(producer, _FakeStreamInfo(uid="uid-abc", source_id="ezmsg-123", host="rpi5"))

    attrs = producer._state.msg_template.attrs
    assert attrs["lsl_uid"] == "uid-abc"
    assert attrs["lsl_source_id"] == "ezmsg-123"
    assert attrs["lsl_hostname"] == "rpi5"


def test_setup_retries_when_the_description_cannot_be_fetched(caplog):
    """The desc is fetched over the wire, so a connect can open but not complete."""

    class TimingOutInlet:
        def info(self, timeout=None):
            raise pylsl.util.TimeoutError("no desc")

    producer = LSLInletProducer(settings=LSLInletSettings())
    producer._state.inlet = TimingOutInlet()

    with caplog.at_level("INFO"):
        assert producer._setup_after_open() is False

    assert producer._state.msg_template is None


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

    _connect(producer)
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


def _template(key="CURSOR_PLAN", n_ch=2):
    return AxisArray(
        data=np.empty((0, n_ch)),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=50.0)},
        key=key,
    )


def test_a_lost_stream_drives_a_reconnect(caplog):
    """LostError is terminal for the connection, so it must reach _produce."""

    class LostInlet:
        def pull_chunk(self, **_kwargs):
            raise pylsl.util.LostError("the stream has been lost.")

    producer = LSLInletProducer(settings=LSLInletSettings())
    producer._state.inlet = LostInlet()
    producer._state.clock_sync = object()
    producer._state.msg_template = _template()
    producer._last_source_id = "src-1"

    with caplog.at_level("INFO"):
        assert asyncio.run(producer._apull(timeout=0.0)) is None
    assert producer._lost is True

    assert asyncio.run(producer._produce()) is None
    # Queued for the top of the next __acall__, where _reset_state tears the
    # connection down and rebuilds the resolver.
    assert producer._hash == -1
    assert producer._lost is False
    assert producer._reconnect_source_id == "src-1"
    assert producer._reconnect_deadline > time.monotonic()


def test_reconnect_holds_out_for_the_original_stream_then_gives_up():
    producer = LSLInletProducer(settings=LSLInletSettings(info=LSLInfo(name="CURSOR_PLAN")))
    opened = []
    producer._open_inlet = opened.append

    replacement = _FakeStreamInfo(uid="uid-2", source_id="src-2")
    producer._state.resolver = type("R", (), {"results": lambda _self: [replacement]})()
    producer._reconnect_source_id = "src-1"
    producer._reconnect_deadline = time.monotonic() + 30.0

    producer._try_connect()
    assert opened == [], "took a different stream while the original might return"

    producer._reconnect_deadline = time.monotonic() - 1.0
    producer._try_connect()
    assert opened == [replacement], "never fell back to the resolver criteria"


def test_reconnect_takes_the_original_stream_when_it_is_back():
    producer = LSLInletProducer(settings=LSLInletSettings(info=LSLInfo(name="CURSOR_PLAN")))
    opened = []
    producer._open_inlet = opened.append

    original = _FakeStreamInfo(uid="uid-9", source_id="src-1")
    other = _FakeStreamInfo(uid="uid-2", source_id="src-2")
    # Resolver order deliberately puts the impostor first.
    producer._state.resolver = type("R", (), {"results": lambda _self: [other, original]})()
    producer._reconnect_source_id = "src-1"
    producer._reconnect_deadline = time.monotonic() + 30.0

    producer._try_connect()
    assert opened == [original]


def test_the_host_criterion_survives_a_reconnect():
    """liblsl's own recovery matches source_id alone and would ignore `host`."""
    producer = LSLInletProducer(settings=LSLInletSettings(info=LSLInfo(name="CURSOR_PLAN", host="rpi5")))
    opened = []
    producer._open_inlet = opened.append

    elsewhere = _FakeStreamInfo(uid="uid-2", source_id="src-1", host="rpi6")
    producer._state.resolver = type("R", (), {"results": lambda _self: [elsewhere]})()
    producer._reconnect_source_id = "src-1"
    producer._reconnect_deadline = time.monotonic() + 30.0

    producer._try_connect()
    assert opened == [], "re-attached to the right source_id on the wrong host"


def test_a_settings_change_drops_the_reconnect_preference():
    """New settings may retarget the inlet, so the old stream is no longer wanted."""
    producer = LSLInletProducer(settings=LSLInletSettings(info=LSLInfo(name="A")))
    producer._reconnect_source_id = "src-1"
    producer._reconnect_deadline = time.monotonic() + 30.0

    producer.update_settings(LSLInletSettings(info=LSLInfo(name="B")))

    assert producer._reconnect_source_id is None
    assert producer._reconnect_deadline == 0.0


def test_a_restarted_upstream_bumps_the_key_epoch():
    producer = LSLInletProducer(settings=LSLInletSettings(distinct_key_per_connection=True))

    _connect(producer, _FakeStreamInfo(uid="uid-1"))
    assert producer._state.msg_template.key == "CURSOR_PLAN"

    # Same outlet instance: a dropped socket, not a restart. State stays valid.
    _connect(producer, _FakeStreamInfo(uid="uid-1"))
    assert producer._state.msg_template.key == "CURSOR_PLAN"

    # New outlet instance behind the same name and shape -- invisible downstream
    # unless the key changes, since the hash is over (shape, rate, key).
    _connect(producer, _FakeStreamInfo(uid="uid-2"))
    assert producer._state.msg_template.key == "CURSOR_PLAN#1"


def test_key_is_stable_across_a_restart_by_default():
    """Off by default: NWB writers name containers by key and would fork one."""
    producer = LSLInletProducer(settings=LSLInletSettings())

    _connect(producer, _FakeStreamInfo(uid="uid-1"))
    _connect(producer, _FakeStreamInfo(uid="uid-2"))

    assert producer._state.msg_template.key == "CURSOR_PLAN"
    assert producer._connection_epoch == 1, "epoch still tracked for attrs/provenance"


def test_reconnecting_to_a_changed_stream_warns(caplog):
    """A different shape or machine still matches the criteria, so nothing else complains."""
    producer = LSLInletProducer(settings=LSLInletSettings())
    _connect(producer, _FakeStreamInfo(n_ch=64, host="rpi5"))

    with caplog.at_level("INFO"):
        _connect(producer, _FakeStreamInfo(n_ch=32, host="rpi6"))

    warned = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warned) == 1
    message = warned[0].getMessage()
    assert "channel_count" in message and "64" in message and "32" in message
    assert "host" in message and "rpi6" in message


def test_reconnecting_to_an_identical_stream_does_not_warn(caplog):
    producer = LSLInletProducer(settings=LSLInletSettings())
    _connect(producer, _FakeStreamInfo(uid="uid-1"))

    with caplog.at_level("INFO"):
        _connect(producer, _FakeStreamInfo(uid="uid-2"))

    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_a_lost_outlet_is_detected_and_replaced_by_a_restarted_one(caplog):
    """End-to-end check of `recover=False`.

    With liblsl's own recovery enabled, a vanished outlet produces no error at
    all -- the inlet retries the original source_id forever and simply stops
    emitting, which is indistinguishable from a quiet stream. Here the loss must
    surface, and a replacement advertising a different source_id and shape (a
    reconfigured upstream, which liblsl would never re-acquire) must be picked up.
    """
    name = f"TESTLOST_{os.getpid()}"
    settings = LSLInletSettings(
        info=LSLInfo(name=name, type="dummy"),
        pull_timeout=0.01,
        reconnect_grace_dur=0.5,
    )
    producer = LSLInletProducer(settings=settings)

    def make_outlet(n_ch, source_id):
        return pylsl.StreamOutlet(
            pylsl.StreamInfo(
                name=name,
                type="dummy",
                channel_count=n_ch,
                nominal_srate=100.0,
                channel_format=pylsl.cf_float32,
                source_id=source_id,
            )
        )

    async def pump(outlet, n_ch, predicate, limit=400):
        """Drive the producer until `predicate` holds, pushing if an outlet exists."""
        for _ in range(limit):
            if outlet is not None:
                outlet.push_chunk(np.zeros((10, n_ch), dtype=np.float32))
            msg = await producer.__acall__()
            if predicate(msg):
                return True
            await asyncio.sleep(0.01)
        return False

    with caplog.at_level("INFO"):
        outlet = make_outlet(4, "src-original")

        connected = asyncio.run(pump(outlet, 4, lambda m: m is not None and np.prod(m.data.shape) > 0))
        assert connected, "never received data from the original outlet"
        assert producer._state.msg_template.attrs["lsl_source_id"] == "src-original"
        first_uid = producer._state.msg_template.attrs["lsl_uid"]

        del outlet
        lost = asyncio.run(pump(None, 4, lambda _m: producer._reconnect_source_id is not None))
        assert lost, "a vanished outlet never surfaced as a lost stream"
        assert producer._reconnect_source_id == "src-original"

        # Restarted upstream: same name/type, new instance, fewer channels.
        replacement = make_outlet(2, "src-restarted")
        back = asyncio.run(pump(replacement, 2, lambda m: m is not None and m.data.shape[1] == 2))
        assert back, "never reconnected to the replacement outlet"

    assert producer._state.msg_template.attrs["lsl_source_id"] == "src-restarted"
    assert producer._state.msg_template.attrs["lsl_uid"] != first_uid
    assert producer._connection_epoch == 1

    assert any("lost the stream" in r.getMessage() for r in caplog.records)
    changed = [r for r in caplog.records if "changed stream" in r.getMessage()]
    assert changed and "channel_count" in changed[0].getMessage()
