"""
These unit tests aren't really testable in a runner without a complicated setup with inlets and outlets.
This code exists mostly to use during development and debugging.
"""

import re
import tempfile
import typing
from pathlib import Path
from unittest import mock

import ezmsg.core as ez
import numpy as np
import pylsl
import pytest
from ezmsg.baseproc.clock import Clock
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis
from ezmsg.util.terminate import TerminateOnTotal
from helpers.synth import Oscillator

from ezmsg.lsl.outlet import LSLOutletSettings, LSLOutletUnit, OutletProcessor


def test_outlet_system():
    n_messages = 10

    file_path = Path(tempfile.gettempdir())
    file_path = file_path / Path("test_outlet_system.txt")

    comps = {
        "CLOCK": Clock(dispatch_rate=100.0),
        "SYNTH": Oscillator(n_time=10, fs=1000, n_ch=32, dispatch_rate="ext_clock"),
        "OUTLET": LSLOutletUnit(stream_name="test_outlet_system", stream_type="EEG"),
        "LOGGER": MessageLogger(output=file_path),
        "TERM": TerminateOnTotal(total=n_messages),
    }
    conns = (
        (comps["CLOCK"].OUTPUT_SIGNAL, comps["SYNTH"].INPUT_SIGNAL),
        (comps["SYNTH"].OUTPUT_SIGNAL, comps["OUTLET"].INPUT_SIGNAL),
        (comps["SYNTH"].OUTPUT_SIGNAL, comps["LOGGER"].INPUT_MESSAGE),
        (comps["LOGGER"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)

    messages: typing.List[AxisArray] = [_ for _ in message_log(file_path)]
    file_path.unlink(missing_ok=True)

    # We merely verify that the messages are being sent to the logger.
    assert len(messages) >= n_messages


def _make_msg() -> AxisArray:
    return AxisArray(
        data=np.zeros((5, 3), dtype=np.int16),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0, offset=0.0),
            "ch": CoordinateAxis(data=np.array(["a", "b", "c"]), dims=["ch"]),
        },
        attrs={},
        key="test",
    )


def _transport_flags_for(sync_blocking: bool) -> int:
    """Build an outlet via OutletProcessor and capture the transport_flags
    passed to pylsl.StreamOutlet, without touching the LSL network."""
    proc = OutletProcessor(
        settings=LSLOutletSettings(stream_name="test", stream_type="EEG", sync_blocking=sync_blocking)
    )
    with mock.patch("ezmsg.lsl.outlet.pylsl.StreamOutlet") as MockOutlet:
        proc._reset_state(_make_msg())
    return MockOutlet.call_args.kwargs["transport_flags"]


def test_outlet_default_transport_flags():
    """By default the outlet is created with the standard async transport."""
    assert _transport_flags_for(sync_blocking=False) == pylsl.transp_default


def test_outlet_sync_blocking_transport_flags():
    """sync_blocking=True sets the transp_sync_blocking transport flag."""
    assert _transport_flags_for(sync_blocking=True) == pylsl.transp_sync_blocking


def test_outlet_string_chunk_roundtrip():
    """A string message goes out in one push_chunk, one timestamp per sample."""
    values = np.array([["a", "bb"], ["ccc", "d"], ["e", "ff"]], dtype="<U3")
    offsets = np.array([10.0, 10.5, 11.25])
    msg = AxisArray(
        data=values,
        dims=["time", "ch"],
        axes={
            "time": CoordinateAxis(data=offsets, dims=["time"], unit="s"),
            "ch": CoordinateAxis(data=np.array(["x", "y"]), dims=["ch"]),
        },
        attrs={},
        key="test_outlet_string_chunk_roundtrip",
    )
    proc = OutletProcessor(
        settings=LSLOutletSettings(
            stream_name="test_outlet_string_chunk_roundtrip",
            stream_type="Markers",
            assume_lsl_clock=True,
        )
    )
    proc(msg)  # Creates the outlet and pushes; the inlet below joins late.

    results = pylsl.resolve_byprop("name", "test_outlet_string_chunk_roundtrip", timeout=5.0)
    assert results, "outlet did not resolve"
    inlet = pylsl.StreamInlet(results[0], processing_flags=0)
    inlet.open_stream(timeout=5.0)
    try:
        proc(msg)
        samples, timestamps = [], []
        for _ in range(20):
            chunk, stamps = inlet.pull_chunk(timeout=0.2, max_samples=16, min_samples=1)
            samples.extend(chunk)
            timestamps.extend(stamps)
            if len(timestamps) >= len(offsets):
                break
    finally:
        inlet.close_stream()
        proc.shutdown()

    assert samples == [["a", "bb"], ["ccc", "d"], ["e", "ff"]]
    assert np.allclose(timestamps, offsets)


def _stream_msg(n: int, dims=("win", "ch"), stream_dim: typing.Optional[str] = "win") -> AxisArray:
    """``n`` samples of 3 channels streaming along ``win`` (not "time")."""
    shape = tuple(n if d == "win" else 3 for d in dims)
    return AxisArray(
        data=np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        dims=list(dims),
        axes={
            "win": AxisArray.LinearAxis(gain=0.5, offset=0.0, unit="s"),
            "ch": CoordinateAxis(data=np.array(["a", "b", "c"]), dims=["ch"]),
        },
        key="test",
        stream_dim=stream_dim,
    )


def _run_outlet(messages: typing.Iterable[AxisArray]):
    """Feed ``messages`` through an OutletProcessor with a mocked StreamOutlet; return the mock class."""
    proc = OutletProcessor(settings=LSLOutletSettings(stream_name="test", stream_type="EEG", assume_lsl_clock=True))
    with mock.patch("ezmsg.lsl.outlet.pylsl.StreamOutlet") as MockOutlet:
        for msg in messages:
            proc(msg)
    return MockOutlet


@pytest.mark.parametrize(
    "dims, stream_dim",
    [(("win", "ch"), "win"), (("ch", "win"), "win"), (("win", "ch"), None)],  # None falls back to dims[0]
)
def test_outlet_non_time_stream_dim(dims, stream_dim):
    """The outlet streams along the declared stream_dim (or dims[0]), so chunk size does not recreate it."""
    msgs = [_stream_msg(n, dims, stream_dim) for n in (4, 7, 2)]
    MockOutlet = _run_outlet(msgs)

    assert MockOutlet.call_count == 1
    info = MockOutlet.call_args.args[0]
    assert info.channel_count() == 3
    assert info.nominal_srate() == 2.0
    pushes = MockOutlet.return_value.push_chunk.call_args_list
    for msg, push in zip(msgs, pushes):
        expected = np.moveaxis(msg.data, dims.index("win"), 0)
        np.testing.assert_array_equal(push.args[0], expected)
        np.testing.assert_allclose(push.kwargs["timestamp"], msg.axes["win"].value(expected.shape[0]))


def _ch_msg(labels=("a", "b", "c"), n: int = 5, dtype=np.float32, fs: float = 100.0, **kwargs) -> AxisArray:
    return AxisArray(
        data=np.zeros((n, len(labels)), dtype=dtype),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=0.0),
            "ch": CoordinateAxis(data=np.array(labels), dims=["ch"]),
        },
        **kwargs,
    )


@pytest.mark.parametrize(
    "second",
    [
        _ch_msg(labels=("x", "y", "z")),
        _ch_msg(dtype=np.int16),
        _ch_msg(labels=("a", "b", "c", "d")),
        _ch_msg(fs=200.0),
    ],
    ids=["relabel", "dtype", "n_ch", "fs"],
)
def test_outlet_recreated_when_advertised_info_changes(second):
    MockOutlet = _run_outlet([_ch_msg(), second])
    assert MockOutlet.call_count == 2
    info = MockOutlet.call_args.args[0]
    assert re.findall(r"<label>(.*?)</label>", info.as_xml()) == list(second.axes["ch"].data)
    assert info.nominal_srate() == second.axes["time"].gain ** -1


@pytest.mark.parametrize(
    "messages",
    [
        [_ch_msg(n=5), _ch_msg(n=2), _ch_msg(n=9)],
        [_ch_msg(segment=0), _ch_msg(segment=1)],
        [_ch_msg(config_id=1), _ch_msg(config_id=2), _ch_msg(config_id=3, n=2)],
        [_ch_msg(key="a"), _ch_msg(key="b")],
        [_ch_msg(dtype="<U1"), _ch_msg(dtype="<U4")],
        [_ch_msg(), _ch_msg(attrs={"unit": "uV"})],
    ],
    ids=["chunk_size", "segment", "config_id", "key", "string_width", "attrs"],
)
def test_outlet_kept_when_advertised_info_unchanged(messages):
    assert _run_outlet(messages).call_count == 1


@pytest.mark.parametrize("config_id", [None, 7], ids=["no_config_id", "constant_config_id"])
def test_outlet_desc_attrs_from_first_message(config_id):
    """Attrs need not be constant: the desc keeps the first message's, and only a layout change recreates."""
    kw = {} if config_id is None else {"config_id": config_id}
    msgs = [_ch_msg(attrs={"trigger": i, "value": i / 2}, **kw) for i in range(3)]

    MockOutlet = _run_outlet(msgs)
    assert MockOutlet.call_count == 1
    assert MockOutlet.call_args.args[0].desc().child_value("trigger") == "0"

    # A declaring source announces the new layout with a new id; under the same id the hash is never consulted.
    kw = {} if config_id is None else {"config_id": config_id + 1}
    MockOutlet = _run_outlet(msgs + [_ch_msg(labels=("a", "b", "c", "d"), attrs={"trigger": 3}, **kw)])
    assert MockOutlet.call_count == 2
    assert MockOutlet.call_args.args[0].desc().child_value("trigger") == "3"
