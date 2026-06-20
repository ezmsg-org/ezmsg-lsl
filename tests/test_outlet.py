"""
These unit tests aren't really testable in a runner without a complicated setup with inlets and outlets.
This code exists mostly to use during development and debugging.
"""

import tempfile
import typing
from pathlib import Path
from unittest import mock

import ezmsg.core as ez
import numpy as np
import pylsl
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
