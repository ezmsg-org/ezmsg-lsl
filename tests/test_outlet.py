"""
These unit tests aren't really testable in a runner without a complicated setup with inlets and outlets.
This code exists mostly to use during development and debugging.
"""

import tempfile
import typing
from pathlib import Path

import ezmsg.core as ez
from ezmsg.baseproc.clock import Clock
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotal

from ezmsg.lsl.outlet import LSLOutletUnit

from helpers.synth import Oscillator


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
