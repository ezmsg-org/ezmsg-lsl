"""
These unit tests aren't really testable in a runner without a complicated setup with inlets and outlets.
This code exists mostly to use during development and debugging.
"""

import asyncio
from pathlib import Path
import tempfile
import typing

import numpy as np
import pylsl
import pytest
import ezmsg.core as ez
from ezmsg.sigproc.synth import Clock, ClockSettings
from ezmsg.util.debuglog import DebugLog, DebugLogSettings
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messagelogger import MessageLogger
from ezmsg.util.messagecodec import message_log
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTimeout, TerminateOnTimeoutSettings, \
    TerminateOnTotalSettings

from ezmsg.lsl.units import LSLInfo, LSLInletSettings, LSLInletUnit


def test_inlet_init_defaults():
    settings = LSLInletSettings(info=LSLInfo(name="", type=""))
    _ = LSLInletUnit(settings)
    assert True


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
        )
        outlet = pylsl.StreamOutlet(info)
        eff_rate = self.SETTINGS.rate or 100.0
        n_interval = int(eff_rate / 10)
        n_pushed = 0
        t0 = pylsl.local_clock()
        while self.SETTINGS.running:
            t_next = t0 + (n_pushed + n_interval) / (self.SETTINGS.rate or 100.0)
            t_now = pylsl.local_clock()
            await asyncio.sleep(t_next - t_now)
            data = np.random.random((n_interval, self.SETTINGS.n_chans))
            outlet.push_chunk(data)
            n_pushed += n_interval


def test_inlet_collection():
    """The primary purpose of this test is to verify that LSLInletUnit can be included in a collection."""

    class LSLTestSystemSettings(ez.Settings):
        stream_name: str = "dummy"
        stream_type: str = "dummy"

    class LSLTestSystem(ez.Collection):
        SETTINGS = LSLTestSystemSettings

        DUMMY = DummyOutlet()
        INLET = LSLInletUnit()
        LOGGER = DebugLog()
        TERM = TerminateOnTotal()

        def configure(self) -> None:
            self.INLET.apply_settings(
                LSLInletSettings(
                    LSLInfo(
                        name=self.SETTINGS.stream_name, type=self.SETTINGS.stream_type
                    )
                )
            )
            self.LOGGER.apply_settings(DebugLogSettings(name="test_inlet_collection"))
            self.TERM.apply_settings(TerminateOnTotalSettings(total=10))

        def network(self) -> ez.NetworkDefinition:
            return (
                (self.INLET.OUTPUT_SIGNAL, self.LOGGER.INPUT),
                (self.LOGGER.OUTPUT, self.TERM.INPUT_MESSAGE),
            )

    # This next line raises an error if the ClockSync object runs its own thread.
    system = LSLTestSystem()
    ez.run(SYSTEM=system)


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
