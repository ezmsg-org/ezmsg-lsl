"""
These unit tests aren't really testable in a runner without a complicated setup with inlets and outlets.
This code exists mostly to use during development and debugging.
"""

import asyncio
import tempfile
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

from ezmsg.lsl.inlet import LSLInfo, LSLInletGenerator, LSLInletSettings, LSLInletUnit


def test_inlet_init_defaults():
    settings = LSLInletSettings(info=LSLInfo(name="", type=""))
    _ = LSLInletUnit(settings)
    assert True


def test_inlet_generator():
    """
    Test the inlet generator object without invoking ezmsg.
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

    gen = LSLInletGenerator(info=LSLInfo(name="dummy", type="dummy"))
    counter = 0
    for msg in gen:
        step_outlet()
        if msg is None or np.prod(msg.data.shape) == 0:
            continue
        assert msg.data.shape[1] == nch
        assert not np.any(msg.data - msg.data[:, :1])
        counter += 1
        if counter > 10:
            break


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
