"""
These unit tests aren't really testable in a runner without a complicated setup with inlets and outlets.
This code exists mostly to use during development and debugging.
"""
import os
import json
from pathlib import Path
import tempfile

import numpy as np

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.lsl.units import LSLInfo, LSLInletSettings, LSLInletUnit


def test_inlet_init_defaults():
    settings = LSLInletSettings(name="", type="")
    _ = LSLInletUnit(settings)
    assert True


class MessageReceiverSettings(ez.Settings):
    num_msgs: int
    output_fn: str


class MessageReceiverState(ez.State):
    num_received: int = 0


class AxarrReceiver(ez.Unit):
    STATE = MessageReceiverState
    SETTINGS = MessageReceiverSettings
    INPUT_SIGNAL = ez.InputStream(AxisArray)

    @ez.subscriber(INPUT_SIGNAL)
    async def on_message(self, msg: AxisArray) -> None:
        self.STATE.num_received += 1
        t_ax = msg.axes["time"]
        tvec = np.arange(msg.data.shape[0]) * t_ax.gain + t_ax.offset
        payload = {self.STATE.num_received: tvec.tolist()}
        with open(self.SETTINGS.output_fn, "a") as output_file:
            output_file.write(json.dumps(payload) + "\n")
        if self.STATE.num_received == self.SETTINGS.num_msgs:
            raise ez.NormalTermination


def test_inlet_init_with_settings():
    test_name = os.environ.get("PYTEST_CURRENT_TEST")
    test_name = test_name.split(":")[-1].split(" ")[0]
    file_path = Path(tempfile.gettempdir())
    file_path = file_path / Path(f"{test_name}.json")

    comps = {
        "SRC": LSLInletUnit(info=LSLInfo(name="BrainVision RDA", type="EEG")),
        "SINK": AxarrReceiver(num_msgs=10_000, output_fn=file_path),
    }
    conns = ((comps["SRC"].OUTPUT_SIGNAL, comps["SINK"].INPUT_SIGNAL),)
    ez.run(components=comps, connections=conns)

    tvecs = []
    with open(file_path, "r") as file:
        for ix, line in enumerate(file.readlines()):
            tvecs.append(json.loads(line)[str(ix + 1)])
    os.remove(str(file_path))
    tvec = np.hstack(tvecs)

    # counts, bins = np.histogram(np.diff(tvec), 20)
    assert np.max(np.diff(tvec)) < 0.003
