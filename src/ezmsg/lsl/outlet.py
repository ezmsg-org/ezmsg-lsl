import typing

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
import numpy as np
import pylsl

from .util import ClockSync


# Reproduce pylsl.string2fmt but add float64 for more familiar numpy usage
string2fmt = {
    "float32": pylsl.cf_float32,
    "double64": pylsl.cf_double64,
    "float64": pylsl.cf_double64,
    "string": pylsl.cf_string,
    "int32": pylsl.cf_int32,
    "int16": pylsl.cf_int16,
    "int8": pylsl.cf_int8,
    "int64": pylsl.cf_int64,
}


class LSLOutletSettings(ez.Settings):
    stream_name: typing.Optional[str] = None
    stream_type: typing.Optional[str] = None
    map_file: typing.Optional[str] = (
        None  # Path to file containing a list of channel names and locations.
    )


class LSLOutletState(ez.State):
    outlet: typing.Optional[pylsl.StreamOutlet] = None


class LSLOutletUnit(ez.Unit):
    """
    Represents a node in a Labgraph graph that subscribes to messages in a
    Labgraph topic and forwards them by writing to an LSL outlet.

    Args:
        stream_name: The `name` of the created LSL outlet.
        stream_type: The `type` of the created LSL outlet.
    """

    INPUT_SIGNAL = ez.InputStream(AxisArray)

    SETTINGS = LSLOutletSettings
    STATE = LSLOutletState

    # Share clock correction across all instances
    clock_sync = ClockSync()

    async def initialize(self) -> None:
        self._stream_created = False
        await self.clock_sync.update(force=True, burst=1000)

    def shutdown(self) -> None:
        del self.STATE.outlet
        self.STATE.outlet = None

    @ez.task
    async def clock_sync_task(self) -> None:
        while True:
            await self.clock_sync.update(force=False, burst=4)

    @ez.subscriber(INPUT_SIGNAL)
    async def lsl_outlet(self, msg: AxisArray) -> None:
        if self.STATE.outlet is None:
            if isinstance(msg.axes["time"], AxisArray.LinearAxis):
                fs = 1 / msg.axes["time"].gain
            else:
                # Coordinate axis because timestamps are irregular
                fs = pylsl.IRREGULAR_RATE
            out_shape = [_[0] for _ in zip(msg.shape, msg.dims) if _[1] != "time"]
            out_size = int(np.prod(out_shape))
            info = pylsl.StreamInfo(
                name=self.SETTINGS.stream_name,
                type=self.SETTINGS.stream_type,
                channel_count=out_size,
                nominal_srate=fs,
                channel_format=string2fmt[str(msg.data.dtype)],
                source_id="",  # TODO: Generate a hash from name, type, channel_count, fs, fmt, other metadata...
            )
            # Add channel labels to the info desc.
            if "ch" in msg.axes and isinstance(
                msg.axes["ch"], AxisArray.CoordinateAxis
            ):
                ch_labels = msg.axes["ch"].data
                # TODO: or get ch_labels from self.SETTINGS.map_file
                # TODO: if msg is multi-dim then construct labels by combining dims.
                #  For now, labels only work if only output dims are "time", "ch"
                if len(ch_labels) == out_size:
                    chans = info.desc().append_child("channels")
                    for ch in ch_labels:
                        chan = chans.append_child("channel")
                        chan.append_child_value("label", ch)
                        # TODO: if self.SETTINGS.map_file: Add channel locations
            self.STATE.outlet = pylsl.StreamOutlet(info)

        if self.STATE.outlet is not None:
            dat = msg.data
            if msg.dims[0] != "time":
                dat = np.moveaxis(dat, msg.dims.index("time"), 0)

            if not dat.flags.c_contiguous or not dat.flags.writeable:
                # TODO: When did this become necessary?
                dat = np.ascontiguousarray(dat).copy()

            if hasattr(msg.axes["time"], "data"):
                ts = msg.axes["time"].data
            else:
                ts = msg.axes["time"].value(dat.shape[0])
            ts = self.clock_sync.system2lsl(ts)
            self.STATE.outlet.push_chunk(dat.reshape(dat.shape[0], -1), timestamp=ts)
