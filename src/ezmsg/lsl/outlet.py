import typing

import ezmsg.core as ez
import numpy as np
import pylsl

from .util import AxisArray


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

    async def initialize(self) -> None:
        self._stream_created = False

    def shutdown(self) -> None:
        del self.STATE.outlet
        self.STATE.outlet = None

    @ez.subscriber(INPUT_SIGNAL)
    async def lsl_outlet(self, arr: AxisArray) -> None:
        if self.STATE.outlet is None:
            fs = 1 / arr.axes["time"].gain
            out_shape = [_[0] for _ in zip(arr.shape, arr.dims) if _[1] != "time"]
            out_size = int(np.prod(out_shape))
            info = pylsl.StreamInfo(
                name=self.SETTINGS.stream_name,
                type=self.SETTINGS.stream_type,
                channel_count=out_size,
                nominal_srate=fs,
                channel_format=string2fmt[str(arr.data.dtype)],
                source_id="",  # TODO: Generate a hash from name, type, channel_count, fs, fmt, other metadata...
            )
            # TODO: if self.SETTINGS.map_file: Add channel labels (and locations?) to the info desc.
            self.STATE.outlet = pylsl.StreamOutlet(info)

        if self.STATE.outlet is not None:
            dat = arr.data
            if arr.dims[0] != "time":
                dat = np.moveaxis(dat, arr.dims.index("time"), 0)

            self.STATE.outlet.push_chunk(dat.reshape(dat.shape[0], -1))
