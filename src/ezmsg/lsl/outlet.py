import asyncio
import hashlib
import time
import typing

import ezmsg.core as ez
import numpy as np
import pylsl
from ezmsg.util.messages.axisarray import AxisArray

from .util import ClockSync

# Reproduce pylsl.string2fmt but add float64 for more familiar numpy usage
string2fmt = {
    "float32": pylsl.cf_float32,
    "double64": pylsl.cf_double64,
    "float64": pylsl.cf_double64,
    "string": pylsl.cf_string,
    "object": pylsl.cf_string,
    "int32": pylsl.cf_int32,
    "int16": pylsl.cf_int16,
    "int8": pylsl.cf_int8,
    "int64": pylsl.cf_int64,
}


def generate_source_id(
    name: typing.Optional[str],
    stream_type: typing.Optional[str],
    channel_count: int,
    nominal_srate: float,
    channel_format: str,
) -> str:
    """Generate a stable source_id hash from stream metadata."""
    components = (
        name or "",
        stream_type or "",
        str(channel_count),
        f"{nominal_srate:.6f}",
        channel_format,
    )
    combined = "|".join(components)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


class LSLOutletSettings(ez.Settings):
    stream_name: typing.Optional[str] = None
    stream_type: typing.Optional[str] = None
    map_file: typing.Optional[str] = None
    """
    Path to file containing a list of channel names and locations.
    This feature is experimental and not tested.
    """

    use_message_timestamp: bool = True
    """
    Whether to push the data with the incoming timestamps (True, default) or to ignore the incoming timestamps
    and push the data with the current pylsl.local_clock (False). When `True`, the incoming data
    must have a "time" dimension.
    """

    assume_lsl_clock: bool = False
    """
    When `use_message_timestamp` is True, this indicates whether the incoming timestamps were already in the
    lsl clock (see :obj:`LslInletSettings`). If False, the incoming timestamps are assumed to be in the system
    time.time clock and are converted to the lsl clock.
    Note: Ignored when use_message_timestamp is False.
    """


class LSLOutletState(ez.State):
    outlet: typing.Optional[pylsl.StreamOutlet] = None
    clock_sync: ClockSync = ClockSync(run_thread=False)
    hash: int = 0


class OutletProcessor:
    def __init__(self, *args, settings: typing.Optional[LSLOutletSettings] = None, **kwargs):
        if settings is None:
            if len(args) > 0 and isinstance(args[0], LSLOutletSettings):
                settings = args[0]
            elif len(args) > 0 or len(kwargs) > 0:
                settings = LSLOutletSettings(*args, **kwargs)
            else:
                settings = LSLOutletSettings()
        self.settings = settings
        self._state: LSLOutletState = LSLOutletState()

    def __del__(self):
        self.shutdown()

    @property
    def state(self) -> LSLOutletState:
        return self._state

    def shutdown(self):
        if self._state.outlet is not None:
            del self._state.outlet
            self._state.outlet = None

    def check_metadata(self, message: AxisArray) -> bool:
        b_reset = False
        b_reset = b_reset or self.state.outlet is None
        fs = pylsl.IRREGULAR_RATE
        sample_shape = message.data.shape
        if "time" in message.axes:
            if hasattr(message.axes["time"], "gain"):
                fs = 1 / message.axes["time"].gain
            time_ix = message.get_axis_idx("time")
            sample_shape = message.data.shape[:time_ix] + message.data.shape[time_ix + 1 :]
        this_hash = hash((message.key, message.data.dtype, fs, sample_shape))
        b_reset = b_reset or this_hash != self._state.hash
        if b_reset:
            self._state.hash = this_hash
        return b_reset

    def reset(self, message: AxisArray) -> None:
        self.shutdown()

        fs = pylsl.IRREGULAR_RATE
        if "time" in message.axes and hasattr(message.axes["time"], "gain"):
            fs = 1 / message.axes["time"].gain
        out_shape = [_[0] for _ in zip(message.shape, message.dims) if _[1] != "time"]
        out_size = int(np.prod(out_shape))
        channel_format = str(message.data.dtype)
        source_id = generate_source_id(
            name=self.settings.stream_name,
            stream_type=self.settings.stream_type,
            channel_count=out_size,
            nominal_srate=fs,
            channel_format=channel_format,
        )
        info = pylsl.StreamInfo(
            name=self.settings.stream_name,
            type=self.settings.stream_type,
            channel_count=out_size,
            nominal_srate=fs,
            channel_format=string2fmt[channel_format],
            source_id="ezmsg-" + source_id,
        )
        # Add channel labels to the info desc.
        if "ch" in message.axes and isinstance(message.axes["ch"], AxisArray.CoordinateAxis):
            ch_labels = message.axes["ch"].data
            # TODO: or get ch_labels from self.settings.map_file
            # TODO: if msg is multi-dim then construct labels by combining dims.
            #  For now, labels only work if only output dims are "time", "ch"
            if len(ch_labels) == out_size:
                chans = info.desc().append_child("channels")
                for ch in ch_labels:
                    chan = chans.append_child("channel")
                    chan.append_child_value("label", ch)
                    # TODO: if self.settings.map_file: Add channel locations
        self._state.outlet = pylsl.StreamOutlet(info)

    def _process(self, message: AxisArray):
        dat = message.data
        if message.dims[0] != "time":
            dat = np.moveaxis(dat, message.dims.index("time"), 0)

        if not dat.flags.c_contiguous:
            dat = np.ascontiguousarray(dat)
        if not dat.flags.writeable:
            # If there is a shared-memory-hop in the processing graph before this node then it has made
            #  the numpy array non-writeable. We need to copy it to a new buffer.
            dat = np.ascontiguousarray(dat).copy()

        if self.settings.use_message_timestamp:
            if hasattr(message.axes["time"], "data"):
                ts = message.axes["time"].data
            else:
                ts = message.axes["time"].value(dat.shape[0])
            if not self.settings.assume_lsl_clock:
                ts = self._state.clock_sync.system2lsl(ts)
        else:
            ts = self._state.clock_sync.system2lsl(time.time())
        dat = dat.reshape(dat.shape[0], -1)

        if self._state.outlet.channel_format == pylsl.cf_string:
            # pylsl requires string data to be passed sample-by-sample
            for ix, row in enumerate(dat):
                self._state.outlet.push_sample(list(row), timestamp=ts[ix] if isinstance(ts, np.ndarray) else ts)
        else:
            self._state.outlet.push_chunk(dat, timestamp=ts)

    def __call__(self, message: AxisArray):
        if self.check_metadata(message):
            self.reset(message)
        return self._process(message)

    def __iter__(self):
        self._state: LSLOutletState = LSLOutletState()
        return self

    send = __call__  # Alias method name


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
        self.create_processor()

    def create_processor(self):
        self.processor = OutletProcessor(settings=self.SETTINGS)

    @ez.task
    async def update_clock(self) -> None:
        while True:
            if self.STATE.outlet is not None:
                self.processor.state.clock_sync.run_once()
            await asyncio.sleep(0.1)

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    async def lsl_outlet(self, msg: AxisArray) -> None:
        self.processor(msg)
