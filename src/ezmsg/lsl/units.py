import asyncio
from dataclasses import dataclass, replace, field, fields
import time
import typing

import numpy as np
import pylsl
import ezmsg.core as ez

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

fmt2npdtype = {
    pylsl.cf_double64: float,  # Prefer native type for float64
    pylsl.cf_int64: int,  # Prefer native type for int64
    pylsl.cf_float32: np.float32,
    pylsl.cf_int32: np.int32,
    pylsl.cf_int16: np.int16,
    pylsl.cf_int8: np.int8,
    # pylsl.cf_string:  # For now we don't provide a pre-allocated buffer for string data type.
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


@dataclass
class LSLInfo:
    name: str = ""
    type: str = ""
    channel_count: typing.Optional[int] = None
    nominal_srate: float = 0.0
    channel_format: typing.Optional[str] = None


class LSLInletSettings(ez.Settings):
    info: LSLInfo = field(default_factory=LSLInfo)
    local_buffer_dur: float = 1.0
    # Whether to ignore the LSL timestamps and use the time.time of the pull (True).
    # If False (default), the LSL timestamps are used, but (optionally) corrected to time.time. See `use_lsl_clock`.
    use_arrival_time: bool = False
    # Whether the AxisArray.Axis.offset should use LSL's clock (True) or time.time's clock (False -- default).
    # This setting is ignored if `use_arrival_time` is True.
    # Setting `use_arrival_time=False, use_lsl_clock=True` is the only way to accommodate playback rate != 1.0 and keep
    # the axis .offset consistent with the original samplerate.
    use_lsl_clock: bool = False
    processing_flags: int = pylsl.proc_ALL
    # The processing flags option passed to pylsl.StreamInlet. Default is proc_ALL which includes all flags.
    # Many users will want to set this to pylsl.proc_clocksync to disable dejittering.


class ClockSync:
    def __init__(self, alpha: float = 0.1, min_interval: float = 0.5):
        self.alpha = alpha
        self.min_interval = min_interval

        self.offset = 0.0
        self.last_update = 0.0
        self.count = 0

    async def update(self, force: bool = False, burst: int = 4) -> None:
        dur_since_last = time.time() - self.last_update
        dur_until_next = self.min_interval - dur_since_last
        if force or dur_until_next <= 0:
            offsets = []
            for _ in range(burst):
                if self.count % 2:
                    y, x = time.time(), pylsl.local_clock()
                else:
                    x, y = pylsl.local_clock(), time.time()
                offsets.append(y - x)
                self.last_update = y
                await asyncio.sleep(0.001)
            offset = np.mean(offsets)

            if self.count > 0:
                # Exponential decay smoothing
                offset = (1 - self.alpha) * self.offset + self.alpha * offset
            self.offset = offset
            self.count += burst
        else:
            await asyncio.sleep(dur_until_next)

    def convert_timestamp(self, lsl_timestamp: float) -> float:
        return lsl_timestamp + self.offset


class LSLInletState(ez.State):
    resolver: typing.Optional[pylsl.ContinuousResolver] = None
    inlet: typing.Optional[pylsl.StreamInlet] = None
    clock_offset: float = 0.0


class LSLInletUnit(ez.Unit):
    """
    Represents a node in a graph that creates an LSL inlet and
    forwards the pulled data to the unit's output.

    Args:
        stream_name: The `name` of the created LSL outlet.
        stream_type: The `type` of the created LSL outlet.
    """

    SETTINGS = LSLInletSettings
    STATE = LSLInletState

    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    # Share clock correction across all instances
    clock_sync = ClockSync()

    def __init__(self, *args, **kwargs) -> None:
        """
        Handle deprecated arguments. Whereas previously stream_name and stream_type were in the
        LSLInletSettings, now LSLInletSettings has info: LSLInfo which has fields for name, type,
        among others.
        """
        if "info" not in kwargs:
            replace_keys = set()
            for k, v in kwargs.items():
                if k.startswith("stream_"):
                    replace_keys.add(k)
            if len(replace_keys) > 0:
                ez.logger.warning(
                    f"LSLInlet kwargs beginning with 'stream_' deprecated. Found {replace_keys}. "
                    f"See LSLInfo dataclass."
                )
                for k in replace_keys:
                    kwargs[k[7:]] = kwargs.pop(k)

            known_fields = [_.name for _ in fields(LSLInfo)]
            info_kwargs = {k: v for k, v in kwargs.items() if k in known_fields}
            for k in info_kwargs.keys():
                kwargs.pop(k)
            kwargs["info"] = LSLInfo(**info_kwargs)
        super().__init__(*args, **kwargs)

    async def initialize(self) -> None:
        # If name, type, and host are all provided, then create the StreamInfo directly and
        #  create the inlet directly from that info.
        if all(
            [
                _ is not None
                for _ in [
                    self.SETTINGS.info.name,
                    self.SETTINGS.info.type,
                    self.SETTINGS.info.channel_count,
                    self.SETTINGS.info.channel_format,
                ]
            ]
        ):
            info = pylsl.StreamInfo(
                name=self.SETTINGS.info.name,
                type=self.SETTINGS.info.type,
                channel_count=self.SETTINGS.info.channel_count,
                channel_format=self.SETTINGS.info.channel_format,
            )
            self.STATE.inlet = pylsl.StreamInlet(
                info, max_chunklen=1, processing_flags=self.SETTINGS.processing_flags
            )
        else:
            # Build the predicate string. This uses XPATH syntax and can filter on anything in the stream info. e.g.,
            # `"name='BioSemi'" or "type='EEG' and starts-with(name,'BioSemi') and count(info/desc/channel)=32"`
            pred = ""
            if self.SETTINGS.info.name:
                pred += f"name='{self.SETTINGS.info.name}'"
            if self.SETTINGS.info.type:
                if len(pred):
                    pred += " and "
                pred += f"type='{self.SETTINGS.info.type}'"
            if not len(pred):
                pred = None
            self.STATE.resolver = pylsl.ContinuousResolver(pred=pred)
        self._fetch_buffer: np.ndarray | None = None

    def shutdown(self) -> None:
        if self.STATE.inlet is not None:
            self.STATE.inlet.close_stream()
        self.STATE.inlet = None

    @ez.task
    async def clock_sync_task(self) -> None:
        while True:
            force = self.clock_sync.count < 1000
            await self.clock_sync.update(force=force, burst=1000 if force else 4)

    @ez.publisher(OUTPUT_SIGNAL)
    async def lsl_pull(self) -> typing.AsyncGenerator:
        while self.STATE.inlet is None:
            results: list[pylsl.StreamInfo] = self.STATE.resolver.results()
            if len(results):
                self.STATE.inlet = pylsl.StreamInlet(
                    results[0], max_chunklen=1, processing_flags=pylsl.proc_ALL
                )
            else:
                await asyncio.sleep(0.5)

        self.STATE.inlet.open_stream()
        inlet_info = self.STATE.inlet.info()
        # If possible, create a destination buffer for faster pulls
        fmt = inlet_info.channel_format()
        n_ch = inlet_info.channel_count()
        if fmt in fmt2npdtype:
            dtype = fmt2npdtype[fmt]
            n_buff = (
                int(self.SETTINGS.local_buffer_dur * inlet_info.nominal_srate()) or 1000
            )
            self._fetch_buffer = np.zeros((n_buff, n_ch), dtype=dtype)
        ch_labels = []
        chans = inlet_info.desc().child("channels")
        if not chans.empty():
            ch = chans.first_child()
            while not ch.empty():
                ch_labels.append(ch.child_value("label"))
                ch = ch.next_sibling()
        while len(ch_labels) < n_ch:
            ch_labels.append(str(len(ch_labels) + 1))
        # Pre-allocate a message template.
        fs = inlet_info.nominal_srate()
        msg_template = AxisArray(
            data=np.empty((0, n_ch)),
            dims=["time", "ch"],
            axes={
                "time": AxisArray.Axis.TimeAxis(
                    fs=fs if fs else 1.0
                ),  # HACK: Use 1.0 for irregular rate.
                "ch": AxisArray.Axis.SpaceAxis(labels=ch_labels),
            },
            key=inlet_info.name(),
        )

        while self.clock_sync.count < 1000:
            # Let the clock_sync task do its job at the beginning.
            await asyncio.sleep(0.001)

        while self.STATE.inlet is not None:
            if self._fetch_buffer is not None:
                samples, timestamps = self.STATE.inlet.pull_chunk(
                    max_samples=self._fetch_buffer.shape[0], dest_obj=self._fetch_buffer
                )
            else:
                samples, timestamps = self.STATE.inlet.pull_chunk()
                samples = np.array(samples)

            # Attempt to update the clock offset (shared across all instances)
            if len(timestamps):
                data = (
                    self._fetch_buffer[: len(timestamps)].copy()
                    if samples is None
                    else samples
                )
                if self.SETTINGS.use_arrival_time:
                    # time.time() gives us NOW, but we want the timestamp of the 0th sample in the chunk
                    t0 = time.time() - (timestamps[-1] - timestamps[0])
                else:
                    t0 = self.clock_sync.convert_timestamp(timestamps[0])
                if fs <= 0.0:
                    # Irregular rate streams need to be streamed sample-by-sample
                    for ts, samp in zip(timestamps, data):
                        out_msg = replace(
                            msg_template,
                            data=samp[None, ...],
                            axes={
                                **msg_template.axes,
                                "time": replace(
                                    msg_template.axes["time"],
                                    offset=t0 + (ts - timestamps[0]),
                                ),
                            },
                        )
                        yield self.OUTPUT_SIGNAL, out_msg
                else:
                    # Regular-rate streams can go in a chunk
                    out_msg = replace(
                        msg_template,
                        data=data,
                        axes={
                            **msg_template.axes,
                            "time": replace(msg_template.axes["time"], offset=t0),
                        },
                    )
                    yield self.OUTPUT_SIGNAL, out_msg
            else:
                await asyncio.sleep(0.001)
