import asyncio
from dataclasses import dataclass, field, fields
import time
import typing

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace
import numpy as np
import numpy.typing as npt
import pylsl

from .util import ClockSync


fmt2npdtype = {
    pylsl.cf_double64: float,  # Prefer native type for float64
    pylsl.cf_int64: int,  # Prefer native type for int64
    pylsl.cf_float32: np.float32,
    pylsl.cf_int32: np.int32,
    pylsl.cf_int16: np.int16,
    pylsl.cf_int8: np.int8,
    # pylsl.cf_string:  # For now we don't provide a pre-allocated buffer for string data type.
}


@dataclass
class LSLInfo:
    name: str = ""
    type: str = ""
    host: str = ""  # Use socket.gethostname() for local host.
    channel_count: typing.Optional[int] = None
    nominal_srate: float = 0.0
    channel_format: typing.Optional[str] = None


def _sanitize_kwargs(kwargs: dict) -> dict:
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
    return kwargs


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

    INPUT_SETTINGS = ez.InputStream(LSLInletSettings)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    # Share clock correction across all instances
    clock_sync = ClockSync()

    def __init__(self, *args, **kwargs) -> None:
        """
        Handle deprecated arguments. Whereas previously stream_name and stream_type were in the
        LSLInletSettings, now LSLInletSettings has info: LSLInfo which has fields for name, type,
        among others.
        """
        kwargs = _sanitize_kwargs(kwargs)
        super().__init__(*args, **kwargs)
        self._msg_template: typing.Optional[AxisArray] = None
        self._fetch_buffer: typing.Optional[npt.NDArray] = None

    def _reset_resolver(self) -> None:
        self.STATE.resolver = pylsl.ContinuousResolver(pred=None, forget_after=30.0)

    def _reset_inlet(self) -> None:
        self._msg_template: typing.Optional[AxisArray] = None
        self._fetch_buffer: typing.Optional[npt.NDArray] = None
        if self.STATE.inlet is not None:
            self.STATE.inlet.close_stream()
            del self.STATE.inlet
        self.STATE.inlet = None
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
            results: list[pylsl.StreamInfo] = self.STATE.resolver.results()
            for strm_info in results:
                b_match = True
                b_match = b_match and (
                    (not self.SETTINGS.info.name)
                    or strm_info.name() == self.SETTINGS.info.name
                )
                b_match = b_match and (
                    (not self.SETTINGS.info.type)
                    or strm_info.type() == self.SETTINGS.info.type
                )
                b_match = b_match and (
                    (not self.SETTINGS.info.host)
                    or strm_info.hostname() == self.SETTINGS.info.host
                )
                if b_match:
                    self.STATE.inlet = pylsl.StreamInlet(
                        strm_info,
                        max_chunklen=1,
                        processing_flags=self.SETTINGS.processing_flags,
                    )
                    break

        if self.STATE.inlet is not None:
            self.STATE.inlet.open_stream()
            inlet_info = self.STATE.inlet.info()
            self.SETTINGS.info.nominal_srate = inlet_info.nominal_srate()
            # If possible, create a destination buffer for faster pulls
            fmt = inlet_info.channel_format()
            n_ch = inlet_info.channel_count()
            if fmt in fmt2npdtype:
                dtype = fmt2npdtype[fmt]
                n_buff = (
                    int(self.SETTINGS.local_buffer_dur * inlet_info.nominal_srate())
                    or 1000
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
            time_ax = (
                AxisArray.TimeAxis(fs=fs)
                if fs
                else AxisArray.CoordinateAxis(
                    data=np.array([]), dims=["time"], unit="s"
                )
            )
            self._msg_template = AxisArray(
                data=np.empty((0, n_ch)),
                dims=["time", "ch"],
                axes={
                    "time": time_ax,
                    "ch": AxisArray.CoordinateAxis(
                        data=np.array(ch_labels), dims=["ch"]
                    ),
                },
                key=inlet_info.name(),
            )

    async def initialize(self) -> None:
        self._reset_resolver()
        self._reset_inlet()
        await self.clock_sync.update(force=True, burst=1000)

    def shutdown(self) -> None:
        if self.STATE.inlet is not None:
            self.STATE.inlet.close_stream()
            del self.STATE.inlet
        self.STATE.inlet = None
        if self.STATE.resolver is not None:
            del self.STATE.resolver
        self.STATE.resolver = None

    @ez.task
    async def clock_sync_task(self) -> None:
        while True:
            await self.clock_sync.update(force=False, burst=4)

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: LSLInletSettings) -> None:
        # The message may be full LSLInletSettings, a dict of settings, just the info, or dict of just info.
        if isinstance(msg, dict):
            # First make sure the info is in the right place.
            msg = _sanitize_kwargs(msg)
            # Next, convert to LSLInletSettings object.
            msg = LSLInletSettings(**msg)
        if msg != self.SETTINGS:
            self.apply_settings(msg)
            self._reset_resolver()
            self._reset_inlet()

    @ez.publisher(OUTPUT_SIGNAL)
    async def lsl_pull(self) -> typing.AsyncGenerator:
        while True:
            if self.STATE.inlet is None:
                # Inlet not yet created, or recently destroyed because settings changed.
                self._reset_inlet()
                await asyncio.sleep(0.1)
                continue

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
                    timestamps = time.time() - (timestamps - timestamps[0])
                else:
                    timestamps = self.clock_sync.lsl2system(timestamps)

                if self.SETTINGS.info.nominal_srate <= 0.0:
                    # Irregular rate stream uses CoordinateAxis for time so each sample has a timestamp.
                    out_time_ax = replace(
                        self._msg_template.axes["time"],
                        data=np.array(timestamps),
                    )
                else:
                    # Regular rate uses a LinearAxis for time so we only need the time of the first sample.
                    out_time_ax = replace(
                        self._msg_template.axes["time"], offset=timestamps[0]
                    )

                out_msg = replace(
                    self._msg_template,
                    data=data,
                    axes={
                        **self._msg_template.axes,
                        "time": out_time_ax,
                    },
                )
                yield self.OUTPUT_SIGNAL, out_msg
            else:
                await asyncio.sleep(0.001)
