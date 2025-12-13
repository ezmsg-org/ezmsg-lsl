import asyncio
import time
import typing
from dataclasses import dataclass, field, fields

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
import pylsl
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

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
                f"LSLInlet kwargs beginning with 'stream_' deprecated. Found {replace_keys}. See LSLInfo dataclass."
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

    use_arrival_time: bool = False
    """
    Whether to ignore the LSL timestamps and use the time.time of the pull (True).
    If False (default), the LSL (send) timestamps are used.
    Send times may be converted from LSL clock to time.time clock. See `use_lsl_clock`.
    """

    use_lsl_clock: bool = False
    """
    Whether the AxisArray.Axis.offset should use LSL's clock (True) or time.time's clock (False -- default).
    """

    processing_flags: int = pylsl.proc_ALL
    """
    The processing flags option passed to pylsl.StreamInlet. Default is proc_ALL which includes all flags.
    Many users will want to set this to pylsl.proc_clocksync to disable dejittering.
    """


class LSLInletState(ez.State):
    resolver: typing.Optional[pylsl.ContinuousResolver] = None
    inlet: typing.Optional[pylsl.StreamInlet] = None
    clock_sync: ClockSync = ClockSync(run_thread=False)
    msg_template: typing.Optional[AxisArray] = None
    fetch_buffer: typing.Optional[npt.NDArray] = None


class LSLInletGenerator:
    def __init__(self, *args, settings: typing.Optional[LSLInletSettings] = None, **kwargs):
        kwargs = _sanitize_kwargs(kwargs)
        if settings is None:
            if len(args) > 0 and isinstance(args[0], LSLInletSettings):
                settings = args[0]
            elif len(args) > 0 or len(kwargs) > 0:
                settings = LSLInletSettings(*args, **kwargs)
            else:
                settings = LSLInletSettings()
        self._state: LSLInletState = LSLInletState()
        self.settings = settings
        self.shutdown()
        self._reset_resolver()

    def __iter__(self):
        # self.shutdown() to reset?
        return self

    @property
    def state(self) -> LSLInletState:
        return self._state

    def _reset_resolver(self) -> None:
        self._state.resolver = pylsl.ContinuousResolver(pred=None, forget_after=30.0)

    def shutdown(self, shutdown_resolver: bool = True):
        self._state.msg_template = None
        self._state.fetch_buffer = None
        if self._state.inlet is not None:
            self._state.inlet.close_stream()
            del self._state.inlet
        self._state.inlet = None
        if shutdown_resolver:
            self._state.resolver = None

    def _reset_inlet(self):
        self.shutdown(shutdown_resolver=False)

        # If name, type, and host are all provided, then create the StreamInfo directly and
        #  create the inlet directly from that info.
        if all(
            [
                _ is not None
                for _ in [
                    self.settings.info.name,
                    self.settings.info.type,
                    self.settings.info.channel_count,
                    self.settings.info.channel_format,
                ]
            ]
        ):
            info = pylsl.StreamInfo(
                name=self.settings.info.name,
                type=self.settings.info.type,
                channel_count=self.settings.info.channel_count,
                channel_format=self.settings.info.channel_format,
            )
            self._state.inlet = pylsl.StreamInlet(info, max_chunklen=1, processing_flags=self.settings.processing_flags)
        elif self._state.resolver is not None:
            results: list[pylsl.StreamInfo] = self._state.resolver.results()
            for strm_info in results:
                b_match = True
                b_match = b_match and ((not self.settings.info.name) or strm_info.name() == self.settings.info.name)
                b_match = b_match and ((not self.settings.info.type) or strm_info.type() == self.settings.info.type)
                b_match = b_match and ((not self.settings.info.host) or strm_info.hostname() == self.settings.info.host)
                if b_match:
                    self._state.inlet = pylsl.StreamInlet(
                        strm_info,
                        max_chunklen=1,
                        processing_flags=self.settings.processing_flags,
                    )
                    break

        if self._state.inlet is not None:
            self._state.inlet.open_stream()
            inlet_info = self._state.inlet.info()
            # It's bad practice to write directly to settings but here we
            #  are filling in a value that was optional.
            self.settings.info.nominal_srate = inlet_info.nominal_srate()
            # If possible, create a destination buffer for faster pulls
            fmt = inlet_info.channel_format()
            n_ch = inlet_info.channel_count()
            if fmt in fmt2npdtype:
                dtype = fmt2npdtype[fmt]
                n_buff = int(self.settings.local_buffer_dur * inlet_info.nominal_srate()) or 1000
                self._state.fetch_buffer = np.zeros((n_buff, n_ch), dtype=dtype)
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
                else AxisArray.CoordinateAxis(data=np.array([]), dims=["time"], unit="s")
            )
            self._state.msg_template = AxisArray(
                data=np.empty((0, n_ch)),
                dims=["time", "ch"],
                axes={
                    "time": time_ax,
                    "ch": AxisArray.CoordinateAxis(data=np.array(ch_labels), dims=["ch"]),
                },
                key=inlet_info.name(),
            )

    def update_settings(self, new_settings: LSLInletSettings) -> None:
        # The message may be full LSLInletSettings, a dict of settings, just the info, or dict of just info.
        if isinstance(new_settings, dict):
            # First make sure the info is in the right place.
            msg = _sanitize_kwargs(new_settings)
            # Next, convert to LSLInletSettings object.
            msg = LSLInletSettings(**msg)
        if msg != self.settings:
            self._reset_resolver()
            self._reset_inlet()

    def __next__(self) -> typing.Optional[AxisArray]:
        if self._state.inlet is None:
            # Inlet not yet created, or recently destroyed because settings changed.
            self._reset_inlet()
            return None

        if self._state.fetch_buffer is not None:
            samples, timestamps = self._state.inlet.pull_chunk(
                max_samples=self._state.fetch_buffer.shape[0],
                dest_obj=self._state.fetch_buffer,
            )
        else:
            samples, timestamps = self._state.inlet.pull_chunk()
            samples = np.array(samples)

        out_msg = self._state.msg_template
        if len(timestamps):
            data = self._state.fetch_buffer[: len(timestamps)].copy() if samples is None else samples

            # `timestamps` is currently in the LSL clock stamped by the sender.
            if self.settings.use_arrival_time:
                # Drop the sender stamps; use "now"
                timestamps = time.time() - (timestamps - timestamps[0])
                if self.settings.use_lsl_clock:
                    timestamps = self._state.clock_sync.system2lsl(timestamps)
            elif not self.settings.use_lsl_clock:
                # Keep the sender clock but convert to system time.
                timestamps = self._state.clock_sync.lsl2system(timestamps)

            if self.settings.info.nominal_srate <= 0.0:
                # Irregular rate stream uses CoordinateAxis for time so each sample has a timestamp.
                out_time_ax = replace(
                    self._state.msg_template.axes["time"],
                    data=np.array(timestamps),
                )
            else:
                # Regular rate uses a LinearAxis for time so we only need the time of the first sample.
                out_time_ax = replace(self._state.msg_template.axes["time"], offset=timestamps[0])

            out_msg = replace(
                self._state.msg_template,
                data=data,
                axes={
                    **self._state.msg_template.axes,
                    "time": out_time_ax,
                },
            )
        return out_msg


class LSLInletUnitState(ez.State):
    generator: typing.Optional[LSLInletGenerator] = None


class LSLInletUnit(ez.Unit):
    """
    Represents a node in a graph that creates an LSL inlet and
    forwards the pulled data to the unit's output.

    Args:
        stream_name: The `name` of the created LSL outlet.
        stream_type: The `type` of the created LSL outlet.
    """

    SETTINGS = LSLInletSettings
    STATE = LSLInletUnitState

    INPUT_SETTINGS = ez.InputStream(LSLInletSettings)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:
        self._create_generator()

    def _create_generator(self):
        self.STATE.generator = LSLInletGenerator(settings=self.SETTINGS)

    def shutdown(self) -> None:
        self.STATE.generator.shutdown()

    @ez.task
    async def update_clock(self) -> None:
        gen = self.STATE.generator
        while True:
            if gen.state.inlet is not None:
                gen.state.clock_sync.run_once()
            await asyncio.sleep(0.1)

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: LSLInletSettings) -> None:
        self.apply_settings(msg)
        self.STATE.generator.update_settings(msg)

    @ez.publisher(OUTPUT_SIGNAL)
    async def lsl_pull(self) -> typing.AsyncGenerator:
        while True:
            out_msg = next(self.STATE.generator)
            if out_msg is not None and np.prod(out_msg.data.shape) > 0:
                yield self.OUTPUT_SIGNAL, out_msg
            else:
                await asyncio.sleep(0.001)
