import asyncio
import time
import typing
import warnings
from dataclasses import dataclass, field, fields

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
import pylsl
import pylsl.util
from ezmsg.baseproc import BaseProducerUnit, BaseStatefulProducer, processor_state
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

# Mapping from LSLInfo.channel_format string values to pylsl channel format constants.
_string2cf = {
    "float32": pylsl.cf_float32,
    "double64": pylsl.cf_double64,
    "float64": pylsl.cf_double64,
    "string": pylsl.cf_string,
    "int32": pylsl.cf_int32,
    "int16": pylsl.cf_int16,
    "int8": pylsl.cf_int8,
    "int64": pylsl.cf_int64,
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


@processor_state
class LSLInletProducerState:
    resolver: typing.Optional[pylsl.ContinuousResolver] = None
    inlet: typing.Optional[pylsl.StreamInlet] = None
    clock_sync: typing.Optional[ClockSync] = None
    msg_template: typing.Optional[AxisArray] = None
    fetch_buffer: typing.Optional[npt.NDArray] = None
    hash: int = -1

    def __init__(self) -> None:
        self.resolver = None
        self.inlet = None
        self.clock_sync = None
        self.msg_template = None
        self.fetch_buffer = None
        self.hash = -1


class LSLInletProducer(BaseStatefulProducer[LSLInletSettings, typing.Optional[AxisArray], LSLInletProducerState]):
    def __init__(self, *args, settings: typing.Optional[LSLInletSettings] = None, **kwargs):
        kwargs = _sanitize_kwargs(kwargs)
        super().__init__(*args, settings=settings, **kwargs)

    def _reset_state(self) -> None:
        self._state.resolver = pylsl.ContinuousResolver(pred=None, forget_after=30.0)
        self._state.clock_sync = ClockSync()

    def _try_connect(self) -> None:
        """Attempt to find and connect to a matching LSL stream.

        If all required fields (name, type, channel_count, channel_format) are provided,
        construct a StreamInfo directly and attempt open_stream with a finite timeout.
        Some streams won't appear via resolve and can only be connected to this way.
        Otherwise, use the ContinuousResolver to discover streams.
        """
        info = self.settings.info
        # Direct-connect path: all required fields are provided.
        if all(
            [
                info.name,
                info.type,
                info.channel_count is not None,
                info.channel_format is not None,
            ]
        ):
            strm_info = pylsl.StreamInfo(
                name=info.name,
                type=info.type,
                channel_count=info.channel_count,
                channel_format=info.channel_format,
            )
            inlet = pylsl.StreamInlet(strm_info, max_chunklen=1, processing_flags=self.settings.processing_flags)
            try:
                inlet.open_stream(timeout=2.0)
            except (pylsl.util.TimeoutError, pylsl.util.LostError):
                return
            self._state.inlet = inlet
            self._setup_after_open()
            return

        # Resolver-based path: match on whichever fields are provided.
        if self._state.resolver is None:
            return
        results: list[pylsl.StreamInfo] = self._state.resolver.results()
        for strm_info in results:
            b_match = True
            b_match = b_match and ((not info.name) or strm_info.name() == info.name)
            b_match = b_match and ((not info.type) or strm_info.type() == info.type)
            b_match = b_match and ((not info.host) or strm_info.hostname() == info.host)
            if info.channel_count is not None:
                b_match = b_match and strm_info.channel_count() == info.channel_count
            if info.channel_format is not None:
                expected_cf = _string2cf.get(info.channel_format)
                if expected_cf is not None:
                    b_match = b_match and strm_info.channel_format() == expected_cf
            if b_match:
                self._open_inlet(strm_info)
                break

    def _open_inlet(self, strm_info: pylsl.StreamInfo) -> None:
        """Create a StreamInlet from a discovered StreamInfo and set up buffers/template."""
        self._state.inlet = pylsl.StreamInlet(
            strm_info,
            max_chunklen=1,
            processing_flags=self.settings.processing_flags,
        )
        self._state.inlet.open_stream(timeout=5.0)
        self._setup_after_open()

    def _setup_after_open(self) -> None:
        """Configure fetch buffer and message template after a stream is opened."""
        # Resolver is no longer needed once connected. Destroy it now (while we're
        # in a background thread via _try_connect) so its destructor doesn't
        # run during shutdown.
        self._state.resolver = None

        inlet_info = self._state.inlet.info()
        # Fill in nominal_srate on settings (it may have been left at default).
        self.settings.info.nominal_srate = inlet_info.nominal_srate()
        # If possible, create a destination buffer for faster pulls.
        fmt = inlet_info.channel_format()
        n_ch = inlet_info.channel_count()
        if fmt in fmt2npdtype:
            dtype = fmt2npdtype[fmt]
            n_buff = int(self.settings.local_buffer_dur * inlet_info.nominal_srate()) or 1000
            self._state.fetch_buffer = np.zeros((n_buff, n_ch), dtype=dtype)
        ch_labels: list[str] = []
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
            AxisArray.TimeAxis(fs=fs) if fs else AxisArray.CoordinateAxis(data=np.array([]), dims=["time"], unit="s")
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

    def _pull(self) -> typing.Optional[AxisArray]:
        """Pull available data from the inlet. Non-blocking (timeout=0.0)."""
        if self._state.inlet is None:
            return None
        try:
            if self._state.fetch_buffer is not None:
                samples, timestamps = self._state.inlet.pull_chunk(
                    max_samples=self._state.fetch_buffer.shape[0],
                    dest_obj=self._state.fetch_buffer,
                )
            else:
                samples, timestamps = self._state.inlet.pull_chunk()
                samples = np.array(samples)
        except Exception:
            # Stream may have been closed concurrently by shutdown.
            return None

        if not len(timestamps):
            return None

        data = self._state.fetch_buffer[: len(timestamps)].copy() if samples is None else samples

        # `timestamps` is currently in the LSL clock stamped by the sender.
        if self.settings.use_arrival_time:
            # Drop the sender stamps; use "now". Useful when playing back old XDF files.
            timestamps = time.monotonic() - (timestamps - timestamps[0])
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

    async def _produce(self) -> typing.Optional[AxisArray]:
        if self._state.inlet is None:
            await asyncio.to_thread(self._try_connect)
            if self._state.inlet is None:
                await asyncio.sleep(0.01)
                return None

        # Update clock sync if its rate limiter has expired.
        await self._state.clock_sync.arun_once()

        # Re-check after the await — shutdown may have closed the inlet.
        if self._state.inlet is None:
            return None

        result = self._pull()
        if result is None:
            await asyncio.sleep(0.001)
        return result

    def shutdown(self) -> None:
        self._state.msg_template = None
        self._state.fetch_buffer = None
        if self._state.inlet is not None:
            self._state.inlet.close_stream()
            del self._state.inlet
        self._state.inlet = None
        if self._state.clock_sync is not None:
            # The thread is not usually started, but in case it is...
            self._state.clock_sync.stop()
        self._state.clock_sync = None


class LSLInletGenerator(LSLInletProducer):
    """Deprecated: use LSLInletProducer instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LSLInletGenerator is deprecated. Use LSLInletProducer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class LSLInletUnit(BaseProducerUnit[LSLInletSettings, typing.Optional[AxisArray], LSLInletProducer]):
    """
    Represents a node in a graph that creates an LSL inlet and
    forwards the pulled data to the unit's output.

    Args:
        stream_name: The `name` of the created LSL outlet.
        stream_type: The `type` of the created LSL outlet.
    """

    SETTINGS = LSLInletSettings

    def create_producer(self) -> None:
        if hasattr(self, "producer") and self.producer is not None:
            self.producer.shutdown()
        super().create_producer()

    async def shutdown(self) -> None:
        if hasattr(self, "producer") and self.producer is not None:
            # Run in a thread so close_stream() doesn't block the event loop.
            # This allows other tasks to process their cancellation during shutdown.
            await asyncio.to_thread(self.producer.shutdown)
