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


def _parse_channel_metadata(chans_elem, n_ch: int) -> typing.Optional[np.ndarray]:
    """Parse a ``<channels>`` XML element into a structured numpy array.

    Returns ``None`` if the element is empty or the channel count doesn't match.
    """
    if chans_elem.empty():
        return None

    # Collect all channel records and discover field names in insertion order.
    ch_records: list[dict[str, str]] = []
    field_order: list[str] = []

    ch_elem = chans_elem.first_child()
    while not ch_elem.empty():
        rec: dict[str, str] = {}
        child = ch_elem.first_child()
        while not child.empty():
            tag = child.name()
            if tag == "location":
                # Flatten: <location><X>→x, <Y>→y, <Z>→z
                loc_child = child.first_child()
                while not loc_child.empty():
                    key = loc_child.name().lower()
                    rec[key] = loc_child.child_value()
                    if key not in field_order:
                        field_order.append(key)
                    loc_child = loc_child.next_sibling()
            else:
                rec[tag] = child.child_value()
                if tag not in field_order:
                    field_order.append(tag)
            child = child.next_sibling()
        ch_records.append(rec)
        ch_elem = ch_elem.next_sibling()

    if not ch_records or len(ch_records) != n_ch:
        return None

    # Infer a numpy dtype for each field.
    dtype_fields: list[tuple[str, str]] = []
    for fname in field_order:
        vals = [rec.get(fname, "") for rec in ch_records]
        non_empty = [v for v in vals if v]
        if non_empty:
            try:
                [int(v) for v in non_empty]
                dtype_fields.append((fname, "i4"))
                continue
            except ValueError:
                pass
            try:
                [float(v) for v in non_empty]
                dtype_fields.append((fname, "f4"))
                continue
            except ValueError:
                pass
        max_len = max((len(v) for v in vals), default=1) or 1
        dtype_fields.append((fname, f"U{max_len}"))

    ch_dtype = np.dtype(dtype_fields)
    ch_data = np.zeros(len(ch_records), dtype=ch_dtype)
    for i, rec in enumerate(ch_records):
        for fname in field_order:
            val = rec.get(fname, "")
            if val:
                ch_data[i][fname] = val
    return ch_data


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


# Never let liblsl recover a lost stream on its own. Its recovery matches on
# source_id alone, so it ignores the `host` criterion and can silently re-attach
# to a same-named stream on a different machine; it also retries forever, so an
# upstream that returns with a different shape is never found and the inlet
# stalls silently. Reconnecting here instead costs no data -- a dropped
# connection discards the outlet's per-consumer queue either way -- and makes
# the loss observable. See `_produce`.
_RECOVER = False

# Bound on fetching the full StreamInfo after opening. The description arrives
# over the wire (discovery results carry only the bare fields), and pylsl would
# otherwise wait forever, stranding a connect attempt that can no longer be
# cancelled.
_INFO_TIMEOUT = 5.0


def _describe_target(info: LSLInfo) -> str:
    """Human-readable form of what an inlet is looking for, for log messages."""
    named = (("name", info.name), ("type", info.type), ("host", info.host))
    parts = [f"{key}={value!r}" for key, value in named if value]
    if info.channel_count is not None:
        parts.append(f"channel_count={info.channel_count}")
    if info.channel_format is not None:
        parts.append(f"channel_format={info.channel_format!r}")
    return ", ".join(parts) if parts else "any stream"


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

    pull_timeout: float = 0.1
    """
    Maximum seconds to wait for the first sample during each steady-state pull.
    Once a sample arrives, ``pull_chunk(min_samples=1)`` immediately drains any
    other available samples up to ``max_pull_samples`` without waiting for the
    buffer to fill. The 0.1 second default therefore reduces idle wakeups
    without adding batching latency.

    Pulling runs on a worker thread via ``asyncio.to_thread``; the liblsl C call
    releases the GIL, so the event loop stays free while the inlet waits.
    """

    max_pull_samples: typing.Optional[int] = None
    """
    Total cap on samples returned per ``pull_chunk``. ``None`` (default) uses the
    full local fetch buffer. The inlet waits only for the first sample and then
    drains immediately available data up to this cap; it does not wait for the
    cap to be reached. A smaller value limits message size when clearing a large
    backlog, but may split that backlog across successive messages.
    """

    reconnect_grace_dur: float = 5.0
    """
    Seconds after losing a stream during which only the *same* stream (matching
    ``source_id``) is accepted. Afterwards any stream matching ``info`` is taken.

    This reproduces liblsl's own recovery, which re-acquires by ``source_id``
    alone, but bounds it: liblsl retries indefinitely, so an upstream that comes
    back with a different shape (and therefore a different ``source_id``) would
    never be found. The grace period prefers the original stream while it might
    still be coming back, then falls back to the resolver criteria so a
    restarted, reconfigured upstream is picked up. Set to 0 to always take the
    first match; the ``host`` criterion is honoured either way, which liblsl's
    recovery does not do.
    """

    distinct_key_per_connection: bool = False
    """
    Whether ``key`` gains a ``#<n>`` suffix that increments each time the inlet
    attaches to a *different outlet instance* (a changed StreamInfo ``uid``).

    Downstream processors key their state on ``(shape, rate, key)``, so a
    restarted upstream that keeps the same name and shape is otherwise invisible
    to them: filter state and partial windows carry across the discontinuity as
    if no gap occurred. Enabling this forces those resets.

    Off by default because it changes the identity that ``key`` denotes: NWB
    writers name containers by it and pipelines route on it, so a reconnect would
    fork the recording. A dropped socket that re-attaches to the *same* outlet
    instance never bumps the suffix, so a brief blip preserves state either way.
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


@dataclass(frozen=True)
class _PullSnapshot:
    """References and scalar settings used by one pull operation."""

    inlet: pylsl.StreamInlet
    fetch_buffer: typing.Optional[npt.NDArray]
    clock_sync: ClockSync
    msg_template: AxisArray
    max_pull_samples: typing.Optional[int]
    use_arrival_time: bool
    use_lsl_clock: bool
    nominal_srate: float


class LSLInletProducer(BaseStatefulProducer[LSLInletSettings, typing.Optional[AxisArray], LSLInletProducerState]):
    def __init__(self, *args, settings: typing.Optional[LSLInletSettings] = None, **kwargs):
        kwargs = _sanitize_kwargs(kwargs)
        # Also set in _reset_state, which only runs on the first __acall__.
        self._logged_searching = False
        self._logged_lost = False
        self._lost = False
        # Describes the *previous* connection, so unlike the flags above these
        # deliberately survive _reset_state -- that reset is how a reconnect
        # happens, and comparing against the old stream is the point.
        self._connection_epoch = 0
        self._last_uid: typing.Optional[str] = None
        self._last_source_id: typing.Optional[str] = None
        self._last_signature: typing.Optional[tuple] = None
        self._reconnect_source_id: typing.Optional[str] = None
        self._reconnect_deadline = 0.0
        super().__init__(*args, settings=settings, **kwargs)

    def update_settings(self, new_settings: LSLInletSettings) -> None:
        # New settings may retarget the inlet entirely, so holding out for the
        # previous stream's source_id would delay connecting to the one just
        # asked for. Both paths reach _reset_state, hence clearing it here.
        self._clear_reconnect_preference()
        super().update_settings(new_settings)

    def _clear_reconnect_preference(self) -> None:
        self._reconnect_source_id = None
        self._reconnect_deadline = 0.0

    def _reset_state(self) -> None:
        # Drop any existing connection and its derived state so a settings
        # change (e.g. a new target stream pushed via INPUT_SETTINGS) forces a
        # fresh resolve/connect. Without this, `_produce` sees a non-None inlet
        # and keeps pulling the previously-connected stream — the settings
        # change would appear to do nothing. An in-flight pull snapshot keeps
        # the old inlet and buffers alive until its bounded wait completes; its
        # result is discarded by `_apull` once this live reference changes.
        self._state.inlet = None
        self._state.msg_template = None
        self._state.fetch_buffer = None
        self._warmed_up = False
        # Log-once flags: both conditions are polled every tick, so log on transition only.
        self._logged_searching = False
        self._logged_lost = False
        self._lost = False
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
            inlet = pylsl.StreamInlet(
                strm_info,
                max_chunklen=1,
                recover=_RECOVER,
                processing_flags=self.settings.processing_flags,
            )
            try:
                inlet.open_stream(timeout=2.0)
            except (pylsl.util.TimeoutError, pylsl.util.LostError):
                return
            self._state.inlet = inlet
            if not self._setup_after_open():
                self._state.inlet = None
            return

        # Resolver-based path: match on whichever fields are provided.
        if self._state.resolver is None:
            return
        results: list[pylsl.StreamInfo] = self._state.resolver.results()
        matches = [strm_info for strm_info in results if self._matches_criteria(strm_info)]
        if not matches:
            return

        # Within the grace window after a loss, hold out for the stream we were
        # on. Matching on source_id is what liblsl's own recovery does; doing it
        # here keeps the `host` criterion applied, which that recovery ignores.
        if self._reconnect_source_id and time.monotonic() < self._reconnect_deadline:
            same_source = [_ for _ in matches if _.source_id() == self._reconnect_source_id]
            if not same_source:
                return
            matches = same_source

        self._open_inlet(matches[0])

    def _matches_criteria(self, strm_info: pylsl.StreamInfo) -> bool:
        """Whether a discovered stream satisfies every field set on ``settings.info``."""
        info = self.settings.info
        if info.name and strm_info.name() != info.name:
            return False
        if info.type and strm_info.type() != info.type:
            return False
        if info.host and strm_info.hostname() != info.host:
            return False
        if info.channel_count is not None and strm_info.channel_count() != info.channel_count:
            return False
        if info.channel_format is not None:
            expected_cf = _string2cf.get(info.channel_format)
            if expected_cf is not None and strm_info.channel_format() != expected_cf:
                return False
        return True

    def _open_inlet(self, strm_info: pylsl.StreamInfo) -> None:
        """Create a StreamInlet from a discovered StreamInfo and set up buffers/template."""
        self._state.inlet = pylsl.StreamInlet(
            strm_info,
            max_chunklen=1,
            recover=_RECOVER,
            processing_flags=self.settings.processing_flags,
        )
        self._state.inlet.open_stream(timeout=5.0)
        if not self._setup_after_open():
            self._state.inlet = None

    def _setup_after_open(self) -> bool:
        """Configure fetch buffer and message template after a stream is opened.

        Returns False if the stream's full description could not be fetched, in
        which case the caller drops the inlet and the connect is retried.
        """
        # Re-thread the first-data warmup on every (re)connect.
        self._warmed_up = False
        # Resolver is no longer needed once connected. Destroy it now (while we're
        # in a background thread via _try_connect) so its destructor doesn't
        # run during shutdown.
        self._state.resolver = None

        self._logged_searching = False
        self._logged_lost = False

        try:
            inlet_info = self._state.inlet.info(timeout=_INFO_TIMEOUT)
        except (pylsl.util.TimeoutError, pylsl.util.LostError) as exc:
            ez.logger.warning("LSL inlet could not fetch the stream description (%s); retrying.", exc)
            return False

        # Fill in nominal_srate on settings (it may have been left at default).
        self.settings.info.nominal_srate = inlet_info.nominal_srate()
        # If possible, create a destination buffer for faster pulls.
        fmt = inlet_info.channel_format()
        n_ch = inlet_info.channel_count()

        # `uid` identifies the outlet *instance* and is regenerated whenever an
        # outlet is constructed, so it -- not `source_id`, which is deliberately
        # stable across restarts so consumers can re-acquire -- is what tells a
        # dropped socket apart from a restarted upstream.
        uid = inlet_info.uid()
        source_id = inlet_info.source_id()
        hostname = inlet_info.hostname()
        if self._last_uid is not None and uid != self._last_uid:
            self._connection_epoch += 1
        self._last_uid = uid
        self._last_source_id = source_id
        self._clear_reconnect_preference()

        # Name the resolved stream: resolution can cross machines and find the
        # wrong one, which otherwise looks like a stream that is merely quiet.
        ez.logger.info(
            "LSL inlet connected to name=%r type=%r on host %r: %d ch @ %g Hz",
            inlet_info.name(),
            inlet_info.type(),
            hostname,
            n_ch,
            inlet_info.nominal_srate(),
        )

        # A reconnect that lands on a different shape or a different machine is
        # still a valid match on the configured criteria, so nothing else will
        # complain -- but it silently changes what the data means.
        signature = (n_ch, fmt, inlet_info.nominal_srate(), hostname)
        if self._last_signature is not None and signature != self._last_signature:
            changes = [
                f"{label}: {old!r} -> {new!r}"
                for label, old, new in zip(
                    ("channel_count", "channel_format", "nominal_srate", "host"),
                    self._last_signature,
                    signature,
                )
                if old != new
            ]
            ez.logger.warning("LSL inlet reconnected to a changed stream (%s).", "; ".join(changes))
        self._last_signature = signature
        if fmt in fmt2npdtype:
            dtype = fmt2npdtype[fmt]
            n_buff = int(self.settings.local_buffer_dur * inlet_info.nominal_srate()) or 1000
            self._state.fetch_buffer = np.zeros((n_buff, n_ch), dtype=dtype)
        # Parse channel metadata into a structured array when possible.
        chans = inlet_info.desc().child("channels")
        ch_info = _parse_channel_metadata(chans, n_ch)
        if ch_info is not None:
            # Fill any missing labels with 1-based indices.
            if "label" in ch_info.dtype.names:
                for i in range(n_ch):
                    if not ch_info[i]["label"]:
                        ch_info[i]["label"] = str(i + 1)
            ch_ax = AxisArray.CoordinateAxis(data=ch_info, dims=["ch"])
        else:
            # No structured metadata — fall back to numeric string labels.
            ch_labels = [str(i + 1) for i in range(n_ch)]
            ch_ax = AxisArray.CoordinateAxis(data=np.array(ch_labels), dims=["ch"])
        # Pre-allocate a message template.
        fs = inlet_info.nominal_srate()
        time_ax = (
            AxisArray.TimeAxis(fs=fs) if fs else AxisArray.CoordinateAxis(data=np.array([]), dims=["time"], unit="s")
        )
        # Epoch 0 renders as the bare name so the common case is unchanged.
        key = inlet_info.name()
        if self.settings.distinct_key_per_connection and self._connection_epoch:
            key = f"{key}#{self._connection_epoch}"
        self._state.msg_template = AxisArray(
            data=np.empty((0, n_ch)),
            dims=["time", "ch"],
            axes={"time": time_ax, "ch": ch_ax},
            key=key,
            attrs={
                "lsl_uid": uid,
                "lsl_source_id": source_id,
                "lsl_hostname": hostname,
            },
        )
        return True

    def _snapshot_pull_state(self) -> typing.Optional[_PullSnapshot]:
        """Capture strong references needed by a pull before entering a worker."""
        state = self._state
        if state.inlet is None or state.clock_sync is None or state.msg_template is None:
            return None

        settings = self.settings
        return _PullSnapshot(
            inlet=state.inlet,
            fetch_buffer=state.fetch_buffer,
            clock_sync=state.clock_sync,
            msg_template=state.msg_template,
            max_pull_samples=settings.max_pull_samples,
            use_arrival_time=settings.use_arrival_time,
            use_lsl_clock=settings.use_lsl_clock,
            nominal_srate=settings.info.nominal_srate,
        )

    def _pull(self, snapshot: _PullSnapshot, timeout: float = 0.0) -> typing.Optional[AxisArray]:
        """Pull available data from the inlet.

        ``timeout`` is the maximum wait for the first sample. ``min_samples=1``
        then makes pylsl drain whatever else is immediately available, up to the
        configured cap, without waiting for the remainder of the chunk. Run via
        ``asyncio.to_thread`` so the event-loop thread remains free to service
        co-located units while liblsl waits.
        """
        inlet = snapshot.inlet
        fetch_buffer = snapshot.fetch_buffer
        cap = snapshot.max_pull_samples
        try:
            if fetch_buffer is not None:
                buf_samples = fetch_buffer.shape[0]
                samples, timestamps = inlet.pull_chunk(
                    timeout=timeout,
                    max_samples=min(buf_samples, cap) if cap else buf_samples,
                    dest_obj=fetch_buffer,
                    min_samples=1,
                )
            elif cap:
                samples, timestamps = inlet.pull_chunk(
                    timeout=timeout,
                    max_samples=cap,
                    min_samples=1,
                )
                samples = np.array(samples)
            else:
                samples, timestamps = inlet.pull_chunk(timeout=timeout, min_samples=1)
                samples = np.array(samples)
        except pylsl.util.LostError:
            # Terminal for this connection: liblsl only raises this once the
            # stream is gone for good (`recover=False`), and every later pull
            # raises too. Flag it for `_produce` to act on rather than tearing
            # down here -- this runs on a worker thread.
            #
            # Nothing is salvageable at this point. liblsl checks the lost state
            # before draining, so buffered samples are unreachable even though
            # `samples_available()` still counts them.
            if self._state.inlet is inlet:
                self._lost = True
                if not self._logged_lost:
                    self._logged_lost = True
                    ez.logger.warning(
                        "LSL inlet lost the stream %r; will re-resolve.",
                        snapshot.msg_template.key,
                    )
            return None
        except Exception:
            # Some other failure -- a closed handle, a malformed chunk. Not known
            # to be terminal, so keep pulling. Log once per connection; stay quiet
            # if this inlet is no longer the live one, which is shutdown or reset
            # racing an in-flight pull.
            if not self._logged_lost and self._state.inlet is inlet:
                self._logged_lost = True
                ez.logger.warning(
                    "LSL inlet pull failed for %r.",
                    snapshot.msg_template.key,
                    exc_info=True,
                )
            return None

        if not len(timestamps):
            return None

        data = fetch_buffer[: len(timestamps)].copy() if samples is None else samples

        # `timestamps` is currently in the LSL clock stamped by the sender.
        if snapshot.use_arrival_time:
            # Drop the sender stamps; use "now". Useful when playing back old XDF files.
            timestamps = time.monotonic() - (timestamps - timestamps[0])
            if snapshot.use_lsl_clock:
                timestamps = snapshot.clock_sync.system2lsl(timestamps)
        elif not snapshot.use_lsl_clock:
            # Keep the sender clock but convert to system time.
            timestamps = snapshot.clock_sync.lsl2system(timestamps)

        if snapshot.nominal_srate <= 0.0:
            # Irregular rate stream uses CoordinateAxis for time so each sample has a timestamp.
            out_time_ax = replace(
                snapshot.msg_template.axes["time"],
                data=np.array(timestamps),
            )
        else:
            # Regular rate uses a LinearAxis for time so we only need the time of the first sample.
            out_time_ax = replace(snapshot.msg_template.axes["time"], offset=timestamps[0])

        out_msg = replace(
            snapshot.msg_template,
            data=data,
            axes={
                **snapshot.msg_template.axes,
                "time": out_time_ax,
            },
        )
        return out_msg

    async def _apull(self, timeout: float = 0.0) -> typing.Optional[AxisArray]:
        """Pull on a worker and discard data from an inlet replaced meanwhile."""
        snapshot = self._snapshot_pull_state()
        if snapshot is None:
            return None

        result = await asyncio.to_thread(self._pull, snapshot, timeout)

        # Reset and shutdown invalidate the live inlet. The snapshot keeps the
        # old inlet and its buffers alive until the worker has safely returned,
        # but data from that old connection must not be published.
        if self._state.inlet is not snapshot.inlet:
            return None
        return result

    async def _produce(self) -> typing.Optional[AxisArray]:
        if self._lost:
            # `_reset_state` already drops the inlet, its buffers and template,
            # and rebuilds the ContinuousResolver that `_setup_after_open`
            # destroyed -- exactly the teardown a reconnect needs. Requesting a
            # reset runs it at the top of the next `__acall__`, on this thread.
            # The old inlet is released rather than closed, so an in-flight pull
            # holding it via its snapshot stays valid until it returns.
            self._lost = False
            self._reconnect_source_id = self._last_source_id or None
            self._reconnect_deadline = time.monotonic() + self.settings.reconnect_grace_dur
            self._request_reset()
            return None

        if self._state.inlet is None:
            await asyncio.to_thread(self._try_connect)
            if self._state.inlet is None:
                # Said once, on the first failed attempt. This line without a
                # later "connected" line is the diagnosis.
                if not self._logged_searching:
                    self._logged_searching = True
                    ez.logger.info(
                        "LSL inlet found no stream matching %s; still looking.",
                        _describe_target(self.settings.info),
                    )
                await asyncio.sleep(0.01)
                return None

        # Update clock sync if its rate limiter has expired.
        if self._state.clock_sync is not None:
            await self._state.clock_sync.arun_once()

        # Re-check after the await — shutdown may have closed the inlet.
        if self._state.inlet is None:
            return None

        if not getattr(self, "_warmed_up", False):
            result = await self._apull(1.0)
            if result is not None:
                self._warmed_up = True
            return result
        # Steady state: block in liblsl on a worker thread up to `pull_timeout`.
        # min_samples=1 returns as soon as data arrives and then drains whatever
        # is available, so a large timeout only caps the idle wait.
        return await self._apull(self.settings.pull_timeout)

    def shutdown(self) -> None:
        # Invalidate in-flight pulls and release the live state. A pull snapshot
        # retains these objects until its bounded wait safely completes, after
        # which `_apull` discards its stale result.
        self._state.inlet = None
        self._state.msg_template = None
        self._state.fetch_buffer = None
        # ClockSync is a singleton shared across all LSL units in the process.
        # Don't stop() it — just drop our reference.
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

    def shutdown(self) -> None:
        if hasattr(self, "producer") and self.producer is not None:
            self.producer.shutdown()
