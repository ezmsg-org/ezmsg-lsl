"""
Test signal generators for ezmsg-lsl integration tests.

These are simplified signal generators intended for testing purposes only.
For production use, see ezmsg-simbiophys package.
"""

import asyncio
import time
import typing

import ezmsg.core as ez
import numpy as np
from ezmsg.util.messages.axisarray import AxisArray


class OscillatorSettings(ez.Settings):
    n_time: int = 100
    fs: float = 1000.0
    n_ch: int = 1
    dispatch_rate: float | str | None = None  # Hz, "realtime", "ext_clock", or None
    freq: float = 10.0  # Hz
    amp: float = 1.0
    phase: float = 0.0
    sync: bool = False  # Adjust freq to sync with sampling rate


class Oscillator(ez.Unit):
    """
    Simple oscillator generator for testing.

    Supports three dispatch modes:
    - None: Produce as fast as possible
    - float: Produce at specified rate in Hz
    - "realtime": Produce at realtime rate matching sample rate
    - "ext_clock": Wait for external clock signal on INPUT_SIGNAL
    """

    SETTINGS = OscillatorSettings

    INPUT_SIGNAL = ez.InputStream(ez.Flag)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:
        self._n_sent = 0
        self._t0 = time.time()

        # Calculate synchronized frequency if requested
        self._freq = self.SETTINGS.freq
        if self.SETTINGS.sync:
            period = 1.0 / self.SETTINGS.freq
            mod = round(period * self.SETTINGS.fs)
            self._freq = 1.0 / (mod / self.SETTINGS.fs)

    def _generate_block(self) -> AxisArray:
        """Generate one block of oscillator data."""
        offset = self._n_sent / self.SETTINGS.fs

        # Generate sinusoidal data
        sample_indices = np.arange(self._n_sent, self._n_sent + self.SETTINGS.n_time)
        t = sample_indices / self.SETTINGS.fs
        data = self.SETTINGS.amp * np.sin(2 * np.pi * self._freq * t + self.SETTINGS.phase)
        data = data[:, np.newaxis]
        data = np.tile(data, (1, self.SETTINGS.n_ch))

        result = AxisArray(
            data=data,
            dims=["time", "ch"],
            axes={
                "time": AxisArray.TimeAxis(fs=self.SETTINGS.fs, offset=offset),
                "ch": AxisArray.CoordinateAxis(
                    data=np.array([f"Ch{_}" for _ in range(self.SETTINGS.n_ch)]),
                    dims=["ch"],
                ),
            },
        )

        self._n_sent += self.SETTINGS.n_time
        return result

    @ez.subscriber(INPUT_SIGNAL)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_clock(self, _: ez.Flag) -> typing.AsyncGenerator:
        """Handle external clock signal - only active when dispatch_rate='ext_clock'."""
        if self.SETTINGS.dispatch_rate == "ext_clock":
            yield self.OUTPUT_SIGNAL, self._generate_block()

    @ez.publisher(OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        """Self-timed production - active when dispatch_rate is not 'ext_clock'."""
        # Skip self-timed production if using external clock
        if self.SETTINGS.dispatch_rate == "ext_clock":
            return

        while True:
            # Calculate timing based on dispatch mode
            if self.SETTINGS.dispatch_rate == "realtime":
                # Realtime mode: sleep until wall-clock time matches sample time
                n_next = self._n_sent + self.SETTINGS.n_time
                t_next = self._t0 + n_next / self.SETTINGS.fs
                sleep_time = t_next - time.time()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            elif self.SETTINGS.dispatch_rate is not None:
                # Manual dispatch rate mode
                n_disp = 1 + self._n_sent / self.SETTINGS.n_time
                t_next = self._t0 + n_disp / self.SETTINGS.dispatch_rate
                sleep_time = t_next - time.time()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            yield self.OUTPUT_SIGNAL, self._generate_block()
