import asyncio
import threading
import time
import typing

import numpy as np
import numpy.typing as npt
import pylsl


def _sample_clock_pair(i: int) -> typing.Tuple[float, float]:
    """Sample one LSL/system clock pair, alternating order to reduce bias."""
    if i % 2:
        y, x = time.monotonic(), pylsl.local_clock()
    else:
        x, y = pylsl.local_clock(), time.monotonic()
    return x, y


def collect_timestamp_pairs(
    npairs: int = 4,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(npairs):
        x, y = _sample_clock_pair(i)
        xs.append(x)
        ys.append(y)
        time.sleep(0.001)  # Usually sleeps more than 1 msec.
    return np.array(xs), np.array(ys)


async def acollect_timestamp_pairs(
    npairs: int = 4,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(npairs):
        x, y = _sample_clock_pair(i)
        xs.append(x)
        ys.append(y)
        await asyncio.sleep(0)
    return np.array(xs), np.array(ys)


class ClockSync:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, alpha: float = 0.1, min_interval: float = 0.1, run_thread: bool = True):
        if not hasattr(self, "_initialized"):
            self._alpha = alpha
            self._interval = min_interval
            self._initialized = True
            self._last_time = time.monotonic() - 1e9
            self._running = False
            self._thread: typing.Optional[threading.Thread] = None
            # Do first burst so we have a real offset even before the thread starts.
            xs, ys = collect_timestamp_pairs(100)
            self._offset: float = np.mean(ys - xs)

            if run_thread:
                self.start()

    def _update_offset(self, xs: np.ndarray, ys: np.ndarray) -> None:
        offset = np.mean(ys - xs)
        self._offset = (1 - self._alpha) * self._offset + self._alpha * offset
        self._last_time = time.monotonic()

    def _should_update(self, force: bool = False) -> bool:
        return force or (time.monotonic() - self._last_time) > self._interval

    def run_once(self, n: int = 4, force: bool = False):
        if self._should_update(force):
            self._update_offset(*collect_timestamp_pairs(n))

    async def arun_once(self, n: int = 4, force: bool = False):
        if self._should_update(force):
            self._update_offset(*await acollect_timestamp_pairs(n))

    def _run(self):
        while self._running:
            time.sleep(self._interval)
            self.run_once(4, True)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._running = False

    @property
    def offset(self) -> float:
        with self._lock:
            return self._offset

    @typing.overload
    def lsl2system(self, lsl_timestamp: float) -> float: ...

    @typing.overload
    def lsl2system(self, lsl_timestamp: npt.NDArray[float]) -> npt.NDArray[float]: ...

    def lsl2system(self, lsl_timestamp):
        # offset = system - lsl --> system = lsl + offset
        with self._lock:
            return lsl_timestamp + self._offset

    @typing.overload
    def system2lsl(self, system_timestamp: float) -> float: ...

    @typing.overload
    def system2lsl(self, system_timestamp: npt.NDArray[float]) -> npt.NDArray[float]: ...

    def system2lsl(self, system_timestamp):
        # offset = system - lsl --> lsl = system - offset
        with self._lock:
            return system_timestamp - self._offset
