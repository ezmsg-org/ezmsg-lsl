import threading
import time
import typing

import numpy as np
import numpy.typing as npt
import pylsl


def collect_timestamp_pairs(
    npairs: int = 4,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for _ in range(npairs):
        if _ % 2:
            y, x = time.time(), pylsl.local_clock()
        else:
            x, y = pylsl.local_clock(), time.time()
        xs.append(x)
        ys.append(y)
        time.sleep(0.001)
    return np.array(xs), np.array(ys)


class ClockSync:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, alpha: float = 0.1, min_interval: float = 0.1):
        if not hasattr(self, "_initialized"):
            self._alpha = alpha
            self._interval = min_interval

            # Do first burst so we have a real offset even before the thread starts.
            xs, ys = collect_timestamp_pairs(100)
            self._offset: float = np.mean(ys - xs)

            self._thread = threading.Thread(target=self._run)
            self._thread.daemon = True
            self._initialized = True
            self._running = True
            self._thread.start()

    def _run(self):
        while self._running:
            time.sleep(self._interval)
            xs, ys = collect_timestamp_pairs(4)
            offset = np.mean(ys - xs)
            self._offset = (1 - self._alpha) * self._offset + self._alpha * offset

    @property
    def offset(self) -> float:
        with self._lock:
            return self._offset

    @typing.overload
    def lsl2system(self, lsl_timestamp: float) -> float:
        ...

    @typing.overload
    def lsl2system(self, lsl_timestamp: npt.NDArray[float]) -> npt.NDArray[float]:
        ...

    def lsl2system(self, lsl_timestamp):
        # offset = system - lsl --> system = lsl + offset
        with self._lock:
            return lsl_timestamp + self._offset

    @typing.overload
    def system2lsl(self, system_timestamp: float) -> float:
        ...

    @typing.overload
    def system2lsl(
            self, system_timestamp: npt.NDArray[float]
    ) -> npt.NDArray[float]:
        ...

    def system2lsl(self, system_timestamp):
        # offset = system - lsl --> lsl = system - offset
        with self._lock:
            return system_timestamp - self._offset
