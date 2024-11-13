import asyncio
import time
import typing

import numpy as np
import numpy.typing as npt
import pylsl


async def collect_timestamp_pairs(
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
        await asyncio.sleep(0.001)
    return np.array(xs), np.array(ys)


class ClockSync:
    def __init__(self, alpha: float = 0.1, min_interval: float = 0.5):
        self.alpha = alpha
        self.min_interval = min_interval

        self._offset: typing.Optional[float] = None
        self._last_update = 0.0
        self.count = 0

    @property
    def offset(self) -> float:
        return self._offset

    @property
    def last_updated(self) -> float:
        return self._last_update

    async def update(self, force: bool = False, burst: int = 4) -> None:
        """
        Update the clock offset estimate. This should be called ~regularly in a long-running
        asynchronous task.

        Args:
            force: Whether to force an update even if the minimum interval hasn't passed.
            burst: The number of pairs to collect for this update.
        """
        dur_since_last = time.time() - self._last_update
        dur_until_next = self.min_interval - dur_since_last
        if force or dur_until_next <= 0:
            xs, ys = await collect_timestamp_pairs(burst)
            self.count += burst
            if burst > 0:
                self._last_update = ys[-1]
            offset = np.mean(ys - xs)
            if self.offset is not None:
                # Do exponential smoothing update.
                self._offset = (1 - self.alpha) * self._offset + self.alpha * offset
            else:
                # First iteration. No smoothing.
                self._offset = offset
        else:
            await asyncio.sleep(dur_until_next)

    @typing.overload
    def lsl2system(self, lsl_timestamp: float) -> float: ...
    @typing.overload
    def lsl2system(self, lsl_timestamp: npt.NDArray[float]) -> npt.NDArray[float]: ...
    def lsl2system(self, lsl_timestamp):
        # offset = system - lsl --> system = lsl + offset
        return lsl_timestamp + self._offset

    @typing.overload
    def system2lsl(self, system_timestamp: float) -> float: ...
    @typing.overload
    def system2lsl(
        self, system_timestamp: npt.NDArray[float]
    ) -> npt.NDArray[float]: ...
    def system2lsl(self, system_timestamp):
        # offset = system - lsl --> lsl = system - offset
        return system_timestamp - self._offset
