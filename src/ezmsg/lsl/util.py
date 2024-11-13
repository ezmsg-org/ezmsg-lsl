import asyncio
import time

import numpy as np
import pylsl


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
            for ix, _ in enumerate(range(burst)):
                if (self.count + ix) % 2:
                    y, x = time.time(), pylsl.local_clock()
                else:
                    x, y = pylsl.local_clock(), time.time()
                # TODO: Use adaptive linear fit instead of simple subtraction.
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

    def lsl2system(self, lsl_timestamp: float) -> float:
        return lsl_timestamp + self.offset
