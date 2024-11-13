import asyncio

import numpy as np

from ezmsg.lsl.util import ClockSync, collect_timestamp_pairs


def test_clock_sync():
    tol = 10e-3  # 1 msec

    # Run a few updates to get a stable estimate.
    clock_sync = ClockSync()
    asyncio.run(clock_sync.update(force=True, burst=1000))
    asyncio.run(clock_sync.update(force=True, burst=10))

    offsets = []
    for _ in range(10):
        xs, ys = asyncio.run(collect_timestamp_pairs(100))
        offsets.append(np.mean(ys - xs))

    est_diff = np.abs(np.mean(offsets) - clock_sync.offset)
    print(est_diff)

    assert est_diff < tol
