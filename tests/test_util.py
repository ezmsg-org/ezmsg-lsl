import time

import numpy as np

from ezmsg.lsl.util import ClockSync, collect_timestamp_pairs


def test_clock_sync():
    tol = 10e-3  # 1 msec

    clock_sync = ClockSync()
    # Let it run a bit to get a stable estimate.
    time.sleep(1.0)

    offsets = []
    for _ in range(10):
        xs, ys = collect_timestamp_pairs(100)
        offsets.append(np.mean(ys - xs))

    est_diff = np.abs(np.mean(offsets) - clock_sync.offset)
    print(est_diff)

    assert est_diff < tol

    # Assert singleton-ness
    clock_sync2 = ClockSync()
    assert clock_sync is clock_sync2
