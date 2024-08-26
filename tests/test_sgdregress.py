import time

import numpy as np
import pylsl

from ezmsg.lsl.util import LinearRegressionSGD


def test_sgdregress():
    n_samples = 1000
    flip = True
    xs = []
    ys = []
    for ix in range(n_samples):
        if flip:
            y, x = time.time(), pylsl.local_clock()
        else:
            x, y = pylsl.local_clock(), time.time()
        flip = not flip
        time.sleep(0.001)
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)[:, None]
    ys = np.array(ys)

    model = LinearRegressionSGD(learning_rate=0.1)
    ypreds = []
    for ix in range(xs.shape[0]):
        model.partial_fit(xs[ix:ix+1], ys[ix:ix+1])
        ypreds.append(model.predict(xs[ix:ix+1]))
    ypreds = np.concatenate(ypreds)
    assert np.max(np.abs(ypreds - ys)) < 1e-4
