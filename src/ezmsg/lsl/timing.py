"""Breaks and jitter in a regularly sampled stream's timestamps.

liblsl can remove timestamp jitter itself (``proc_dejitter``), but its filter has no notion
of a break: when a sender pauses and resumes, the smoothed timestamps carry on as if no time
had passed and take minutes to recover, so the break is invisible and the samples after it
are mis-timed. A threshold that restarts the filter exists from liblsl 1.18, but it is
process-wide configuration, off by default, and cannot be read back.

:class:`StreamTiming` therefore does both jobs here, per inlet, on timestamps pulled without
liblsl's smoothing: it finds the breaks first, then removes jitter from each run between them.
"""

import numpy as np
import numpy.typing as npt


class StreamTiming:
    """Split a stream's chunks at breaks and time each run without jitter.

    A break is an interval between consecutive samples whose size exceeds both
    ``break_seconds`` and ``break_samples`` sample periods: pyxdf's rule and defaults, so a
    live stream and its recording agree on where the breaks are.

    Jitter is removed as liblsl does it: a straight line fitted to timestamp against sample
    count, weighting each sample half as much for every ``halftime`` seconds of age. The fit
    is done a chunk at a time and restarts at a break.

    Args:
        srate: The stream's nominal sampling rate, in Hz.
        break_seconds: See above. Non-positive (with ``break_samples``) disables break detection.
        break_samples: See above.
        dejitter: Whether to remove jitter. If False, each run is timed by its first raw timestamp.
        halftime: Age, in seconds, at which a sample's weight in the fit has halved.
    """

    def __init__(
        self,
        srate: float,
        break_seconds: float = 1.0,
        break_samples: float = 500,
        dejitter: bool = True,
        halftime: float = 90.0,
    ):
        self.period = 1.0 / srate
        threshold = max(break_seconds, break_samples * self.period)
        self.threshold = threshold if threshold > 0 else np.inf
        self.dejitter = dejitter
        self._lam = 2.0 ** (-1.0 / (srate * halftime))
        self.last: float | None = None
        """The last raw timestamp seen."""
        self._restart()

    def _restart(self) -> None:
        # Weighted sums over the samples seen, for sample count u and time y (relative to
        # `_t_ref`), with u measured from the first sample of the chunk about to be added.
        self._s0 = self._s1 = self._s2 = self._y0 = self._y1 = 0.0
        self._t_ref: float | None = None

    def split(self, timestamps: npt.ArrayLike) -> list[tuple[int, int, float, bool]]:
        """Take the next chunk's raw timestamps, in order.

        Returns ``(start, stop, time, is_break)`` for each run in the chunk: the samples
        ``[start, stop)``, the time of the first of them, and whether a break precedes it. A
        chunk with no break in it is one run.
        """
        ts = np.asarray(timestamps, dtype=float)
        if ts.size == 0:
            return []
        prev = ts[0] - self.period if self.last is None else self.last
        breaks = set(np.flatnonzero(np.abs(np.diff(ts, prepend=prev)) > self.threshold).tolist())
        self.last = float(ts[-1])
        bounds = sorted({0, *breaks, ts.size})
        runs = []
        for start, stop in zip(bounds[:-1], bounds[1:]):
            is_break = start in breaks
            if is_break:
                self._restart()
            runs.append((start, stop, self._fit(ts[start:stop]) if self.dejitter else float(ts[start]), is_break))
        return runs

    def _fit(self, ts: npt.NDArray) -> float:
        """Add a run of timestamps to the fit; return the fitted time of its first sample."""
        if self._t_ref is None:
            self._t_ref = float(ts[0])
        n = ts.size
        u = np.arange(n, dtype=float)
        y = ts - self._t_ref
        w = self._lam ** (n - 1 - u)  # the newest sample has weight 1
        decay = self._lam**n
        s0 = decay * self._s0 + w.sum()
        s1 = decay * self._s1 + (w * u).sum()
        s2 = decay * self._s2 + (w * u * u).sum()
        y0 = decay * self._y0 + (w * y).sum()
        y1 = decay * self._y1 + (w * u * y).sum()

        det = s0 * s2 - s1 * s1
        if det > 1e-9 * s0 * s2:
            slope = (s0 * y1 - s1 * y0) / det
        else:
            # One sample, or too few to say: the nominal rate is all there is to go on.
            slope = self.period
        fitted = self._t_ref + (y0 - slope * s1) / s0  # at u = 0, this run's first sample

        # Re-measure u from the first sample of the next chunk (u -> u - n), which keeps the
        # sums small however long the stream runs.
        self._s0, self._y0 = s0, y0
        self._s1 = s1 - n * s0
        self._s2 = s2 - 2 * n * s1 + n * n * s0
        self._y1 = y1 - n * y0
        return float(fitted)
