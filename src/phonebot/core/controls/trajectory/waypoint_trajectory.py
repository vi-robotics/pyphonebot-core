#!/usr/bin/env python3

__all__ = ['WaypointTrajectory']

import numpy as np

from phonebot.core.common.logger import get_default_logger
from phonebot.core.controls.trajectory.trajectory import Trajectory
from phonebot.core.common.math.utils import lerp

logger = get_default_logger()


# TODO(yycho0108): Rename -> CyclicWaypointTrajectory
class WaypointTrajectory(Trajectory):
    """
    Piecewise upsampled waypoint trajectory.

    NOTE(yycho0108) :
        Currently, the input times and values must be supplied according to the following specification:
        1. times[-1] - times[0] == period
        2. values[-1] == values[0]
        i.e. the input values must be specified in a cyclic manner.
    """

    def __init__(self, times, values, period):
        times, values, period = WaypointTrajectory.validate(
            times, values, period)
        self.times_ = times
        self.values_ = values
        self.period_ = period
        self.length_ = len(times)

    @staticmethod
    def validate(times, values, period):
        # Standardize arguments.
        if times is None:
            times = np.linspace(0, period, len(values))
        if period is None:
            period = times[-1] - times[0]

        # Validate arguments.
        if len(times) != len(values):
            raise ValueError(
                "Same number of sample times and values must be supplied")
        if not np.allclose(times[-1] - times[0], period):
            raise ValueError(
                "Supplied argument times must meet periodic constraint")
        if not np.allclose(values[-1], values[0]):
            logger.warn("Value endpoints must be equivalent : {} != {}"
                        .format(values[0], values[-1]))

        return times, values, period

    def evaluate(self, stamp):
        # Wrap stamp with period.
        stamp = stamp % self.period_

        # Search two endpoints indices.
        i0 = (np.searchsorted(self.times_, stamp, side='right') - 1)
        i1 = (i0 + 1) % self.length_

        # Extract values and corresponding weights.
        v0, v1 = self.values_[i0], self.values_[i1]
        t0, t1 = self.times_[i0], self.times_[i1]
        dt = (t1 - t0) % self.period_

        # Apply linear interpolation and return result.
        # TODO(yycho0108): consider alternative interpolation schemes.
        v = lerp(v0, v1, (stamp - t0) / dt)
        return v


def main():
    from matplotlib import pyplot as plt
    times = np.linspace(0, 4, num=10, endpoint=True)
    values = np.sin(np.linspace(-np.pi, np.pi, num=10, endpoint=True))
    trajectory = WaypointTrajectory(times, values, period=4.0)
    plt.plot(times, values, 'o', label='sample')
    t_interp = np.linspace(0, 4, num=100)
    plt.plot(t_interp, [trajectory.evaluate(t)
                        for t in t_interp], 'x--', label='target')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
