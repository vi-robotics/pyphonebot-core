#!/usr/bin/env python3

__all__ = [
    'CyclicKernel',
    'EllipticalKernel',
    'NegativeKernel',
    'UnaryKernel',
    'ConstantKernel',
    'PrintKernel',
    'ParametricTrajectory']

import numpy as np

from phonebot.core.controls.trajectory.trajectory import Trajectory
from phonebot.core.common.math.transform import Transform, Position, Rotation


class UnaryKernel(object):
    def __init__(self, fun):
        self.fun_ = fun

    def __call__(self, value):
        return self.fun_(value)


class PrintKernel(object):
    def __init__(self):
        pass

    def __call__(self, value):
        print(value)
        return value


class ConstantKernel(object):
    def __init__(self, value):
        self.value_ = value

    def __call__(self, value):
        return self.value_


class NegativeKernel(object):
    def __init__(self):
        pass

    def __call__(self, value):
        return -value


class CyclicKernel(object):
    """
    Map a scalar value to a periodic value.
    """

    def __init__(self, offset=0.0, period=2 * np.pi, scale=1.0):
        self.offset_ = offset
        self.period_ = period
        self.scale_ = scale

    def __call__(self, value):
        return (self.scale_ * value - self.offset_) % (self.period_)


class EllipticalKernel(object):
    """
    Maps angle to a corresponding point in an ellipse.
    Note that the ellipse defined here rotates about the +Z axis(0,0,1),
    and @angle=0 points to (1,0,0) (scaled by dimension).
    """

    def __init__(self, pose=Transform.identity(), dimensions=np.ones(3)):
        self.pose_ = pose
        self.dimensions_ = dimensions

    def __call__(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        k0 = np.zeros_like(c)
        point = np.stack([c, s, k0], axis=-1)
        point = point * self.dimensions_
        return self.pose_ * Position(point)


class SequentialKernel(object):
    def __init__(self, kernels):
        self.kernels_ = kernels

    def __call__(self, value):
        for kernel in self.kernels_:
            value = kernel(value)
        return value


class ParametricTrajectory(Trajectory):
    """
    Parametric trajectory based on kernel reduction.
    """

    def __init__(self, kernel):
        if isinstance(kernel, list):
            kernel = SequentialKernel(kernel)
        self.kernel_ = kernel

    def evaluate(self, time):
        return self.kernel_(time)


def main():
    import time
    from phonebot.vis.viewer import PrimitiveViewer

    data_queue, event_queue = PrimitiveViewer.create()

    if False:
        period = 4.0
        trajectory = ParametricTrajectory(
            [CyclicKernel(scale=2 * np.pi / period), EllipticalKernel()])
    else:
        # configure...
        period = 4.0
        times = np.linspace(-period, period)
        offset = 0.0
        # pose = Transform([0, 0, 0], Rotation.from_quaternion([0, 0, 0, 1]))
        pose = Transform.identity()
        rw = 1.0
        rh = 1.0
        scale = 1.0
        negate = True

        kernels = [
            # ( 2 * np.pi * x / period - offset) % period
            CyclicKernel(scale=(2 * np.pi / period), offset=offset),
            EllipticalKernel(pose, dimensions=np.asarray(
                [rw * scale * 0.5, rh * scale * 0.5, 0.0]))
        ]
        if negate:
            kernels = [NegativeKernel()] + kernels
        trajectory = ParametricTrajectory(kernels)

    line = []
    for i in range(100):
        point = trajectory.evaluate(time.time())
        if 0 >= len(line):
            line.extend([point, point])
        else:
            line.extend([line[-1], point])
        linedata = np.stack(line, axis=0)
        if not data_queue.full():
            data_queue.put_nowait({'line': dict(pos=linedata[-32:])})
        time.sleep(0.05)


if __name__ == '__main__':
    main()
