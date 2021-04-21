#!/usr/bin/env python3

import sys
import numpy as np
from typing import Tuple

from phonebot.core.controls.trajectory.trajectory import Trajectory
from phonebot.core.common.math.utils import anorm
from phonebot.core.common.math.transform import Position, Rotation, Transform
from phonebot.core.controls.trajectory.parametric_trajectory import *

from phonebot.core.common.config import PhonebotSettings
from phonebot.core.kinematics.workspace import get_workspace, max_rect


def get_elliptical_trajectory(rect: Tuple[float, float, float, float],
                              period: float, offset: float, flip: bool,
                              scale: float = 1.0) -> Trajectory:
    # TODO(yycho0108): remove dependencies on rcx, rcy, ...
    rcx, rcy, rw, rh = rect

    position = Position([rcx, rcy, 0.0])
    rotation = Rotation.from_axis_angle(
        [0, 1, 0, np.pi]) if flip else Rotation.identity()
    pose = Transform(position, rotation)

    # Reverse direction if scale < 0.0.
    # NOTE(yycho0108):
    # For `EllipticalKernel`, scale < 0 == 180' rotation,
    # Which is not equivalent to reversing the direction of motion.
    negate = (scale < 0.0)
    scale = np.abs(scale)

    # TODO(ycho): Be a little bit smarter about timing the footing.
    kernels = [
        # ( 2 * np.pi * x / period - offset) % period
        CyclicKernel(scale=(2 * np.pi / period), offset=offset),
        EllipticalKernel(pose, dimensions=np.asarray(
            [rw * scale * 0.5, rh * scale * 0.5, 0.0]))
    ]
    if negate:
        kernels.insert(-1, NegativeKernel())

    # NOTE(yycho0108): Pose here is specified with respect to leg origin.
    return ParametricTrajectory(kernels)


def is_cyclic(data: np.ndarray, eps=None):
    if eps is None:
        eps = np.finfo(data.dtype).eps
    delta = data[..., -1] - data[..., 0]
    return np.all(np.abs(delta) < eps)


def freq_from_time(points: np.ndarray, resolution: int = 3,
                   num_samples: int = 50, times: np.ndarray = None):
    # If spacing information is not available, compute time based on distance
    # between points.
    if times is None:
        diffs = np.linalg.norm(
            np.diff(points, axis=0, prepend=points[..., 0]), axis=-1)
        times = np.cumsum(diffs)

    # Validate that the input points is cyclic.
    if not is_cyclic(points):
        raise ValueError('points must be cyclic.')

    # Tmax == trajecotry period.
    tmax = times[-1]


def parametrize_cyclic_trajectory(points, resolution=3, num=50, times=None):
    """
    Simplify a periodic trajectory to a parametric form.
    (Concise frequency/ampliutude-based representation)
    TODO(yycho0108): Support multidimensional input (currently hardcoded to 2)

    Args:
        points: the points on which to apply simplification (must be cyclic, p[0]==p[-1]).
        resolution: the number of frequency components to utilize.
        num: size of uniform samples that the trajectory will consist of.
    Returns:
        p_out
    """

    # If spacing information is not available, compute it based on distance
    # between points.
    if times is None:
        diffs = np.linalg.norm(np.diff(points, axis=0), axis=-1)
        times = np.cumsum(np.insert(diffs, 0, 0.0))
    tmax = times[-1]

    # Generate uniform points along trajectory.
    utimes = np.linspace(0, tmax, num=num)

    # Compute FFT over uniform samples.
    x_u = [np.interp(utimes, times, points[..., i], period=tmax)
           for i in range(points.shape[-1])]
    x_u = np.stack(x_u, axis=-1)
    X_u = np.fft.rfft(x_u, axis=0)

    # Simplify trajectory.
    sample_freq = np.fft.rfftfreq(num, 1.0 / (num - 1))  # cycles/second
    fmag = np.linalg.norm(X_u, axis=-1)
    sel = np.argsort(fmag)[-resolution:]
    msk = np.zeros(fmag.shape, dtype=np.bool)
    msk[0] = 1
    msk[sel] = 1
    if False:
        msk[:] = 1

    # Apply mask.
    X_c = (1.0 / num) * X_u[msk]
    f_c = sample_freq[msk]

    return X_c, f_c


def time_from_freq(coefs, freqs, times):
    """
    Compute irfft from individual components.

    coefs: Complex Fourier coefficient.
    freqs: Frequency. (where period = 1)
    times: Times at which to evaluate the fft.
    """
    phase = np.angle(coefs[None, :])
    value = np.abs(coefs[None, :]) * np.cos(phase +
                                            freqs[None, :] * times[:, None])
    return value[..., 0] + (2.0 * value[..., 1:].sum(axis=-1))


def mutate_fourier(coef, freq):
    # FIXME(ycho): This function does not belong here.
    mag = np.random.normal(
        loc=np.abs(coef),
        scale=0.01)
    ang = np.random.normal(
        loc=np.angle(coef),
        scale=0.0)
    # freq = np.exp(np.random.normal(loc=0, scale=0.1, size=freq.shape)) * freq
    coef = mag * (np.cos(ang) + 1j * np.sin(ang))
    return coef, freq


def generate_fourier_function(
        num_samples: int, num_components: int, num_dimensions: int,
        period: float = 2 * np.pi):
    # Generate sample times at which to evaluate the function.
    times = np.linspace(0, period, num=num_samples, endpoint=True)
    mag = np.random.normal(scale=1.0, size=(num_components, num_dimensions))
    ang = np.random.uniform(low=-np.pi, high=np.pi,
                            size=(num_components, num_dimensions))
    if period <= 0:
        # Generate aperiodic function.
        frq = np.random.uniform(low=-10 * np.pi, high=10 * np.pi,
                                size=(num_components, num_dimensions))
    else:
        # Generate periodic function.
        frq = (2 * np.pi) * np.random.randint(1, 16, size=(
            num_components, num_dimensions)) / period
    # Values = (num_samples, num_components, num_dimensions)
    values = (mag[None, :, :] * np.cos(ang[None, :, :] +
                                       frq[None, :, :] * times[:, None, None]))
    values = values.sum(axis=1)
    return times, values


def make_angle_continugous(x):
    return np.cumsum(anorm(np.diff(x, prepend=x[..., 0])))


def visualize_parametric_kernels():
    from matplotlib import pyplot as plt
    # configure...
    period = 2.0
    times = np.linspace(-period, period, 256)
    offset = 0.0
    # pose = Transform([0, 0, 0], Rotation.from_quaternion([0, 0, 0, 1]))
    pose = Transform.identity()
    rw = 1.0
    rh = 1.0
    scale = 1.0
    negate = False

    kernels = [
        # ( 2 * np.pi * x / period - offset) % period
        # map x ->  (s*x - o) % p
        CyclicKernel(scale=(2 * np.pi / period), offset=offset),
        # UnaryKernel(lambda x: np.power(x-np.pi, 3) / (np.pi**2)),
        # UnaryKernel(lambda x: x*x / (2 * np.pi)),
        # UnaryKernel(lambda x : 1.0 / (1.0 + np.exp(-x))),
        UnaryKernel(lambda x: x / (2 * np.pi)),
        # UnaryKernel(lambda x : (-2*x*x*x + 3*x*x)),
        # UnaryKernel(lambda x: 6 * x**5 - 15*x**4 + 10*x**3),
        UnaryKernel(lambda x: np.where(
            x < 0.75, 0.5 / 0.75 * x, 0.5 / 0.25 * (x - 0.75) + 0.5)),
        # UnaryKernel(np.sin),
        UnaryKernel(lambda x: x * (2 * np.pi)),

        EllipticalKernel(pose, dimensions=np.asarray(
            [rw * scale * 0.5, rh * scale * 0.5, 0.0]))
    ]
    if negate:
        # kernels = [NegativeKernel()] + kernels
        kernels.insert(-1, NegativeKernel())

    # NOTE(yycho0108): Pose here is specified with respect to leg origin.
    trajectory = ParametricTrajectory(kernels)
    values = trajectory.evaluate(times)
    # values = make_angle_continugous(values)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(times, values, label='values')
    ax.grid()

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(values[..., 0], values[..., 1], label='ellipse')
    ax.set_aspect('equal')
    plt.legend()
    plt.grid()
    plt.show()


def visualize_elliptical_trajectory():
    from matplotlib import pyplot as plt
    config = PhonebotSettings()
    workspace = get_workspace(0.0, config, return_poly=True)
    mr = max_rect(workspace, 4096)
    (x0, y0), (x1, y1) = mr
    mr = (0.5 * (x0 + x1), 0.5 * (y0 + y1),
          abs(x1 - x0), abs(y1 - y0))

    period = 2.0
    for x in [False, True]:
        for y in [-1, +1]:
            t = np.linspace(0.0, 2.0)
            tr = get_elliptical_trajectory(
                mr, period, y * np.pi / 2, x, 0.7).evaluate(t)
            plt.plot(t, tr[..., 1], label='{}{}'.format(x, y))
    plt.legend()
    plt.show()


def visualize_path_simplification():
    from matplotlib import pyplot as plt
    # Generate a random function.
    times, points = generate_fourier_function(1024, 3, 2, 2 * np.pi)
    x = points[..., 0]
    y = points[..., 1]

    # Parametrize into fourier coefficients.
    Xc, fc = parametrize_cyclic_trajectory(
        points, resolution=3, num=len(points), times=times)
    if True:
        Xc, fc = mutate_fourier(Xc, fc)
    xc = Xc[..., 0]
    yc = Xc[..., 1]

    # Compute real-valued inverse dft.
    t = np.linspace(0, 2 * np.pi, num=1024, endpoint=True)
    xo = time_from_freq(xc, fc, t)
    yo = time_from_freq(yc, fc, t)

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(times, points[..., 0], label='in')
    ax.plot(t, np.real(xo), label='out-real')
    ax.plot(t, np.imag(xo), label='out-imag')
    ax.legend()
    ax.grid()
    ax.set_title('X')

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(times, points[..., 1], label='in')
    ax.plot(t, np.real(yo), label='out-real')
    ax.plot(t, np.imag(yo), label='out-imag')
    ax.legend()
    ax.grid()
    ax.set_title('Y')

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x[:5], y[:5], label='in')
    ax.plot(xo[:5], yo[:5], label='out')
    ax.legend()
    ax.grid()

    plt.show()


def main():
    if True:
        visualize_parametric_kernels()
    if False:
        visualize_elliptical_trajectory()
    if False:
        visualize_path_simplification()


if __name__ == '__main__':
    main()
