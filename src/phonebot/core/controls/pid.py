#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PIDSettings:
    """ PID Controller Settings. """
    kp: float
    ki: float
    kd: float

    max_i: float  # windup
    max_u: float  # max effort

    cutoff_freq: float  # used for derivative filter coef


def _filter_coefs(c: float):
    """ Premultiplied butterworth filter coefs """
    s = (1 / (1 + c * c + 1.414 * c))
    w = [1.0, 2.0, 1.0, -(c * c - 1.414 * c + 1), -(-2 * c * c + 2)]
    return np.multiply(w, s)


class PID(object):
    """
    Simple PID class.

    Supported features:

    * Max Windup clamping
    * Smooth derivative (2nd order butterworth)
    * Vectorized operation - (input doesn't need to be scalars)

    Reference:
    https://bitbucket.org/AndyZe/pid/src/master/
    """

    def __init__(self, settings: PIDSettings):
        self.settings = settings

        # ~raw state variables...
        self.error_ = None
        self.error_i_ = None
        self.error_d_ = None

        # filtered (smooth) state.
        self.f_error_ = None
        self.f_error_d_ = None

    @property
    def kp(self):
        return self.settings.kp

    @property
    def ki(self):
        return self.settings.ki

    @property
    def kd(self):
        return self.settings.kd

    def set_gains(self, kp: float, ki: float, kd: float):
        self.settings.kp = kp
        self.settings.ki = ki
        self.settings.kd = kd

    def set_max_i(self, max_i: float):
        self.settings.max_i = max_i

    def reset(self, soft=True):
        if soft:
            if self.error_ is not None:
                self.error_.fill(0.0)
                self.error_i_.fill(0.0)
                self.error_d_.fill(0.0)
                self.f_error_.fill(0.0)
                self.f_error_d_.fill(0.0)
        else:
            self.error_ = None
            self.error_i_ = None
            self.error_d_ = None
            self.f_error_ = None
            self.f_error_d_ = None

    def _allocate(self, shape: Tuple[int, ...]):
        # NOTE(ycho): `3` here is just the buffer length for
        # maintaining a smooth derivative.
        self.error_ = np.zeros((3,) + shape, np.float32)  # 3xN
        self.error_i_ = np.zeros(shape, np.float32)  # N
        self.error_d_ = np.zeros_like(self.error_)  # 3xN
        # Filtered ...
        self.f_error_ = np.zeros_like(self.error_)  # 3xN
        self.f_error_d_ = np.zeros_like(self.error_d_)  # 3xN

    def __call__(self, err: float, dt: float):

        # If this is the first invocation since reset,
        # Configure the controller buffers.
        if self.error_ is None:
            self._allocate(np.shape(err))

        # Set the current error.
        self.error_ = np.roll(self.error_, -1, axis=0)
        self.error_[-1] = err

        # Apply numerical integration and clip the results.
        self.error_i_ += self.error_[-1] * dt
        self.error_i_ = np.clip(
            self.error_i_, -self.settings.max_i, self.settings.max_i,
            out=self.error_i_)

        # Apply (smooth) numerical differentiation.
        t = np.tan((self.settings.cutoff_freq * 2 * np.pi) * 0.5 * dt)
        # FIXME(ycho): Remove hardcoded epsilon (0.01),
        # Or reparametrize filter coefficients to be numerically stable at or
        # near 0 (if applicable).
        if np.abs(t) <= 0.01:
            t = 0.01 * np.sign(t)

        c = 1.0 / t
        k = _filter_coefs(c)
        self.f_error_ = np.roll(self.f_error_, -1, axis=0)
        self.f_error_[-1] = k.dot(np.r_[self.error_, self.f_error_[:2]])

        self.error_d_ = np.roll(self.error_d_, -1, axis=0)
        self.error_d_[-1] = (1.0 / dt) * (self.error_[2] - self.error_[1])

        self.f_error_d_ = np.roll(self.f_error_d_, -1, axis=0)
        self.f_error_d_[-1] = k.dot(np.r_[self.error_d_,
                                          self.f_error_d_[:2]])

        # Collect contributions.
        u_p = self.kp * self.error_[-1]
        u_i = self.ki * self.error_i_
        u_d = self.kd * self.f_error_d_[-1]

        u = np.zeros_like(u_p)
        if self.kp > 0:
            u += u_p
        if self.ki > 0:
            u += u_i
        if np.abs(self.kd) > 0:
            u += u_d

        # Clip output, and return.
        u = np.clip(u, -self.settings.max_u, self.settings.max_u, out=u)
        return u
