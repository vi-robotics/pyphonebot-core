#!/usr/bin/env python3

import numpy as np
import logging

from phonebot.core.common.math.transform import Transform, Rotation, Position


def norm(x, out=None):
    """ Norm of x about the last axis. """
    return np.linalg.norm(x, axis=-1, keepdims=True)


def normalize(x):
    """ Normalize the quantity such that its magnitude is 1 at the last axis. """
    return x / norm(x)


def lerp(a, b, w):
    """
    Linear interpoation between a and b with weight w.
    a @ w = 0,  b @ w=1.
    """
    return a + (b - a) * w


def alerp(a, b, w):
    """
    Interpolate between angles.
    """
    return a + adiff(b, a) * w


def slerp(q0, q1, w, eps=np.finfo(float).eps):
    """
    Quaternion-based slerp(spherical linear interpolation)
    """

    # Sanitize input.
    q0 = normalize(q0)
    q1 = normalize(q1)

    # Handle fast exist cases.
    if w == 0.0:
        return q0
    elif w == 1.0:
        return q1

    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < eps:
        return q0
    if d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0  # != q1.inverse()
    d = np.clip(d, 0.0, 1.0)
    angle = np.arccos(d)
    if abs(angle) < eps:
        return q0
    isin = 1.0 / np.sin(angle)
    q0 *= np.sin((1.0 - w) * angle) * isin
    q1 *= np.sin(w * angle) * isin
    q0 += q1
    return q0


def tlerp(a: Transform, b: Transform, w: float = 0.5) -> Transform:
    """
    Interpolate transforms.
    """
    position = lerp(a.position, b.position, w)
    rotation = slerp(a.rotation, b.rotation, w)
    return Transform(position, rotation)


def se3_from_SE3(T: Transform):
    """
    SE3 logarithm.
    NOTE(yycho0108): Does not handle small angles correctly right now.
    """
    # SO3
    r = T.rotation.to_axis_angle()
    axis = r[..., :3]
    # NOTE(yycho0108): selecting shortest path here (anorm)
    angle = anorm(r[..., -1])
    rlog = axis * angle
    # is_small = np.less_equal(np.abs(r[...,-1]), np.finfo(r).eps)

    # SE3
    w = rlog
    p = T.position
    t, t2 = angle, angle*angle
    st, ct = np.sin(angle), np.cos(angle)
    alpha = t * st / (2.0 * (1.0 - ct))
    beta = 1.0 / t2 - st / (2.0 * t * (1.0 - ct))
    plog = (alpha * p - 0.5 * np.cross(w, p) + beta * np.dot(w, p)*w)

    return np.concatenate([plog, rlog], axis=-1)


def SE3_from_se3(T):
    """
    se3 exponential.
    NOTE(yycho0108): Does not handle small angles correctly right now.
    """
    v = T[..., :3]
    w = T[..., 3:]

    t2 = np.inner(w, w)
    t = np.sqrt(t2)
    rotation = Rotation.from_axis_angle(
        np.concatenate([w/t, t[..., None]], axis=-1))

    ct, st = np.cos(t), np.sin(t)
    awxv = (1.0 - ct) / t2
    av = st / t
    aw = (1.0 - av) / t2
    position = av*v + aw * np.dot(w, v)*w + awxv * np.cross(w, v)
    return Transform(position, rotation)


def tlerp_geodesic(source: Transform, target: Transform, weight: float):
    """
    Geodesic interpolation.
    TODO(yycho0108): consider using this instead of plain tlerp?
    """
    delta = target * source.inverse()
    return SE3_from_se3(weight * se3_from_SE3(delta)) * source


def anorm(x):
    """
    Normalize angle from -pi ~ pi.
    """
    x = np.asarray(x)
    return (x + np.pi) % (2 * np.pi) - np.pi


def adiff(a, b):
    """
    Angular difference accounting for angle wrap.
    """
    return anorm(a - b)


def rotation_matrix_2d(x):
    """
    Two-dimensional rotation matrix about z axis.
    TODO(yycho0108): return phonebot.core.common.math.Rotation instead.
    """
    c, s = np.cos(x), np.sin(x)
    shape = np.shape(x)[:-1] + (2, 2)
    return np.stack([c, -s, s, c], axis=-1).reshape(shape)


def skew(x):
    """
    Skew-symmetric matrix from vector x.
    """
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, x[0]],
                     [x[1], x[0], 0]])


def rotation_between_vectors(a, b, out=None):
    """
    Rotation quaternion that aligns a to b.
    @see https://stackoverflow.com/a/1171995 
    TODO(yycho0108): return phonebot.core.common.math.Rotation instead.
    """

    # Normalize input just in case.
    a = normalize(a)
    b = normalize(b)

    xyz = np.cross(a, b)
    w = 1.0 + np.dot(a, b)
    if out is None:
        out = np.empty(shape=np.shape(a)[:-1] + (4,))
    out[..., :3] = xyz
    out[..., 3:] = w.reshape(out[..., 3:].shape)
    return normalize(out)
