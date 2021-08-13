#!/usr/bin/env python3
from typing import Union

import numpy as np
from numbers import Number
import logging

from phonebot.core.common.math.transform import Transform, Rotation, Position


def norm(x: np.ndarray):
    """Norm of x about the last axis.

    Args:
        x (np.ndarray): An ndarray

    Returns:
        np.ndarray: the norm of the array along the last axis.
    """
    return np.linalg.norm(x, axis=-1, keepdims=True)


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize the quantity such that its magnitude is 1 at the last axis.

    Args:
        x (np.ndarray): normalize the array using the norm along the last axis.

    Returns:
        np.ndarray: the normalized array
    """
    return x / norm(x)


def lerp(a: Union[np.ndarray, Number],
         b: Union[np.ndarray, Number],
         weight: Union[np.ndarray, Number]) -> Union[np.ndarray, Number]:
    """Linear interpoation between a and b with weight. Interpolant is a when
    weight = 0, and b when weight = 1.    


    Args:
        a (Union[np.ndarray, Number]): Start element to lerp
        b (Union[np.ndarray, Number]): End element to lerp
        weight (Union[np.ndarray, Number]): Weight value or array of values
            between 0 and 1.

    Returns:
        Union[np.ndarray, Number]: The linearly interpolated number or array.
    """
    return a + (b - a) * weight


def alerp(a: Union[np.ndarray, Number],
          b: Union[np.ndarray, Number],
          weight: Union[np.ndarray, Number]) -> Union[np.ndarray, Number]:
    """Interpolate between angles.

    Args:
        a (Union[np.ndarray, Number]): Start angle to alerp.
        b (Union[np.ndarray, Number]): End angle to alerp. 
        weight (Union[np.ndarray, Number]): Weight value or array of values
            between 0 and 1.

    Returns:
        Union[np.ndarray, Number]: The linearly interpolated angle or array.
    """
    return a + adiff(b, a) * weight


def slerp(q0: np.ndarray, q1: np.ndarray,
          weight: float, eps=np.finfo(float).eps):
    """Quaternion-based slerp (spherical linear interpolation).


    Args:
        q0 (np.ndarray): An array with shape (..., 4) representing the start
            quaternions.
        q1 (np.ndarray): An array with shape (..., 4) representing the end
            quaternions.
        weight (float): A value from 0 to 1 represnting the slerp weight (0 is
            q0 and 1 is q1, values in between are rotations between).
        eps ([type], optional): Machine epsilon to handle "close" quaternions.
            Defaults to np.finfo(float).eps.

    Returns:
        np.ndarray: The slerped quaternion of same shape as q0 and q1 (..., 4).
    """

    # Sanitize input.
    q0 = normalize(q0)
    q1 = normalize(q1)

    # Handle fast exist cases.
    if weight == 0.0:
        return q0
    elif weight == 1.0:
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
    q0 *= np.sin((1.0 - weight) * angle) * isin
    q1 *= np.sin(weight * angle) * isin
    q0 += q1
    return q0


def tlerp(a: Transform, b: Transform, w: float = 0.5) -> Transform:
    """Interpolate transforms, using lerp for translation and slerp for
    for rotation.

    Args:
        a (Transform): The start transform.
        b (Transform): The end transform.
        w (float, optional): The weight (0 is a, 1 is b, values between are
            transforms between). Defaults to 0.5.

    Returns:
        Transform: the resulting Transform
    """
    position = lerp(a.position, b.position, w)
    rotation = slerp(a.rotation, b.rotation, w)
    return Transform(position, rotation)


def se3_from_SE3(T: Transform) -> np.ndarray:
    """Special Euclidean Group logarithmic map.
    NOTE(yycho0108): Does not handle small angles correctly right now.

    Args:
        T (Transform): transform to take the logarithm of.

    Returns:
        np.ndarray: The logarithmic map.
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
    t, t2 = angle, angle * angle
    st, ct = np.sin(angle), np.cos(angle)
    alpha = t * st / (2.0 * (1.0 - ct))
    beta = 1.0 / t2 - st / (2.0 * t * (1.0 - ct))
    plog = (alpha * p - 0.5 * np.cross(w, p) + beta * np.dot(w, p) * w)

    return np.concatenate([plog, rlog], axis=-1)


def SE3_from_se3(T: Transform) -> Transform:
    """Special Euclidean Group exponential map.

    NOTE(yycho0108): Does not handle small angles correctly right now.

    Args:
        T (Transform): the transform to take the exponential map of.

    Returns:
        Transform: the exponent of the transform.
    """
    v = T[..., :3]
    w = T[..., 3:]

    t2 = np.inner(w, w)
    t = np.sqrt(t2)
    rotation = Rotation.from_axis_angle(
        np.concatenate([w / t, t[..., None]], axis=-1))

    ct, st = np.cos(t), np.sin(t)
    awxv = (1.0 - ct) / t2
    av = st / t
    aw = (1.0 - av) / t2
    position = av * v + aw * np.dot(w, v) * w + awxv * np.cross(w, v)
    return Transform(position, rotation)


def tlerp_geodesic(source: Transform, target: Transform, weight: float
                   ) -> Transform:
    """Geodesic interpolation.

    TODO(yycho0108): consider using this instead of plain tlerp?

    Args:
        source (Transform): The start transform.
        target (Transform): The end transform.
        weight (float): A value between 0 and 1 (0 is source, 1 is target).

    Returns:
        Transform: the interpolated Transform.
    """
    delta = target * source.inverse()
    return SE3_from_se3(weight * se3_from_SE3(delta)) * source


def anorm(x: np.ndarray) -> np.ndarray:
    """Normalize angle from -pi ~ pi.

    Args:
        x (np.ndarray): the input array of radians

    Returns:
        np.ndarray: the resulting normalized array.
    """
    x = np.asarray(x)
    return (x + np.pi) % (2 * np.pi) - np.pi


def adiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angular difference accounting for angle wrap.

    Args:
        a (np.ndarray): the first angle
        b (np.ndarray): the second angle

    Returns:
        np.ndarray: the difference between the angles.
    """
    return anorm(a - b)


def rotation_matrix_2d(x: np.ndarray) -> np.ndarray:
    """Two-dimensional rotation matrix about z axis for an angle x.
    TODO(yycho0108): return phonebot.core.common.math.Rotation instead.


    Args:
        x (np.ndarray): the array of angles (in radians) the rotation matrices
            should rotate.

    Returns:
        np.ndarray: An array of shape (*x.shape, 2, 2)
    """
    c, s = np.cos(x), np.sin(x)
    shape = np.shape(x)[:-1] + (2, 2)
    return np.stack([c, -s, s, c], axis=-1).reshape(shape)


def skew(x: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from vector x.

    Args:
        x (np.ndarray): A vector of shape (3,)

    Returns:
        np.ndarray: A 3x3 skew symmetric matrix created by x.
    """
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, x[0]],
                     [x[1], x[0], 0]])


def rotation_between_vectors(a: np.ndarray, b: np.ndarray,
                             out: np.ndarray = None) -> np.ndarray:
    """Rotation quaternion that aligns a to b.

    @see https://stackoverflow.com/a/1171995 
    TODO(yycho0108): return phonebot.core.common.math.Rotation instead.

    Args:
        a (np.ndarray): An array of shape (..., 3)
        b (np.ndarray): An array the same shape as `a`
        out (np.ndarray, optional): Optional output array of shape `a`. Defaults
            to None.

    Returns:
        np.ndarray: The resulting quaternion array of shape (*a.shape, 4)
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
