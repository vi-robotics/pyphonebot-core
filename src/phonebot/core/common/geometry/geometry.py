#!/usr/bin/env python3
from typing import Tuple, Optional
import numpy as np
from phonebot.core.common.math.utils import normalize


def parallel_component(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Project vector a onto b

    Args:
        a (np.ndarray): A vector of length #N to project
        b (np.ndarray): A vector of length #N to project onto.

    Returns:
        np.ndarray: The vector a projected onto vector b.
    """

    nb = normalize(b)
    return nb * nb.dot(a)


def perpendicular_component(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Perpendicular component of a with respect to b 

    Args:
        a (np.ndarray): A vector of length #N to extract the perpendicular
            component to b of.
        b (np.ndarray): A vector of length #N.

    Returns:
        np.ndarray: A vector of length #N which is the perpendicular component
            of a with respect to b.
    """
    return a - parallel_component(a, b)


def plane_plane_intersection(
        p1: np.ndarray,
        n1: np.ndarray,
        p2: np.ndarray,
        n2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the intersection of two planes, choosing an optimal point for
    paramaterizing the line which is closest to midpoint of the line between p1
    and p2.

    See https://math.stackexchange.com/a/1937116 for more details.

    Args:
        p1 (np.ndarray): A vector of shape (3,) representing a point on plane 1.
        n1 (np.ndarray): A vector of shape (3,) represnting the normal of plane
            1.
        p2 (np.ndarray): A vector of shape (3,) representing a point on plane 2.
        n2 (np.ndarray): A vector of shape (3,) represnting the normal of plane
            2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple comprising:
            (np.ndarray): A vector of shape (3,) representing a point on the
                line.
            (np.ndarray): A vector of shape (3,) representing the direction of
                the line of intersection of the planes.
    """
    N = np.stack([n1, n2], axis=0)
    A = np.block([
        [2 * np.eye(3), N.T],
        [N, np.zeros((2, 2))]
    ])
    b = np.reshape([0, 0, 0, np.dot(p1, n1), np.dot(p2, n2)], 5)
    return np.linalg.solve(A, b)[:3], normalize(np.cross(n1, n2))


def circle_intersection(circle1: Tuple[float, float, float],
                        circle2: Tuple[float, float, float],
                        epsilon=0.002) -> Optional[Tuple[Tuple[float, float],
                                                         Tuple[float, float]]]:
    """Calculates intersection points of two circles

    Source:
        https://gist.github.com/xaedes/974535e71009fa8f090e

    Args:
        circle1 (Tuple[float, float, float]): A tuple of (x, y, radius)
        circle2 (Tuple[float, float, float]): A tuple of (x, y, radius)
        epsilon (float, optional): Tolerance to detect circle tangency. Defaults
            to 0.002.

    Returns:
        Optional[Tuple[Tuple[float, float], Tuple[float, float]]]: If no
            intersection exists, then None is returned. Else, a tuple is
            returned comprising:
            (Tuple[float, float]): The first intersection point
            (Tuple[float, float]): The second intersection point
    """
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    # http://stackoverflow.com/a/3349134/798588
    dx, dy = x2 - x1, y2 - y1
    d = np.sqrt(dx * dx + dy * dy)
    if d > r1 + r2:
        return None  # no solutions, the circles are separate
    dr = abs(r1 - r2)
    if d < dr:
        # FIXME(yycho0108): Curses on this shitty program
        if d + epsilon > dr:
            d = dr
        else:
            return None  # no solutions because one circle is contained within the other
    if d == 0 and r1 == r2:
        return None  # circles are coincident and there are an infinite number of solutions

    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h = np.sqrt(r1 * r1 - a * a)
    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    xs1 = xm + h * dy / d
    xs2 = xm - h * dy / d
    ys1 = ym - h * dx / d
    ys2 = ym + h * dx / d
    return (xs1, ys1), (xs2, ys2)
