#!/usr/bin/env python3

from typing import List
import numpy as np


def sphere_inertia(radius: float):
    r = radius
    return 0.4 * r * r * np.eye(3)


def cylinder_inertia(radius: float, height: float):
    r, h = radius, height
    ixx = iyy = 0.08333333333333333 * (r * r + h * h)
    izz = 0.5 * r * r
    return np.diag([ixx, iyy, izz])


def box_inertia(dims: List[float]):
    sq = np.square(dims)
    return 0.08333333333333333 * np.diag(sq.sum() - sq)
