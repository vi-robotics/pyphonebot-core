#!/usr/bin/env python3

import numpy as np
from phonebot.core.common.math.utils import normalize


def parallel_component(a, b):
    """ Project vector a to b """
    nb = normalize(b)
    return nb * nb.dot(a)


def tangent_component(a, b):
    """ tangent component of a wrt b """
    return a - parallel_component(a, b)


def plane_plane_intersection(p1, n1, p2, n2):
    """
    https://math.stackexchange.com/a/1937116
    """
    N = np.stack([n1, n2], axis=0)
    A = np.block([
        [2*np.eye(3), N.T],
        [N, np.zeros((2, 2))]
    ])
    b = np.reshape([0, 0, 0, np.dot(p1, n1), np.dot(p2, n2)], 5)
    return np.linalg.solve(A, b)[:3]


def circle_intersection(circle1, circle2, epsilon=0.002):
    '''
    Source:
        https://gist.github.com/xaedes/974535e71009fa8f090e

    @summary: calculates intersection points of two circles
    @param circle1: tuple(x,y,radius)
    @param circle2: tuple(x,y,radius)
    @result: tuple of intersection points (which are (x,y) tuple)
    '''
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    # http://stackoverflow.com/a/3349134/798588
    dx, dy = x2-x1, y2-y1
    d = np.sqrt(dx*dx+dy*dy)
    if d > r1+r2:
        return None  # no solutions, the circles are separate
    dr = abs(r1-r2)
    if d < dr:
        # FIXME(yycho0108): Curses on this shitty program
        if d + epsilon > dr:
            d = dr
        else:
            return None  # no solutions because one circle is contained within the other
    if d == 0 and r1 == r2:
        return None  # circles are coincident and there are an infinite number of solutions

    a = (r1*r1-r2*r2+d*d)/(2*d)
    h = np.sqrt(r1*r1-a*a)
    xm = x1 + a*dx/d
    ym = y1 + a*dy/d
    xs1 = xm + h*dy/d
    xs2 = xm - h*dy/d
    ys1 = ym - h*dx/d
    ys2 = ym + h*dx/d
    return (xs1, ys1), (xs2, ys2)