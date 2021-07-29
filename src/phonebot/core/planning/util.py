#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

from phonebot.core.common.math.utils import anorm


def _angular_neighbors(i0, h0, i1, h1):
    # insertion index ...
    i_prv = np.searchsorted(h0, h1, side='right') - 1
    i_prv[i_prv == -1] = len(h0) - 1
    # i_prv[i_prv == len(h0)-1] = 0
    ii = i0[i_prv]

    # if (ii[0] == ii[-1]):
    #    ii[0] = i0[np.searchsorted(anorm(h0 - h1[0]), 0, side='right') - 1]
    # ii[0] = 5

    # dup = (ii[1:] == ii[:-1])
    # dup = (ii[1:] == ii[:-1])
    dup = np.roll
    ii[1:][dup] = i1[:-1][dup]
    # ii[0]=100
    return ii


def main():
    np.random.seed(2)
    n0 = 100
    n1 = 200
    h0 = np.sort(np.random.uniform(-0.5 * np.pi, 0.5 * np.pi, size=n0))
    h1 = np.sort(np.random.uniform(-np.pi, np.pi, size=n1))
    i0 = np.arange(len(h0))
    i1 = len(i0) + np.arange(len(h1))
    i_lo = _angular_neighbors(i0, h0, i1, h1)

    h = np.r_[h0, h1]
    i = np.r_[i0, i1]

    c = np.cos(h)
    s = np.sin(h)

    plt.plot(c[i0], s[i0], 'k.')
    plt.plot(c[i1], s[i1], 'r.')
    plt.quiver(c[i1], s[i1],
               c[i_lo] - c[i1], s[i_lo] - s[i1],
               # units='xy',
               angles='xy',
               scale_units='xy',
               scale=1,
               alpha=0.5)
    for i, x, y in zip(i0, c[i0], s[i0]):
        plt.text(x, y, F'i0/{i:02d}')
    for i, x, y in zip(i1, c[i1], s[i1]):
        plt.text(x, y, F'i1/{i:02d}')
    plt.show()


if __name__ == '__main__':
    main()
