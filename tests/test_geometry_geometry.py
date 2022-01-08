import unittest
import numpy as np

from src.phonebot.core.common.geometry.geometry import (
    circle_intersection,
    plane_plane_intersection
)


class TestGeometry(unittest.TestCase):

    def test_plane_intersection(self):
        p0 = np.array([1, 0, 0])
        p1 = np.array([-1, 0, 0])
        n0 = np.array([-0.5, 0, 0.5])
        n1 = np.array([0.5, 0, 0.5])

        pi, ni = plane_plane_intersection(p0, n0, p1, n1)

        self.assertTrue(np.allclose(pi, np.array([0, 0, -1])))
        self.assertTrue(np.allclose(ni, np.array([0, 1, 0])))

    def test_circle_intersection(self):
        center_1 = np.array([0, 0])
        center_2 = np.array([1, 0])
        radius_1 = 0.75
        radius_2 = 0.75
        # We expect y values of intersection to be 0.5
        p1, p2 = circle_intersection(
            (*center_1, radius_1), (*center_2, radius_2))
        xs = np.array([p1[0], p2[0]])
        self.assertTrue(np.allclose(xs, 0.5))

        res = circle_intersection(
            (*center_1, radius_1), (*center_2, 100 * radius_2))
        # We expect circle 1 to be contained by circle 2.
        self.assertTrue(res is None)


if __name__ == '__main__':
    unittest.main()
