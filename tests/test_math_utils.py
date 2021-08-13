import unittest

import numpy as np

from src.phonebot.core.common.math.transform import Transform
from src.phonebot.core.common.math.utils import normalize, adiff


class TestUtils(unittest.TestCase):

    def test_normalize(self):
        """Test normalizing a multi-dimensional array works.
        """
        arr = np.random.rand(10, 3)
        arr_norm = normalize(arr)
        self.assertTrue(np.allclose(np.linalg.norm(arr_norm, axis=1), 1))

    def test_adiff(self):
        delta = 0.01
        a0 = -np.pi + delta
        a1 = np.pi - delta
        self.assertTrue(np.isclose(adiff(a0, a1), 2 * delta))


if __name__ == '__main__':
    unittest.main()
