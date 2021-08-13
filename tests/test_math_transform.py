import unittest

import numpy as np

from src.phonebot.core.common.math.transform import (
    Position, Rotation, Transform
)


class TestPosition(unittest.TestCase):

    def test_initialization(self):
        """Test initializing a Position works properly
        """
        p = Position(np.random.rand(3))
        x, y, z = p

        self.assertTrue(np.isclose(x, p.x))
        self.assertTrue(np.isclose(y, p.y))
        self.assertTrue(np.isclose(z, p.z))

        def error():
            Position(np.random.rand(4))

        self.assertRaises(ValueError, error)

    def test_identity_init(self):
        """Test initializing a Position with it's identity method works.
        """
        p = Position.identity(size=(5, 4))

        self.assertTrue(p.shape == (3, 5, 4))

    def test_setting(self):
        """Test that setting values works correctly
        """
        p = Position(np.zeros(3))

        x_new, y_new, z_new = [1, 2, 3]

        p.x = x_new
        p.y = y_new
        p.z = z_new
        self.assertTrue(np.allclose(p, np.array([x_new, y_new, z_new])))

    def test_sum(self):
        """Test that adding two Positions works correctly
        """
        p1 = Position(np.array([1, 2, 3]))
        p2 = Position(np.array([4, 5, 6]))

        self.assertTrue(np.allclose(p1 + p2, np.array([5, 7, 9])))

    def test_encode_decode(self):
        """Test that encoding and decoding Position works.
        """
        p1 = Position(np.array([1, 2, 3]))

        data = p1.encode()

        p_dec = Position.decode(data)
        self.assertTrue(np.allclose(p1, p_dec))

        p_rest = Position(np.zeros(3))
        p_rest.restore(data)
        self.assertTrue(np.allclose(p1, p_rest))


class TestRotation(unittest.TestCase):

    def test_init(self):
        """Test initializing a rotation
        """
        axis = np.array([np.pi / 4, np.pi / 3, np.pi / 2])
        r = Rotation.from_axis_angle(axis)

        targ_mat = np.array([[-0.24150405, -0.39705091, 0.88545263],
                             [0.97005278, -0.07437851, 0.23122595],
                             [-0.02594983, 0.91477779, 0.40312305]])
        self.assertTrue(np.allclose(targ_mat, r.to_matrix()))

    def test_rot_mul(self):
        """Test rotating rotations by eachother works.
        """
        axis = np.array([np.pi / 4, np.pi / 3, np.pi / 2])
        r = Rotation.from_axis_angle(axis)

        r_sq_mat = r.rotate(r).to_matrix()
        targ_mat = np.matmul(r.to_matrix(), r.to_matrix())
        self.assertTrue(np.allclose(r_sq_mat, targ_mat))


class TestTransform(unittest.TestCase):

    def test_to_matrix(self):
        """Test that converting to a transformation matrix works
        """
        p = np.array([1, 2, 3])
        r = Rotation.from_axis_angle(
            np.array([np.pi / 5, np.pi / 4, np.pi / 3]))

        r_mat = r.to_matrix()
        trans_mat_targ = np.eye(4)
        trans_mat_targ[:3, :3] = r_mat
        trans_mat_targ[:3, -1] = p

        ts = Transform(p, r)
        self.assertTrue(np.allclose(ts.to_matrix(), trans_mat_targ))

        # Test matrix conversion
        for _ in range(100):
            T = Transform.random()
            Tm = T.to_matrix()
            p = Position.random()
            a = T * p
            b = Tm.dot(np.append(p, [1.0], axis=-1))[..., :3]
            self.assertTrue(np.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
