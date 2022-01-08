#!/usr/bin/env python3
from typing import Union, Tuple
import numpy as np

from cho_util.math import transform as tx

from phonebot.core.common.serial import Serializable, decode, encode

__all__ = ['Position', 'Rotation', 'Transform']


class Position(np.ndarray, Serializable):
    """3-element Vector wrapper around ndarray.
    """
    def __new__(cls, input_array: np.ndarray) -> "Position":
        """Initialize a Position instance

        Args:
            input_array (np.ndarray): An array of shape (3,...) representing the
                position.

        Raises:
            ValueError: input array is of the wrong shape

        Returns:
            Position: The resulting Position instance.
        """
        if input_array.shape[0] != 3:
            raise ValueError("Input array is of shape"
                             f" {input_array.shape}, expected (3,...)")
        obj = np.asarray(input_array, dtype=np.float64).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def x(self):
        return super().__getitem__(0)

    @x.setter
    def x(self, x):
        super().__setitem__(0, x)

    @property
    def y(self):
        return super().__getitem__(1)

    @y.setter
    def y(self, y):
        super().__setitem__(1, y)

    @property
    def z(self):
        return super().__getitem__(2)

    @z.setter
    def z(self, z):
        super().__setitem__(2, z)

    @classmethod
    def identity(cls, size: Tuple = ()):
        """The additive identity of position.

        Args:
            size (Tuple): The shape to give the axes after the first 3 (
                which are x, y, z).

                Example:
                    >>> Position.identity(size=(4,5)).shape
                    (3, 4, 5)

        Returns:
            Position: initialized with zeros.
        """
        # Handle size=(X) or size=(X,)
        size = (3,) + tuple(np.reshape(size, [-1]))
        return cls(np.zeros(size))

    @classmethod
    def random(cls, size: Tuple = (), scale: float = 1):
        """Initialize a random position using a scaled normal distribution.

        Args:
            size (Tuple): The shape to give the axes after the first 3 (
                which are x, y, z).

                Example:
                    >>> Position.random(size=(4,5)).shape
                    (3, 4, 5)

            scale (float, optional): A scalar to multiply the sampled values by.
                Defaults to 1.

        Returns:
            Position: The randomly initialized Position object.
        """
        size = (3,) + tuple(np.reshape(size, [-1]))
        return cls(np.random.normal(size=size) * scale)

    def encode(self, *args, **kwargs) -> bytes:
        return encode(self.tobytes(), *args, **kwargs)

    def restore(self, data: bytes, *args, **kwargs):
        self[...] = np.frombuffer(decode(data, *args, **kwargs))

    @classmethod
    def decode(cls, data: bytes, *args, **kwargs):
        a = np.frombuffer(decode(data, *args, **kwargs))
        return cls(a)


class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


class Rotation(np.ndarray, Serializable):
    """
    Thin wrapper around ndarray with rotation type information.
    NOTE(yycho0108): Do NOT override operators such as __mul__.
    Underlying POD computation functions will be affected by the change.
    """

    _quaternion = 0
    _axis_angle = 1
    _euler = 2
    _matrix = 3

    @classproperty
    def quaternion(cls):
        return tx.rotation.quaternion

    @classproperty
    def axis_angle(cls):
        return tx.rotation.axis_angle

    @classproperty
    def euler(cls):
        return tx.rotation.euler

    @classproperty
    def matrix(cls):
        return tx.rotation.matrix

    @classproperty
    def base_map(cls):
        """ enum -> compute module """
        return {
            Rotation._quaternion: Rotation.quaternion,
            Rotation._axis_angle: Rotation.axis_angle,
            Rotation._euler: Rotation.euler,
            Rotation._matrix: Rotation.matrix}

    @classproperty
    def base_inverse_map(cls):
        """ compute module -> enum """
        return {
            Rotation.quaternion: Rotation._quaternion,
            Rotation.axis_angle: Rotation._axis_angle,
            Rotation.euler: Rotation._euler,
            Rotation.matrix: Rotation._matrix}

    @property
    def base(self):
        """ compute module """
        return Rotation.base_map[self.rtype]

    def __new__(cls, input_array, rtype=None):
        if isinstance(input_array, Rotation):
            rtype = input_array.rtype
        if rtype in Rotation.base_inverse_map:
            rtype = Rotation.base_inverse_map[rtype]
        if rtype is None:
            rtype = Rotation._quaternion
        if rtype not in Rotation.base_map:
            raise ValueError('rtype must be one of {}; got: {}'.format(
                list(Rotation.base_map.keys()), rtype))

        obj = np.asarray(input_array, dtype=np.float64).view(cls)
        obj.rtype = rtype
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.rtype = getattr(obj, 'rtype', None)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super().__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.rtype,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.rtype = state[-1]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super().__setstate__(state[0:-1])

    @classmethod
    def identity(cls) -> "Rotation":
        """The Identity rotation (as a quaternion)

        Returns:
            Rotation: The identity quaternion.
        """
        return cls(Rotation.quaternion.identity(), Rotation._quaternion)

    @classmethod
    def random(cls, *args, **kwargs):
        """Create a random Rotation using a random quaternion.

        Returns:
            Rotation: a randomly generated Rotation.
        """
        return cls(Rotation.quaternion.random(*args, **kwargs), Rotation._quaternion)

    @classmethod
    def from_quaternion(cls, value: np.ndarray):
        """Create a Rotation from a quaternion

        Args:
            value (np.ndarray): an array of shape (4,) representing a
                unit quaternion.

        Returns:
            Rotation: The resulting Rotation
        """
        return cls(value, rtype=Rotation._quaternion)

    @classmethod
    def from_axis_angle(cls, value: np.ndarray):
        """Create a Rotation from a rotation vector.

        Args:
            value (np.ndarray): an array of shape (3,) representing a rotation
                vector.

        Returns:
            Rotation: the resulting Rotation
        """
        return cls(value, rtype=Rotation._axis_angle)

    @classmethod
    def from_euler(cls, value: np.ndarray):
        """Create a Rotation from euler angles (extrinsic xyz). This is
        equivalent to using 'xyz' for seq with scipy.spatial.transform.Rotation:

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html

        Args:
            value (np.ndarray): an array of shape (3,) representing extrinsic
                xyz euler angles.

        Returns:
            Rotation: the resulting Rotation
        """
        return cls(value, rtype=Rotation._euler)

    @classmethod
    def from_matrix(cls, value: np.ndarray):
        """Create a Rotation from a rotation matrix.

        Args:
            value (np.ndarray): A 3x3 rotation matrix.

        Returns:
            Rotation: the resulting Rotation
        """
        return cls(value, rtype=Rotation._matrix)

    def to_quaternion(self) -> "Rotation":
        """Converts the Rotation to another Rotation with a quaternion base.

        Returns:
            Rotation: the resulting Rotation
        """
        if self.rtype == Rotation._quaternion:
            return self
        return Rotation(self.base.to_quaternion(self), Rotation._quaternion)

    def to_axis_angle(self) -> "Rotation":
        """Converts the Rotation to another Rotation with a rotation vector
        base.

        Returns:
            Rotation: the resulting Rotation
        """
        if self.rtype == Rotation._axis_angle:
            return self
        return Rotation(self.base.to_axis_angle(self), Rotation._axis_angle)

    def to_euler(self) -> "Rotation":
        """Converts the Rotation to another Rotation with an euler base (of the
        convention extrinsic xyz).

        Returns:
            Rotation: the resulting Rotation
        """
        if self.rtype == Rotation._euler:
            return self
        return Rotation(self.base.to_euler(self), Rotation._euler)

    def to_matrix(self, *args, **kwargs) -> "Rotation":
        """Converts the Rotation to another Rotation with a matrix base.

        Returns:
            Rotation: the resulting Rotation
        """
        if self.rtype == Rotation._matrix:
            return self
        return Rotation(self.base.to_matrix(self, *args, **kwargs), Rotation._matrix)

    def rotate(self, value: Union["Rotation", Position]) -> Union["Rotation",
                                                                  Position]:
        """Rotate a Rotation or a Position with the given Rotation. This is
        equivalent to left multipling the Rotation by value.

        Args:
            value (Union[Rotation, Position]): A Position or Rotation to rotate.

        Returns:
            Union["Rotation", Position]: The rotated Position or Rotation (same
                as input type of value).
        """
        if isinstance(value, Rotation):
            # Compose rotations after mapping to quaternion.
            q0 = self.to_quaternion()
            q1 = value.to_quaternion()
            q0q1 = Rotation.quaternion.multiply(q0, q1)
            return Rotation(q0q1, Rotation._quaternion)
        elif isinstance(value, Position):
            return Position(self.base.rotate(self, value))
        else:
            return NotImplemented

    def inverse(self):
        """The inverse rotation, of the same type.

        Returns:
            Rotation: the resulting Rotation
        """
        return Rotation(self.base.inverse(self), rtype=self.rtype)

    def __repr__(self):
        rtype_map = {
            Rotation._quaternion: 'quaternion',
            Rotation._axis_angle: 'axis_angle',
            Rotation._euler: 'euler',
            Rotation._matrix: 'matrix',
        }
        return ('{}(rtype={})'.format(super().__repr__(), rtype_map[self.rtype]))

    def encode(self, *args, **kwargs) -> bytes:
        data = (self.rtype, self.tobytes())
        return encode(data, *args, **kwargs)

    def restore(self, data: bytes, *args, **kwargs):
        rtype, b = decode(data, *args, **kwargs)
        self[...] = np.frombuffer(b)
        self.rtype = rtype

    @classmethod
    def decode(cls, data: bytes, *args, **kwargs):
        rtype, b = decode(data, *args, **kwargs)
        a = np.frombuffer(b)
        out = cls(a, rtype)
        return out


class Transform(Serializable):
    """
    Simple transform utility class.
    """

    def __init__(self, position: np.ndarray, rotation: np.ndarray,
                 inverted: bool = False):
        """Initialize a Transform with a position and rotation.

        Args:
            position (np.ndarray): An array of shape (3,) representing the
                position (translation component of the transform).
            rotation (np.ndarray): A unit quaternion array of shape (4,)
                representing the rotation.
            inverted (bool, optional): If True, then this transform is marked as
                inverted and the inverse will be calculated lazily. Defaults to
                False.
        """
        self.position_ = Position(position)
        self.rotation_ = Rotation(rotation)
        self.inverted_ = inverted

    @classmethod
    def identity(cls):
        """Return the idenity transform (no translation or rotation)

        Returns:
            Transform: the resulting Transform
        """
        return cls(Position.identity(), Rotation.identity())

    @classmethod
    def random(cls, position_scale: float = 1):
        """Return a random transform by uniformly sampling rotations, and
        sampling Position from a normal distribution

        Args:
            position_scale (float, optional): A scalar to multiply the sampled
                values of Position by. Defaults to 1.

        Returns:
            Transform: the resulting Transform
        """
        return cls(Position.random(scale=position_scale), Rotation.random())

    @classmethod
    def from_rotation(cls, rotation: Rotation):
        """Create a Transform using the identity Position and the provided
        rotation.

        Args:
            rotation (Rotation): The rotation to set for the transform.

        Returns:
            Transform: the resulting Transform
        """
        return cls(Position.identity(), rotation)

    @classmethod
    def from_position(cls, position: Position):
        """Create a Transform using the identity Rotation and the provided
        position.

        Args:
            position (Position): The position to set for the transform.

        Returns:
            Transform: the resulting Transform
        """
        return cls(position, Rotation.identity())

    @classmethod
    def from_transform(cls, transform: "Transform"):
        """Copy an existing transform.

        Args:
            transform (Transform): the Transform to copy.

        Returns:
            Transform: the copied Transform.
        """
        return cls(transform.position.copy(), transform.rotation)

    def to_matrix(self) -> np.ndarray:
        """Convert the Transform into a transformation matrix.

        Returns:
            np.ndarray: an array of transformation matrices of shape (..., 4, 4)
        """
        T = np.zeros(
            shape=(self.position_.shape[:-1]) + (4, 4), dtype=np.float64)
        self.rotation.to_matrix(out=T[..., :3, :3])
        T[..., :3, -1] = self.position
        T[..., -1, -1] = 1.0
        return T

    @property
    def position(self) -> Position:
        """The position of the transform

        Returns:
            Position: the resulting Position
        """
        if self.inverted_:
            return -self.rotation_.inverse().rotate(self.position_)
        else:
            return self.position_

    @position.setter
    def position(self, value: Position):
        """Set the position of the transform

        Args:
            value (Position): the Position object to set
        """
        if self.inverted_:
            # a = -Ri * b
            # b = R * -a
            self.position_ = self.rotation_.rotate(-1 * value)
        else:
            self.position_ = value

    @property
    def rotation(self) -> Rotation:
        """The rotation of the transform

        Returns:
            Rotation: the resulting Rotation
        """
        if self.inverted_:
            return self.rotation_.inverse()
        else:
            return self.rotation_

    @rotation.setter
    def rotation(self, value: Rotation):
        """Set the rotation of the transform

        Args:
            value (Rotation): the Rotation to set.

        Raises:
            ValueError: value is not a Rotation
        """
        if not isinstance(value, Rotation):
            raise ValueError('Cannot set rotation component to non-rotation!')

        if self.inverted_:
            self.rotation_ = value.inverse()
        else:
            self.rotation_ = value

    def rotate(self, other: Union[Position, Rotation]) -> Union[Position,
                                                                Rotation]:
        """Rotate a Position or Rotation by the transform's rotation component.

        Args:
            other (Union[Position, Rotation]): a Position or Rotation to rotate.

        Returns:
            Union[Position, Rotation]: the resulting Position or Rotation (same
                as other).
        """
        return self.rotation.rotate(other)

    def inverse(self):
        """Return the inverse of the transform (lazily evaluated, simpy inverts
        the 'inverted' property).

        Returns:
            Transform: the resulting Transform
        """
        return Transform(self.position_, self.rotation_, (not self.inverted_))

    def encode(self, *args, **kwargs) -> bytes:
        data = (self.position_, self.rotation_, self.inverted_)
        return encode(data, *args, **kwargs)

    def restore(self, data: bytes):
        pos, rot, inv = decode(data, *args, **kwargs)
        self.position_ = pos
        self.rotation_ = rot
        self.inverted_ = inv

    @classmethod
    def decode(cls, data: bytes, *args, **kwargs):
        pos, rot, inv = decode(data, *args, **kwargs)
        return cls(pos, rot, inv)

    def __imul__(self, other):
        self.position += self.rotation.rotate(other.position)
        self.rotation = self.rotation.rotate(other.rotation)
        return self

    def __mul__(self, other):
        if isinstance(other, Transform):
            cpy = Transform.from_transform(self)
            cpy *= other
            return cpy
        elif isinstance(other, Position):
            return self.position + self.rotation.rotate(other)
        else:
            return NotImplemented

    def __str__(self):
        return '({},{})'.format(self.position, self.rotation)

    def __repr__(self):
        return '({},{},{})'.format(repr(self.position_), repr(self.rotation_), self.inverted_)
