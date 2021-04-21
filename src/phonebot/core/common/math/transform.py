#!/usr/bin/env python3

import numpy as np
import pickle

from cho_util.math import transform as tx

from phonebot.core.common.serial import Serializable, decode, encode

__all__ = ['Position', 'Rotation', 'Transform']


class Position(np.ndarray, Serializable):
    """
    3-element Vector wrapper around ndarray.
    """
    def __new__(cls, input_array):
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
    def y(self, x):
        super().__setitem__(1, y)

    @property
    def z(self):
        return super().__getitem__(2)

    @z.setter
    def z(self, z):
        super().__setitem__(2, z)

    @classmethod
    def identity(cls):
        return cls(np.zeros(3))

    @classmethod
    def random(cls, size=(), *args, **kwargs):
        size = tuple(np.reshape(size, [-1])) + (3,)
        return cls(np.random.normal(size=size))

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
    def identity(cls):
        return cls(Rotation.quaternion.identity(), Rotation._quaternion)

    @classmethod
    def random(cls, *args, **kwargs):
        return cls(Rotation.quaternion.random(*args, **kwargs), Rotation._quaternion)

    @classmethod
    def from_quaternion(cls, value):
        return cls(value, rtype=Rotation._quaternion)

    @classmethod
    def from_axis_angle(cls, value):
        return cls(value, rtype=Rotation._axis_angle)

    @classmethod
    def from_euler(cls, value):
        return cls(value, rtype=Rotation._euler)

    @classmethod
    def from_matrix(cls, value):
        return cls(value, rtype=Rotation._matrix)

    def to_quaternion(self):
        if self.rtype == Rotation._quaternion:
            return self
        return Rotation(self.base.to_quaternion(self), Rotation._quaternion)

    def to_axis_angle(self):
        if self.rtype == Rotation._axis_angle:
            return self
        return Rotation(self.base.to_axis_angle(self), Rotation._axis_angle)

    def to_euler(self):
        if self.rtype == Rotation._euler:
            return self
        return Rotation(self.base.to_euler(self), Rotation._euler)

    def to_matrix(self, *args, **kwargs):
        if self.rtype == Rotation._matrix:
            return self
        return Rotation(self.base.to_matrix(self, *args, **kwargs), Rotation._matrix)

    def rotate(self, value):
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

    def restore(self, data: bytes):
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

    def __init__(self, position, rotation, inverted=False):
        self.position_ = Position(position)
        self.rotation_ = Rotation(rotation)
        self.inverted_ = inverted

    @classmethod
    def identity(cls):
        return cls(Position.identity(), Rotation.identity())

    @classmethod
    def random(cls):
        return cls(Position.random(), Rotation.random())

    @classmethod
    def from_rotation(cls, rotation):
        return cls(Position.identity(), rotation)

    @classmethod
    def from_position(cls, position):
        return cls(position, Rotation.identity())

    @classmethod
    def from_transform(cls, transform):
        return cls(transform.position.copy(), transform.rotation)

    def to_matrix(self):
        T = np.zeros(
            shape=(self.position_.shape[:-1]) + (4, 4), dtype=np.float64)
        R = self.rotation.to_matrix(out=T[..., :3, :3])
        T[..., :3, -1] = self.position
        T[..., -1, -1] = 1.0
        return T

    @property
    def position(self):
        if self.inverted_:
            return -self.rotation_.inverse().rotate(self.position_)
        else:
            return self.position_

    @position.setter
    def position(self, value):
        if self.inverted_:
            # a = -Ri * b
            # b = R * -a
            self.position_ = self.rotation_.rotate(-value)
        else:
            self.position_ = value

    @property
    def rotation(self):
        if self.inverted_:
            return self.rotation_.inverse()
        else:
            return self.rotation_

    @rotation.setter
    def rotation(self, value):
        if not isinstance(value, Rotation):
            raise ValueError('Cannot set rotation component to non-rotation!')

        if self.inverted_:
            self.rotation_ = value.inverse()
        else:
            self.rotation_ = value

    def rotate(self, other):
        return self.rotation.rotate(other)

    def inverse(self):
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
        # if self.inverted_:
        #    if other.inverted_:
        #        # (B*A)^-1
        #        self.position_ = other.position_ + \
        #            other.rotation_.rotate(self.position_)
        #        self.rotation_ = other.rotation_.rotate(self.rotation_)
        #        return self

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


def main():
    # Test pickle
    import pickle
    pickle.dumps(Position.identity())
    pickle.dumps(Rotation.identity())
    pickle.dumps(Transform.identity())

    # Test broadcasting
    poss = Position.random(size=(5,))
    rots = Rotation.random(size=(1,))
    print(rots.rotate(poss).shape)

    # Test matrix conversion
    for _ in range(100):
        T = Transform.random()
        Tm = T.to_matrix()
        p = Position.random()
        a = T * p
        b = Tm.dot(np.append(p, [1.0], axis=-1))[..., :3]
        if not np.allclose(a, b):
            print('{} != {}'.format(a, b))


if __name__ == '__main__':
    main()
