#!/usr/bin/env python3

__all__ = ['FrameEdge', 'SimpleFrameEdge', 'StaticFrameEdge',
           'ParametricFrameEdge', 'RevoluteJointEdge', 'BufferedFrameEdge', 'InvertedFrameEdge']

import time
import numpy as np
import logging

from collections import deque
from typing import Any, List, Dict
from abc import abstractmethod, ABCMeta

from phonebot.core.common.serial import Serializable, encode, decode
from phonebot.core.common.util import find_nearest_index, find_k_nearest_indices, get_time
from phonebot.core.common.math.utils import normalize, adiff, rotation_between_vectors, tlerp
from phonebot.core.common.geometry.geometry import perpendicular_component
from phonebot.core.common.math.transform import Position, Rotation, Transform

logger = logging.getLogger(__name__)


class FrameEdge(Serializable):
    """
    Abstract class that defines the interface to obtain a transform
    from source frame to target frame at a given time.
    """

    def __init__(self, source_frame, target_frame):
        super().__init__()
        self.source_frame_ = source_frame
        self.target_frame_ = target_frame

    @property
    def source(self) -> str:
        return self.source_frame_

    @property
    def target(self) -> str:
        return self.target_frame_

    @abstractmethod
    def update(self, stamp: float):
        return NotImplemented

    @abstractmethod
    def get_transform(self, stamp: float) -> Transform:
        return NotImplemented

    @property
    @abstractmethod
    def stamp(self) -> float:
        return NotImplemented

    def has_transform(self, stamp: float, tol: float = 1e-2) -> bool:
        return abs(self.stamp - stamp) <= tol

    def has_frame(self, frame: str) -> bool:
        return frame in [self.source_frame_, self.target_frame_]

    def __str__(self):
        return '({}->{})'.format(self.source, self.target)

    def __repr__(self):
        return '{}({}->{})'.format(self.__class__.__name__, self.source, self.target)


class BufferedFrameEdge(FrameEdge):
    """
    FrameEdge class with buffered transforms, allowing for lookups between known poses.
    """
    def __new__(cls, edge, *args, **kwargs):
        if isinstance(edge, StaticFrameEdge):
            # No need to buffer static frames.
            # NOTE(yycho0108): better way to handle this?
            return edge
        elif isinstance(edge, BufferedFrameEdge):
            # Avoid buffering twice.
            return edge
        return super().__new__(cls)

    def __getnewargs__(self):
        return (self.edge_, self.queue_size_, self.timeout_)

    def __init__(self, edge, queue_size=4, timeout=0.1):
        super().__init__(edge.source, edge.target)
        self.edge_ = edge
        self.queue_ = deque(maxlen=queue_size)
        self.queue_size_ = queue_size
        self.timeout_ = timeout

    def __getattr__(self, attr):
        """ Delegation to underyling edge """
        if attr in ['__getstate__', '__setstate__']:
            return object.__getattr__(self, attr)
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.edge_, attr)

    def has_transform(self, stamp: float, tol: float = None) -> Transform:
        """
        Check if edge has transform at the given timestamp.
        """
        tol = tol if (tol is not None) else self.timeout_
        # No element in queue.
        if len(self.queue_) <= 0:
            return False

        # Single element in queue.
        if len(self.queue_) == 1:
            ref_stamp, _ = self.queue_[0]
            return np.abs(ref_stamp - stamp) < tol

        indices = find_k_nearest_indices(self.stamps, stamp, k=2)
        prev_index, next_index = sorted(indices)

        # Check if interpolation is possible.
        if self.stamps[prev_index] <= stamp <= self.stamps[next_index]:
            return True

        # Otherwise, check if extrapolation is possible
        # within the `timeout` window.
        return np.abs(self.stamps[indices[0]] - stamp) < tol

    def get_transform(self, stamp: float) -> Transform:
        # Check if transform is possible.

        # Handle a case with exactly one element.
        if len(self.queue_) == 1:
            return self.transforms[0]

        # Get reference transforms from which to build transform.
        indices = find_k_nearest_indices(self.stamps, stamp, k=2)
        prev_index, next_index = sorted(indices)
        numer = (stamp - self.stamps[prev_index])
        denom = (self.stamps[next_index] - self.stamps[prev_index])
        if denom <= np.finfo(float).eps:
            # Two stamps are really close to each other.
            # To avoid dividing by zero, return arbitrary selection.
            return self.transforms[indices[0]]

        # Return interpolated transform.
        # NOTE(yycho018): tlerp also handles extrapolation.
        weight = numer / denom
        transform = tlerp(
            self.transforms[prev_index], self.transforms[next_index],
            weight)
        return transform

    def update(self, stamp: float, *args, **kwargs):
        self.edge_.update(stamp, *args, **kwargs)
        transform = self.edge_.get_transform(stamp)
        self.queue_.append((stamp, transform))

    @property
    def stamp(self):
        try:
            return self.stamps[-1]
        except IndexError:
            return -np.inf

    def encode(self, *args, **kwargs) -> bytes:
        data = (
            self.edge_,
            self.queue_,
            self.queue_size_,
            self.timeout_
        )
        return encode(data, *args, **kwargs)

    def restore(self, data: bytes, *args, **kwargs):
        data = decode(data, *args, **kwargs)
        self.edge_, self.queue_, self.queue_size_, self.timeout_ = data

    @classmethod
    def decode(cls, data: bytes, *args, **kwargs):
        data = decode(data, *args, **kwargs)
        edge, queue, queue_size, timeout = data
        out = cls(edge, queue_size, timeout)
        out.queue_ = queue
        return out

    @property
    def stamps(self):
        return [stamp for stamp, _ in self.queue_]

    @property
    def transforms(self):
        return [transform for _, transform in self.queue_]

    def __str__(self):
        return '({}->{})'.format(self.source, self.target)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.edge_.__repr__())


class SimpleFrameEdge(FrameEdge):
    """
    General frame edge.
    """

    def __init__(self, source_frame, target_frame):
        super().__init__(source_frame, target_frame)
        self.stamp_ = 0.0
        self.transform_ = Transform.identity()

    def update(self, stamp: float, transform: Transform):
        # TODO(yycho0108): if the frame edge was obtained through
        # graph.get_edge()..., naively calling update()
        # may in fact apply an inverse transform.
        self.stamp_ = stamp
        self.transform_ = transform

    def get_transform(self, stamp):
        # TODO(yych0108): return copy / indicate immutable?
        return self.transform_

    @property
    def stamp(self):
        return self.stamp_

    def encode(self, *args, **kwargs) -> bytes:
        data = (self.source_frame_, self.target_frame_, self.stamp_,
                self.transform_)
        out = encode(data, *args, **kwargs)
        return out

    def restore(self, data: bytes, *args, **kwargs):
        data = decode(data, *args, **kwargs)
        src, dst, stamp, xfm = data
        self.source_frame_ = src
        self.target_frame_ = dst
        self.stamp_ = stamp
        self.transform_ = xfm

    @classmethod
    def decode(cls, data: bytes, *args, **kwargs):
        data = decode(data, *args, **kwargs)
        src, dst, stamp, xfm = data
        out = cls(src, dst)
        out.stamp_ = stamp
        out.transform_ = xfm
        return out

    def __str__(self):
        return '({}->{})'.format(self.source, self.target)

    def __repr__(self):
        return '{}({}->{})'.format(self.__class__.__name__, self.source, self.target)


class InvertedFrameEdge(SimpleFrameEdge):
    """
    Lightweight inversion wrapper to a SimpleFrameEdge.

    NOTE(yycho0108): As of current implementation,
    InvertedFrameEdge will NOT handle ParametricFrameEdge correctly.
    """
    def __new__(cls, edge, *args, **kwargs):
        if isinstance(edge, InvertedFrameEdge):
            # No need to invert twice.
            return edge.edge_
        return super().__new__(cls)

    def __getnewargs__(self):
        return (self.edge_,)

    def __init__(self, edge):
        super().__init__(edge.target, edge.source)
        self.edge_ = edge

    def __getattr__(self, attr):
        """ Delegation to underyling edge """
        if attr in ['__getstate__', '__setstate__']:
            return object.__getattr__(self, attr)
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.edge_, attr)

    def get_transform(self, stamp: float) -> Transform:
        return self.edge_.get_transform(stamp).inverse()

    def update(self, stamp: float, transform: Transform, *args, **kwargs):
        return self.edge_.update(stamp, transform.inverse(), *args, **kwargs)

    def encode(self, *args, **kwargs) -> bytes:
        return encode(self.edge_, *args, **kwargs)

    def restore(self, data: bytes, *args, **kwargs):
        edge = decode(data, *args, **kwargs)
        self.edge_ = edge

    @classmethod
    def decode(cls, data: bytes, *args, **kwargs):
        edge = decode(data, *args, **kwargs)
        return cls(edge)

    def __str__(self):
        return '({}->{})'.format(self.source, self.target)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.edge_.__repr__())


class StaticFrameEdge(SimpleFrameEdge):
    """
    Frame edge that does not change.
    """

    def __init__(self, source_frame, target_frame, transform):
        super().__init__(source_frame, target_frame)
        self.transform_ = transform

    def update(self, stamp: float, transform: Transform):
        self.transform_ = transform

    def has_transform(self, stamp, tol=1e-2):
        return True

    def get_transform(self, stamp):
        # TODO(yych0108): return copy / indicate immutable?
        return self.transform_

    @property
    def stamp(self):
        """ Static frame = newest stamp """
        # TODO(yycho0108): return special stamp value instead?
        # return time.time()
        return get_time()

    def encode(self, *args, **kwargs) -> bytes:
        data = (self.source_frame_, self.target_frame_,
                self.transform_)
        return encode(data, *args, **kwargs)

    def restore(self, data: bytes, *args, **kwargs):
        src, dst, xfm = decode(data, *args, **kwargs)
        self.source_frame_ = src
        self.target_frame_ = dst
        self.transform_ = xfm

    @classmethod
    def decode(cls, data: bytes, *args, **kwargs):
        src, dst, xfm = decode(data, *args, **kwargs)
        out = cls(src, dst, xfm)
        return out

    def __str__(self):
        return '({}->{})'.format(self.source, self.target)

    def __repr__(self):
        return '{}({}->{})'.format(self.__class__.__name__, self.source, self.target)


class ParametricFrameEdge(SimpleFrameEdge):
    """
    Frame whose transform is parametrically defined.
    """

    def __init__(self, source_frame, target_frame, param):
        super().__init__(source_frame, target_frame)
        self.param_ = param
        self.stamp_ = 0.0

    @abstractmethod
    def from_param(self, param) -> Transform:
        """ Map parameter to transform. """
        return NotImplemented

    @abstractmethod
    def to_param(self, transform: Transform):
        """ Map the transform to parameter. """
        return NotImplemented

    def update(self, stamp: float, param):
        if isinstance(param, Transform):
            logger.warn(
                """
                Transform {} provided as update to ParametricFrameEdge;
                Attempting default conversion as fallback.
                May result in unexpected behavior.
                """.format(param))
            return self.update(stamp, self.to_param(param))
        self.param_ = param
        return super().update(stamp, self.from_param(param))

    def encode(self, *args, **kwargs) -> bytes:
        data = (self.source_frame_, self.target_frame_,
                self.stamp_, self.param_)
        return encode(data, *args, **kwargs)

    def restore(self, data: bytes, *args, **kwargs):
        src, dst, stamp, param = decode(data, *args, **kwargs)
        self.source_frame_ = src
        self.target_frame_ = dst
        self.stamp_ = stamp
        self.param_ = param

    @classmethod
    def decode(cls, data: bytes, *args, **kwargs):
        src, dst, stamp, param = decode(data, *args, **kwargs)
        out = cls(src, dst, param)
        out.stamp_ = stamp
        return out

    @property
    def param(self):
        return self.param_

    def __str__(self):
        return '{}({}->{})'.format(self.__class__.__name__, self.source, self.target)

    def __repr__(self):
        return '{}({}->{})'.format(self.__class__.__name__, self.source, self.target)


class RevoluteJointEdge(ParametricFrameEdge):
    """
    Simple representation of a revolute joint.
    """

    def __init__(self, source_frame, target_frame, axis, offset):
        self.axis_ = axis
        self.offset_ = offset
        self.offset_transform_ = Transform.from_position(
            Position(self.offset_))
        super().__init__(source_frame, target_frame, 0.0)

    def from_param(self, param):
        rotation_vector = list(self.axis_) + [self.angle]
        rotation = Rotation.from_axis_angle(rotation_vector)
        # TODO(yycho0108): There's probably a bit more elegant way to specify this.
        transform = Transform.from_rotation(rotation) * self.offset_transform_
        return transform

    def to_param(self, transform):
        """
        Get the angle from the transform.
        Note that the rotation component of the transform is ignored.
        """
        source = perpendicular_component(self.offset_, self.axis_)
        if isinstance(transform, Transform):
            target = perpendicular_component(transform.position, self.axis_)
        # elif isinstance(transform, Position):
        else:
            target = perpendicular_component(transform, self.axis_)

        rotation = rotation_between_vectors(source, target)
        # validation ...
        axa = Rotation.from_quaternion(rotation).to_axis_angle()
        if not np.allclose(axa.rotate(Position(source)), target):
            logger.warn("Rotation may be ill-defined!")
        return np.sign(axa[:3].dot(self.axis_)) * axa[-1]

    @property
    def axis(self):
        return self.axis_

    @property
    def angle(self):
        """ alias to provide shorthand for obtaining joint angle """
        return self.param_

    def encode(self, *args, **kwargs) -> bytes:
        data = (
            self.source_frame_,
            self.target_frame_,
            self.stamp_,
            self.param_,
            self.axis_,
            self.offset_,
            self.offset_transform_)
        return encode(data, *args, **kwargs)

    def restore(self, data: bytes, *args, **kwargs):
        (self.source_frame_,
         self.target_frame_,
         self.stamp_,
         self.param_,
         self.axis_,
         self.offset_,
         self.offset_transform_) = decode(data, *args, **kwargs)

    @classmethod
    def decode(cls, data: bytes, *args, **kwargs):
        source, target, stamp, param, axis, offset, xfm = decode(
            data, *args, **kwargs)
        out = cls(source, target, axis, offset)
        out.stamp_ = stamp
        out.param_ = param
        out.offset_transform_ = xfm
        return out

    def __str__(self):
        return '{}({}->{})'.format(self.__class__.__name__, self.source, self.target)

    def __repr__(self):
        return '{}({}->{})'.format(self.__class__.__name__, self.source, self.target)


def main():
    """
    Test revolute joint edge forwards<->backwards.
    """

    import pickle
    data = pickle.dumps(BufferedFrameEdge(SimpleFrameEdge('source', 'target')))
    print('pickle : {}'.format(data))

    import time
    from phonebot.vis.viewer import PrimitiveViewer
    num_tests = 16384
    data_queue, event_queue = PrimitiveViewer.create()
    points = []
    for i in range(num_tests):
        stamp = 0.0
        axis = normalize(np.random.normal(size=3))
        offset = np.random.normal(size=3)
        edge = RevoluteJointEdge('source', 'target', axis, offset)

        angle = np.random.uniform(-np.pi, np.pi)

        edge.update(stamp, angle)
        t = edge.get_transform(stamp)
        t2 = Transform(t.position, Rotation.random())
        delta = np.random.normal(size=3, scale=0.1)

        # Uncomment following code to test invalid backwards calc
        # t2.position += delta

        # print('t<->t2', t, t2)
        error = adiff(angle, edge.to_param(t2))
        pos = edge.get_transform(stamp).position
        points.append(pos)

        if not data_queue.full():
            data_queue.put_nowait(
                {'point': dict(pos=np.reshape(points, [-1, 3]))})

        if not np.isclose(error, 0.0):
            logger.warn('Inconsistent transform : {} {} {}'.format(
                axis, offset, angle))

        time.sleep(0.001)


if __name__ == '__main__':
    main()
