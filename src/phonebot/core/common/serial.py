#!/usr/bin/env python3

from abc import abstractmethod, ABCMeta

import zlib
import sys
import pickle
from collections import deque


class Serializable(metaclass=ABCMeta):
    @abstractmethod
    def encode(self, *args, **kwargs) -> bytes:
        raise NotImplementedError("encode")

    @abstractmethod
    def restore(self, data: bytes):
        raise NotImplementedError("restore")

    @classmethod
    @abstractmethod
    def decode(cls, data: bytes):
        raise NotImplementedError("decode")


def _encode(data) -> bytes:
    return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)


def _decode(data: bytes):
    return pickle.loads(data)


def encode(data: Serializable, Z=None) -> bytes:
    # Resolve global object
    is_root = False
    if Z is None:
        is_root = True
        Z = {}

    cls = data.__class__
    dbytes = None
    if isinstance(data, Serializable):
        dbytes = data.encode(Z)
    elif isinstance(data, bytes):
        # No need to further encode
        dbytes = data
    elif isinstance(data, str):
        dbytes = str.encode(data)
    elif isinstance(data, dict):
        dbytes = {encode(k, Z): encode(v, Z) for k, v in data.items()}
    elif isinstance(data, (tuple, list, deque)):
        # Handle explicit list-like iterables.
        # NOTE(yycho0108): Not checking for __iter__
        # To prevent possible loss of information.
        dbytes = tuple([encode(x, Z) for x in data])
    elif isinstance(data, (int, float)):
        # Handle ~builtin data objects.
        dbytes = _encode(data)
        # Direct dumps do not require class information.
        cls = None
    else:
        # raise(TypeError('fallback = {}'.format(data.__class__)))
        dbytes = _encode(data)
        cls = None

    # Register class to class map.
    if cls not in Z:
        Z[cls] = len(Z)

    if is_root:
        # export the inverse map.
        Zi = {v: k for k, v in Z.items()}
        out = _encode((Zi, Z[cls], dbytes))
    else:
        out = (Z[cls], dbytes)
    return out


def compress(data: bytes) -> bytes:
    return zlib.compress(data)


def decode(data: bytes, Z=None) -> Serializable:
    # Resolve global object
    is_root = (Z is None)
    if is_root:
        Z, cls, d = _decode(data)
    else:
        cls, d = data  # d=dbytes

    # Lookup class from class map.
    cls = Z[cls]

    if cls is None:
        return _decode(d)
    elif issubclass(cls, Serializable):
        return cls.decode(d, Z)
    elif issubclass(cls, str):
        return d.decode()
    elif issubclass(cls, dict):
        return {decode(k, Z): decode(v, Z) for k, v in d.items()}
    elif issubclass(cls, list):
        return [decode(x, Z) for x in d]
    elif issubclass(cls, tuple):
        return tuple(decode(x, Z) for x in d)
    elif issubclass(cls, deque):
        return deque(decode(x, Z) for x in d)
    elif issubclass(cls, bytes):
        return d
    # elif hasattr(cls, '__iter__'):
    #    # warn
    #    return cls([decode(x) for x in data])
    else:
        # print('fallback = {} {}'.format(cls, d))
        return _decode(d)
        # raise NotImplementedError(
        #    'unsupported class = {}'.format(cls, d))
