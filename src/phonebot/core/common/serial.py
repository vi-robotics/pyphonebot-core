#!/usr/bin/env python3

from abc import abstractmethod, ABCMeta
from typing import Any, Dict

import zlib
import pickle
from collections import deque
from collections.abc import Iterable


class Serializable(metaclass=ABCMeta):
    """Abstract class describing a serializable object. To inherit, implement
    the abstract methods, using each methods documentation as a guide.
    """

    @abstractmethod
    def encode(self, *args, **kwargs) -> bytes:
        """Encode all relevant attributes of the instance to a bytes object,
        which decode and restore will be able to reverse.

        Raises:
            NotImplementedError: Override this method

        Returns:
            bytes: The encoded data in bytes object.
        """
        raise NotImplementedError("encode: Override this method")

    @abstractmethod
    def restore(self, data: bytes, *args, **kwargs):
        """Restore the current instance from a bytes array to one previously
        encoded using the encode method.

        Args:
            data (bytes): The bytes data to decode which was previously created
                by an encode call.

        Raises:
            NotImplementedError: Override this method
        """
        raise NotImplementedError("restore: Override this method")

    @classmethod
    @abstractmethod
    def decode(cls, data: bytes) -> "Serializable":
        """Decode the bytes object into a new instance of the class.

        Args:
            data (bytes): The encoded bytes to decode.

        Raises:
            NotImplementedError: Override this method

        Returns:
            Serializable: The new instance of the class created from the
                decoded bytes.
        """
        raise NotImplementedError("decode: Override this method")


def _encode(data: Any) -> bytes:
    """Wrapper for encoding data using the pickle module.

    Args:
        data (Any): The data to encode with pickle.

    Returns:
        bytes: The resulting bytes object representing the pickled data.
    """
    return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)


def _decode(data: bytes) -> Any:
    """Wrapper for decoding bytes using pickle

    Args:
        data (bytes): The bytes object to decode

    Returns:
        Any: The result of pickle decoding
    """
    return pickle.loads(data)


def encode(data: Serializable, class_map: Dict[Any, int] = None) -> bytes:
    """Encode a serializable data

    Args:
        data (Serializable): An instance of Serializable to encode.
        class_map (Dict[Any, int], optional): An existing mapping (dict) from
            class to int. If none is provided, then one is created and it's
            inverse is passed . Defaults to None.

    Returns:
        bytes: A tuple of class map value, optional inverse (if root), and the
            encoded data. 
    """
    # Resolve global object
    is_root = False
    if class_map is None:
        is_root = True
        class_map = {}

    cls = data.__class__
    dbytes = None
    if isinstance(data, Serializable):
        dbytes = data.encode(class_map)
    elif isinstance(data, bytes):
        # No need to further encode
        dbytes = data
    elif isinstance(data, str):
        dbytes = str.encode(data)
    elif isinstance(data, dict):
        dbytes = {encode(k, class_map): encode(v, class_map)
                  for k, v in data.items()}
    elif isinstance(data, (tuple, list, deque)):
        # Handle explicit list-like iterables.
        # NOTE(yycho0108): Not checking for __iter__
        # To prevent possible loss of information.
        dbytes = tuple([encode(x, class_map) for x in data])
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
    if cls not in class_map:
        class_map[cls] = len(class_map)

    if is_root:
        # export the inverse map.
        class_map_inverse = {v: k for k, v in class_map.items()}
        out = _encode((class_map_inverse, class_map[cls], dbytes))
    else:
        out = (class_map[cls], dbytes)
    return out


def compress(data: bytes) -> bytes:
    """Compress a byte array using zlib

    Args:
        data (bytes): Binary data to be compressed

    Returns:
        bytes: A byte array of compressed data.
    """
    return zlib.compress(data)


def decompress(data: bytes) -> bytes:
    """Decompress a byte array using zlib

    Args:
        data (bytes): Binary data to decompress

    Returns:
        bytes: A byte array of uncompressed data
    """
    return zlib.decompress(data)


def decode(data: bytes, class_map=None) -> Serializable:
    """Decode a serializable data

    Args:
        data (bytes): An byte array to decode.
        class_map (Dict[Any, int], optional): An existing mapping (dict) from
            class to int. If none is provided, then one is created and it's
            inverse is passed . Defaults to None.
    Returns:
        Serializable: The resulting decoded data.
    """

    # Resolve global object
    is_root = (class_map is None)
    if is_root:
        class_map, cls, d = _decode(data)
    else:
        cls, d = data  # d=dbytes

    # Lookup class from class map.
    cls = class_map[cls]

    if cls is None:
        return _decode(d)
    elif issubclass(cls, Serializable):
        return cls.decode(d, class_map)
    elif issubclass(cls, str):
        return d.decode()
    elif issubclass(cls, dict):
        return {decode(k, class_map): decode(v, class_map) for k, v in d.items()}
    elif issubclass(cls, list):
        return [decode(x, class_map) for x in d]
    elif issubclass(cls, tuple):
        return tuple(decode(x, class_map) for x in d)
    elif issubclass(cls, deque):
        return deque(decode(x, class_map) for x in d)
    elif issubclass(cls, bytes):
        return d
    elif issubclass(cls, Iterable):
        return cls([decode(x) for x in data])
    else:
        return _decode(d)
