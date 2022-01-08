#!/usr/bin/env python3

from typing import Any, Union
import json
import pickle
import io

_HEX_TAG = '__hex__'


class Encoder(json.JSONEncoder):
    """Extension of JSON encoder which handles Settings objects, or uses
    pickle as a fallback option.

    """

    def default(self, o: Any) -> Any:
        """Generate a serializable object to encode

        Args:
            o (Any): A Settings object, or a pickleable object, or an instance
                with default implemented.

        Returns:
            Any: A serializable object
        """
        if isinstance(o, Settings):
            return o.__dict__
        else:
            try:
                return super().default(o)
            except TypeError:
                # Fallback to pickle.
                return {'__str__': o.__str__(), _HEX_TAG: pickle.dumps(o).hex()}


class Decoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: Any) -> Any:
        """Recovers pickled data if decoded.

        Args:
            obj (Any): If a dictionary with a key "_HEX_TAG" then the data held
                by the value is depickeled and returned, else the object is
                retuned.

        Returns:
            Any: The decoded JSON object
        """
        # Recover from pickle-based fallback.
        if isinstance(obj, dict) and _HEX_TAG in obj:
            return pickle.loads(bytes.fromhex(obj[_HEX_TAG]))
        return obj


class Settings(object):
    """Settings base, supporting the following set of features:

        - Hierarchical composition.
        - JSON-based file IO.
    """

    def __init__(self, **kwargs):
        # NOTE(yycho0108): The assumption is subclasses will add properties here.
        self.update(kwargs)
        super().__init__()

    @classmethod
    def from_file(cls, path: str) -> "Settings":
        """Construct a Settings object from a JSON file.

        Args:
            path (str): The path to the JSON file.

        Returns:
            Settings: The constructed Settings object.
        """
        with open(path, 'r') as f:
            data = json.load(f, cls=Decoder)
            return cls(**data)

    @classmethod
    def from_string(cls, string: str) -> "Settings":
        """Construct a Settings object from a JSON string.

        Returns:
            "Settings": The constructed Settings object.
        """
        return cls(**json.loads(string, cls=Decoder))

    def update(self, data: Union[dict, "Settings"]):
        """Batch update the Settings object from a dictionary or another
        Settings object. See the @set function for more information."""

        # TODO(yycho0108): Cleanup this awkward back-and-forth with __dict__.
        if isinstance(data, Settings):
            return self.update(data.__dict__)
        for k, v in data.items():
            self.set(k, v)

    def set(self, key: str, value: Any):
        """Set the internal dictionary key to a given value. This sets the
        instance attribute to the value:

            Settings.key = value

        Args:
            key (str): The key to set. This will be the key of self.__dict__
                to which the value will be assigned.
            value (Any): The value to assign to the key.

        Raises:
            ValueError: Key is not supported.
        """
        # Key does not exist : apply subkey matching.
        if key not in self.__dict__:
            # Attempt subkey update.
            if ('.' in key):
                key, subkey = key.split('.', 1)
                if key in self.__dict__ and isinstance(self.__dict__[key],
                                                       Settings):
                    self.__dict__[key].set(subkey, value)
                    return

            raise ValueError(
                f"Supplied key [{key}] not supported for {self.__class__.__name__}. "
                f"Valid options : {list(self.__dict__.keys())}"
            )

        # Key exists: commit results.
        if isinstance(self.__dict__[key], Settings):
            self.__dict__[key].update(value)
        else:
            # Actual raw-value update.
            self.__dict__[key] = value

    def save(self, path: Union[io.IOBase, str]):
        """Save the Settings object to a JSON file.

        Args:
            path (Union[io.IOBase, str]): The path to write the file to.
        """
        if isinstance(path, io.IOBase):
            json.dump(self.__dict__, path, indent=4,
                      separators=(',', ':'), cls=Encoder)
        else:
            with open(path, 'w') as fp:
                self.save(fp)

    def load(self, path: Union[io.IOBase, str]):
        """Load the current instance from a JSON file.

        Args:
            path (Union[io.IOBase, str]): The path to load the settings from.
        """
        if isinstance(path, io.IOBase):
            data = json.load(path, cls=Decoder)
            for k, v in data.items():
                if isinstance(self.__dict__[k], Settings):
                    self.__dict__[k].update(v)
                else:
                    self.__dict__[k] = v
        else:
            with open(path, 'r') as f:
                self.load(f)

    def __str__(self):
        return json.dumps(self.__dict__, indent=4, separators=(',', ':'),
                          cls=Encoder)

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4, separators=(',', ':'),
                          cls=Encoder)
