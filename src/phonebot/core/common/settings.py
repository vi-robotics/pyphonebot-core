#!/usr/bin/env python3

import json
import pickle
import io

_HEX_TAG = '__hex__'


class Encoder(json.JSONEncoder):
    def default(self, o):
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

    def object_hook(self, obj):
        # Recover from pickle-based fallback.
        if isinstance(obj, dict) and _HEX_TAG in obj:
            return pickle.loads(bytes.fromhex(obj[_HEX_TAG]))
        return obj


class Settings(object):
    """
    Settings base, supporting the following set of features:
    * Hierarchical composition.
    * JSON-based file IO.
    """

    def __init__(self, **kwargs):
        # NOTE(yycho0108): The assumption is subclasses will add properties here.
        self.update(kwargs)
        super().__init__()

    def update(self, data: dict):
        """ Batch update, @see set()"""
        # TODO(yycho0108): Cleanup this awkward back-and-forth with __dict__.
        if isinstance(data, Settings):
            return self.update(data.__dict__)
        for k, v in data.items():
            self.set(k, v)

    def set(self, key, value):
        """
        Settings.key = value
        """
        # Key does not exist : apply subkey matching.
        if key not in self.__dict__:
            # Attempt subkey update.
            if ('.' in key):
                key, subkey = key.split('.', 1)
                if key in self.__dict__ and isinstance(self.__dict__[key], Settings):
                    self.__dict__[key].set(subkey, value)
                    return

            raise ValueError(
                "Supplied key [{}] not supported for {}. "
                "Valid options : {}"
                .format(key, self.__class__.__name__, list(self.__dict__.keys())))

        # Key exists: commit results.
        if isinstance(self.__dict__[key], Settings):
            self.__dict__[key].update(value)
        else:
            # Actual raw-value update.
            self.__dict__[key] = value

    def __str__(self):
        return json.dumps(self.__dict__, indent=4, separators=(',', ':'), cls=Encoder)

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4, separators=(',', ':'), cls=Encoder)

    def save(self, path: str):
        if isinstance(path, io.IOBase):
            json.dump(self.__dict__, path, indent=4,
                      separators=(',', ':'), cls=Encoder)
        else:
            with open(path, 'w') as fp:
                self.save(fp)

    def load(self, path: str):
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

    @classmethod
    def from_file(cls, path):
        with open(path, 'r') as f:
            data = json.load(f, cls=Decoder)
            return cls(**data)

    @classmethod
    def from_string(cls, string):
        return cls(**json.loads(string, cls=Decoder))
