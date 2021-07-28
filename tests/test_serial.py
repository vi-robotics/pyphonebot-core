import unittest

from phonebot.core.common.serial import Serializable, decode, encode


class TestSerial(unittest.TestCase):

    def test_serialize_dummy(self):

        class DummySerializable(Serializable):
            def __init__(self, num: int) -> None:
                self.num = num

            def encode(self, *args, **kwargs) -> bytes:
                return encode(self.num, *args, **kwargs)

            @classmethod
            def decode(cls, data: bytes, *args, **kwargs) -> "Serializable":
                num = decode(data, *args, **kwargs)
                return cls(num)

            def restore(self, data: bytes, *args, **kwargs):
                num = decode(data, *args, **kwargs)
                self.num = num

        ds = DummySerializable(3)
        ds_bytes = ds.encode()
        ds_decode = DummySerializable.decode(ds_bytes)

        self.assertTrue(ds.num == ds_decode.num)


if __name__ == "__main__":
    unittest.main()
