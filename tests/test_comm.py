import unittest
import threading
import time
import numpy as np
import asyncio
from src.phonebot.core.common.comm.server import SimpleServer
from src.phonebot.core.common.comm.client import SimpleClient


class TestClientServer(unittest.TestCase):

    def setUp(self):
        self.stop_event = asyncio.Event()
        self.data_arr = []
        self.server = SimpleServer(
            self.on_data, addr='127.0.0.1', stop_event=self.stop_event)
        self.client = SimpleClient()

    def test_client_send(self):
        """Test that the server receives data the client sends.
        """
        b = np.random.normal(size=(10, 10)).tobytes()
        for _ in range(5):
            self.client.send(b, self.on_done)

        time.sleep(0.1)
        for data in self.data_arr:
            self.assertTrue(b == data)

    def tearDown(self):
        self.stop_event.set()

    def on_data(self, data):
        self.data_arr.append(data)

    def on_done(self, data):
        pass


if __name__ == '__main__':
    unittest.main()
