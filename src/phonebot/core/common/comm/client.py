#!/usr/bin/env python3

import time
import asyncio
import concurrent.futures
import logging

from collections import deque
from phonebot.core.common.comm.protocol import send, recv


class SimpleClient(object):
    def __init__(self, addr='127.0.0.1', port=11411, start=True):
        self.addr = addr
        self.port = port
        self.loop = asyncio.get_event_loop()
        # NOTE(yycho0108): StreamWriter is NOT thread safe.
        self.pool = concurrent.futures.ThreadPoolExecutor(1)
        self.reader = None
        self.writer = None
        if start:
            self.start()

    @property
    def connected(self) -> bool:
        return (self.writer is not None)

    def __del__(self):
        if self.reader is not None:
            self.reader.close()
        if self.writer is not None:
            self.writer.close()
        self.loop.close()
        self.pool.shutdown()

    def start(self):
        if self.reader is None and self.writer is None:
            try:
                task = self.loop.create_task(asyncio.open_connection(self.addr, self.port,
                                                                     loop=self.loop))
                self.reader, self.writer = self.loop.run_until_complete(task)
            except ConnectionRefusedError as e:
                logging.warn('Connection refused : {}'.format(e))
        else:
            print('Already started')

    def _send(self, message: bytes, on_done=None):
        task = self.loop.create_task(send(message, self.writer))
        result = self.loop.run_until_complete(task)
        if on_done is not None:
            on_done(result)

    def send(self, message: bytes, on_done=None, wait=False):
        if wait:
            # synchronous execution
            self._send(message, on_done)
        else:
            # asynchronous execution
            self.pool.submit(self._send, message, on_done)


def main():
    import numpy as np

    def on_done(x):
        print('on_done')
        print(x)

    client = SimpleClient()
    # client.start()
    b = np.random.normal(size=(512, 512)).tobytes()
    for _ in range(5):
        client.send(b, on_done)

    # wait 5 sec.
    now = time.time()
    while time.time() < (now + 5):
        time.sleep(0.1)


if __name__ == '__main__':
    main()
