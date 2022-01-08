#!/usr/bin/env python3

import time
from typing import Callable
import asyncio
import concurrent.futures
import logging

from collections import deque
from phonebot.core.common.comm.protocol import send, recv


class SimpleClient(object):
    def __init__(self, addr: str = '127.0.0.1', port: int = 11411,
                 start: bool = True):
        """A simple client instance which sends data from a StreamWriter
        connected on the given address and port of a server. The server
        should be started first.

        Args:
            addr (str, optional): The address to connect to. Defaults to
                '127.0.0.1'.
            port (int, optional): The port to connect to. Defaults to 11411.
            start (bool, optional): If True, then start the client on
                initialization. This will start the asyncio event loop.
                Defaults to True.
        """
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
        """True if the client is connected to the server. False otherwise.


        Returns:
            bool: Connection status
        """
        return (self.writer is not None)

    def __del__(self):
        if self.writer is not None:
            self.writer.close()
        self.loop.close()
        self.pool.shutdown()

    def start(self):
        """Start the client and asyncio event loop.
        """
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

    def send(self, message: bytes, on_done: Callable[[bytes], None] = None,
             wait: bool = False):
        """Send data to the server.

        Args:
            message (bytes): A byte array to send.
            on_done (Callable[[bytes], None], optional): An optional callback
                to call when send is complete. Defaults to None.
            wait (bool, optional): If True, then perform the action
                synchronously, else asynchronous. Defaults to False.
        """
        if wait:
            # synchronous execution
            self._send(message, on_done)
        else:
            # asynchronous execution
            self.pool.submit(self._send, message, on_done)
