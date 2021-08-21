#!/usr/bin/env python3

from typing import Any, Callable
import asyncio
import time
from phonebot.core.common.comm.protocol import send, recv


class SimpleServer(object):
    def __init__(self, on_data: Callable[[bytes], None], addr='0.0.0.0', port=11411,
                 stop_event: asyncio.Event = None):
        """A simple server instance which reads data from a StreamReader
        connected on the given address and port.

        Args:
            on_data (Callable[[bytes], None]): A callback which accepts the
                bytes data received by the server's StreamReader.
            addr (str, optional): The address to serve on. Defaults to
                '0.0.0.0'.
            port (int, optional): The port to serve on. Defaults to 11411.
            stop_event (asyncio.Event, optional): If provided, then stop the
                server if the event is set. Defaults to None.
        """

        self.on_data = on_data
        self.addr = addr
        self.port = port
        self.stop_event = stop_event

        self.loop = asyncio.get_event_loop()
        task = asyncio.start_server(
            self.on_client, self.addr, self.port, loop=self.loop)

        self.server = self.loop.run_until_complete(task)

    def run(self):
        """Run the server main loop, and exit on a shutdown event or 
        keyboard interupt.
        """
        try:
            while True:
                # Run the event loop one step at a time in order to catch
                # stop events.
                if self.stop_event.is_set():
                    self.stop()
                    break
                self.loop.call_soon(self.loop.stop)
                self.loop.run_forever()

        except KeyboardInterrupt:
            pass

    def stop(self):
        """Stop the server and close the loop.
        """
        self.loop.stop()
        self.server.close()
        self.loop.close()

    async def on_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        while True:
            # if self.stop_event.is_set():
            #     break
            msg = recv(reader)
            try:
                data = await msg
            except asyncio.CancelledError:
                break
            if data is None:
                break
            self.on_data(data)
        writer.close()
