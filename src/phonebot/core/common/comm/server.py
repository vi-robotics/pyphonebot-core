#!/usr/bin/env python3

import asyncio
from phonebot.core.common.comm.protocol import send, recv


class SimpleServer(object):
    def __init__(self, on_data, addr='0.0.0.0', port=11411):
        self.on_data = on_data
        self.addr = addr
        self.port = port

        self.loop = asyncio.get_event_loop()
        task = asyncio.start_server(
            self.on_client, self.addr, self.port, loop=self.loop)
        self.server = self.loop.run_until_complete(task)

    def __del__(self):
        self.server.close()
        self.loop.run_until_complete(self.server.wait_closed())
        self.loop.close()

    def run(self):
        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            pass

    async def on_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        while True:
            msg = recv(reader)
            data = await msg
            if data is None:
                break
            self.on_data(data)
        writer.close()


def on_data(data):
    print('Received : {}'.format(hash(data)))


def main():
    server = SimpleServer(on_data)
    server.run()


if __name__ == '__main__':
    main()
