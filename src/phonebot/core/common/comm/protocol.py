#!/usr/bin/env python3

import asyncio
import sys


async def send(data: bytes, writer: asyncio.StreamWriter):
    header = b'%d\n' % len(data)
    writer.write((header) + (data))
    return await writer.drain()


async def recv(reader: asyncio.StreamReader):
    header = await reader.readline()
    if not header:
        return
    msglen = int(header)
    return await reader.readexactly(msglen)
