#!/usr/bin/env python3

from typing import Coroutine, Any
import asyncio


async def send(data: bytes, writer: asyncio.StreamWriter
               ) -> None:
    """Send a byte array using a stream writer.

    Args:
        data (bytes): The bytes data to send.
        writer (asyncio.StreamWriter): The stream writer to send data
            on.
    """
    header = b'%d\n' % len(data)
    writer.write((header) + (data))
    return await writer.drain()


async def recv(reader: asyncio.StreamReader) -> bytes:
    """Receive data using a stream reader.

    Args:
        reader (asyncio.StreamReader): The stream reader to receive data from.

    Returns:
        bytes: The received bytes.
    """
    header = await reader.readline()
    if not header:
        return b''
    msglen = int(header)
    return await reader.readexactly(msglen)
