#!/usr/bin/env python3

from multiprocessing import Queue
from queue import Empty
from threading import Thread, Event
from typing import Callable
import time
import logging


class QueueListener(Thread):
    """
    Generic queue listener.
    """

    def __init__(self, queue: Queue,
                 data_cb: Callable[[object], None],
                 timeout: float = 1.0,
                 *args, **kwargs):
        self.queue_ = queue
        self.data_cb = data_cb
        self.timeout = timeout
        self.stop_event_ = Event()
        super().__init__(*args, **kwargs)

    def stop(self):
        self.stop_event_.set()

    def stopped(self):
        return self.stop_event_.is_set()

    def run(self):
        while not self.stopped():
            try:
                data = self.queue_.get(timeout=self.timeout)
                self.data_cb(data)
            except Empty:
                # Only allowable exception is queue.Empty
                continue


def main():
    indices = []

    def data_cb(i):
        indices.append(i)

    queue = Queue()
    queue_listener = QueueListener(queue, data_cb)
    queue_listener.start()

    for i in range(64):
        queue.put_nowait(i)

    while len(indices) < 64:
        time.sleep(1e-9)

    print(indices)


if __name__ == '__main__':
    main()
