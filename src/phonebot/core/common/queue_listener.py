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
        """A thread which listens for additions to a queue in a list and then
        calls a provided data processing function.

        Args:
            queue (Queue): The Queue to listen to.
            data_cb (Callable[[object], None]): The call back for processing
                the data in the queue.
            timeout (float, optional): The timeout for each queue get call,
                where the processing callback is called after the timeout.
                Defaults to 1.0.
        """
        self.queue_ = queue
        self.data_cb = data_cb
        self.timeout = timeout
        self.stop_event_ = Event()
        super().__init__(*args, **kwargs)

    def stop(self):
        """Stop the thread
        """
        self.stop_event_.set()

    def stopped(self) -> bool:
        """The stop event state

        Returns:
            bool: True if the stopped event is set.
        """
        return self.stop_event_.is_set()

    def run(self):
        """Run the queue get in a loop with a timeout. The only way to exit
        the loop is to raise the stop event (call self.stop()).
        """
        while not self.stopped():
            try:
                data = self.queue_.get(timeout=self.timeout)
                self.data_cb(data)
            except Empty:
                # Only allowable exception is queue.Empty
                continue
