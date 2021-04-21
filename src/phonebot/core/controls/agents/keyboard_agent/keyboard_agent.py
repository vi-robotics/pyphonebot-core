#!/usr/bin/env python3

import numpy as np
import time
import sys
from typing import Dict, Tuple

from multiprocessing import Process, Manager, Queue
from threading import Thread, Event

from phonebot.core.common.settings import Settings
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.controls.agents.base_agent.base_agent import BaseAgent
from phonebot.core.common.queue_listener import QueueListener
from phonebot.core.controls.agents.keyboard_agent.common import spawn_opencv_key_window


class KeyboardAgent(BaseAgent):
    """
    Sample - Basic keyboard agent that controls periodic offset of each leg.
    Requires keyboard event registration.
    """

    def __init__(self, event_source: str):
        # Setup according gui source that will generate the keyboard events.
        self.event_source_ = event_source
        self.listener_ = None
        self.setup_input(event_source)
        super().__init__()

    def setup_input(self, event_source: str):
        if event_source is None:
            # on_key will be called manually.
            return

        if event_source == 'opencv':
            self.queue_ = Queue()
            self.listener_ = QueueListener(self.queue_, self.on_key)
            spawn_opencv_key_window(self.queue_)
            self.listener_.start()
