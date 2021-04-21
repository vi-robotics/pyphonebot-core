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
from phonebot.core.common.logger import get_default_logger
from phonebot.core.common.queue_listener import QueueListener
from phonebot.core.controls.agents.keyboard_agent.common import spawn_opencv_key_window
from phonebot.core.controls.agents.keyboard_agent.keyboard_agent import KeyboardAgent

logger = get_default_logger()


class JointKeyboardAgentSettings(Settings):
    """
    Joint keyboard control settings.
    """

    key_map: Dict[str, Tuple[np.ndarray, np.ndarray]]
    event_source: str

    def __init__(self, **kwargs):
        self.key_map = {}
        self.event_source = None
        super().__init__(**kwargs)


class JointKeyboardAgent(KeyboardAgent):
    """
    Sample - Basic keyboard agent.
    Requires keyboard event registration.
    """

    def __init__(self, settings: PhonebotSettings, key_settings: JointKeyboardAgentSettings):
        self.settings_ = settings
        self.key_settings_ = key_settings

        self.num_control_ = len(self.settings_.active_joint_names)
        self.control_ = np.zeros(self.num_control_)
        logger.info('KeyMap : {}'.format(self.key_settings_.key_map))

        super().__init__(key_settings.event_source)

    def on_key(self, key):
        """
        Update action value according to the registered keymap.
        """
        if key not in self.key_settings_.key_map:
            # raise ValueError('JointKeyboardAgent called with unknown key : {}')
            return

        idx, val = self.key_settings_.key_map[key]
        self.control_[idx] = val

    def __call__(self, state, time_step):
        return self.control_


def main():
    settings = PhonebotSettings()

    key_map = {
        'w': ([1, 2, 3], [1, 2, 3]),
        'a': ([0], [1]),
        '0': (np.arange(8), np.zeros(8))
    }

    key_settings = JointKeyboardAgentSettings(
        key_map=key_map,
        event_source='opencv')

    agent = JointKeyboardAgent(settings, key_settings)

    s, t = None, None
    for _ in range(100):
        ctrl = agent(s, t)
        print(ctrl)
        time.sleep(0.01)


if __name__ == '__main__':
    main()
