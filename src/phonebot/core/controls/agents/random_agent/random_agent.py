#!/usr/bin/env python3

import numpy as np
import time

from phonebot.core.common.settings import Settings
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.controls.agents.base_agent.base_agent import BaseAgent


class RandomAgentSettings(Settings):
    """
    Random agent settings.
    Configures the generated random distribution.
    """
    mean: float
    scale: float

    def __init__(self, **kwargs):
        self.mean = 0.0
        self.scale = np.pi
        super().__init__(**kwargs)


class RandomAgent(BaseAgent):
    """
    Basic random agent.
    """

    def __init__(self, settings: PhonebotSettings, agent_settings: RandomAgentSettings):
        self.settings_ = settings
        self.agent_settings_ = agent_settings
        self.num_control_ = len(self.settings_.active_joint_names)
        super().__init__()

    def __call__(self, state, time_step):
        return np.random.normal(self.agent_settings_.mean, self.agent_settings_.scale, self.num_control_)
