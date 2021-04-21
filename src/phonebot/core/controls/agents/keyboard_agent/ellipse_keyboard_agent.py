#!/usr/bin/env python3

import numpy as np
import time
import sys
from typing import Dict, Tuple, List

from phonebot.core.common.math.utils import alerp
from phonebot.core.common.settings import Settings
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
from phonebot.core.controls.trajectory.trajectory_utils import get_elliptical_trajectory
from phonebot.core.controls.agents.keyboard_agent.keyboard_agent import KeyboardAgent
from phonebot.core.controls.controllers.trajectory_controller import EndpointTrajectoryGraphControllerManual
from phonebot.core.kinematics.workspace import get_workspace, max_rect
from phonebot.core.common.logger import get_default_logger

logger = get_default_logger()


class EllipseKeyboardAgentSettings(Settings):
    """
    Joint keyboard control settings.
    """

    key_map: Dict[str, Tuple[str, float]]
    event_source: str

    def __init__(self, **kwargs):
        self.key_map = {
            'w': ('FL', 0.03),
            'e': ('FR', 0.03),
            'r': ('HL', 0.03),
            't': ('HR', 0.03),
            's': ('FL', -0.03),
            'd': ('FR', -0.03),
            'f': ('HL', -0.03),
            'g': ('HR', -0.03),
        }
        self.event_source = None
        super().__init__(**kwargs)


class EllipseKeyboardAgent(KeyboardAgent):
    """
    Sample - Basic keyboard agent that controls periodic offset of each leg.
    Requires keyboard event registration.
    """

    def __init__(self, graph: PhonebotGraph, settings: PhonebotSettings, agent_settings: EllipseKeyboardAgentSettings):
        # Save arguments.
        self.settings_ = settings
        self.agent_settings_ = agent_settings
        self.graph_ = graph

        # Get maximal ellipse bounds.
        if settings.use_cached_rect:
            rect = settings.max_rect
        else:
            workspace = get_workspace(0.0, settings, return_poly=True)
            rect = max_rect(workspace, 4096)

        (x0, y0), (x1, y1) = rect
        rect = (0.5 * (x0+x1), 0.5 * (y0 + y1),
                abs(x1-x0), abs(y1-y0))

        # FIXME(yycho0108): Remove ugly sign remapping here.
        self.trajectories_ = {
            'FL': get_elliptical_trajectory(rect, 1.0, -np.pi/2, False, 0.8),
            'FR': get_elliptical_trajectory(rect, 1.0, np.pi/2, True, 0.8),
            'HL': get_elliptical_trajectory(rect, 1.0, np.pi/2, False, 0.8),
            'HR': get_elliptical_trajectory(rect, 1.0, -np.pi/2, True, 0.8)
        }

        # Setup controllers.
        self.controllers_ = [
            EndpointTrajectoryGraphControllerManual(
                graph,
                '{}_leg_origin'.format(leg_prefix),
                self.trajectories_[leg_prefix],
                settings, 0.5)
            for leg_prefix in settings.order]
        # self.control_ = [ctrl.get_position() for ctrl in self.controllers_]
        self.control_ = [0.5, 0.5, 0.5, 0.5]

        super().__init__(agent_settings.event_source)
        logger.debug('Key Map : {}'.format(agent_settings.key_map))

    def on_key(self, key):
        """
        Update action value according to the registered keymap.
        """
        if key not in self.agent_settings_.key_map:
            # raise ValueError('JointKeyboardAgent called with unknown key : {}')
            return

        # Apply increment based on the key.
        leg, inc = self.agent_settings_.key_map[key]
        idx = self.settings_.index[leg]
        self.control_[idx] += inc

        # Update position accordingly.
        for controller, value in zip(self.controllers_, self.control_):
            controller.set_position(value)

    def __call__(self, state, stamp):
        commands = []
        # TODO(yycho0108): either enforce strict compliance to joint order,
        # Or return a dict instead.
        for controller in self.controllers_:
            command = controller.control(stamp)
            commands.extend(command)
        x = state[self.settings_.active_joint_indices]
        y = commands
        out = alerp(x, y, 0.2)
        logger.debug('ctrl : {}'.format(out))
        return out


def main():
    pass


if __name__ == '__main__':
    main()
