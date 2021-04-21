#!/usr/bin/env python3

__all__ = ['PointTrackerAgent']

from typing import List, Dict, Tuple
import numpy as np

from phonebot.core.controls.controllers.base_rotation_controller import BaseRotationController
from phonebot.core.controls.controllers.base_cached_rotation_controller import BaseCachedRotationController
from phonebot.core.common.math.utils import alerp, normalize
from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
from phonebot.core.common.math.transform import Transform, Position, Rotation
from phonebot.core.common.config import PhonebotSettings, FrameName
from phonebot.core.controls.agents.base_agent.base_agent import BaseAgent


def compute_orientation(graph: PhonebotGraph, stamp,
                        config: PhonebotSettings) -> Tuple[float, float, float]:
    # Compute the plane assuming legs are all affixed to ground.
    # This assumption allows us to compute roll and pitch.
    foot_pos = []
    for leg_prefix in config.order:
        frame = FrameName(prefix=leg_prefix, suffix='a')
        xfm = graph.get_transform(frame.FOOT, frame.BODY, stamp)
        foot_pos.append(xfm.position)

    # TODO(yycho0108):
    # Evaluate all four possible planes, and select the solution
    # for which the COM is within the triangle.
    ux = normalize(foot_pos[config.index['FR']] -
                   foot_pos[config.index['HR']])
    uy = normalize(foot_pos[config.index['HL']] -
                   foot_pos[config.index['HR']])
    n = normalize(np.cross(ux, uy))  # z-vec IN body frame

    # Compute roll-pitch from normal vector.
    roll = np.arctan2(n[1], n[2])
    pitch = np.arctan2(-n[0] * np.cos(roll), n[2])

    # Compute distance of body frame from ground.
    # NOTE(yycho0108): This defines the local frame
    # as the "footprint" immediately below the body frame
    # in the +z axis direction of the local frame.
    z = np.dot(n, -foot_pos[config.index['FL']])

    return (roll, pitch, z)


class PointTrackerAgent(BaseAgent):
    def __init__(self, graph: PhonebotGraph, config=PhonebotSettings(),
                 use_cached_controller: bool = False):
        self.graph_ = graph
        self.config_ = config

        # TODO(ycho): Better architecture for maintaining generic cache
        if use_cached_controller:
            self.controller_ = BaseCachedRotationController(graph)
        else:
            self.controller_ = BaseRotationController(graph)

        self.roll_ = 0.0
        self.pitch_ = 0.0
        self.update_target(0, 0)

    def update_target(self, dx, dy, relative=True):
        if not relative:
            target_rotation = Rotation.from_euler(
                [dx, dy, 0.0]
            ).to_quaternion()
        else:
            # TODO(yycho0108): Figure out the negative for dx and dy
            target_rotation = Rotation.from_euler(
                [self.roll_ + dx, self.pitch_ + dy, 0.0]).to_quaternion()
        self.controller_.update(target_rotation)

    def update_orientation(self, roll, pitch, alpha=0.5):
        # NOTE(yycho0108): BaseCachedRotationController is open loop,
        # So updates to pitch/roll internal states doesn't really do anything.
        # they are only meaningful in the context of setting target angles.
        self.roll_ = alerp(self.roll_, roll, alpha)
        self.pitch_ = alerp(self.pitch_, pitch, alpha)

    def update_state_from_graph(self, stamp):
        graph = self.graph_
        config = self.config_

        # Orientation is computed from endpoint positions.
        roll, pitch, z = compute_orientation(graph, stamp, config)
        self.update_orientation(roll, pitch)
        # print(self.roll_, self.pitch_)

        # The graph is also updated.
        graph.get_edge(
            FrameName.BODY, FrameName.LOCAL).update(
            0.0,
            Transform(
                Position([0, 0, z]),
                Rotation.from_euler([self.roll_, self.pitch_, 0])))

    def __call__(self, state, stamp):
        self.update_state_from_graph(stamp)
        commands = self.controller_.control(stamp)
        return commands
