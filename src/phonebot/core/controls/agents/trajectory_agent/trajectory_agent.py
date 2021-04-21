#!/usr/bin/env python3

import sys

import time
import numpy as np
import pickle
import os

from typing import Dict, List, Tuple

from phonebot.core.common.math.transform import Transform, Position, Rotation
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.controls.trajectory import WaypointTrajectory, ParametricTrajectory, Trajectory
from phonebot.core.controls.trajectory import CyclicKernel, EllipticalKernel, NegativeKernel, UnaryKernel, ConstantKernel, PrintKernel
from phonebot.core.controls.trajectory.trajectory_utils import get_elliptical_trajectory
from phonebot.core.controls.controllers.trajectory_controller import TrajectoryController, EndpointTrajectoryGraphController
from phonebot.core.kinematics.workspace import get_workspace, max_rect
from phonebot.core.controls.agents.base_agent.base_agent import BaseAgent


class TrajectoryAgent(BaseAgent):
    def __init__(self, length=200, period=4.0):
        # Save input parameters.
        self.length_ = length
        self.period_ = period

        # Compute elliptical trajectories accordingly.
        leg_trajectories = get_elliptical_trajectories(length)
        self.fl_, self.fr_, self.bl_, self.br_ = leg_trajectories

        # Independently setup controllers for each leg.
        self.controllers_ = [
            TrajectoryController(Trajectory(
                times=None, values=v, period=period))
            for v in [self.fl_, self.fr_, self.bl_, self.br_]]

    def __call__(self, state, stamp):
        # FIXME(yycho0108): arbitrary parsing logic as such should be fixed.
        j_act = state.reshape(-1, 2, 2)[:, :, 0]
        j_psv = state.reshape(-1, 2, 2)[:, :, 1]
        ctrl = []

        # TODO(yycho0108): write more robust environment parse_state()
        for controller, (h1, h2), (h3, h4) in zip(
                self.controllers_, j_act, j_psv):
            seed = dict(h1=h1, h2=h2, h3=h3, h4=h4)
            ctrl.extend(controller.control(seed, t))
        return ctrl


class TrajectoryAgentGraph(BaseAgent):
    """
    Trajectory Agent with automated feedback using the global frame graph.
    FIXME(yycho0108): I think all trajectory agents should be refactored not to rely on its own handle to the frame graph?
    """

    def __init__(self, graph, period=4.0, config=PhonebotSettings(),
                 trajectories: Dict[str, Trajectory] = None):
        self.graph_ = graph
        self.period_ = period

        # Get maximal ellipse bounds.
        if config.use_cached_rect:
            self.max_rect_ = config.max_rect
        else:
            self.workspace_ = get_workspace(0.0, config, return_poly=True)
            self.max_rect_ = max_rect(self.workspace_, 4096)
        (x0, y0), (x1, y1) = self.max_rect_
        self.max_rect_ = (0.5 * (x0 + x1), 0.5 * (y0 + y1),
                          abs(x1 - x0), 1.25 * abs(y1 - y0))

        if trajectories is None:
            # FIXME(yycho0108): Remove ugly sign remapping here.
            tau = 2 * np.pi
            trajectories = {
                'FL': get_elliptical_trajectory(
                    self.max_rect_,
                    self.period_,
                    0.25 * tau,
                    False,
                    0.7),
                'FR': get_elliptical_trajectory(
                    self.max_rect_,
                    self.period_,
                    0.75 * tau,
                    True,
                    0.7),
                'HL': get_elliptical_trajectory(
                    self.max_rect_,
                    self.period_,
                    0.0 * tau,
                    False,
                    0.7),
                'HR': get_elliptical_trajectory(
                    self.max_rect_,
                    self.period_,
                    0.5 * tau,
                    True,
                    0.7)}
        self.trajectories_ = trajectories

        self.controllers_ = [
            EndpointTrajectoryGraphController(graph,
                                              '{}_leg_origin'.format(
                                                  leg_prefix),
                                              self.trajectories_[leg_prefix],
                                              config)
            for leg_prefix in config.order]

    def __call__(self, state, stamp):
        commands = []
        # TODO(yycho0108): either enforce strict compliance to joint order,
        # Or return a dict instead.
        for controller in self.controllers_:
            command = controller.control(stamp)
            # commands.append(command)
            commands.extend(command)
        return commands


def visualize_default_gait_policy():
    # Importing in inner scope to prevent
    # visualization imports from creeping into global scope.
    from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
    from matplotlib import pyplot as plt
    period = 16.0
    times = np.linspace(0.0, 2 * period)
    config = PhonebotSettings()
    graph = PhonebotGraph(config)
    agent = TrajectoryAgentGraph(graph, period, config)
    positions = {name: trajectory.evaluate(times)
                 for name, trajectory in agent.trajectories_.items()}
    i = 0
    for name, pos in positions.items():
        ax = plt.gcf().add_subplot(4, 1, 1 + i)
        x = pos[..., 1]
        ax.plot(times, pos[..., 1], label=name)
        ax.grid()
        ax.legend()
        ax.set_title(name)
        i += 1
    plt.show()


def main():
    visualize_default_gait_policy()


if __name__ == '__main__':
    main()
