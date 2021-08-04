#!/usr/bin/env python3

import numpy as np

from phonebot.core.common.config import PhonebotSettings
from phonebot.core.common.math.transform import Position, Rotation, Transform
from phonebot.core.controls.trajectory import Trajectory
from phonebot.core.frame_graph.frame_graph import FrameGraph
from phonebot.core.frame_graph.graph_utils import solve_inverse_kinematics
from phonebot.core.common.math.utils import lerp, norm
from phonebot.core.kinematics.workspace import get_workspace
# from phonebot.core.controls.pid import PID

from phonebot.core.common.logger import get_default_logger
logger = get_default_logger()

from phonebot.core.planning.cwp import CircleWorldPlanner


def _clamp_target(source: Position, target: Position,
                  max_distance=0.01) -> Position:
    """Clamp target position to a maximum distance from the source position."""
    # Currently sets the output position to a small increment from current
    # position instead.

    # FIXME(yycho0108): Instead of a naive distance-based increment
    # and fallback behavior to prior target (which may still be unstable),
    # Apply a policy where the set target is a point on a spline
    # Connecting from source to target, that has been offset
    # from source by a certain distance.

    delta = (target - source)
    dist = np.linalg.norm(delta)
    if dist > max_distance:
        target = source + (max_distance / dist) * delta
    return target


class TrajectoryController(object):
    """Open loop controller that follows the supplied trajectory evaluated
    according to the timestamp."""

    def __init__(self, trajectory: Trajectory):
        self.trajectory_ = trajectory
        pass

    def control(self, stamp):
        target = self.trajectory_.evaluate(stamp)
        return target


class EndpointTrajectoryGraphController(object):
    """Alternative version of EndpointTrajectoryController that automatically
    sets up required transform lookups from the provided frame graph.

    FIXME(yycho0108): Wrap around EndpointTrajectoryController instead?
    """

    def __init__(
            self, graph: FrameGraph, frame: str, trajectory: Trajectory,
            config: PhonebotSettings = PhonebotSettings()):
        # TODO(ycho): [REFACTOR] suffix-underscore -> prefix-underscore;
        # leaving as-is for now in order to not clutter the PR with multiple features.
        self.graph_ = graph
        self.frame_ = frame
        # FIXME(yycho0108): Avoid parsing frame string.
        self.leg_prefix_ = frame.split('_')[0]
        self.trajectory_ = trajectory
        self.config_ = config
        self.planner_ = CircleWorldPlanner.from_phonebot(self.config_)

        # TODO(yycho0108): enable workspace querying once path-planning option is implemented.
        #self.workspace_ = get_workspace(0.0, config, return_poly=True)
        self.prev_stamp = 0.0

    def control(self, stamp):
        # Compute current endpoint.
        leg_from_foot = self.graph_.get_transform(
            '{}_foot_a'.format(
                self.leg_prefix_), '{}_leg_origin'.format(
                self.leg_prefix_), stamp)
        source = leg_from_foot.position

        # Compute target endpoint.
        foot_origin = self.trajectory_.evaluate(stamp)

        # Option 1)
        # Directly set target to trajectory evaluation output.
        target = foot_origin

        # Option 2)
        # Apply slight interpolation on the result,
        # mostly intended to prevent "jumps" resulting in degeneracy.
        # target = lerp(source, foot_origin, 0.5)

        # NOTE(yycho0108): If source-target distance is too high, rectify target position.
        # FIXME(yycho0108): Remove implicit 0.01 value.

        # Option 3.a) clip by heuristic (position difference between trajectory evaluations)
        # prev_foot_frame = self.trajectory_.evaluate(self.prev_stamp)
        # clip_distance = norm(foot_origin - prev_foot_frame)
        # Option 3.b) clip by hardcoded constant
        # clip_distance = 0.01
        # clipped_target = _clamp_target(source, target, clip_distance)
        # Option 3.c) no clipping
        clipped_target = target

        # Attempt 1 - solve IK on modified target.
        angles = None
        try:
            angles = solve_inverse_kinematics(
                self.graph_, stamp, self.leg_prefix_, clipped_target, self.config_)
        except Exception:
            # NOTE(ycho): Figure out why exceptions are being thrown
            # and remove this try-catch if possible.
            pass

        # NOTE(ycho): In the current code, Attempt 1 == Attempt 2.
        # Attempt 2 - solve IK directly to target.
        # This should ideally never happen.
        if angles is None:
            angles = solve_inverse_kinematics(
                self.graph_, stamp, self.leg_prefix_, target, self.config_)

        # FIXME(ycho): stateful controller.
        # Ideally, `prev_stamp` would either be passed in as part of a universal interface,
        # or eliminated completely.
        self.prev_stamp = stamp

        return angles


class EndpointTrajectoryGraphControllerManual(object):
   """Alternative version of EndpointTrajectoryControllerManual that enables
   manually setting the current trajectory position.

   FIXME(yycho0108): Reduce code duplication here.
   FIXME(ycho): Consider removing this class entirely?
   """

   def __init__(
           self, graph: FrameGraph, frame: str, trajectory: Trajectory,
           config: PhonebotSettings = PhonebotSettings(),
           initial_position=0.0):
       self.graph_ = graph
       self.frame_ = frame
       # FIXME(yycho0108): Avoid parsing frame string.
       # FIXME(ycho): Refactor the whole frame reference architecture.
       self.leg_prefix_ = frame.split('_')[0]
       self.trajectory_ = trajectory
       self.config_ = config
       self.pos_ = initial_position

   def set_position(self, pos):
       self.pos_ = pos

   def get_position(self):
       return self.pos_

   def control(self, stamp):
       # Compute current endpoint.
       body_from_foot = self.graph_.get_transform(
           '{}_foot_a'.format(self.leg_prefix_), 'body', stamp)
       source = body_from_foot.position

       # Compute target endpoint.
       foot_origin = self.trajectory_.evaluate(self.pos_)
       body_from_origin = self.graph_.get_transform(
           self.frame_, 'body', stamp)
       foot_body = body_from_origin * foot_origin

       # Apply slight interpolation on the result,
       # mostly intended to prevent "jumps" resulting in degeneracy.
       # target = lerp(body_from_foot.position, foot_body, 0.5)
       target = foot_body

       # FIXME(yycho0108): Remove implicit 0.01 value.
       # target = _clamp_target(source, target)
       angles = solve_inverse_kinematics(
           self.graph_, stamp, self.leg_prefix_, target, self.config_)

       if angles is None:
           angles = solve_inverse_kinematics(
               self.graph_, stamp, self.leg_prefix_, target, self.config_)
       return angles
