#!/usr/bin/env python3

import numpy as np

from phonebot.core.common.config import PhonebotSettings
from phonebot.core.common.math.utils import anorm, adiff
from phonebot.core.common.geometry.geometry import circle_intersection
from phonebot.core.common.math.transform import Position, Rotation, Transform
from phonebot.core.common.logger import get_default_logger

logger = get_default_logger()


def hip_ik(
        foot_position: Position,
        hip_position: Position,
        knee_position: Position,
        hip_sign: int,
        config=PhonebotSettings):
    """
    Solve for hip-joint angles based on the desired foot_position.

    Note:
        Only the x-y values for the respective positions are utilized,
        since the assumption here is that all positions lie on the
        z-plane as defined with respect to the leg origin frame.

    Arguments:
        foot_position: Endpoint in leg frame.
        hip_position: Hip joint in leg frame.
        knee_position: Knee joint position in leg frame. (Needed for solution selection)
        config: Phonebot settings.

    Returns:
        joint angle to reach the specified foot_position.

    """
    # Parse inputs.
    x, y = foot_position[:2]

    # Compute analytic intersection.
    knee_solutions = circle_intersection(
        (hip_position.x, hip_position.y, config.hip_link_length),
        (x, y, config.knee_link_length))
    if knee_solutions is None:
        logger.warn(
            '{} {} {} {} {} {}'.format(
                hip_position.x,
                hip_position.y,
                config.hip_link_length,
                x,
                y,
                config.knee_link_length))
        logger.warn("Intersection compute somehow failed!")
        return None
    knee_solutions = np.asarray(knee_solutions, dtype=np.float32)

    # Reject solutions that cross the collision boundary.
    knee_solutions = [s for s in knee_solutions
                      if (hip_sign * s[0] >= 0)]

    # Select nearest solution.
    select_index = np.argmin(np.linalg.norm(
        knee_solutions - knee_position[:2], axis=-1))

    knee_solution = knee_solutions[select_index]

    # Convert back to angle.
    offset = Position([knee_solution[0], knee_solution[1], 0.0]) - hip_position
    return np.arctan2(offset[1], offset[0])
