#!/usr/bin/env python3

__all__ = ['get_graph_geometries']

import numpy as np

from phonebot.core.common.config import PhonebotSettings, FrameName
from phonebot.core.common.math.transform import Position, Rotation, Transform
from phonebot.core.common.math.utils import anorm, adiff, alerp
from phonebot.core.frame_graph.frame_graph import FrameGraph
from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
from phonebot.core.kinematics.inverse_kinematics import hip_ik
from phonebot.core.common.logger import get_default_logger
from phonebot.core.common.geometry.geometry import circle_intersection

logger = get_default_logger()


def get_graph_geometries(
        graph: FrameGraph, stamp: float, target_frame=FrameName.BODY,
        tol: float = 1e-2):
    poses = {}
    edges = []
    for frame in graph.frames:
        if not graph.has_transform(frame, target_frame, stamp, tol):
            continue
        pose = graph.get_transform(frame, target_frame, stamp, tol)
        poses[frame] = pose

        for neighbor in graph.get_connected_frames(frame):
            edge = graph.get_edge(frame, neighbor)
            if edge in edges:
                continue
            edges.append(edge)
    return poses, edges


def initialize_graph_vector(
        graph: PhonebotGraph, stamp: float, values,
        config=PhonebotSettings()):
    index = 0
    for leg_prefix in config.order:
        for leg_suffix in 'ab':
            frame = FrameName(prefix=leg_prefix, suffix=leg_suffix)
            graph.get_edge(frame.KNEE, frame.HIP).update(stamp, values[index])
            index += 1

    update_passive_joints(graph, stamp, config)


def initialize_graph_zero(graph: PhonebotGraph,
                          stamp: float, config=PhonebotSettings()):
    """ Initialize the graph to nominal joint positions. """
    logger.warn("Zero initialization is currently not well-supported!!")
    for leg_prefix in config.order:
        for leg_suffix in 'ab':
            frame = FrameName(prefix=leg_prefix, suffix=leg_suffix)
            graph.get_edge(frame.KNEE, frame.HIP).update(
                stamp, 0)
            graph.get_edge(frame.FOOT, frame.KNEE).update(
                stamp, 0)


def initialize_graph_nominal(
        graph: PhonebotGraph, stamp: float, config=PhonebotSettings()):
    """ Initialize the graph to nominal joint positions. """
    for leg_prefix in config.order:
        for leg_suffix in 'ab':
            frame = FrameName(prefix=leg_prefix, suffix=leg_suffix)
            graph.get_edge(frame.KNEE, frame.HIP).update(
                stamp, config.nominal_hip_angle)
            graph.get_edge(frame.FOOT, frame.KNEE).update(
                stamp, config.nominal_knee_angle)
    update_passive_joints(graph, stamp, config)


def get_joint_edges(graph: PhonebotGraph, config=PhonebotSettings()):
    """
    Get edges that pertain to a revolute joint.
    Currently all such joints are hardcoded.
    """
    # TODO(yycho0108): Consider caching into config settings.
    joint_edges = []
    for leg_prefix in config.order:
        for leg_suffix in 'ab':
            frame = FrameName(prefix=leg_prefix, suffix=leg_suffix)
            joint_edge = graph.get_edge(frame.KNEE, frame.HIP)
            joint_edges.append(joint_edge)
    return joint_edges


def update_passive_joints(graph: PhonebotGraph,
                          stamp: float, config=PhonebotSettings()):
    """ Update the angles passive joints after a change in active joints.
    Typically invoked after joint_edge.update(..., angle).
    """
    # Update passive joints.
    for leg_prefix in config.order:

        frame = FrameName(prefix=leg_prefix)
        frame_a = FrameName(prefix=leg_prefix, suffix='a')
        frame_b = FrameName(prefix=leg_prefix, suffix='b')

        knee_angle_a, knee_angle_b = solve_knee_angle(
            graph, leg_prefix, stamp, config=config)

        knee_edge_a = graph.get_edge(frame_a.FOOT, frame_a.KNEE)
        knee_edge_b = graph.get_edge(frame_b.FOOT, frame_b.KNEE)

        knee_edge_a.update(stamp, alerp(
            knee_edge_a.angle, knee_angle_a, 1.0))
        knee_edge_b.update(stamp, alerp(
            knee_edge_b.angle, knee_angle_b, 1.0))


def solve_inverse_kinematics_half(
        graph: PhonebotGraph, stamp: float, leg_prefix: str, leg_suffix: str,
        foot_position: Position, config=PhonebotSettings()):
    """
    Simpler IK alternative via solving circle intersections.
    TODO(yycho0108):
    Currently, the stamp lookup tolerance setting is set to inf.
    Is there some better way of handling this?
    """

    # Parse inputs and query current transforms.
    frame = FrameName(prefix=leg_prefix, suffix=leg_suffix)
    leg_from_hip = graph.get_transform(
        frame.HIP, frame.LEG, stamp, tol=np.inf)
    leg_from_knee = graph.get_transform(
        frame.KNEE, frame.LEG, stamp, tol=np.inf)

    # Extract relevant parameters.
    hip_position = leg_from_hip.position
    knee_position = leg_from_knee.position
    foot_leg = foot_position

    return hip_ik(foot_leg, hip_position, knee_position,
                  config.hip_sign[leg_suffix], config)


def solve_inverse_kinematics(
        graph: PhonebotGraph, stamp: float, leg_prefix: str,
        foot_position: Position, config=PhonebotSettings()):
    # Solve IK in the A-subassembly.
    sol_a = solve_inverse_kinematics_half(
        graph, stamp, leg_prefix, 'a', foot_position, config)
    if sol_a is None:
        return None

    # Solve IK in the B-subassembly.
    sol_b = solve_inverse_kinematics_half(
        graph, stamp, leg_prefix, 'b', foot_position, config)
    if sol_b is None:
        return None

    # After computing both solutions, format the output and return the solution.
    # NOTE(yycho0108): negation and offset in `sol_b` are due to frame
    # definitions in config.
    return anorm([sol_a, -(np.pi + sol_b)])


def solve_knee_angle(
        graph: PhonebotGraph, leg_prefix: str, stamp: float,
        config=PhonebotSettings(),
        foot_position=None):
    """
    Solve passive knee angle given the hip angles.
    Should be able to replace `solve_passive`.

    NOTE(yycho0108): Also accepts cached foot position, which MUST be specified in leg frame.
    """
    frame = FrameName(prefix=leg_prefix)
    frame_a = FrameName(prefix=leg_prefix, suffix='a')
    frame_b = FrameName(prefix=leg_prefix, suffix='b')

    leg_from_knee_a = graph.get_transform(frame_a.KNEE, frame.LEG, stamp)
    leg_from_knee_b = graph.get_transform(frame_b.KNEE, frame.LEG, stamp)

    knee_pos_a = leg_from_knee_a.position
    knee_pos_b = leg_from_knee_b.position
    radius = config.knee_link_length

    # Extract circle centers.
    ax = knee_pos_a.x
    ay = knee_pos_a.y
    bx = knee_pos_b.x
    by = knee_pos_b.y

    # TODO(yycho0108: Technically, there's no need to guess
    # foot_positions if info is already available.
    if foot_position is None:
        foot_positions = circle_intersection(
            (ax, ay, radius), (bx, by, radius))
        if foot_positions is None:
            logger.warn('{} {} {} {} {} {}', ax, ay, radius, bx, by, radius)
            logger.warn("Intersection compute somehow failed!")
            return

        if not config.select_nearest_ik:
            # If nearest-ik config is not enabled,
            # Of the >=1 solutions, accept the one closer to the ground.
            # NOTE(yycho0108):+y points downwards.
            foot_positions = np.asarray(foot_positions)
            foot_position = foot_positions[np.argmax(foot_positions[..., 1])]
            foot_positions = [foot_position]
    else:
        foot_positions = [foot_position]

    joint_a = graph.get_edge(frame_a.KNEE, frame_a.FOOT)
    joint_b = graph.get_edge(frame_b.KNEE, frame_b.FOOT)

    angles = []
    for x, y in foot_positions:
        pa = leg_from_knee_a.inverse() * Position([x, y, 0.0])
        pb = leg_from_knee_b.inverse() * Position([x, y, 0.0])
        a = joint_a.to_param(pa)
        b = joint_b.to_param(pb)
        angles.append([a, b])

    angles = np.float32(angles)

    # Debugging notes ...
    # print('----')
    # print(foot_positions)
    # print(graph.get_transform(frame_a.FOOT, frame.LEG, stamp))
    # print('====')
    # print(angles)
    # print(graph.get_edge(frame_a.KNEE, frame_a.FOOT).angle)
    # print(graph.get_edge(frame_b.KNEE, frame_b.FOOT).angle)
    # print('----')

    # print('angles')
    # print(angles)

    prv_angle_a = graph.get_edge(frame_a.KNEE, frame_a.FOOT).angle
    prv_angle_b = graph.get_edge(frame_b.KNEE, frame_b.FOOT).angle

    select_index = np.argmin(np.linalg.norm(
        adiff(angles, [prv_angle_a, prv_angle_b]), axis=-1))

    #print('angles comp')
    #print(prv_angle_a, prv_angle_b)
    # print(angles[select_index])

    return angles[select_index]


def main():
    pass


if __name__ == '__main__':
    main()
