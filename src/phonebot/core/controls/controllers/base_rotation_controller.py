#!/usr/bin/env python3

import numpy as np
import shapely.ops
import shapely.geometry

from phonebot.core.common.config import PhonebotSettings
from phonebot.core.common.logger import get_default_logger
from phonebot.core.common.geometry.geometry import parallel_component, tangent_component, plane_plane_intersection
from phonebot.core.common.math.utils import normalize, lerp
from phonebot.core.common.math.transform import Position, Rotation, Transform
from phonebot.core.kinematics.workspace import get_workspace
from phonebot.core.frame_graph.graph_utils import solve_inverse_kinematics

logger = get_default_logger()


class BaseRotationController(object):
    """
    Controller that commands the local rotation of the base.
    """

    def __init__(self, graph, frame='body', config=PhonebotSettings()):
        self.graph_ = graph
        self.frame_ = frame
        self.config_ = config

        self.rotation_ = Rotation.identity()

        self.distance_ = None
        self.normal_ = None

        # Compute target.
        self.base_workspace_ = get_workspace(-0.005)
        self.base_workspace_poly_ = shapely.geometry.Polygon(
            np.asarray(self.base_workspace_, dtype=np.float32)[..., :2])
        self.workspace_ = {}

        for leg_prefix in self.config_.order:
            # Get static transform.
            leg_origin = '{}_leg_origin'.format(leg_prefix)
            body_from_leg = self.graph_.get_transform(
                leg_origin, self.frame_, 0.0)

            # append to workspace.
            self.workspace_[leg_prefix] = body_from_leg * \
                Position(self.base_workspace_)

    def update(self, rotation):
        """ Set rotation target. """
        # rotation = local_from_body
        # body.rotation() = rotation * local.rotation()
        # Obtain plane normal vector from rotation:
        # Essentially removes the yaw component.
        normal = rotation.inverse().rotate(Position([0, 0, 1]))

        # Find the feasibility boundary:
        # find max-min and min-max projections of workspaces
        d_max_min = -np.inf
        d_min_max = np.inf
        for leg_prefix, workspace in self.workspace_.items():
            distances = workspace.dot(normal)
            d_min = np.min(distances)
            d_max = np.max(distances)

            d_min_max = min(d_min_max, d_max)
            d_max_min = max(d_max_min, d_min)

        if d_max_min > d_min_max:
            # not possible
            logger.warn(
                'Rotation not possible : {}-{}'.format(d_max_min, d_min_max))
            return

        # NOTE(yycho0108):
        # Deteremine distance of local from body within feasible bounds.
        # Currently selecting the point fourth of the way from the minimum,
        # Since we don't want to get too close to the base itself.
        self.normal_ = rotation.rotate(Position([0, 0, 1]))
        self.distance_ = lerp(d_max_min, d_min_max, 0.25)

        # Project plane to lines in leg frame
        self.cache_ = {}
        visuals = {}
        for leg_prefix in self.config_.order:
            # Build primitives.
            leg_origin = '{}_leg_origin'.format(leg_prefix)
            leg_from_body = self.graph_.get_transform(
                self.frame_, leg_origin, 0.0)
            body_from_leg = leg_from_body.inverse()

            # Get intersection point.
            p_start = plane_plane_intersection(
                self.distance_ * normal, normal,
                body_from_leg.position, [0, 1, 0]
            )
            p_start = leg_from_body * Position(p_start)

            projected_tangent_leg = leg_from_body * \
                Position(np.cross(normal, [0, 1, 0]))

            pa = p_start + 1000 * projected_tangent_leg
            pb = p_start - 1000 * projected_tangent_leg
            path = shapely.geometry.LineString([(pa.x, pa.y), (pb.x, pb.y)])
            intersections = path.intersection(self.base_workspace_poly_)
            self.cache_[leg_prefix] = intersections

            # test
            # print(shapely.ops.nearest_points(
            #     self.cache_[leg_prefix], shapely.geometry.Point(0, 0)))

            intersection_points = [Position([p.x, p.y, 0.0]) for p in
                                   intersections.boundary]
            visuals[leg_prefix] = intersection_points

        return visuals

    def control(self, stamp):
        # Extract foot poses.
        commands = []
        for leg_prefix in self.config_.order:
            # TODO(yycho0108) : change to FL_foot instead of
            # FL_foot_a/FL_foot_b
            leg_origin = '{}_leg_origin'.format(leg_prefix)
            foot_frame = '{}_foot_a'.format(leg_prefix)

            # Attempt to get transforms at current timestamp.
            body_from_leg = self.graph_.get_transform(
                leg_origin, 'body', stamp)
            leg_from_foot = self.graph_.get_transform(
                foot_frame, leg_origin, stamp)

            # If the plan falls through, resolve to latest feasible option.
            if body_from_leg is None:
                logger.warn(
                    'Transform {} not available at stamp {}: fallback to using any available transform'.format(
                        leg_origin, stamp))
                metadata = {}
                body_from_leg = self.graph_.get_transform(
                    leg_origin, 'body', stamp, tol=np.inf, metadata=metadata)
                logger.warn(
                    'Alternative transform found at {} (diff={})'.format(
                        metadata['stamp'],
                        stamp - metadata['stamp']))
            if leg_from_foot is None:
                logger.warn(
                    'Transform {} not available at stamp {}: fallback to using any available transform'.format(
                        foot_frame, stamp))
                metadata = {}
                leg_from_foot = self.graph_.get_transform(
                    foot_frame, leg_origin, stamp, tol=np.inf,
                    metadata=metadata)
                logger.warn(
                    'Alternative transform found at {} (diff={})'.format(
                        metadata['stamp'],
                        stamp - metadata['stamp']))

            # Compute closest target to defined plane.
            foot_point = leg_from_foot.position
            cur = shapely.geometry.Point(foot_point.x, foot_point.y)
            references = self.cache_[leg_prefix]
            target, _ = shapely.ops.nearest_points(references, cur)

            # Solve IK for computed target.
            angles = solve_inverse_kinematics(
                self.graph_, stamp, leg_prefix,
                Position([target.x, target.y, 0.0]),
                self.config_)
            commands.extend(angles)
        return commands


def main():
    from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
    graph = PhonebotGraph()
    controller = BaseRotationController(graph)
    controller.update(Rotation.identity())


if __name__ == '__main__':
    main()
