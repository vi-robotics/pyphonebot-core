#!/usr/bin/env python3

import numpy as np
import os
import pickle
import pkg_resources
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


class BaseCachedRotationController(object):
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

        # TODO(ycho): Validate that this loading scheme works.
        # TODO(ycho): Better architecture for caching (potentially expensive)
        # config-dependent data at startup?
        raise IOError(
            'Please implement phonebot.core.resource before using cache-related functions.')
        #trajectory_cache_file = pkg_resources.resource_filename(
        #    'phonebot.core.common', 'data/face_follow_trajectories.p')
        #with open(trajectory_cache_file, 'rb') as f:
        #    data = pickle.load(f)
        self._cached_cmds = data['commands']
        self._cached_rv = data['roll_grid']
        self._cached_pv = data['pitch_grid']

    def update(self, rotation):
        """ Set rotation target. """
        # rotation = local_from_body
        # body.rotation() = rotation * local.rotation()
        # Obtain plane normal vector from rotation:
        # Essentially removes the yaw component.
        self.rotation_ = rotation

    def control(self, stamp):
        euler = self.rotation_.to_euler()
        roll, pitch = [euler[0], euler[1]]

        roll_idx = self.find_nearest(self._cached_rv[:, 0], roll)
        pitch_idx = self.find_nearest(self._cached_pv[0, :], pitch)

        # print(self._cached_cmds[0][0])
        # print(self.rotation_)
        # TODO(Max): Get nearest grid command
        return self._cached_cmds[roll_idx][pitch_idx]
        # Extract foot poses.

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx


def main():
    from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
    graph = PhonebotGraph()
    controller = BaseCachedRotationController(graph)
    controller.update(Rotation.identity())


if __name__ == '__main__':
    main()
