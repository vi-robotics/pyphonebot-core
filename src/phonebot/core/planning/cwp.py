#!/usr/bin/env python3

import numpy as np

from _lib.cwp import CircleWorldPlanner

from phonebot.core.controls.trajectory import Trajectory
from phonebot.core.common.math.utils import lerp, alerp, adiff


class Line2D(Trajectory):
    def __init__(self, points: Tuple[float, float]):
        self._points = points

    def evaluate(self, time: float):
        return lerp(self._points[0], self._points[1], time)

    @property
    def length(self):
        return np.linalg.norm(self._points[1] - self._points[1],
                              axis=-1)


class MinorArc2D(Trajectory):
    def __init__(self,
                 center: Tuple[float, float],
                 radius: float,
                 angles: Tuple[float, float]):
        self._center = np.asarray(center)
        self._radius = np.asarray(radius)
        self._angles = np.asarray(angles)

    def evaluate(self, time: float):
        return (
            self._center +
            self._radius * alerp(self._angles[0], self._angles[1], time)
        )

    @property
    def length(self):
        theta = np.abs(adiff(self._angles[1], self._angles[0]))
        return theta * self._radius


class TimeScaleTrajectory(Trajectory):
    def __init__(self, trajectory: Trajectory, scale: float):
        self._trajectory = trajectory
        self._scale = scale

    def evaluate(self, time: float):
        return self._trajectory.evaluate(time * self._scale)


class PiecewiseTrajectory(Trajectory):
    def __init__(self, trajectories: Tuple[Trajectory, ...],
                 durations: Tuple[float, ...]):
        assert(len(durations) == len(trajectories))
        self._trajectories = trajectories
        self._upper_bound = np.cumsum(durations)
        self._lower_bound = np.insert(self._upper_bound[1:], 0, 0)

    def evaluate(self, time: float):
        assert(time <= self._upper_bound[-1])
        index = np.searchsorted(self._upper_bound, time, side='left')
        rel_time = time - self._lower_bound[index]
        return self._trajectories[index].evaluate(rel_time)


class LegTrajectoryPlanner(CWPBase):
    """Plan leg endpoint trajectories within the valid workspace."""

    def __init__(self, cfg: PhonebotSettings):
        self._planner = self._create_planner(cfg)

    def _create_planner(self) -> CircleWorldPlanner:
        """Instantiate `CircleWorldPlanner` from phonebot config."""
        small_radius = cfg.knee_link_length - cfg.hip_link_length
        sqr0 = np.square(small_radius)
        circles = (
            (cfg.hip_joint_offset, 0, small_radius),
            (-cfg.hip_joint_offset, 0, small_radius)
        )
        return CircleWorldPlanner(circles)

    def _map_to_trajectory(self, G: nx.Graph, path: Tuple[int, ...]):
        """Convert the list of edges to a spatial path.

        Args:
            G: base graph structure containing node/edge data.
            path: discrete path; list of node ids for `G`.

        Returns:
            Array of spatial coordinates mapepd from `path`.
        """
        out = []
        for n0, n1 in pairwise(path):
            p0 = G.nodes[n0]['pos']
            p1 = G.nodes[n1]['pos']

            e = G.get_edge_data(n0, n1)
            if e['center'] is None:
                # Straight line
                dp = p1 - p0
                out.append(p0 + np.linspace(0.0, 1.0)[:, None] * dp)
            else:
                # Arc
                c = self.circles[e['center']]
                h0 = np.arctan2(*(p0 - c[:2])[::-1])
                h1 = np.arctan2(*(p1 - c[:2])[::-1])
                dh = adiff(h1, h0)
                h = h0 + np.linspace(0, dh)
                p = c[:2] + c[2] * np.c_[np.cos(h), np.sin(h)]
                out.append(p)
        out = np.concatenate(out, axis=0)
        return out

    def plan(self, waypoints: Tuple[Tuple[float, float], ...]):
        return self._planner.plan(waypoints,
