#!/usr/bin/env python3
"""2D Circular world shortest-path planner.

Decomposes the circular world into line- and arc-based edges, and runs
shortest-path graph search on pre-built graph with additional waypoints.
"""

import numpy as np
from typing import Tuple, Iterable
import itertools
import networkx as nx
from collections import defaultdict

from phonebot.core.common.math.utils import normalize, norm, adiff
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.common.geometry.geometry import (
    circle_point_intersects, circle_segment_intersects)


def pairwise(seq: Iterable):
    """Iterate through adjacent pairs: (x[i],x[i+1])"""
    # TODO(ycho): Dump these utility functions in core/common.
    it = iter(seq)
    a = next(it, None)
    for b in it:
        yield (a, b)
        a = b


def cycle(seq: Iterable):
    """Iterate and wrap to first element: x[:] + x[:1]"""
    # TODO(ycho): Dump these utility functions in core/common.
    it = iter(seq)
    a = next(it, None)
    yield a
    for b in it:
        yield b
    yield a


class CircleWorldPlanner:
    """Optimal shortest-path planner in a circular world.

    NOTE(ycho): Doesn't implement the full algorithm including
    e.g. culling and robustness against overlapping inputs.

    Reference:
    https://redblobgames.github.io/circular-obstacle-pathfinding/
    """

    def __init__(self, circles: Tuple[Tuple[float, float, float], ...]):
        """
        Args:
            circles: array(...,3) List of circles encoded as (x,y,radius)
        """
        self.circles = np.asarray(circles, dtype=np.float32)
        self.G, self.cache = self._build_graph(nx.Graph())

    def _build_graph(self, G: nx.Graph) -> nx.Graph:
        """Append path-planning nodes and edges to graph.

        Args:
            G: base graph, expected to be empty.
        """
        # Compute index pairs.
        # NOTE(ycho): working with vectorized indices incurs memory cost:
        # n -> nC2. If efficiency becomes important,
        # than just go ahead and implement this in C/C++.
        n = len(self.circles)
        indices = itertools.combinations(range(n), 2)  # nC2,2
        indices = list(indices)
        i0, i1 = np.transpose(indices)

        c0 = self.circles[i0]
        c1 = self.circles[i1]
        r0 = c0[..., 2]
        r1 = c1[..., 2]
        dp = c1[..., :2] - c0[..., :2]
        d = norm(dp)

        # Define relative coordinate system.
        u = dp / d
        ux, uy = u[..., 0], u[..., 1]
        R = np.asarray([[ux, -uy], [uy, ux]], dtype=np.float32)

        src, dst = [], []

        # Internal(+1)/External(-1) bi-tangents
        for sign in [+1, -1]:
            r1_ = sign * r1
            c = (r0 + r1_)
            c /= d.squeeze(axis=-1)
            s = np.sqrt(1 - np.square(c))

            u0 = np.einsum('abn, bn -> na', R, [c, s])
            u1 = np.einsum('abn, bn -> na', R, [c, -s])

            src.append(c0[..., :2] + r0[..., None] * u0)  # C
            src.append(c0[..., :2] + r0[..., None] * u1)  # D
            dst.append(c1[..., :2] - r1_[..., None] * u0)  # F
            dst.append(c1[..., :2] - r1_[..., None] * u1)  # E

        # Finalize surfing edges.
        rpos = np.concatenate(src + dst, axis=0)
        pos = rpos.reshape(2, -1, 2)
        dist = norm(pos[0] - pos[1]).squeeze(axis=-1)
        m = len(dist)
        node_index_map = defaultdict(list)
        for i, e in enumerate(dist):
            invalid = circle_segment_intersects(
                self.circles[:, None],
                pos[None, None, :, i]).any()
            if invalid:
                continue

            # NOTE(ycho): nodes and edges are indexed by combinatoric order.
            G.add_node(i, pos=pos[0, i])
            G.add_node(i + m, pos=pos[1, i])
            G.add_edge(i, i + m, length=e, center=None)

            # Add node to circle.
            i_perm = i % len(indices)
            i_circle0 = i0[i_perm]
            i_circle1 = i1[i_perm]
            node_index_map[i_circle0].append(i)
            node_index_map[i_circle1].append(i + m)

        # Add `hugging` edges to the graph.
        node_ang_map = {}
        for i in range(n):
            # Circle index -> edge indices that touch said circle
            circle = self.circles[i]
            center = circle[:2]
            radius = circle[2]

            # Nodes that belong to circle
            #if len(node_index_map[i]) <= 0:
            #    continue
            node_indices = np.asarray(node_index_map[i])

            node_rel_pos = rpos[node_indices] - center
            node_ang = np.arctan2(
                node_rel_pos[..., 1],
                node_rel_pos[..., 0])

            node_order = np.argsort(node_ang)
            node_indices = node_indices[node_order]
            node_ang = node_ang[node_order]

            # Update data in cache.
            node_ang_map[i] = node_ang
            node_index_map[i] = node_indices

            for (i0, h0), (i1, h1) in pairwise(
                    cycle(zip(node_indices, node_ang))):
                arclen = radius * np.abs(adiff(h1, h0))
                G.add_edge(i0, i1, length=arclen, center=i)

        cache = dict(
            node_index_map=node_index_map,
            node_ang_map=node_ang_map,
            node_offset=(2 * m)
        )
        return G, cache

    def _spatial_path(self, G: nx.Graph, path: Tuple[int, ...]):
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
        """

        Args:
            waypoints: array(..., 2) list of spatial waypoints to traverse through.

        Returns:
            The plan through the waypoints.

        """
        # TODO(ycho): What if the plan for supplied waypoints are infeasible?

        # TODO(ycho): Consider not using networkx, since
        # - graph copy takes forever (it really shouldn't)
        # - shortest path query takes forever
        # - Can't use implicit nodes/edges
        # - does not expose useful data structure (opaque API)
        G = self.G.copy()

        waypoints = np.asarray(waypoints)

        # Compute some intermediate geometries ...
        c0 = self.circles[:, None, ...]  # N,1,2
        c1 = waypoints[None, :, ...]  # 1,M,2
        r0 = c0[..., 2]  # N,1
        dp = c1[..., :2] - c0[..., :2]  # N,M,2
        d = norm(dp)  # N,M,1

        # Define relative coordinate system.
        u = dp / d  # N,M,2
        ux, uy = u[..., 0], u[..., 1]  # N,M
        R = np.asarray([[ux, -uy], [uy, ux]], dtype=np.float32)  # 2,2,N,M

        # Connect to (internal=external) tangents
        src = []
        c = r0 / d.squeeze(axis=-1)  # N,M
        s = np.sqrt(1 - np.square(c))  # N,M

        u0 = np.einsum('abnm, bnm -> nma', R, [c, s])  # N,M,2
        u1 = np.einsum('abnm, bnm -> nma', R, [c, -s])  # N,M,2

        src.append(c0[..., :2] + r0[..., None] * u0)  # N,M,2
        src.append(c0[..., :2] + r0[..., None] * u1)  # N,M,2

        src = np.stack(src, axis=0)  # 2,N,M,2
        dist = norm(src - c1[None, ..., :2]).squeeze(axis=-1)  # 2,N,M
        m = dist.size

        # Add waypoint nodes.
        # NOTE(ycho): can't just naively use num_nodes() here,
        # Since we omit certain "invalid" nodes in node indexing.
        wpt_idx0 = self.cache['node_offset']
        for i, wpt in enumerate(waypoints):
            G.add_node(wpt_idx0 + i, pos=wpt)

        # Add tangent nodes + edges.
        idx0 = wpt_idx0 + len(waypoints)
        node_index_map = defaultdict(list)
        iii = np.indices(dist.shape)
        iii = iii.reshape(iii.shape[0], -1).T
        for ni, i in enumerate(iii):
            _, ci, wpi = i

            node_index = idx0 + ni
            waypoint_index = wpt_idx0 + wpi

            # Bookkeeping index map ...
            node_index_map[ci].append(node_index)

            segment = np.stack([src[tuple(i)], waypoints[wpi]])
            invalid = circle_segment_intersects(self.circles[:, None],
                                                segment[None, :]).any()
            if invalid:
                continue

            # Add node tangent to circle.
            G.add_node(node_index, pos=src[tuple(i)])

            # Add surfing edge.
            G.add_edge(node_index, waypoint_index,
                       length=dist[tuple(i)], center=None)

        # NOTE(ycho): Add edges `among` waypoint nodes.
        # This would be the equivalent of `surfing` edges (bitangents)
        # occurring amongst elements of `waypoint` nodes.
        for i0, i1 in itertools.combinations(range(len(waypoints)), 2):
            segment = waypoints[(i0, i1), :2]
            invalid = circle_segment_intersects(
                self.circles[:, None],
                segment[None, :]).any()
            if invalid:
                continue
            G.add_edge(
                wpt_idx0 + i0, wpt_idx0 + i1,
                length=norm(waypoints[i0, :2] - waypoints[i1, :2]).squeeze(),
                center=None)

        # Add hugging edges.
        n = len(self.circles)
        for ci in range(n):
            node_indices = np.asarray(node_index_map[ci])
            if node_indices.size <= 0:
                continue

            # Compute angles from relative positions to anchored circles.
            # FIXME(ycho): Technically, we're duplicating work from above.
            rel_pos = src[:, ci] - c0[None, ci, ..., :2]  # 2,M,2 - 1,1,2
            node_ang = np.arctan2(rel_pos[..., 1], rel_pos[..., 0])
            node_ang = node_ang.ravel()

            # Sort and update angles/indices
            node_order = np.argsort(node_ang)
            node_indices = node_indices[node_order]
            node_ang = node_ang[node_order]

            # Find nodes to Connect to
            prev_node_indices = self.cache['node_index_map'][ci]
            prev_node_ang = self.cache['node_ang_map'][ci]

            # Find additional adjaceny nodes based on
            # newly added nodes; most simply achieved by sorting.
            all_indices = np.r_[prev_node_indices, node_indices]
            all_angs = np.r_[prev_node_ang, node_ang]
            order = np.argsort(all_angs)
            for ii0, ii1 in pairwise(cycle(order)):
                i0, i1 = all_indices[ii0], all_indices[ii1]
                h0, h1 = all_angs[ii0], all_angs[ii1]
                if i0 not in G.nodes or i1 not in G.nodes:
                    continue
                if i0 >= idx0 or i1 >= idx0:
                    arclen = self.circles[ci, 2] * np.abs(adiff(h1, h0))
                    G.add_edge(i0, i1, length=arclen, center=ci)

        # Stitch together a path ...
        path = []
        # NOTE(ycho): takes forever
        for (wpi0, wpi1) in pairwise(range(len(waypoints))):
            # TODO(ycho): For len(waypoints)>2,
            # intermediate waypoints may be duplicated in the
            # below path.
            path.extend(
                nx.shortest_path(
                    G,
                    wpt_idx0 + wpi0,
                    wpt_idx0 + wpi1,
                    weight='length'))
        # TODO(ycho): Consider arg to pass in for _spatial_path alternative
        # to convert to `phonebot..Trajectory` rather than
        # a set of discretized waypoints
        path = self._spatial_path(G, path)
        return np.asarray(path)
