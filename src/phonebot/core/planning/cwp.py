#!/usr/bin/env python3

import numpy as np
from typing import Tuple, Iterable
import itertools
import networkx as nx
from collections import defaultdict

from phonebot.core.common.math.utils import normalize, norm, adiff
from phonebot.core.common.config import PhonebotSettings

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Arc
from matplotlib.colors import TABLEAU_COLORS

from functools import cmp_to_key


def _sq_norm(x: np.ndarray):
    """x^T@x; when in doubt, use einsum."""
    return np.einsum('...i,...i->...', x, x)


def circle_point_intersects(circle: np.ndarray, point: np.ndarray):
    """Boolean check for circle-point intersection.

    circle: array(..., 3) encoded as (x,y,r)
    point: array(..., 2) encoded as (x',y')
    """
    sqr = _sq_norm(circle[..., :2] - point)
    return sqr < np.square(circle[..., 2])


def circle_segment_intersects(circle: np.ndarray, segment: np.ndarray):
    """Boolean check for circle-segment intersection.

    circle: array(..., 3) encoded as (x,y,r)
    segment: array(..., 2, 2) encoded as ((x1,y1), (x2,y2))
    """

    center = circle[..., :2]
    radius = circle[..., 2]
    source = segment[..., 0, :2]
    target = segment[..., 1, :2]

    ba = target - source
    ca = center - source
    u = np.einsum('...a,...a->...', ca, ba) / _sq_norm(ba)
    u = np.clip(u, 0.0, 1.0)
    e = source + u[..., None] * ba
    sqd = _sq_norm(center - e)
    out = (sqd + 1e-6) <= np.square(radius)
    return out


def pairwise(seq: Iterable):
    """adjacent pairs: (x[i],x[i+1])"""
    # TODO(ycho): Dump these utility functions in core/common.
    it = iter(seq)
    a = next(it, None)
    for b in it:
        yield (a, b)
        a = b


def cycle(seq: Iterable):
    """Wrap to first element: x[:] + x[:1]"""
    # TODO(ycho): Dump these utility functions in core/common.
    it = iter(seq)
    a = next(it, None)
    yield a
    for b in it:
        yield b
    yield a


def _angular_neighbors(i0, h0, i1, h1):
    # Insertion index
    i_ins = np.searchsorted(h0, h1, side='right') - 1
    i_lo = i0[i_ins]

    # NOTE(ycho): `roll` required to deal with wraps.
    i_lo_prv = np.roll(i_lo, 1)
    dup = np.argwhere(i_lo == i_lo_prv).ravel()
    i_lo[dup] = i1[dup - 1]

    return i_lo


class CircleWorldPlanner:
    """Optimal shortest-path planner in a circular world.

    NOTE(ycho): Doesn't implement the full algorithm including
    e.g. culling and robustness against overlapping inputs.

    Reference:
    https://redblobgames.github.io/circular-obstacle-pathfinding/
    """

    def __init__(self, circles: Tuple[Tuple[float, float, float], ...]):
        self.circles = np.asarray(circles, dtype=np.float32)
        self.G, self.cache = self._generate_graph(nx.Graph())

    def _generate_graph(self, G: nx.Graph) -> nx.Graph:
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

        # Internal bi-tangents
        if True:
            c = (r0 + r1) / d.squeeze(axis=-1)
            s = np.sqrt(1 - np.square(c))

            u0 = np.einsum('abn, bn -> na', R, [c, s])
            u1 = np.einsum('abn, bn -> na', R, [c, -s])

            src.append(c0[..., :2] + r0[..., None] * u0)  # C
            src.append(c0[..., :2] + r0[..., None] * u1)  # D
            dst.append(c1[..., :2] - r1[..., None] * u0)  # F
            dst.append(c1[..., :2] - r1[..., None] * u1)  # E

        # External bi-tangents
        if True:
            c = (r0 - r1) / d.squeeze(axis=-1)
            s = np.sqrt(1 - np.square(c))

            u0 = np.einsum('abn, bn -> na', R, [c, s])
            u1 = np.einsum('abn, bn -> na', R, [c, -s])

            src.append(c0[..., :2] + r0[..., None] * u0)  # C
            src.append(c0[..., :2] + r0[..., None] * u1)  # D
            dst.append(c1[..., :2] + r1[..., None] * u0)  # F
            dst.append(c1[..., :2] + r1[..., None] * u1)  # E

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

        # Add hugging edges to the graph.
        node_ang_map = {}
        if True:
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

                if True:
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
                else:
                    for (i0, h0), (i1, h1) in itertools.combinations(
                            zip(node_indices, node_ang), 2):
                        arclen = radius * np.abs(adiff(h1, h0))
                        G.add_edge(i0, i1, length=arclen, center=i)

        cache = dict(
            node_index_map=node_index_map,
            node_ang_map=node_ang_map,
            node_offset=(2 * m)
        )
        return G, cache

    @classmethod
    def from_phonebot(cls, cfg: PhonebotSettings):
        small_radius = cfg.knee_link_length - cfg.hip_link_length
        sqr0 = np.square(small_radius)
        circles = (
            (cfg.hip_joint_offset, 0, small_radius),
            (-cfg.hip_joint_offset, 0, small_radius)
        )
        return cls(circles)

    def _spatial_path(self, G: nx.Graph, path: Tuple[int, ...]):
        """Convert the list of edges to a spatial path."""
        out = []
        for n0, n1 in pairwise(path):
            p0 = G.nodes[n0]['pos']
            p1 = G.nodes[n1]['pos']

            e = G.get_edge_data(n0, n1)
            if e['center'] is None:
                # straight line
                dp = p1 - p0
                out.append(p0 + np.linspace(0.0, 1.0)[:, None] * dp)
            else:
                # arc
                c = self.circles[e['center']]
                h0 = np.arctan2(*(p0 - c[:2])[::-1])
                h1 = np.arctan2(*(p1 - c[:2])[::-1])
                dh = adiff(h1, h0)
                h = h0 + np.linspace(0, dh)
                p = c[:2] + c[2] * np.c_[np.cos(h), np.sin(h)]
                out.append(p)
        out = np.concatenate(out, axis=0)
        return out

    def hack(self):
        n = nx.number_of_nodes(self.G)
        nim = self.cache['node_index_map']
        source = np.random.choice(nim[0])
        target = np.random.choice(nim[1])
        #source, target = np.random.choice(
        #    self.G.nodes, 2,
        #    replace=False)
        edges = self.G.edges(data=True)
        path = nx.shortest_path(
            self.G,
            source,
            target,
            weight='length'
            # self.G.nodes[source],
            # self.G.nodes[target]
        )

        # graph-valued path --> spatial trajectory
        # FIXME(ycho): Only for debugging, for now.
        out = self._spatial_path(self.G, path)
        return out

    def plan(
            self, waypoints: Tuple[Tuple[float, float], ...], dbg: dict = None):
        # TODO(ycho): as_view = True or False ??
        # G = self.G.copy(as_view=True)
        G = self.G.copy()
        waypoints = np.asarray(waypoints)

        c0 = self.circles[:, None, ...]  # N,1,2
        c1 = waypoints[None, :, ...]  # 1,M,2
        r0 = c0[..., 2]  # N,1
        r1 = np.zeros_like(c1[..., 0])  # 1,M
        dp = c1[..., :2] - c0[..., :2]  # N,M,2
        d = norm(dp)  # N,M,1

        # Define relative coordinate system.
        u = dp / d  # N,M,2
        ux, uy = u[..., 0], u[..., 1]  # N,M
        R = np.asarray([[ux, -uy], [uy, ux]], dtype=np.float32)  # 2,2,N,M

        # Connect to (internal=external) tangents
        src = []
        if True:
            c = r0 / d.squeeze(axis=-1)  # N,M
            s = np.sqrt(1 - np.square(c))  # N,M

            u0 = np.einsum('abnm, bnm -> nma', R, [c, s])  # N,M,2
            u1 = np.einsum('abnm, bnm -> nma', R, [c, -s])  # N,M,2

            src.append(c0[..., :2] + r0[..., None] * u0)  # C; N,M,2
            src.append(c0[..., :2] + r0[..., None] * u1)  # D; N,M,2

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
            # _, circle, waypoint
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

            if dbg is not None:
                if ('edge' not in dbg):
                    dbg['edge'] = []
                dbg['edge'].append((src[tuple(i)], waypoints[wpi]))

        # NOTE(ycho): Add edges `between` waypoint nodes.
        # This would be the equivalent of `surfing` edges (bitangents)
        # occurring amongst elements of `waypoint` nodes.
        for i0, i1 in itertools.combinations(range(len(waypoints)), 2):
            # print(self.circles.shape)
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
        if True:
            n = len(self.circles)
            for ci in range(n):
                node_indices = np.asarray(node_index_map[ci])
                if node_indices.size <= 0:
                    continue

                # Compute angles.
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

                # Hmm.....
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

                        if dbg is not None:
                            if ('edge' not in dbg):
                                dbg['edge'] = []
                            p0 = G.nodes[i0]['pos']
                            p1 = G.nodes[i1]['pos']
                            dbg['edge'].append((p0, p1))

        # Stitch together a path ...
        path = []
        for (wpi0, wpi1) in pairwise(range(len(waypoints))):
            path.extend(
                nx.shortest_path(
                    G,
                    wpt_idx0 + wpi0,
                    wpt_idx0 + wpi1,
                    weight='length'))
        path = self._spatial_path(G, path)
        return np.asarray(path)


def generate_non_overlapping_circles(n: int):
    out = np.zeros((n, 3), dtype=np.float32)
    index = 0
    while index < n:
        circle = np.random.normal(size=3)
        circle[2] = np.abs(circle[2])

        #if index < 2:
        #    circle[2] = 1e-8

        if (_sq_norm(out[:index, :2] - circle[None, :2])
                > np.square(circle[2] + out[:index, 2])).all():
            out[index] = circle
            index += 1
    return out


def main():
    seed = np.random.randint(2**16 - 1)
    # seed = 31331
    print(F'seed:{seed}')
    np.random.seed(seed)
    # np.random.seed(6)

    # circles = np.random.normal(size=(16, 3))
    # circles[..., 2] = 0.4 * np.abs(circles[..., 2])
    circles = generate_non_overlapping_circles(64)
    cwp = CircleWorldPlanner(circles)

    for c in circles:
        p = plt.Circle((c[0], c[1]), radius=c[2], fill=False, color='c')
        plt.gca().add_patch(p)
    plt.gca().set_aspect('equal', adjustable='datalim')
    dbg = {}

    while True:
        min_bound = np.min(circles[..., :2] - circles[..., 2:], axis=0)
        max_bound = np.max(circles[..., :2] + circles[..., 2:], axis=0)
        wpts = np.random.uniform(
            low=min_bound, high=max_bound,
            size=(2, 2))
        if not circle_point_intersects(circles[:, None], wpts[None, :]).any():
            break

    path = cwp.plan(wpts, dbg=dbg)
    # path = cwp.hack()
    segments = []
    for e in cwp.G.edges(data=True):
        ed = e[2]

        p0 = cwp.G.nodes[e[0]]['pos']
        p1 = cwp.G.nodes[e[1]]['pos']
        ps = np.stack([p0, p1], axis=0)

        if ed['center'] is None:
            # straight line
            # plt.plot(ps[..., 0], ps[..., 1], 'k*--', alpha=0.5)
            segments.append(ps)
        else:
            continue
            # arc (hugging edge)
            c = circles[ed['center']]
            if not np.isfinite(c).all():
                continue
            h0 = np.arctan2(*(p0 - c[:2])[::-1])
            h1 = np.arctan2(*(p1 - c[:2])[::-1])
            if not np.isfinite([h0, h1]).all():
                continue
            dh = adiff(h1, h0)
            href = h1 if dh < 0 else h0

            hmax = np.maximum(0.0, np.rad2deg(np.abs(dh)) - 2.0)
            hmin = np.minimum(2.0, hmax)

            arc = Arc(
                c[: 2],
                2 * c[2],
                2 * c[2],
                np.rad2deg(href),
                hmin, hmax,
                edgecolor=list(TABLEAU_COLORS.values())
                [np.random.choice(len(TABLEAU_COLORS))],
                linewidth=8, linestyle='--')
            plt.gca().add_patch(arc)
            # plt.plot([x0, x1], [y0, y1], 'r*--', alpha=0.5)
    col = LineCollection(segments, color='gray', alpha=0.5, linestyle='--')
    plt.gca().add_collection(col)

    print(len(dbg['edge']))
    col = LineCollection(dbg['edge'], color='blue')
    plt.gca().add_collection(col)
    #for p0, p1 in dbg['edge']:
    #    x = p0[0], p1[0]
    #    y = p0[1], p1[1]
    #    plt.plot(x, y, label='extra-edge')
    plt.plot(path[..., 0], path[..., 1], 'r-')
    plt.plot(wpts[..., 0], wpts[..., 1], 'ro', markersize=10)
    # plt.plot(*path[-1], 'bo', markersize=10)
    plt.gca().plot()
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
