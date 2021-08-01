#!/usr/bin/env python3
"""PRM Implementation.

Mostly adapted from yycho0108/PRM, intended for brute-force testing in
path-planning scenarios.
"""

from typing import Tuple, Callable
import numpy as np
import networkx as nx
import logging

has_kdtree = False
try:
    from scipy.spatial import cKDTree
    has_kdtree = True
except ImportError:
    # TODO(ycho): It's technically possible to supplant KDTree queries
    # either with non-scipy backends or brute-force search.
    logging.warn('PRM planner will be unusable since KDTree is unavailable.')


class PRM:
    """Probabilistic Roadmap Planner."""

    def __init__(self,
                 sample_fn: Callable[[int], np.ndarray],
                 query_fn: Callable[[np.ndarray], bool],
                 plan_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 N: int = 128,
                 K: int = 8):

        # NOTE(ycho): Sample from workspace.
        self._sample = sample_fn
        # NOTE(ycho): Query occupancy of spatial position.
        self._query = query_fn
        # NOTE(ycho): Compute simple local plan between two points.
        self._plan_local = plan_fn

        self.N = N  # num vertices
        self.K = K  # num neighbors

        # Graph
        self._vertices = None  # vertices
        self._edges = None  # edges
        self._graph = None  # graph

        # Helpers
        self._edge_dists = None  # edge distances
        self.tree = None  # KDTree for fast spatial neighborhood queries

    def _construct_vertices(self):
        """Randomly generate vertices from sampler."""
        N = self.N

        count = 0
        while count < N:
            n = N - count
            q = self._sample(n)  # n,?
            occ = self._query(q)  # n
            q = q[~occ]

            # NOTE(ycho): lazy construction of V
            # based on runtime dimensions
            if self._vertices is None:
                self._vertices = np.empty((N, q.shape[-1]), q.dtype)

            # Update V with newly queried stuff
            new_count = count + len(q)
            self._vertices[count: new_count] = q
            count = new_count

    def _construct_edges(self):
        """Add locally-connected edges to graph."""
        self.tree = cKDTree(self._vertices)
        dist, inds = self.tree.query(self._vertices, k=self.K + 1)

        # Check connectivity between nodes
        src = np.arange(len(self._vertices))  # (N, 1)
        dst = inds[..., 1:]  # (N, K)
        con = self._plan_local(
            self._vertices[:, None, :],
            self._vertices[dst, :])

        # if connected, then create edges between.
        src = np.broadcast_to(src[:, None], dst.shape)
        nbr = np.stack([src, dst], axis=-1)

        self._edges = nbr[con]
        self._edge_dists = dist[..., 1:][con]

    def _construct_graph(self):
        """Construct a networkx graph."""
        G = nx.Graph()
        G.add_weighted_edges_from(
            [(i0, i1, d) for(i0, i1), d
                in zip(self._edges, self._edge_dists)],
            axis=-1)
        self._graph = G

    def construct(self):
        """Shorthand for calling all internal builder functions."""
        self._construct_vertices()
        self._construct_edges()
        self._construct_graph()

    def plan(self, q0, q1):
        """Path from `q0` to `q1`"""
        q0 = np.asarray(q0)
        q1 = np.asarray(q1)
        _, (i0, i1) = self.tree.query([q0, q1], k=self.K)

        # Connect q0,q1 to roadmap
        src = np.stack([q0, q1])[:, None, :]  # 2,1,2
        dst = self._vertices[np.stack(
            [i0, i1], axis=0), :]  # 2,4,2 == 2,K,D
        con = self._plan_local(src, dst)

        # Require that at least one connection was established
        # on both ends.
        if not con.any(axis=-1).all():
            return None

        # Connect to roadmap, nearest preferred
        nbr = con.argmax(axis=-1)
        i0, i1 = i0[nbr[0]], i1[nbr[1]]

        # now to connect q0 -- i0 -- i1 -- q1
        try:
            p = nx.shortest_path(self._graph, source=i0, target=i1)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return np.concatenate(
            [q0[None], self._vertices[p], q1[None]], axis=0)
