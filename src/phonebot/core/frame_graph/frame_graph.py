#!/usr/bin/env python3

__all__ = ['FrameGraph']

import numpy as np
import networkx as nx
import time
import itertools
import logging
import pickle

from typing import List, Dict, Any

from phonebot.core.common.serial import encode, decode
from phonebot.core.common.math.transform import Rotation, Position, Transform
from phonebot.core.frame_graph.frame_edges import *
from phonebot.core.common.logger import get_default_logger

logger = get_default_logger(logging.WARN)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class FrameGraph(object):
    """
    Generalized frame graph.
    """

    def __init__(self):
        self.edges_ = {}

    def reset(self):
        self.edges_ = {}

    @property
    def edges(self):
        return self.edges_.values()

    @property
    def frames(self):
        result = set()
        for edge in self.edges:
            result.add(edge.source)
            result.add(edge.target)
        return result

    def add_edge(self, edge):
        has_edge_fwd = self.get_edge(edge.source, edge.target)
        has_edge_bwd = self.get_edge(edge.target, edge.source)
        if has_edge_fwd or has_edge_bwd:
            # TODO(yycho0108): Potentially allow duplicate edges?
            # If so, would requires updates for get_edge() syntax.
            logger.warn(
                'Edge {}-{} already exists!'.format(edge.source, edge.target))
            return
        self.edges_[edge.source, edge.target] = edge
        self.edges_[edge.target, edge.source] = InvertedFrameEdge(edge)

    def get_connected_frames(self, frame):
        frames = []
        for edge in self.edges:
            if edge.has_frame(frame):
                frames.append(edge.target if edge.source ==
                              frame else edge.source)
        return frames

    def get_edge(self, source, target):
        key = (source, target)
        if key in self.edges_:
            return self.edges_[key]
        return None

    def has_transform(self, source_frame, target_frame, stamp, tol=0.1):
        if source_frame not in self.frames:
            return False
        if target_frame not in self.frames:
            return False
        # Collect edges that have the transform.
        edges = [e for e in self.edges if e.has_transform(stamp, tol)]

        # Build graph.
        graph = nx.Graph()
        for edge in edges:
            graph.add_edge(edge.source, edge.target)
        return nx.has_path(graph, source_frame, target_frame)

    def get_transform(self, source_frame, target_frame, stamp, tol=0.1, metadata={}) -> Transform:
        """
        Get the most recent known transform w.r.t. stamp.
        TODO(yycho0108): Enable interpolation.
        """
        metadata['stamp'] = None
        if source_frame == target_frame:
            metadata['stamp'] = stamp
            return Transform.identity()

        # Collect edges that have the transform.
        # NOTE(yycho0108): skipping the collection step here for now.
        # not sure if it's of any value.
        # edges = [e for e in self.edges if e.has_transform(stamp, tol)]
        edges = self.edges

        # for edge in self.edges_:
        #    if not edge.has_transform(stamp, tol):
        #        logger.debug(
        #            'Edge {} does not have transform at stamp {}'.format(edge, stamp))

        # Sort edges by distance from target stamp.
        # NOTE(yycho0108): treating static frames in a special way.
        edges = sorted(edges, key=lambda edge: 0.0 if isinstance(
            edge, StaticFrameEdge) else abs(stamp - edge.stamp))

        # Build graph.
        worst_stamp = 0.0
        graph = nx.Graph()
        for edge in edges:
            graph.add_edge(edge.source, edge.target, edge=edge)
            if not graph.has_node(source_frame) or not graph.has_node(target_frame):
                continue
            if nx.has_path(graph, source_frame, target_frame):
                worst_stamp = stamp if isinstance(
                    edge, StaticFrameEdge) else edge.stamp
                break
        else:
            # Path not found!
            logger.debug('No path from {} to {}'.format(
                source_frame, target_frame))
            for cc in nx.connected_components(graph):
                if source_frame in cc:
                    logger.info('Known frames w.r.t source : {}'.format(cc))
                if target_frame in cc:
                    logger.info('Known frames w.r.t target : {}'.format(cc))
            return None
        metadata['stamp'] = worst_stamp

        # if np.abs(stamp - worst_stamp) > tol:
        #    logger.warn('{}-{}, {} vs {} -> {}'.format(source_frame, target_frame, worst_stamp,
        #                                               stamp, stamp - worst_stamp))

        # Get paths.
        paths = nx.all_simple_paths(graph, target_frame, source_frame)

        # Obtained Sequence:
        # first -> ... -> last

        # Correct sequence:
        # transform(...->last) * transform(first->...)

        if paths is None:
            # Should never really reach here.
            logger.warn('No path from {} to {}'.format(
                source_frame, target_frame))
            return None

        # TODO(yycho0108): consider evaluating metrics such as uncertainty
        # instead of accepting the first path.
        path = None
        for path_option in paths:
            path = path_option
            break
        result = Transform.identity()

        # dbg = ''
        # for target, source in pairwise(reversed(path)):
        #    dbg += ',[{}<-{}]'.format(target, source)
        # print('path pair : {}'.format(dbg))

        for target, source in pairwise(path):
            frame_edge = graph[source][target]['edge']
            should_invert = (frame_edge.source != source)
            transform = frame_edge.get_transform(stamp)
            if should_invert:
                transform = transform.inverse()
            result *= transform
        return result

    def encode(self, *args, **kwargs) -> bytes:
        return encode(self.edges_, *args, **kwargs)

    def restore(self, data: bytes):
        self.edges_ = decode(data)

    @classmethod
    def decode(cls, data: bytes):
        out = cls()
        out.edges_ = decode(data)
        return out

    def save_all(self) -> bytes:
        return pickle.dumps(self.__dict__, protocol=-1)

    def load_all(self, data: bytes):
        b = pickle.loads(data)
        self.__dict__.update(b)

    @classmethod
    def create_from_pickle(cls, data: bytes):
        return pickle.loads(data)


def main():
    """
    Simple test.
    """
    graph = FrameGraph()
    xfm = Transform(
        Position.random(),
        Rotation.random().to_euler())
    local_from_global = StaticFrameEdge(
        'global', 'local', xfm)
    body_from_local = SimpleFrameEdge('body', 'local')
    body_from_local.update(time.time(), Transform.random())
    graph.add_edge(local_from_global)
    graph.add_edge(body_from_local)
    graph.add_edge(StaticFrameEdge('global', 'body', Transform.random()))
    transform = graph.get_transform('global', 'body', time.time())
    print(repr(transform))

    b = graph.encode()
    g2 = FrameGraph()
    g2.decode(b)

    transform = graph.get_transform('global', 'body', time.time())
    print(repr(transform))

    transform = g2.get_transform('global', 'body', time.time())
    print(repr(transform))


if __name__ == '__main__':
    main()
