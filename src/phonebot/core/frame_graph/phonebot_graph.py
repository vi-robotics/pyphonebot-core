#!/usr/bin/env python3

import sys
import numpy as np
import time
# import bz2
import zlib

from phonebot.core.common.config import PhonebotSettings, FrameName
from phonebot.core.common.math.transform import Position, Rotation, Transform
from phonebot.core.frame_graph import FrameGraph, StaticFrameEdge, SimpleFrameEdge, RevoluteJointEdge, BufferedFrameEdge

from phonebot.core.common.logger import get_default_logger
logger = get_default_logger()


class PhonebotGraph(FrameGraph):
    def __init__(self, config=PhonebotSettings()):
        super().__init__()
        self.config_ = config
        self.build()

    def build(self):
        # Unroll some parameters.
        config = self.config_

        # TODO(yycho0108): replace global-local-body frame chain
        # with pose estimators when they are implemented and available.
        self.add_edge(StaticFrameEdge(FrameName.LOCAL,
                                      FrameName.GLOBAL, Transform.identity()))
        self.add_edge(StaticFrameEdge(FrameName.BODY,
                                      FrameName.LOCAL, Transform.identity()))

        leg_offset = Position(config.leg_offset)
        for leg_index, leg_prefix in enumerate(config.order):
            frame = FrameName(prefix=leg_prefix)
            frame_a = FrameName(prefix=leg_prefix, suffix='a')
            frame_b = FrameName(prefix=leg_prefix, suffix='b')
            # Add static transforms.
            self.add_edge(StaticFrameEdge(frame.LEG, frame.BODY,
                                          Transform(config.leg_sign[leg_index] * leg_offset,
                                                    config.leg_rotation[leg_index])))

            # Connect foot frames.
            # TODO(ycho): Restore below edge at some point...?
            #self.add_edge(StaticFrameEdge(
            #    frame_a.FOOT, frame_b.FOOT, Transform.identity()))

            for suffix in 'ab':
                # Define frame namespace of the half assembly.
                subframe = FrameName(leg_prefix, suffix)
                self.add_edge(StaticFrameEdge(subframe.HIP, subframe.LEG,
                                              Transform(
                                                  [config.hip_sign[suffix] *
                                                   config.hip_joint_offset, 0.0, 0.0],
                                                  Rotation.from_axis_angle(
                                                      [0, 1, 0, config.hip_angle[suffix]]).to_quaternion()
                                              )))

                # Also add dynamic joint edges.
                self.add_edge(RevoluteJointEdge(
                    subframe.KNEE, subframe.HIP, [0, 0, 1], [config.hip_link_length, 0, 0]))
                self.add_edge(RevoluteJointEdge(
                    subframe.FOOT, subframe.KNEE, [0, 0, 1], [-config.knee_link_length, 0, 0]))

        # NOTE(yycho0108): Defining a derived frame (ground)
        # Defined as the point on the ground plane directly below the body frame.
        # self.add_edge(StaticFrameEdge('ground', 'body', Transform.identity()))

        # NOTE(yycho0108): Defining a camera frame,
        # such that +z points outwards.
        # TODO(yycho0108): Camera position should be included in config.
        self.add_edge(StaticFrameEdge(FrameName.CAMERA, FrameName.BODY, Transform(
            [0.5*config.body_dim[0], 0.5 *
             config.body_dim[1], 0.5*config.body_dim[2]],
            Rotation.from_axis_angle(
                [0, 0, 1, np.deg2rad(90)]).to_quaternion()
        )))

    def add_edge(self, edge):
        """ Add wrapped edges """
        config = self.config_
        super().add_edge(BufferedFrameEdge(edge,
                                           queue_size=config.queue_size,
                                           timeout=config.timeout
                                           ))
        # super().add_edge(BufferedFrameEdge(edge))


def main():
    graph = PhonebotGraph()
    d1 = graph.save_all()
    # print(sys.getsizeof(d1))
    # print(d1)
    data = graph.encode()
    print('compressed-custom')
    print(sys.getsizeof(zlib.compress(data)))
    print('compressed-pkl')
    print(sys.getsizeof(zlib.compress(d1)))
    # with bz2.BZ2File('/tmp/graph2.pbz2', 'wb') as f:
    #    f.write(data)
    print('custom')
    print(sys.getsizeof(data))
    print('pickle')
    print(sys.getsizeof(d1))
    # print(data)
    # print(data)
    graph.restore(data)


if __name__ == '__main__':
    main()
