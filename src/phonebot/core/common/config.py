#!/usr/bin/env python3

import numpy as np

from phonebot.core.common.math.transform import Rotation
from phonebot.core.common.settings import Settings
from typing import List, Tuple, Dict
from enum import Enum


class LegOrder(Enum):
    FL = 0
    FR = 1
    HL = 2
    HR = 3


class FrameName():
    """
    Class for managing frame names without hardcoding or string parsing.
    TODO(yycho0108): Figure out a better architecture.
    """
    # These are always used without modification..
    GLOBAL = 'global'
    LOCAL = 'local'
    BODY = 'body'
    CAMERA = 'camera'

    # These are overridden with modifiers.
    LEG = 'leg_origin'
    HIP = 'hip_joint'
    KNEE = 'knee_joint'
    FOOT = 'foot'

    # Metadata - delimiter
    DELIM = '_'

    # Suffixes.
    A = 'a'
    B = 'b'

    def __init__(self, prefix: str='', suffix: str=''):
        """Initialize a FrameName instance

        Args:
            prefix (str, optional): The prefix string to use for the joints.
                Defaults to ''.
            suffix (str, optional): The suffix to use for the joints. Defaults
                to ''.
        """
        self.GLOBAL = FrameName.GLOBAL
        self.LOCAL = FrameName.LOCAL
        self.BODY = FrameName.BODY

        prefix = prefix + (FrameName.DELIM if len(prefix) > 0 else '')
        suffix = (FrameName.DELIM if len(suffix) > 0 else '') + suffix
        self.LEG = f'{prefix}{FrameName.LEG}'
        self.HIP = f'{prefix}{FrameName.HIP}{suffix}'
        self.KNEE = f'{prefix}{FrameName.KNEE}{suffix}'
        self.FOOT = f'{prefix}{FrameName.FOOT}{suffix}'


class PhonebotSettings(Settings):
    """
    Parametric phonebot configuration.
    """
    leg_offset: Tuple[float, float, float]
    leg_sign: List[Tuple[int, int, int]]
    leg_rotation: List[Rotation]
    axis_sign: Tuple[int, int, int]
    hip_joint_offset: float
    hip_link_length: float
    knee_link_length: float
    body_dim: Tuple[float, float, float]
    leg_radius: float
    order: Tuple[str, str, str, str]
    index: Dict[str, int]
    queue_size: int
    timeout: float
    joint_names: List[str]
    active_joint_names: List[str]
    active_joint_indices: List[int]
    passive_joint_names: List[str]
    passive_joint_indices: List[int]
    nominal_hip_angle: float
    nominal_knee_angle: float
    hip_sign: Dict[str, int]
    hip_angle: Dict[str, float]
    camera_position: Tuple[float, float, float]
    select_nearest_ik: bool
    use_cached_rect: bool
    max_rect: Tuple[Tuple[float, float], Tuple[float, float]]

    body_mass: float
    hip_mass: float
    knee_mass: float
    foot_mass: float
    servo_torque: float

    def __init__(self, **kwargs):
        """Initialize the PhonebotSettings with default values which
        can be overriden.
        """
        # Displacement from body origin to leg origin.
        self.leg_offset = (0.041418, 0.0425, -0.010148)
        self.leg_sign = [(1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1)]

        rz = Rotation.from_axis_angle([0, 0, 1, np.pi])
        rx = Rotation.from_axis_angle([1, 0, 0, -np.pi / 2])
        rzrx = rz.rotate(rx)

        self.leg_rotation = (rx, rzrx, rx, rzrx)
        self.axis_sign = (1, 1, 1, 1)

        # Transformation parameters for a-b subassemblies.
        self.hip_sign = {'a': +1, 'b': -1}
        self.hip_angle = {'a': 0.0, 'b': np.pi}

        # Distance from leg origin to hip joint.
        self.hip_joint_offset = 0.0110
        self.hip_link_length = 0.0175
        self.knee_link_length = 0.0285

        # Geometry
        self.body_dim = np.asarray([0.15, 0.07635, 0.021612])
        self.leg_radius = 0.004

        # Leg order
        # self.FL, self.FR, self.HL, self.HR = range(4)
        # FIXME(yycho0108): rename order -> leg_order for clarity.
        self.order = ('FL', 'FR', 'HL', 'HR')
        # self.order = (LegOrder.FL, LegOrder.FR, LegOrder.HL, LegOrder.HR)
        self.index = {k: i for i, k in enumerate(self.order)}

        # Define joint angles at nominal position.
        self.nominal_hip_angle = 0.5710274713594905
        self.nominal_knee_angle = -1.0161038637161255

        # Define Frame graph options.
        self.queue_size = 4
        self.timeout = 0.1

        self.camera_position = [0.067900, 0.01191000, self.body_dim[2] / 2]
        self.select_nearest_ik = False

        self.use_cached_rect = True
        self.max_rect = (
            (-0.021572265625000003, 0.010986328125),
            (0.021396484374999997, 0.032470703125)
        )

        # Phone + Case(PLA) + 8xServo
        self.body_mass = 0.138 + 0.100 + 0.009 * 8
        self.hip_mass = 0.030
        self.knee_mass = 0.030
        self.foot_mass = 0.010
        self.servo_torque = 0.235

        super().__init__(**kwargs)
        self.build()

    def build(self):
        """Build derived properties
        """
        joint_names = []
        active_joint_names = []
        passive_joint_names = []
        for prefix in self.order:
            for suffix in 'ab':
                for jtype in (FrameName.HIP, FrameName.KNEE):
                    jname = '{}_{}_{}'.format(prefix, jtype, suffix)
                    joint_names.append(jname)
                    if jtype == FrameName.HIP:
                        active_joint_names.append(jname)
                    else:
                        passive_joint_names.append(jname)

        active_joint_indices = []
        for i, j in enumerate(joint_names):
            if j in active_joint_names:
                active_joint_indices.append(i)

        passive_joint_indices = []
        for i, j in enumerate(joint_names):
            if j in passive_joint_names:
                passive_joint_indices.append(i)

        self.joint_names = joint_names
        self.active_joint_names = active_joint_names
        self.active_joint_indices = np.asarray(active_joint_indices)
        self.passive_joint_names = passive_joint_names
        self.passive_joint_indices = np.asarray(passive_joint_indices)
