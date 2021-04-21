#!/usr/bin/env python3

__all__ = ['Trajectory']

from abc import ABCMeta, abstractmethod


class Trajectory(metaclass=ABCMeta):
    """
    Abstract representation of sequential data (position, transform, etc.),
    evaluated across multiple points in time and can be queried as such.
    """
    @abstractmethod
    def evaluate(self, time):
        return NotImplemented
