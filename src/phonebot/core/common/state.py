#!/usr/bin/env python3

import numpy as np
import logging


class PhonebotLegState(object):
    # joint angles + chirality
    STATE_SIZE = 3

    def __init__(self, param=np.zeros(3)):
        self.param_ = param

    @staticmethod
    def identity():
        pass

    def to_vector(state=None):
        if state is None:
            try:
                return self.param_
            except NameError as e:
                logging.error(
                    'Called as instance method without initialization -> [{}]'.format(e))
                raise
        return state.param_

    def from_vector(param):
        try:
            # instance method version
            self.param_ = param
            return self
        except NameError:
            pass
        # static method version
        return PhonebotLegState(param=param)


class PhonebotState(object):
    """
    Current phonebot state, including body pose and joint angles.
    """

    def __init__(self):
        # NOTE(yycho0108): pose_ is the transform from body frame to local frame.
        self.pose_ = None
        self.leg_states_ = [PhonebotLegState() for _ in range(4)]
        pass

    @staticmethod
    def to_vector(state):
        leg_params = [leg_state.to_vector()
                      for leg_state in self.leg_states_]
        return np.concatenate(leg_params + [self.pose_])

    @staticmethod
    def from_vector(param):
        pass


def main():
    v = np.zeros(3)
    pbs = PhonebotLegState(v)
    print(PhonebotLegState.to_vector(pbs))
    print(pbs.to_vector())
    print(PhonebotLegState.from_vector(v))
    print(pbs.from_vector(v))


if __name__ == "__main__":
    main()
