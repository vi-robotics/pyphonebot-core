#!/usr/bin/env python3

from abc import ABC, abstractmethod
import inspect


class BaseAgent(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, state, stamp):
        pass


def main():
    class DummyAgent(BaseAgent):
        def __init__(self):
            super().__init__()

        def __call__(self, state, stamp):
            pass

    ba = DummyAgent()


if __name__ == '__main__':
    main()
