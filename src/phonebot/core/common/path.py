#!/usr/bin/env python3
import os


class PhonebotPath(object):
    @staticmethod
    def root():
        common = os.path.dirname(__file__)
        return os.path.abspath(os.path.dirname(os.path.join(common, '../', '../')))

    @staticmethod
    def assets():
        return os.path.join(PhonebotPath.root(), 'assets')


def main():
    print(PhonebotPath.assets())


if __name__ == '__main__':
    main()
