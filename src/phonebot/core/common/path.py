#!/usr/bin/env python3
import os
from os.path import abspath
from typing import AnyStr


class PhonebotPath():
    """Path configuration contains absolute paths to Phonebot directories.
    """
    @staticmethod
    def root() -> AnyStr @ abspath:
        """The root directory.

        Returns:
            str: The absolute path to the phonebot source directory.
        """
        common = os.path.dirname(__file__)
        return os.path.abspath(os.path.dirname(os.path.join(common, '../', '../')))

    @staticmethod
    def assets() -> AnyStr @ abspath:
        """The assets directory, which might not necessarily exist.

        Returns:
            str: The absolute path to the phonebot assets directory.
        """
        return os.path.join(PhonebotPath.root(), 'assets')
