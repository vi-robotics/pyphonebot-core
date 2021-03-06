#!/usr/bin/env python3

import inspect
import logging
from typing import Union


def _caller_name(skip: int = 2) -> str:
    """Get a name of a caller in the format module.class.method

    Args:
        skip (int, optional): specifies how many levels of stack to skip while
            getting caller name. skip=1 means "who calls me", skip=2 "who calls
            my caller" etc.. Defaults to 2.

    Returns:
        str: An empty string is returned if skipped levels exceed stack height

    Source:
        http://code.activestate.com/recipes/578352-get-full-caller-name-packagemodulefunction/
    """
    stack = inspect.stack()
    start = 0 + skip
    if len(stack) < start + 1:
        return ''
    parentframe = stack[start][0]

    name = []
    module = inspect.getmodule(parentframe)

    if module:
        name.append(module.__name__)
    # detect classname
    if 'self' in parentframe.f_locals:
        name.append(parentframe.f_locals['self'].__class__.__name__)
    codename = parentframe.f_code.co_name
    if codename != '<module>':  # top level usually
        name.append(codename)  # function or a method
    del parentframe
    return ".".join(name)


def get_root_logger(level: int = logging.NOTSET) -> logging.Logger:
    """Get the root phonebot logger and attach a StreamHandler.

    Args:
        level (int, optional): The log level of the logger. Defaults to
            logging.NOTSET.

    Returns:
        logging.Logger: The resulting logger.
    """
    # FIXME(yycho0108): hard-coded root module name
    logger = logging.getLogger('phonebot')

    # Default format ...
    fmt = '[%(asctime)s] %(name)s:%(levelname)s> %(message)s'

    # Formatter
    formatter = logging.Formatter(fmt=fmt)

    # Handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)

    # Setup
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def set_log_level(level: Union[int, str]):
    """Set the log level of the root logger.

    Args:
        level (Union[int, str]): An integer or string representing the log
            level.
    """
    # Map level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    root_logger = get_root_logger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


def get_default_logger(level: Union[int, str] = logging.NOTSET
                       ) -> logging.Logger:
    """Get the default logger.

    Args:
        level (Union[int, str], optional): Optionally set the log level.
            Defaults to logging.NOTSET.

    Returns:
        logging.Logger: The resulting logger.
    """
    # Map level
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Configure
    name = _caller_name()

    # Initialize
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Return
    return logger
