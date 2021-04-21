#!/usr/bin/env python3

import inspect
import logging


def _caller_name(skip=2):
    """Get a name of a caller in the format module.class.method

       `skip` specifies how many levels of stack to skip while getting caller
       name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.

       An empty string is returned if skipped levels exceed stack height

       Source : http://code.activestate.com/recipes/578352-get-full-caller-name-packagemodulefunction/
    """
    stack = inspect.stack()
    start = 0 + skip
    if len(stack) < start + 1:
        return ''
    parentframe = stack[start][0]

    name = []
    module = inspect.getmodule(parentframe)
    # `modname` can be None when frame is executed directly in console
    # TODO(techtonik): consider using __main__
    if module:
        name.append(module.__name__)
    # detect classname
    if 'self' in parentframe.f_locals:
        # I don't know any way to detect call from the object method
        # XXX: there seems to be no way to detect static method call - it will
        #      be just a function call
        name.append(parentframe.f_locals['self'].__class__.__name__)
    codename = parentframe.f_code.co_name
    if codename != '<module>':  # top level usually
        name.append(codename)  # function or a method
    del parentframe
    return ".".join(name)


def get_root_logger(level=logging.NOTSET):
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


def set_log_level(level):
    # Map level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    root_logger = get_root_logger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


def get_default_logger(level=logging.NOTSET):
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
