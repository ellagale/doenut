"""
DoENUT.utils

Internal helper functions for the module.
"""
import logging


def initialise_log(log_name: str, initial_level: "int|str") -> logging.Logger:
    """Sets up a given logger. Should be invoked per module with __name__
    Handles making sure we don't duplicate logs due to multiple invocations
    from jupyter imports

    Parameters
    ----------
    log_name: str :

    initial_level: "int|str" :


    Returns
    -------

    """
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    result = logging.getLogger(log_name)
    if result.hasHandlers():
        result.handlers.clear()
    result.setLevel(initial_level)
    result.addHandler(handler)
    return result
