import logging


def initialise_log(log_name: str, initial_level: int) -> logging.Logger:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    result = logging.getLogger(log_name)
    if result.hasHandlers():
        result.handlers.clear()
    result.setLevel(initial_level)
    result.addHandler(handler)
    return result
