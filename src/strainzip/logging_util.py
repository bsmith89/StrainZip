import logging
import time
from contextlib import contextmanager

from tqdm import tqdm


@contextmanager
def phase_info(name):
    start_time = time.time()
    logging.info(f"START: {name}")
    yield None
    end_time = time.time()
    delta_time = end_time - start_time
    delta_time_rounded = round(delta_time)
    logging.info(f"END: {name} (took {delta_time_rounded} seconds)")


@contextmanager
def phase_debug(name):
    start_time = time.time()
    logging.debug(f"START: {name}")
    yield None
    end_time = time.time()
    delta_time = end_time - start_time
    delta_time_rounded = round(delta_time)
    logging.debug(f"END: {name} (took {delta_time_rounded} seconds)")


def tqdm_info(*args, **kwargs):
    return tqdm(
        *args, disable=(not logging.getLogger().isEnabledFor(logging.INFO)), **kwargs
    )


def tqdm_debug(*args, **kwargs):
    return tqdm(
        *args, disable=(not logging.getLogger().isEnabledFor(logging.DEBUG)), **kwargs
    )
