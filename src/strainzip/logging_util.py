import logging
import time
from contextlib import contextmanager

from tqdm import tqdm

global_phase_level = 0


def _phase_pre_and_post_fixes(i):
    prefix = ">" * i + f"({i}) START:"
    pstfix = "<" * i + f"({i})   END:"
    return prefix, pstfix


@contextmanager
def phase_info(name):
    global global_phase_level
    prefix, pstfix = _phase_pre_and_post_fixes(global_phase_level)
    start_time = time.time()
    logging.info(f"{prefix} {name}")
    global_phase_level += 1
    yield None
    global_phase_level -= 1
    end_time = time.time()
    delta_time = end_time - start_time
    delta_time_rounded = round(delta_time)
    logging.info(f"{pstfix} {name} ({delta_time_rounded} seconds)")


@contextmanager
def phase_debug(name):
    global global_phase_level
    prefix, pstfix = _phase_pre_and_post_fixes(global_phase_level)
    start_time = time.time()
    logging.debug(f"{prefix} {name}")
    global_phase_level += 1
    yield None
    global_phase_level -= 1
    end_time = time.time()
    delta_time = end_time - start_time
    delta_time_rounded = round(delta_time)
    logging.debug(f"{pstfix} {name} ({delta_time_rounded} seconds)")


def tqdm_info(*args, **kwargs):
    return tqdm(
        *args, disable=(not logging.getLogger().isEnabledFor(logging.INFO)), **kwargs
    )


def tqdm_debug(*args, **kwargs):
    return tqdm(
        *args, disable=(not logging.getLogger().isEnabledFor(logging.DEBUG)), **kwargs
    )
