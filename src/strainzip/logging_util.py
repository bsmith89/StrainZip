import logging
import time
from contextlib import contextmanager

from tqdm import tqdm

global_phase_stack = [0] * 100  # Fixed maximum depth
global_phase_level = 0


def global_phase_stack_as_string():
    global global_phase_stack
    global global_phase_level
    pass


@contextmanager
def push_to_phase_id_stack():
    global global_phase_stack
    global global_phase_level
    global_phase_stack[global_phase_level] += 1
    global_phase_level += 1
    phase_stack_string = ".".join(
        [str(i) for i in global_phase_stack[:global_phase_level]]
    )
    # TODO: Increment the phase stack
    yield phase_stack_string
    global_phase_stack[global_phase_level] = 0
    global_phase_level -= 1
    # TODO: Decrement the phase stack


@contextmanager
def phase_info(name):
    start_time = time.time()
    with push_to_phase_id_stack() as phase_id:
        logging.info(f"({phase_id}) {name}")
        yield None
        end_time = time.time()
        delta_time = end_time - start_time
        delta_time_rounded = round(delta_time)
        logging.info(f"({phase_id}) {name} (DONE {delta_time_rounded} sec)")


@contextmanager
def phase_debug(name):
    start_time = time.time()
    with push_to_phase_id_stack() as phase_id:
        logging.debug(f"({phase_id}) {name}")
        yield None
        end_time = time.time()
        delta_time = end_time - start_time
        delta_time_rounded = round(delta_time)
        logging.debug(f"({phase_id}) {name} ({delta_time_rounded} sec)")


def tqdm_info(*args, **kwargs):
    return tqdm(
        *args, disable=(not logging.getLogger().isEnabledFor(logging.INFO)), **kwargs
    )


def tqdm_debug(*args, **kwargs):
    return tqdm(
        *args, disable=(not logging.getLogger().isEnabledFor(logging.DEBUG)), **kwargs
    )
