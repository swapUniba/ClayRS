import contextlib
import os
from collections.abc import Iterable

from tqdm.contrib.logging import logging_redirect_tqdm

from orange_cb_recsys.utils.custom_logger import getCustomLogger
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(THIS_DIR, '../../')
contents_path = os.path.join(root_path, 'contents/')
datasets_path = os.path.join(root_path, 'datasets/')

logger = getCustomLogger('custom_logger')


@contextlib.contextmanager
def get_progbar(iterator: Iterable) -> tqdm:
    pbar = tqdm(iterator)
    pbar.bar_format = "{desc} {percentage:.0f}%|{bar}| {n:}/{total_fmt} [{elapsed}<{remaining}]"
    with logging_redirect_tqdm(loggers=[logger]):
        yield pbar
