import os
from pathlib import Path
import logging
import progressbar
from progressbar import progressbar as progbar      #!!IMPORTANT DO NOT CANCEL
from orange_cb_recsys.utils.custom_logger import CustomFormatter

home_path = str(Path.home())
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(THIS_DIR, '../../')

DEVELOPING = True

progressbar.streams.wrap_stderr()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)
