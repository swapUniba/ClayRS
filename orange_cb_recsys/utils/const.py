from pathlib import Path
import logging
import progressbar
from progressbar import progressbar as progbar      #!!IMPORTANT DO NOT CANCEL
from orange_cb_recsys.utils.custom_logger import CustomFormatter

home_path = str(Path.home())
DEVELOPING = True

progressbar.streams.wrap_stderr()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)
