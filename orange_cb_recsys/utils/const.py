from pathlib import Path
import logging

home_path = str(Path.home())
DEVELOPING = True

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('logger')

logger.setLevel(logging.INFO)
