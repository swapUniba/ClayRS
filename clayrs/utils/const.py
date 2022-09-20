import os
from clayrs.utils.custom_logger import getCustomLogger


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(THIS_DIR, '../../')
contents_path = os.path.join(root_path, 'contents/')
datasets_path = os.path.join(root_path, 'datasets/')

logger = getCustomLogger('custom_logger')
