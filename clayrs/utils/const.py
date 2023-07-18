import os
from clayrs.utils.custom_logger import get_custom_logger

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(THIS_DIR, '../../')
contents_path = os.path.join(root_path, 'contents/')
datasets_path = os.path.join(root_path, 'datasets/')

logger = get_custom_logger('custom_logger')
