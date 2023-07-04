import os
from pathlib import Path

dir_test_files = os.path.join(os.path.dirname(__file__), 'test_files')
dir_root_repo = Path(os.path.join(os.path.dirname(__file__), '..')).resolve()
