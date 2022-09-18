import subprocess

import os
print(os.environ.get('COVERAGE_PROCESS_START'))


def test_via_subproc():
    proc = subprocess.Popen(["py.test", "test_recsys.py", "-v"])
    proc.wait()
