import time
from unittest import TestCase

import tqdm

from clayrs.utils.context_managers import get_progbar, get_iterator_parallel


class TestContextManagers(TestCase):

    def test_get_progbar(self):

        with get_progbar(range(50), total=50) as pbar:
            self.assertIsInstance(pbar, tqdm.tqdm)

            expected_bar_format = "{desc} {percentage:.0f}%|{bar}| {n:}/{total_fmt} [{elapsed}<{remaining}]"
            result_bar_format = pbar.bar_format

            expected_list = list(range(50))
            result_list = list(pbar)

        self.assertEqual(expected_bar_format, result_bar_format)
        self.assertEqual(expected_list, result_list)

    def test_get_iterator_parallel(self):

        def f(x):
            time.sleep(1)

            return x

        expected_list = list(range(5))

        # single cpu
        with get_iterator_parallel(1, f, list(range(5))) as it:
            result_list = list(it)

        self.assertEqual(expected_list, result_list)

        # multi cpu
        with get_iterator_parallel(2, f, list(range(5))) as it:
            result_list = list(it)

        self.assertEqual(expected_list, result_list)

        # multi cpu with progbar
        with get_iterator_parallel(2, f, list(range(5)), progress_bar=True, total=5) as pbar:

            self.assertIsInstance(pbar, tqdm.tqdm)

            result_list = list(pbar)

        self.assertEqual(expected_list, result_list)
