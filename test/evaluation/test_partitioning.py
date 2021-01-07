from unittest import TestCase
import pandas as pd

from orange_cb_recsys.evaluation import KFoldPartitioning


class TestKFoldPartitioning(TestCase):
    def test_set_dataframe(self):

        a = KFoldPartitioning()
        a.set_dataframe(pd.DataFrame.from_dict({'from_id': ["001", "001", "002", "002", "002", "003", "004", "004"],
                                                'to_id': ["aaa", "bbb", "aaa", "ddd", "ccc", "ccc", "ddd", "ccc"],
                                                'rating': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7]}))
        for partition in a:
            pass
