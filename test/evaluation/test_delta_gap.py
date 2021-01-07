from unittest import TestCase
from orange_cb_recsys.evaluation.delta_gap import calculate_gap


class Test(TestCase):
    def test_calculate_gap(self):
        group = {'aaa', 'bbb', 'ccc'}
        avg_pop = {'aaa': 0.5, 'bbb': 0.7}
        calculate_gap(group, avg_pop)

