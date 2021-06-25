import unittest
import pandas as pd
from sklearn.metrics import ndcg_score
import numpy as np

from orange_cb_recsys.evaluation.exceptions import KError
from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split
from orange_cb_recsys.evaluation.metrics.ranking_metrics import NDCG, Correlation, MRR, NDCGAtK, MRRAtK

user_pred_only_new_items = pd.DataFrame(
    {'from_id': ['u1', 'u1', 'u2', 'u2'],
     'to_id': ['inew1', 'inew2', 'inew3', 'inew4'],
     'score': [650, 600, 500, 650]})

user_pred_w_new_items = pd.DataFrame(
    {'from_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2'],
     'to_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i4', 'i6', 'i1', 'i8'],
     'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50]})

user_pred_only_one_item = pd.DataFrame(
    {'from_id': ['u1', 'u2'],
     'to_id': ['i4', 'i8'],
     'score': [650, 600]})

user_truth = pd.DataFrame({'from_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
                           'to_id': ['i1', 'i2', 'i3', 'i4', 'i6', 'i1', 'i8', 'i4'],
                           'score': [3, 2, 3, 1, 2, 4, 3, 3]})

split_only_new = Split(user_pred_only_new_items, user_truth)
split_w_new_items = Split(user_pred_w_new_items, user_truth)
split_only_one = Split(user_pred_only_one_item, user_truth)


def for_each_method(test_func):
    def wrapper(self, *args, **kwargs):
        for method in self.methods_list:
            with self.subTest(current_method=method):
                test_func(*((self, method) + args), **kwargs)

    return wrapper


class TestNDCG(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = NDCG()

    def test_perform_only_new(self):
        metric = self.metric

        result_only_new = metric.perform(split_only_new)

        ndcgs_predicted = set(result_only_new[str(metric)])
        ndcgs_expected = {0}
        self.assertEqual(ndcgs_expected, ndcgs_predicted)

    def test_perform_w_new_items(self):
        metric = self.metric

        result_w_new_items = metric.perform(split_w_new_items)

        u1_actual = [[2, 3, 1, 0, 2, 3, 0, 0]]
        u1_ideal = [[3, 3, 2, 2, 1, 0, 0, 0]]

        u1_expected_ndcg = ndcg_score(u1_ideal, u1_actual)
        u1_result_ndcg = float(result_w_new_items.query('from_id == "u1"')[str(metric)])

        self.assertAlmostEqual(u1_expected_ndcg, u1_result_ndcg)

        u2_actual = [[3, 0, 4, 3]]
        u2_ideal = [[4, 3, 3, 0]]

        u2_expected_ndcg = ndcg_score(u2_ideal, u2_actual)
        u2_result_ndcg = float(result_w_new_items.query('from_id == "u2"')[str(metric)])

        self.assertAlmostEqual(u2_expected_ndcg, u2_result_ndcg)

        sys_expected_ndcg = (u1_expected_ndcg + u2_expected_ndcg) / 2
        sys_result_ndcg = float(result_w_new_items.query('from_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected_ndcg, sys_result_ndcg)

    def test_perform_only_one(self):
        metric = self.metric

        result_only_one = metric.perform(split_only_one)

        ndcgs_predicted = set(result_only_one[str(metric)])
        ndcgs_expected = {1}

        self.assertEqual(ndcgs_expected, ndcgs_predicted)


class TestNDCGAtK(unittest.TestCase):

    def test_perform(self):
        k = 2
        metric = NDCGAtK(k=k)

        result_w_new_items = metric.perform(split_w_new_items)

        u1_actual = [[2, 3, 1, 0, 2, 3, 0, 0]]
        u1_ideal = [[3, 3, 2, 2, 1, 0, 0, 0]]

        u1_expected_ndcg = ndcg_score(u1_ideal, u1_actual, k=k)
        u1_result_ndcg = float(result_w_new_items.query('from_id == "u1"')[str(metric)])

        self.assertAlmostEqual(u1_expected_ndcg, u1_result_ndcg)

        u2_actual = [[3, 0, 4, 3]]
        u2_ideal = [[4, 3, 3, 0]]

        u2_expected_ndcg = ndcg_score(u2_ideal, u2_actual, k=k)
        u2_result_ndcg = float(result_w_new_items.query('from_id == "u2"')[str(metric)])

        self.assertAlmostEqual(u2_expected_ndcg, u2_result_ndcg)

        sys_expected_ndcg = (u1_expected_ndcg + u2_expected_ndcg) / 2
        sys_result_ndcg = float(result_w_new_items.query('from_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected_ndcg, sys_result_ndcg)

    def test_perform_w_greater_k(self):
        k = 99
        metric = NDCGAtK(k=k)

        result = metric.perform(split_w_new_items)

        expected = NDCG().perform(split_w_new_items)

        result = np.sort(result, axis=0)
        expected = np.sort(expected, axis=0)

        self.assertTrue(np.array_equal(expected, result))


class TestMRR(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = MRR(relevant_threshold=3)

        cls.metric_mean_threshold = MRR()
        # mean u1 = 2.2; mean u2 = 3.33

    def test_perform_only_new(self):
        metric = self.metric

        result_only_new = metric.perform(split_only_new)

        mrr_expected = 0
        mrr_predicted = float(result_only_new[str(metric)])

        self.assertEqual(mrr_expected, mrr_predicted)

        # u1 = [0, 0], u2 = [0, 0]

    def test_perform_only_new_mean(self):
        metric = self.metric_mean_threshold

        result_only_new = metric.perform(split_only_new)

        mrr_expected = 0
        mrr_predicted = float(result_only_new[str(metric)])

        self.assertEqual(mrr_expected, mrr_predicted)

        # u1 = [0, 0], u2 = [0, 0]

    def test_perform_w_new_items(self):
        metric = self.metric

        result_w_new_items = metric.perform(split_w_new_items)

        u1_expected_rr = 1/2
        u2_expected_rr = 1

        sys_expected_mrr = (u1_expected_rr + u2_expected_rr) / 2

        sys_result_mrr = float(result_w_new_items.query('from_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected_mrr, sys_result_mrr)

        # u1 = [0, 1, 0, 0, 0, 1, 0, 0], u2 = [1, 0, 1, 1]

    def test_perform_w_new_items_mean(self):
        metric = self.metric_mean_threshold

        result_w_new_items = metric.perform(split_w_new_items)

        u1_expected_rr = 1/2
        u2_expected_rr = 1/3

        sys_expected_mrr = (u1_expected_rr + u2_expected_rr) / 2

        sys_result_mrr = float(result_w_new_items.query('from_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected_mrr, sys_result_mrr)

        # u1 = [0, 1, 0, 0, 0, 1, 0, 0], u2 = [0, 0, 1, 0]

    def test_perform_only_one(self):
        metric = self.metric

        result_only_one = metric.perform(split_only_one)

        u1_expected_rr = 0
        u2_expected_rr = 1/1

        sys_expected_mrr = (u1_expected_rr + u2_expected_rr) / 2
        sys_result_mrr = float(result_only_one[str(metric)])

        self.assertEqual(sys_expected_mrr, sys_result_mrr)

        # u1 = [0], u2 = [1]

    def test_perform_only_one_mean(self):
        metric = self.metric_mean_threshold

        result_only_one = metric.perform(split_only_one)

        u1_expected_rr = 0
        u2_expected_rr = 0

        sys_expected_mrr = (u1_expected_rr + u2_expected_rr) / 2
        sys_result_mrr = float(result_only_one[str(metric)])

        self.assertEqual(sys_expected_mrr, sys_result_mrr)

        # u1 = [0], u2 = [0]


class TestMRRAtK(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.k = 2

        cls.metric_mean_threshold = MRRAtK(cls.k)

    def test_perform(self):
        metric = MRRAtK(self.k, relevant_threshold=3)

        result_w_new_items = metric.perform(split_w_new_items)

        u1_expected_rr = 1/2
        u2_expected_rr = 1

        sys_expected_mrr = (u1_expected_rr + u2_expected_rr) / 2

        sys_result_mrr = float(result_w_new_items.query('from_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected_mrr, sys_result_mrr)

        # u1_total = [0, 1, 0, 0, 0, 1, 0, 0], u2_total = [1, 0, 1, 1]
        # u1@k = [0, 1], u2@k = [1, 0]

    def test_perform_w_greater_k(self):
        metric = MRRAtK(99, relevant_threshold=3)

        result = metric.perform(split_w_new_items)
        expected = MRR(relevant_threshold=3).perform(split_w_new_items)

        result = np.sort(result, axis=0)
        expected = np.sort(expected, axis=0)

        self.assertTrue(np.array_equal(expected, result))

    def test_perform_w_new_items_mean(self):
        metric = MRRAtK(self.k)

        result_w_new_items = metric.perform(split_w_new_items)

        u1_expected_rr = 1/2
        u2_expected_rr = 0

        sys_expected_mrr = (u1_expected_rr + u2_expected_rr) / 2

        sys_result_mrr = float(result_w_new_items.query('from_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected_mrr, sys_result_mrr)

        # mean u1 = 2.2; mean u2 = 3.33
        # u1_total = [0, 1, 0, 0, 0, 1, 0, 0], u2_total = [1, 0, 1, 1]
        # u1@k = [0, 1], u2@k = [0, 0]

    def test_k_not_valid(self):
        with self.assertRaises(KError):
            MRRAtK(k=-2)
            MRRAtK(k=0)


class TestCorrelation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.methods_list = ['pearson', 'spearman', 'kendall']
        cls.top_n = 2

    @for_each_method
    def test_perform_only_new(self, method: str):
        metric = Correlation(method)

        result_only_new = metric.perform(split_only_new)

        pearsons_predicted = result_only_new[str(metric)]

        self.assertTrue(all(pd.isna(pearsons_predicted)))

    @for_each_method
    def test_perform_only_new_top_n(self, method: str):
        metric = Correlation(method, top_n=self.top_n)

        result_only_new = metric.perform(split_only_new)

        pearsons_predicted = result_only_new[str(metric)]

        self.assertTrue(all(pd.isna(pearsons_predicted)))

    @for_each_method
    def test_perform_w_new_items(self, method: str):
        metric = Correlation(method)

        result_w_new_items = metric.perform(split_w_new_items)

        u1_actual = pd.Series([1, 5, 0, 4, 2])
        u1_ideal = pd.Series([0, 1, 2, 3, 4])

        u1_expected_pearson = u1_actual.corr(u1_ideal, method)
        u1_result_pearson = float(result_w_new_items.query('from_id == "u1"')[str(metric)])

        self.assertAlmostEqual(u1_expected_pearson, u1_result_pearson)

        u2_actual = pd.Series([2, 0, 3])
        u2_ideal = pd.Series([0, 1, 2])

        u2_expected_pearson = u2_actual.corr(u2_ideal, method)
        u2_result_pearson = float(result_w_new_items.query('from_id == "u2"')[str(metric)])

        self.assertAlmostEqual(u2_expected_pearson, u2_result_pearson)

        sys_expected_pearson = (u1_expected_pearson + u2_expected_pearson) / 2
        sys_result_pearson = float(result_w_new_items.query('from_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected_pearson, sys_result_pearson)

    @for_each_method
    def test_perform_w_new_items_top_n(self, method: str):
        metric = Correlation(method, top_n=self.top_n)

        result_w_new_items = metric.perform(split_w_new_items)

        u1_actual = pd.Series([1, 5])
        u1_ideal = pd.Series([0, 1, 2, 3, 4])

        u1_expected_pearson = u1_actual.corr(u1_ideal, method)
        u1_result_pearson = float(result_w_new_items.query('from_id == "u1"')[str(metric)])

        self.assertAlmostEqual(u1_expected_pearson, u1_result_pearson)

        u2_actual = pd.Series([2, 0])
        u2_ideal = pd.Series([0, 1, 2])

        u2_expected_pearson = u2_actual.corr(u2_ideal, method)
        u2_result_pearson = float(result_w_new_items.query('from_id == "u2"')[str(metric)])

        self.assertAlmostEqual(u2_expected_pearson, u2_result_pearson)

        sys_expected_pearson = (u1_expected_pearson + u2_expected_pearson) / 2
        sys_result_pearson = float(result_w_new_items.query('from_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected_pearson, sys_result_pearson)

    @for_each_method
    def test_perform_only_one(self, method: str):
        metric = Correlation(method)

        result_only_one = metric.perform(split_only_one)

        pearsons_predicted = result_only_one[str(metric)]

        self.assertTrue(all(pd.isna(pearsons_predicted)))

    @for_each_method
    def test_perform_only_one_top_n(self, method: str):
        metric = Correlation(method, top_n=self.top_n)

        result_only_one = metric.perform(split_only_one)

        pearsons_predicted = result_only_one[str(metric)]

        self.assertTrue(all(pd.isna(pearsons_predicted)))
