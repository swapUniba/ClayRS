import unittest
import pandas as pd
from sklearn.metrics import ndcg_score
import numpy as np

from clayrs.content_analyzer import Ratings
from clayrs.evaluation.metrics.ranking_metrics import NDCG, Correlation, MRR, NDCGAtK, MRRAtK, MAP, MAPAtK
from clayrs.evaluation.eval_pipeline_modules.metric_evaluator import Split

pred_only_new_items = pd.DataFrame(
    {'user_id': ['u1', 'u1', 'u2', 'u2'],
     'item_id': ['inew1', 'inew2', 'inew3', 'inew4'],
     'score': [650, 600, 500, 650]})
pred_only_new_items = Ratings.from_dataframe(pred_only_new_items)

pred_w_new_items = pd.DataFrame(
    {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2'],
     'item_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i4', 'i6', 'i1', 'i8'],
     'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50]})
pred_w_new_items = Ratings.from_dataframe(pred_w_new_items)

pred_only_one_item = pd.DataFrame(
    {'user_id': ['u1', 'u2'],
     'item_id': ['i4', 'i8'],
     'score': [650, 600]})
pred_only_one_item = Ratings.from_dataframe(pred_only_one_item)

truth = pd.DataFrame({'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
                      'item_id': ['i1', 'i2', 'i3', 'i4', 'i6', 'i1', 'i8', 'i4'],
                      'score': [3, 2, 3, 1, 2, 4, 3, 3]})
truth = Ratings.from_dataframe(truth)


split_only_new = Split(pred_only_new_items, truth)
split_w_new_items = Split(pred_w_new_items, truth)
split_only_one = Split(pred_only_one_item, truth)


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

        # no items predicted is in truth, so ndcg can't be computed
        for result in result_only_new[str(metric)]:
            self.assertTrue(np.isnan(result))

    def test_perform_w_new_items(self):
        metric = self.metric

        result_w_new_items = metric.perform(split_w_new_items)

        u1_actual = [[600, 650, 220, 500, 300]]
        u1_truth = [[3, 2, 3, 1, 2]]

        u1_expected_ndcg = ndcg_score(u1_truth, u1_actual)
        u1_result_ndcg = float(result_w_new_items.query('user_id == "u1"')[str(metric)])

        self.assertAlmostEqual(u1_expected_ndcg, u1_result_ndcg)

        u2_actual = [[100, 50, 350]]
        u2_ideal = [[4, 3, 3]]

        u2_expected_ndcg = ndcg_score(u2_ideal, u2_actual)
        u2_result_ndcg = float(result_w_new_items.query('user_id == "u2"')[str(metric)])

        self.assertAlmostEqual(u2_expected_ndcg, u2_result_ndcg)

        sys_expected_ndcg = (u1_expected_ndcg + u2_expected_ndcg) / 2
        sys_result_ndcg = float(result_w_new_items.query('user_id == "sys"')[str(metric)])

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

        u1_actual_all = [[600, 650, 220, 500, 300]]
        u1_truth_all = [[3, 2, 3, 1, 2]]

        u1_expected_ndcg = ndcg_score(u1_truth_all, u1_actual_all, k=k)
        u1_result_ndcg = float(result_w_new_items.query('user_id == "u1"')[str(metric)])

        self.assertAlmostEqual(u1_expected_ndcg, u1_result_ndcg)

        u2_actual_all = [[100, 50, 350]]
        u2_truth_all = [[4, 3, 3]]

        u2_expected_ndcg = ndcg_score(u2_truth_all, u2_actual_all, k=k)
        u2_result_ndcg = float(result_w_new_items.query('user_id == "u2"')[str(metric)])

        self.assertAlmostEqual(u2_expected_ndcg, u2_result_ndcg)

        sys_expected_ndcg = (u1_expected_ndcg + u2_expected_ndcg) / 2
        sys_result_ndcg = float(result_w_new_items.query('user_id == "sys"')[str(metric)])

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

        u1_expected_rr = 1 / 2
        u2_expected_rr = 1

        sys_expected_mrr = (u1_expected_rr + u2_expected_rr) / 2

        sys_result_mrr = float(result_w_new_items.query('user_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected_mrr, sys_result_mrr)

        # u1 = [0, 1, 0, 0, 0, 1, 0, 0], u2 = [1, 0, 1, 1]

    def test_perform_w_new_items_mean(self):
        metric = self.metric_mean_threshold

        result_w_new_items = metric.perform(split_w_new_items)

        u1_expected_rr = 1 / 2
        u2_expected_rr = 1 / 3

        sys_expected_mrr = (u1_expected_rr + u2_expected_rr) / 2

        sys_result_mrr = float(result_w_new_items.query('user_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected_mrr, sys_result_mrr)

        # u1 = [0, 1, 0, 0, 0, 1, 0, 0], u2 = [0, 0, 1, 0]

    def test_perform_only_one(self):
        metric = self.metric

        result_only_one = metric.perform(split_only_one)

        u1_expected_rr = 0
        u2_expected_rr = 1 / 1

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

    def test_perform_no_relevant_u1(self):
        metric_no_relevant_u1 = MRR(relevant_threshold=4)

        result = metric_no_relevant_u1.perform(split_w_new_items)

        expected_sys = 1 / 3  # only u2 result will be considered
        result_sys = float(result.query('user_id == "sys"')[str(metric_no_relevant_u1)])
        self.assertAlmostEqual(expected_sys, result_sys)

        # u1 has no relevant items, u2 = [0, 0, 1, 0]

    def test_no_relevant_items(self):
        truth_no_rel = pd.DataFrame({'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
                                     'item_id': ['i1', 'i2', 'i3', 'i4', 'i6', 'i1', 'i8', 'i4'],
                                     'score': [3, 2, 3, 1, 2, 2, 3, 3]})
        truth_no_rel = Ratings.from_dataframe(truth_no_rel)

        split_no_rel = Split(pred_w_new_items, truth_no_rel)

        # any ClassificationMetric will work
        metric = MRR(relevant_threshold=4)

        with self.assertRaises(ValueError):
            metric.perform(split_no_rel)


class TestMRRAtK(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        k = 2
        cls.k = k

        cls.metric_mean_threshold = MRRAtK(k)

    def test_perform(self):
        metric = MRRAtK(self.k, relevant_threshold=3)

        result_w_new_items = metric.perform(split_w_new_items)

        u1_expected_rr = 1 / 2
        u2_expected_rr = 1

        sys_expected_mrr = (u1_expected_rr + u2_expected_rr) / 2

        sys_result_mrr = float(result_w_new_items.query('user_id == "sys"')[str(metric)])

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

        u1_expected_rr = 1 / 2
        u2_expected_rr = 0

        sys_expected_mrr = (u1_expected_rr + u2_expected_rr) / 2

        sys_result_mrr = float(result_w_new_items.query('user_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected_mrr, sys_result_mrr)

        # mean u1 = 2.2; mean u2 = 3.33
        # u1_total = [0, 1, 0, 0, 0, 1, 0, 0], u2_total = [1, 0, 1, 1]
        # u1@k = [0, 1], u2@k = [0, 0]

    def test_k_not_valid(self):
        with self.assertRaises(ValueError):
            MRRAtK(k=-2)
            MRRAtK(k=0)


class TestMAP(unittest.TestCase):

    def test_perform_w_new(self):
        split = Split(pred_w_new_items, truth)

        metric = MAP(relevant_threshold=3)
        df_result = metric.perform(split)

        u1_result = float(df_result.query('user_id == "u1"')['AP'])
        u1_expected = 1/2 * (1/2 + 2/6)
        self.assertAlmostEqual(u1_expected, u1_result)

        u2_result = float(df_result.query('user_id == "u2"')['AP'])
        u2_expected = 1/3 * (1/1 + 2/3 + 3/4)
        self.assertAlmostEqual(u2_expected, u2_result)

        sys_result = float(df_result.query('user_id == "sys"')['MAP'])
        sys_expected = (u1_expected + u2_expected) / 2
        self.assertAlmostEqual(sys_expected, sys_result)

        # WITH MEAN as relevant threshold
        metric = MAP(relevant_threshold=None)
        df_result = metric.perform(split)

        u1_result = float(df_result.query('user_id == "u1"')['AP'])
        u1_expected = 1/2 * (1/2 + 2/6)
        self.assertAlmostEqual(u1_expected, u1_result)

        u2_result = float(df_result.query('user_id == "u2"')['AP'])
        u2_expected = 1/1 * (1/3)
        self.assertAlmostEqual(u2_expected, u2_result)

        sys_result = float(df_result.query('user_id == "sys"')['MAP'])
        sys_expected = (u1_expected + u2_expected) / 2
        self.assertAlmostEqual(sys_expected, sys_result)

    def test_perform_only_one(self):
        split = Split(pred_only_one_item, truth)

        metric = MAP(relevant_threshold=3)
        df_result = metric.perform(split)

        u1_result = float(df_result.query('user_id == "u1"')['AP'])
        u1_expected = 0
        self.assertAlmostEqual(u1_expected, u1_result)

        u2_result = float(df_result.query('user_id == "u2"')['AP'])
        u2_expected = 1/3 * (1/1)
        self.assertAlmostEqual(u2_expected, u2_result)

        sys_result = float(df_result.query('user_id == "sys"')['MAP'])
        sys_expected = (u1_expected + u2_expected) / 2
        self.assertAlmostEqual(sys_expected, sys_result)

    def test_perform_only_new(self):

        split = Split(pred_only_new_items, truth)

        metric = MAP(relevant_threshold=2)
        df_result = metric.perform(split)

        u1_result = float(df_result.query('user_id == "u1"')['AP'])
        u1_expected = 0
        self.assertEqual(u1_expected, u1_result)

        u2_result = float(df_result.query('user_id == "u2"')['AP'])
        u2_expected = 0
        self.assertEqual(u2_expected, u2_result)

        sys_result = float(df_result.query('user_id == "sys"')['MAP'])
        sys_expected = 0
        self.assertEqual(sys_expected, sys_result)

    def test_perform_nan(self):
        # RELEVANT THRESHOLD greater than all rating given by u1
        split = Split(pred_w_new_items, truth)
        metric = MAP(relevant_threshold=4)
        df_result = metric.perform(split)

        u1_result = df_result.query('user_id == "u1"')['AP'].values
        self.assertTrue(pd.isna(u1_result))

        u2_result = float(df_result.query('user_id == "u2"')['AP'])
        u2_expected = 1/1 * (1/3)
        self.assertAlmostEqual(u2_expected, u2_result)

        sys_result = float(df_result.query('user_id == "sys"')['MAP'])
        sys_expected = u2_expected  # since u1 is nan only u2 will matter for MAP computation
        self.assertAlmostEqual(sys_expected, sys_result)


class TestMAPAtK(unittest.TestCase):

    def test__compute_ap(self):
        relevant_threshold = 3

        u1_uidx_pred = pred_w_new_items.user_map["u1"]
        user_predictions_items_uidx = pred_w_new_items.get_user_interactions(u1_uidx_pred)[:, 1].astype(int)
        user_predictions_items_str = pred_w_new_items.item_map[user_predictions_items_uidx]

        u1_uidx_truth = truth.user_map["u1"]
        user_truth = truth.get_user_interactions(u1_uidx_truth)
        user_truth_relevant_items_uidx = user_truth[np.where(user_truth[:, 2] >= relevant_threshold)][:, 1].astype(int)
        user_truth_relevant_items_str = truth.item_map[user_truth_relevant_items_uidx]

        metric = MAPAtK(k=2)

        result_u1_ap = metric._compute_ap(user_predictions_items_str, user_truth_relevant_items_str)
        expected_u1_ap = 1/2 * 1/2
        self.assertAlmostEqual(expected_u1_ap, result_u1_ap)


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
    def test_perform_w_new_items(self, method):
        metric = Correlation(method)

        result_w_new_items = metric.perform(split_w_new_items)

        u1_truth = pd.Series([3, 2, 3, 1, 2])
        u1_pred = pd.Series([600, 650, 220, 500, 300])

        u1_expected_corr = u1_truth.corr(u1_pred, method)
        u1_result_corr = float(result_w_new_items.query('user_id == "u1"')[str(metric)])

        self.assertAlmostEqual(u1_expected_corr, u1_result_corr)

        u2_truth = pd.Series([4, 3, 3])
        u2_pred = pd.Series([100, 50, 350])

        u2_expected_corr = u2_truth.corr(u2_pred, method)
        u2_result_corr = float(result_w_new_items.query('user_id == "u2"')[str(metric)])

        self.assertAlmostEqual(u2_expected_corr, u2_result_corr)

        sys_expected_pearson = (u1_expected_corr + u2_expected_corr) / 2
        sys_result_pearson = float(result_w_new_items.query('user_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected_pearson, sys_result_pearson)

    @for_each_method
    def test_perform_w_new_items_top_n(self, method: str):
        metric = Correlation(method, top_n=self.top_n)

        result_w_new_items = metric.perform(split_w_new_items)

        # top_2 items i2==650, i1==600
        u1_truth = pd.Series([3, 2])
        u1_pred = pd.Series([600, 650])

        u1_expected_corr = u1_truth.corr(u1_pred, method)
        u1_result_corr = float(result_w_new_items.query('user_id == "u1"')[str(metric)])

        self.assertAlmostEqual(u1_expected_corr, u1_result_corr)

        # top_2 items i4=350, i6==200 which does not appear in truth
        u2_truth = pd.Series([3])
        u2_pred = pd.Series([350])

        u2_expected_corr = u2_truth.corr(u2_pred, method)
        u2_result_corr = float(result_w_new_items.query('user_id == "u2"')[str(metric)])

        self.assertTrue(np.isnan(u2_expected_corr))
        self.assertTrue(np.isnan(u2_result_corr))

        sys_expected_corr = u1_expected_corr  # the mean doesn't consider nan values
        sys_result_corr = float(result_w_new_items.query('user_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected_corr, sys_result_corr)

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


if __name__ == '__main__':
    unittest.main()
