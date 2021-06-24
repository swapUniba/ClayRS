import unittest
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split
from orange_cb_recsys.evaluation.metrics.error_metrics import MSE, MAE, RMSE


user_pred_only_new_items = pd.DataFrame(
    {'from_id': ['u1', 'u1', 'u2', 'u2'],
     'to_id': ['inew1', 'inew2', 'inew3', 'inew4'],
     'score': [4.23, 3.55, 2.12, 4.56]})

user_pred_w_new_items = pd.DataFrame(
    {'from_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2'],
     'to_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i4', 'i6', 'i1', 'i8'],
     'score': [3.23, 2.33, 1.58, 4.23, 3.32, 4.8, 3.45, 2.56, 4.1, 3.21, 2.8, 1.57]})

user_pred_only_one_item = pd.DataFrame(
    {'from_id': ['u1', 'u2'],
     'to_id': ['i4', 'i8'],
     'score': [4.5, 3.66]})

user_truth = pd.DataFrame({'from_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
                           'to_id': ['i1', 'i2', 'i3', 'i4', 'i6', 'i1', 'i8', 'i4'],
                           'score': [3, 2, 3, 1, 2, 4, 2, 3]})

split_only_new = Split(user_pred_only_new_items, user_truth)
split_w_new_items = Split(user_pred_w_new_items, user_truth)
split_only_one = Split(user_pred_only_one_item, user_truth)


class TestMAE(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = MAE()

    def test_perform_only_new(self):
        metric = self.metric

        result = metric.perform(split_only_new)

        scores = result[str(metric)]

        self.assertTrue(all(pd.isna(scores)))

    def test_perform_w_new_items(self):
        metric = self.metric

        result = metric.perform(split_w_new_items)

        # We exclude predicted score for items not in the truth
        u1_predicted_scores = [2.33, 3.23, 4.8, 1.58, 3.32]
        u1_actual_scores = user_truth.query('from_id == "u1"')['score']

        u1_expected = mean_absolute_error(u1_actual_scores, u1_predicted_scores)
        u1_result = float(result.query('from_id == "u1"')[str(metric)])

        self.assertAlmostEqual(u1_expected, u1_result)

        # We exclude predicted score for items not in the truth
        u2_predicted_scores = [2.8, 1.57, 4.1]
        u2_actual_scores = user_truth.query('from_id == "u2"')['score']

        u2_expected = mean_absolute_error(u2_actual_scores, u2_predicted_scores)
        u2_result = float(result.query('from_id == "u2"')[str(metric)])

        self.assertAlmostEqual(u2_expected, u2_result)

        sys_expected = (u1_expected + u2_expected) / 2
        sys_result = float(result.query('from_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected, sys_result)

    def test_perform_only_one(self):
        metric = self.metric

        result = metric.perform(split_only_one)

        # We exclude predicted score for items not in the truth
        u1_predicted_scores = [4.5]
        u1_actual_scores = [1]

        u1_expected = mean_absolute_error(u1_actual_scores, u1_predicted_scores)
        u1_result = float(result.query('from_id == "u1"')[str(metric)])

        self.assertAlmostEqual(u1_expected, u1_result)

        # We exclude predicted score for items not in the truth
        u2_predicted_scores = [3.66]
        u2_actual_scores = [2]

        u2_expected = mean_absolute_error(u2_actual_scores, u2_predicted_scores)
        u2_result = float(result.query('from_id == "u2"')[str(metric)])

        self.assertAlmostEqual(u2_expected, u2_result)

        sys_expected = (u1_expected + u2_expected) / 2
        sys_result = float(result.query('from_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected, sys_result)


class TestMSE(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = MSE()

    def test_perform_only_new(self):
        metric = self.metric

        result = metric.perform(split_only_new)

        scores = result[str(metric)]

        self.assertTrue(all(pd.isna(scores)))

    def test_perform_w_new_items(self):
        metric = self.metric

        result = metric.perform(split_w_new_items)

        # We exclude predicted score for items not in the truth
        u1_predicted_scores = [2.33, 3.23, 4.8, 1.58, 3.32]
        u1_actual_scores = user_truth.query('from_id == "u1"')['score']

        u1_expected = mean_squared_error(u1_actual_scores, u1_predicted_scores)
        u1_result = float(result.query('from_id == "u1"')[str(metric)])

        self.assertAlmostEqual(u1_expected, u1_result)

        # We exclude predicted score for items not in the truth
        u2_predicted_scores = [2.8, 1.57, 4.1]
        u2_actual_scores = user_truth.query('from_id == "u2"')['score']

        u2_expected = mean_squared_error(u2_actual_scores, u2_predicted_scores)
        u2_result = float(result.query('from_id == "u2"')[str(metric)])

        self.assertAlmostEqual(u2_expected, u2_result)

        sys_expected = (u1_expected + u2_expected) / 2
        sys_result = float(result.query('from_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected, sys_result)

    def test_perform_only_one(self):
        metric = self.metric

        result = metric.perform(split_only_one)

        # We exclude predicted score for items not in the truth
        u1_predicted_scores = [4.5]
        u1_actual_scores = [1]

        u1_expected = mean_squared_error(u1_actual_scores, u1_predicted_scores)
        u1_result = float(result.query('from_id == "u1"')[str(metric)])

        self.assertAlmostEqual(u1_expected, u1_result)

        # We exclude predicted score for items not in the truth
        u2_predicted_scores = [3.66]
        u2_actual_scores = [2]

        u2_expected = mean_squared_error(u2_actual_scores, u2_predicted_scores)
        u2_result = float(result.query('from_id == "u2"')[str(metric)])

        self.assertAlmostEqual(u2_expected, u2_result)

        sys_expected = (u1_expected + u2_expected) / 2
        sys_result = float(result.query('from_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected, sys_result)


class TestRMSE(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = RMSE()

    def test_perform_only_new(self):
        metric = self.metric

        result = metric.perform(split_only_new)

        scores = result[str(metric)]

        self.assertTrue(all(pd.isna(scores)))

    def test_perform_w_new_items(self):
        metric = self.metric

        result = metric.perform(split_w_new_items)

        # We exclude predicted score for items not in the truth
        u1_predicted_scores = [2.33, 3.23, 4.8, 1.58, 3.32]
        u1_actual_scores = user_truth.query('from_id == "u1"')['score']

        u1_expected = mean_squared_error(u1_actual_scores, u1_predicted_scores, squared=False)
        u1_result = float(result.query('from_id == "u1"')[str(metric)])

        self.assertAlmostEqual(u1_expected, u1_result)

        # We exclude predicted score for items not in the truth
        u2_predicted_scores = [2.8, 1.57, 4.1]
        u2_actual_scores = user_truth.query('from_id == "u2"')['score']

        u2_expected = mean_squared_error(u2_actual_scores, u2_predicted_scores, squared=False)
        u2_result = float(result.query('from_id == "u2"')[str(metric)])

        self.assertAlmostEqual(u2_expected, u2_result)

        sys_expected = (u1_expected + u2_expected) / 2
        sys_result = float(result.query('from_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected, sys_result)

    def test_perform_only_one(self):
        metric = self.metric

        result = metric.perform(split_only_one)

        # We exclude predicted score for items not in the truth
        u1_predicted_scores = [4.5]
        u1_actual_scores = [1]

        u1_expected = mean_squared_error(u1_actual_scores, u1_predicted_scores, squared=False)
        u1_result = float(result.query('from_id == "u1"')[str(metric)])

        self.assertAlmostEqual(u1_expected, u1_result)

        # We exclude predicted score for items not in the truth
        u2_predicted_scores = [3.66]
        u2_actual_scores = [2]

        u2_expected = mean_squared_error(u2_actual_scores, u2_predicted_scores, squared=False)
        u2_result = float(result.query('from_id == "u2"')[str(metric)])

        self.assertAlmostEqual(u2_expected, u2_result)

        sys_expected = (u1_expected + u2_expected) / 2
        sys_result = float(result.query('from_id == "sys"')[str(metric)])

        self.assertAlmostEqual(sys_expected, sys_result)

