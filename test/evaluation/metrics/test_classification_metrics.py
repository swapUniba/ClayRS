import unittest
from unittest import TestCase
import pandas as pd
import numpy as np

from clayrs.content_analyzer import Ratings
from clayrs.evaluation.metrics.classification_metrics import Precision, Recall, FMeasure, PrecisionAtK, \
    RPrecision, RecallAtK, FMeasureAtK
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

pred_i1_i4_missing = pd.DataFrame(
    {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
     'item_id': ['i2', 'i5', 'i6', 'i3', 'i8', 'i9', 'i6', 'i1', 'i8'],
     'score': [600, 400, 300, 220, 100, 50, 200, 100, 50]})
pred_i1_i4_missing = Ratings.from_dataframe(pred_i1_i4_missing)

truth = pd.DataFrame({'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
                      'item_id': ['i1', 'i2', 'i3', 'i4', 'i6', 'i1', 'i8', 'i4'],
                      'score': [3, 2, 3, 1, 2, 4, 3, 3]})
truth = Ratings.from_dataframe(truth)

split_only_new = Split(pred_only_new_items, truth)
split_w_new_items = Split(pred_w_new_items, truth)
split_only_one = Split(pred_only_one_item, truth)
split_missing = Split(pred_i1_i4_missing, truth)


class TestClassificationMetric(TestCase):

    def test_perform_no_relevant_u1(self):
        # any classification metric will work, you just need to change the ratio in the test
        metric_no_relevant_u1 = Precision(relevant_threshold=4)

        result = metric_no_relevant_u1.perform(split_w_new_items)

        result_u1 = result.query('user_id == "u1"')[str(metric_no_relevant_u1)].values[0]
        self.assertTrue(np.isnan(result_u1))

        expected_u2 = 2 / 8
        result_u2 = float(result.query('user_id == "u2"')[str(metric_no_relevant_u1)])
        self.assertAlmostEqual(expected_u2, result_u2)

        expected_sys = expected_u2
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
        metric = Precision(relevant_threshold=4)

        with self.assertRaises(ValueError):
            metric.perform(split_no_rel)

    def test_exception_different_users(self):

        # test case in which user appears in prediction set but not in truth
        pred_all_users = pred_w_new_items

        truth_missing_u2 = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1'],
             'item_id': ['i1', 'i2', 'i3', 'i4', 'i6'],
             'score': [3, 2, 3, 1, 2]})
        truth_missing_u2 = Ratings.from_dataframe(truth_missing_u2)

        with self.assertRaises(ValueError):
            Precision(relevant_threshold=2).perform(Split(pred_all_users, truth_missing_u2))

        # test case in which user appears in truth but not in prediction set
        pred_missing_u1 = pd.DataFrame(
            {'user_id': ['u2', 'u2', 'u2', 'u2'],
             'item_id': ['i4', 'i6', 'i1', 'i8'],
             'score': [350, 200, 100, 50]})
        pred_missing_u1 = Ratings.from_dataframe(pred_missing_u1)

        truth_all_users = truth

        with self.assertRaises(ValueError):
            Precision(relevant_threshold=2).perform(Split(pred_missing_u1, truth_all_users))

class TestPrecision(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.metric_macro = Precision(relevant_threshold=3, sys_average='macro')
        cls.metric_micro = Precision(relevant_threshold=3, sys_average='micro')
        cls.metric_mean = Precision(sys_average='macro')

    def test_perform_only_new(self):
        metric_macro = self.metric_macro
        metric_micro = self.metric_micro
        metric_mean = self.metric_mean

        result = metric_macro.perform(split_only_new)[str(metric_macro)]
        self.assertEqual(0, all(result))

        result = metric_micro.perform(split_only_new)[str(metric_micro)]
        self.assertEqual(0, all(result))

        result = metric_mean.perform(split_only_new)[str(metric_mean)]
        self.assertEqual(0, all(result))

        # u1 = [0, 0], u2 = [0, 0]

    def test_perform_w_new_items_macro(self):
        metric_macro = self.metric_macro

        result_macro = metric_macro.perform(split_w_new_items)

        expected_u1 = 2 / 8
        result_macro_u1 = float(result_macro.query('user_id == "u1"')[str(metric_macro)])
        self.assertAlmostEqual(expected_u1, result_macro_u1)

        expected_u2 = 3 / 4
        result_macro_u2 = float(result_macro.query('user_id == "u2"')[str(metric_macro)])
        self.assertAlmostEqual(expected_u2, result_macro_u2)

        expected_macro_sys = (expected_u1 + expected_u2) / 2
        result_macro_sys = float(result_macro.query('user_id == "sys"')[str(metric_macro)])
        self.assertAlmostEqual(expected_macro_sys, result_macro_sys)

        # u1 = [0, 1, 0, 0, 0, 1, 0, 0], u2 = [1, 0, 1, 1]

    def test_perform_w_new_items_micro(self):
        metric_micro = self.metric_micro

        result_micro = metric_micro.perform(split_w_new_items)

        expected_u1 = 2 / 8
        result_micro_u1 = float(result_micro.query('user_id == "u1"')[str(metric_micro)])
        self.assertAlmostEqual(expected_u1, result_micro_u1)

        expected_u2 = 3 / 4
        result_micro_u2 = float(result_micro.query('user_id == "u2"')[str(metric_micro)])
        self.assertAlmostEqual(expected_u2, result_micro_u2)

        expected_micro_sys = (2 + 3) / ((2 + 3) + (6 + 1))
        result_micro_sys = float(result_micro.query('user_id == "sys"')[str(metric_micro)])
        self.assertAlmostEqual(expected_micro_sys, result_micro_sys)

        # u1 = [0, 1, 0, 0, 0, 1, 0, 0], u2 = [1, 0, 1, 1]

    def test_perform_w_new_items_mean(self):
        metric_mean = self.metric_mean

        result_mean = metric_mean.perform(split_w_new_items)

        expected_u1 = 2 / 8
        result_mean_u1 = float(result_mean.query('user_id == "u1"')[str(metric_mean)])
        self.assertAlmostEqual(expected_u1, result_mean_u1)

        expected_u2 = 1 / 4
        result_mean_u2 = float(result_mean.query('user_id == "u2"')[str(metric_mean)])
        self.assertAlmostEqual(expected_u2, result_mean_u2)

        expected_mean_sys = (expected_u1 + expected_u2) / 2
        result_mean_sys = float(result_mean.query('user_id == "sys"')[str(metric_mean)])
        self.assertAlmostEqual(expected_mean_sys, result_mean_sys)

        # u1 = [0, 1, 0, 0, 0, 1, 0, 0], u2 = [0, 0, 1, 0]
        # mean_u1 = 2.2, mean_u2 = 3.33

    def test_perform_only_one_macro(self):
        metric_macro = self.metric_macro

        result_macro = metric_macro.perform(split_only_one)

        expected_u1 = 0 / 1
        result_macro_u1 = float(result_macro.query('user_id == "u1"')[str(metric_macro)])
        self.assertAlmostEqual(expected_u1, result_macro_u1)

        expected_u2 = 1 / 1
        result_macro_u2 = float(result_macro.query('user_id == "u2"')[str(metric_macro)])
        self.assertAlmostEqual(expected_u2, result_macro_u2)

        expected_macro_sys = (expected_u1 + expected_u2) / 2
        result_macro_sys = float(result_macro.query('user_id == "sys"')[str(metric_macro)])
        self.assertAlmostEqual(expected_macro_sys, result_macro_sys)

        # u1 = [0], u2 = [1]

    def test_perform_only_one_micro(self):
        metric_micro = self.metric_micro

        result_micro = metric_micro.perform(split_only_one)

        expected_u1 = 0 / 1
        result_micro_u1 = float(result_micro.query('user_id == "u1"')[str(metric_micro)])
        self.assertAlmostEqual(expected_u1, result_micro_u1)

        expected_u2 = 1 / 1
        result_micro_u2 = float(result_micro.query('user_id == "u2"')[str(metric_micro)])
        self.assertAlmostEqual(expected_u2, result_micro_u2)

        expected_micro_sys = (0 + 1) / ((0 + 1) + (1 + 0))
        result_micro_sys = float(result_micro.query('user_id == "sys"')[str(metric_micro)])
        self.assertAlmostEqual(expected_micro_sys, result_micro_sys)

        # u1 = [0], u2 = [1]

    def test_perform_only_one_mean(self):
        metric_mean = self.metric_mean

        result_mean = metric_mean.perform(split_only_one)

        expected_u1 = 0 / 1
        result_mean_u1 = float(result_mean.query('user_id == "u1"')[str(metric_mean)])
        self.assertAlmostEqual(expected_u1, result_mean_u1)

        expected_u2 = 0 / 1
        result_mean_u2 = float(result_mean.query('user_id == "u2"')[str(metric_mean)])
        self.assertAlmostEqual(expected_u2, result_mean_u2)

        expected_mean_sys = (expected_u1 + expected_u2) / 2
        result_mean_sys = float(result_mean.query('user_id == "sys"')[str(metric_mean)])
        self.assertAlmostEqual(expected_mean_sys, result_mean_sys)

        # u1 = [0], u2 = [0]
        # mean_u1 = 2.2, mean_u2 = 3.33


# Only new tests other than those of TestPrecision, to remove redundancy
class TestPrecisionAtK(TestCase):
    def test_perform(self):
        metric = PrecisionAtK(k=3, relevant_threshold=3)

        result = metric.perform(split_w_new_items)

        expected_u1 = 1 / 3
        result_macro_u1 = float(result.query('user_id == "u1"')[str(metric)])
        self.assertAlmostEqual(expected_u1, result_macro_u1)

        expected_u2 = 2 / 3
        result_macro_u2 = float(result.query('user_id == "u2"')[str(metric)])
        self.assertAlmostEqual(expected_u2, result_macro_u2)

        expected_macro_sys = (expected_u1 + expected_u2) / 2
        result_macro_sys = float(result.query('user_id == "sys"')[str(metric)])
        self.assertAlmostEqual(expected_macro_sys, result_macro_sys)

        # u1_total = [0, 1, 0, 0, 0, 1, 0, 0], u2_total = [1, 0, 1, 1]
        # u1@k = [0, 1, 0], u2@k = [1, 0, 1]

    def test_perform_greater_k(self):
        metric = PrecisionAtK(k=999, relevant_threshold=3)

        result = metric.perform(split_w_new_items)
        expected = Precision(relevant_threshold=3).perform(split_w_new_items)

        result = np.sort(result, axis=0)
        expected = np.sort(expected, axis=0)

        # If k > than number of rows, then it's a regular Precision
        self.assertTrue(np.array_equal(expected, result))

    def test_perform_mean(self):
        metric = PrecisionAtK(k=3)

        result = metric.perform(split_w_new_items)

        expected_u1 = 1 / 3
        result_mean_u1 = float(result.query('user_id == "u1"')[str(metric)])
        self.assertAlmostEqual(expected_u1, result_mean_u1)

        expected_u2 = 1 / 3
        result_mean_u2 = float(result.query('user_id == "u2"')[str(metric)])
        self.assertAlmostEqual(expected_u2, result_mean_u2)

        expected_mean_sys = (expected_u1 + expected_u2) / 2
        result_mean_sys = float(result.query('user_id == "sys"')[str(metric)])
        self.assertAlmostEqual(expected_mean_sys, result_mean_sys)

        # u1_total = [0, 1, 0, 0, 0, 1, 0, 0], u2_total = [0, 0, 1, 0]
        # u1@k = [0, 1, 0], u2@k = [0, 0, 1]
        # mean_u1 = 2.2, mean_u2 = 3.33

    def test_perform_i1_i4_missing(self):
        metric = PrecisionAtK(k=3, relevant_threshold=3)

        result_macro = metric.perform(split_missing)

        expected_u1 = 0 / 3
        result_macro_u1 = float(result_macro.query('user_id == "u1"')[str(metric)])
        self.assertAlmostEqual(expected_u1, result_macro_u1)

        expected_u2 = 2 / 3
        result_macro_u2 = float(result_macro.query('user_id == "u2"')[str(metric)])
        self.assertAlmostEqual(expected_u2, result_macro_u2)

        expected_macro_sys = (expected_u1 + expected_u2) / 2
        result_macro_sys = float(result_macro.query('user_id == "sys"')[str(metric)])
        self.assertAlmostEqual(expected_macro_sys, result_macro_sys)

        # u1_total = [0, 0, 0, 1, 0, 0], u2_total = [0, 1, 1]
        # u1@k = [0, 0, 0], u2@k = [0, 1, 1]


# Only new tests other than those of TestPrecision, to remove redundancy
class TestRPrecision(TestCase):
    def test_perform(self):
        metric = RPrecision(relevant_threshold=3)

        result = metric.perform(split_w_new_items)

        expected_u1 = 1 / 2
        result_macro_u1 = float(result.query('user_id == "u1"')[str(metric)])
        self.assertAlmostEqual(expected_u1, result_macro_u1)

        expected_u2 = 2 / 3
        result_macro_u2 = float(result.query('user_id == "u2"')[str(metric)])
        self.assertAlmostEqual(expected_u2, result_macro_u2)

        expected_macro_sys = (expected_u1 + expected_u2) / 2
        result_macro_sys = float(result.query('user_id == "sys"')[str(metric)])
        self.assertAlmostEqual(expected_macro_sys, result_macro_sys)

        # u1_total = [0, 1, 0, 0, 0, 1, 0, 0], u2_total = [1, 0, 1, 1]
        # u1@r = [0, 1], u2@r = [1, 0, 1]

    def test_perform_mean(self):
        metric = RPrecision()

        result = metric.perform(split_w_new_items)

        expected_u1 = 1 / 2
        result_mean_u1 = float(result.query('user_id == "u1"')[str(metric)])
        self.assertAlmostEqual(expected_u1, result_mean_u1)

        expected_u2 = 0 / 1
        result_mean_u2 = float(result.query('user_id == "u2"')[str(metric)])
        self.assertAlmostEqual(expected_u2, result_mean_u2)

        expected_mean_sys = (expected_u1 + expected_u2) / 2
        result_mean_sys = float(result.query('user_id == "sys"')[str(metric)])
        self.assertAlmostEqual(expected_mean_sys, result_mean_sys)

        # u1_total = [0, 1, 0, 0, 0, 1, 0, 0], u2_total = [0, 0, 1, 0]
        # u1@r = [0, 1], u2@r = [0]
        # mean_u1 = 2.2, mean_u2 = 3.33

    def test_perform_i1_i4_missing(self):
        metric = RPrecision(relevant_threshold=3)

        result_macro = metric.perform(split_missing)

        expected_u1 = 0 / 2
        result_macro_u1 = float(result_macro.query('user_id == "u1"')[str(metric)])
        self.assertAlmostEqual(expected_u1, result_macro_u1)

        expected_u2 = 2 / 3
        result_macro_u2 = float(result_macro.query('user_id == "u2"')[str(metric)])
        self.assertAlmostEqual(expected_u2, result_macro_u2)

        expected_macro_sys = (expected_u1 + expected_u2) / 2
        result_macro_sys = float(result_macro.query('user_id == "sys"')[str(metric)])
        self.assertAlmostEqual(expected_macro_sys, result_macro_sys)

        # u1_total = [0, 0, 0, 0, 1, 0, 0], u2_total = [0, 0, 1, 1]
        # u1@r = [0, 0], u2@r = [0, 0]


class TestRecall(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.metric_macro = Recall(relevant_threshold=3, sys_average='macro')
        cls.metric_micro = Recall(relevant_threshold=3, sys_average='micro')
        cls.metric_mean = Recall(sys_average='macro')

    def test_perform_only_new(self):
        metric_macro = self.metric_macro
        metric_micro = self.metric_micro
        metric_mean = self.metric_mean

        result = metric_macro.perform(split_only_new)[str(metric_macro)]
        self.assertEqual(0, all(result))

        result = metric_micro.perform(split_only_new)[str(metric_micro)]
        self.assertEqual(0, all(result))

        result = metric_mean.perform(split_only_new)[str(metric_mean)]
        self.assertEqual(0, all(result))

        # u1 = [0, 0], u2 = [0, 0]

    def test_perform_w_new_items_macro(self):
        metric_macro = self.metric_macro

        result_macro = metric_macro.perform(split_w_new_items)

        expected_u1 = 2 / 2
        result_macro_u1 = float(result_macro.query('user_id == "u1"')[str(metric_macro)])
        self.assertAlmostEqual(expected_u1, result_macro_u1)

        expected_u2 = 3 / 3
        result_macro_u2 = float(result_macro.query('user_id == "u2"')[str(metric_macro)])
        self.assertAlmostEqual(expected_u2, result_macro_u2)

        expected_macro_sys = (expected_u1 + expected_u2) / 2
        result_macro_sys = float(result_macro.query('user_id == "sys"')[str(metric_macro)])
        self.assertAlmostEqual(expected_macro_sys, result_macro_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 3
        # u1 = [0, 1, 0, 0, 0, 1, 0, 0], u2 = [1, 0, 1, 1]

    def test_perform_w_new_items_micro(self):
        metric_micro = self.metric_micro

        result_micro = metric_micro.perform(split_w_new_items)

        expected_u1 = 2 / 2
        result_micro_u1 = float(result_micro.query('user_id == "u1"')[str(metric_micro)])
        self.assertAlmostEqual(expected_u1, result_micro_u1)

        expected_u2 = 3 / 3
        result_micro_u2 = float(result_micro.query('user_id == "u2"')[str(metric_micro)])
        self.assertAlmostEqual(expected_u2, result_micro_u2)

        expected_micro_sys = (2 + 3) / ((2 + 3) + (0 + 0))
        result_micro_sys = float(result_micro.query('user_id == "sys"')[str(metric_micro)])
        self.assertAlmostEqual(expected_micro_sys, result_micro_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 3
        # u1 = [0, 1, 0, 0, 0, 1, 0, 0], u2 = [1, 0, 1, 1]

    def test_perform_w_new_items_mean(self):
        metric_mean = self.metric_mean

        result_mean = metric_mean.perform(split_w_new_items)

        expected_u1 = 2 / 2
        result_mean_u1 = float(result_mean.query('user_id == "u1"')[str(metric_mean)])
        self.assertAlmostEqual(expected_u1, result_mean_u1)

        expected_u2 = 1 / 1
        result_mean_u2 = float(result_mean.query('user_id == "u2"')[str(metric_mean)])
        self.assertAlmostEqual(expected_u2, result_mean_u2)

        expected_mean_sys = (expected_u1 + expected_u2) / 2
        result_mean_sys = float(result_mean.query('user_id == "sys"')[str(metric_mean)])
        self.assertAlmostEqual(expected_mean_sys, result_mean_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 1
        # u1 = [0, 1, 0, 0, 0, 1, 0, 0], u2 = [0, 0, 1, 0]
        # mean_u1 = 2.2, mean_u2 = 3.33

    def test_perform_only_one_macro(self):
        metric_macro = self.metric_macro

        result_macro = metric_macro.perform(split_only_one)

        expected_u1 = 0 / 2
        result_macro_u1 = float(result_macro.query('user_id == "u1"')[str(metric_macro)])
        self.assertAlmostEqual(expected_u1, result_macro_u1)

        expected_u2 = 1 / 3
        result_macro_u2 = float(result_macro.query('user_id == "u2"')[str(metric_macro)])
        self.assertAlmostEqual(expected_u2, result_macro_u2)

        expected_macro_sys = (expected_u1 + expected_u2) / 2
        result_macro_sys = float(result_macro.query('user_id == "sys"')[str(metric_macro)])
        self.assertAlmostEqual(expected_macro_sys, result_macro_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 3
        # u1 = [0], u2 = [1]

    def test_perform_only_one_micro(self):
        metric_micro = self.metric_micro

        result_micro = metric_micro.perform(split_only_one)

        expected_u1 = 0 / 2
        result_micro_u1 = float(result_micro.query('user_id == "u1"')[str(metric_micro)])
        self.assertAlmostEqual(expected_u1, result_micro_u1)

        expected_u2 = 1 / 3
        result_micro_u2 = float(result_micro.query('user_id == "u2"')[str(metric_micro)])
        self.assertAlmostEqual(expected_u2, result_micro_u2)

        expected_micro_sys = (0 + 1) / ((0 + 1) + (2 + 2))
        result_micro_sys = float(result_micro.query('user_id == "sys"')[str(metric_micro)])
        self.assertAlmostEqual(expected_micro_sys, result_micro_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 3
        # u1 = [0], u2 = [1]

    def test_perform_only_one_mean(self):
        metric_mean = self.metric_mean

        result_mean = metric_mean.perform(split_only_one)

        expected_u1 = 0 / 2
        result_mean_u1 = float(result_mean.query('user_id == "u1"')[str(metric_mean)])
        self.assertAlmostEqual(expected_u1, result_mean_u1)

        expected_u2 = 0 / 1
        result_mean_u2 = float(result_mean.query('user_id == "u2"')[str(metric_mean)])
        self.assertAlmostEqual(expected_u2, result_mean_u2)

        expected_mean_sys = (expected_u1 + expected_u2) / 2
        result_mean_sys = float(result_mean.query('user_id == "sys"')[str(metric_mean)])
        self.assertAlmostEqual(expected_mean_sys, result_mean_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 1
        # u1 = [0], u2 = [0]
        # mean_u1 = 2.2, mean_u2 = 3.33


# Only new tests other than those of TestRecall, to remove redundancy
class TestRecallAtK(TestCase):
    def test_perform(self):
        metric = RecallAtK(k=3, relevant_threshold=3)

        result = metric.perform(split_w_new_items)

        expected_u1 = 1 / 2
        result_macro_u1 = float(result.query('user_id == "u1"')[str(metric)])
        self.assertAlmostEqual(expected_u1, result_macro_u1)

        expected_u2 = 2 / 3
        result_macro_u2 = float(result.query('user_id == "u2"')[str(metric)])
        self.assertAlmostEqual(expected_u2, result_macro_u2)

        expected_macro_sys = (expected_u1 + expected_u2) / 2
        result_macro_sys = float(result.query('user_id == "sys"')[str(metric)])
        self.assertAlmostEqual(expected_macro_sys, result_macro_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 3
        # u1_total = [0, 1, 0, 0, 0, 1, 0, 0], u2_total = [1, 0, 1, 1]
        # u1@k = [0, 1, 0], u2@k = [1, 0, 1]

    def test_perform_greater_k(self):
        metric = RecallAtK(k=999, relevant_threshold=3)

        result = metric.perform(split_w_new_items)
        expected = Recall(relevant_threshold=3).perform(split_w_new_items)

        result = np.sort(result, axis=0)
        expected = np.sort(expected, axis=0)

        # If k > than number of rows, then it's a regular Precision
        self.assertTrue(np.array_equal(expected, result))

    def test_perform_mean(self):
        metric = RecallAtK(k=3)

        result = metric.perform(split_w_new_items)

        expected_u1 = 1 / 2
        result_mean_u1 = float(result.query('user_id == "u1"')[str(metric)])
        self.assertAlmostEqual(expected_u1, result_mean_u1)

        expected_u2 = 1 / 1
        result_mean_u2 = float(result.query('user_id == "u2"')[str(metric)])
        self.assertAlmostEqual(expected_u2, result_mean_u2)

        expected_mean_sys = (expected_u1 + expected_u2) / 2
        result_mean_sys = float(result.query('user_id == "sys"')[str(metric)])
        self.assertAlmostEqual(expected_mean_sys, result_mean_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 1
        # u1_total = [0, 1, 0, 0, 0, 1, 0, 0], u2_total = [0, 0, 1, 0]
        # u1@k = [0, 1, 0], u2@k = [0, 0, 1]
        # mean_u1 = 2.2, mean_u2 = 3.33

    def test_perform_i1_i4_missing(self):
        metric = RecallAtK(k=3, relevant_threshold=3)

        result_macro = metric.perform(split_missing)

        expected_u1 = 0 / 2
        result_macro_u1 = float(result_macro.query('user_id == "u1"')[str(metric)])
        self.assertAlmostEqual(expected_u1, result_macro_u1)

        expected_u2 = 2 / 3
        result_macro_u2 = float(result_macro.query('user_id == "u2"')[str(metric)])
        self.assertAlmostEqual(expected_u2, result_macro_u2)

        expected_macro_sys = (expected_u1 + expected_u2) / 2
        result_macro_sys = float(result_macro.query('user_id == "sys"')[str(metric)])
        self.assertAlmostEqual(expected_macro_sys, result_macro_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 3
        # u1_total = [0, 0, 0, 1, 0, 0], u2_total = [0, 1, 1]
        # u1@k = [0, 0, 0], u2@k = [0, 1, 1]


class TestFMeasure(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.metric_macro = FMeasure(relevant_threshold=3, sys_average='macro')
        cls.metric_micro = FMeasure(relevant_threshold=3, sys_average='micro')
        cls.metric_mean = FMeasure(sys_average='macro')

        cls.metric_f2 = FMeasure(beta=2, relevant_threshold=3)
        cls.metric_f3_2 = FMeasure(beta=3.2, relevant_threshold=3)

    @staticmethod
    def fscore(beta: float, prec: float, reca: float):
        beta_2 = beta ** 2
        num = prec * reca
        den = (beta_2 * prec) + reca

        fbeta = (1 + beta_2) * (den and num / den or 0)

        fbeta = fbeta

        return fbeta

    def test_perform_only_new(self):
        metric_macro = self.metric_macro
        metric_micro = self.metric_micro
        metric_mean = self.metric_mean

        result = metric_macro.perform(split_only_new)[str(metric_macro)]
        self.assertEqual(0, all(result))

        result = metric_micro.perform(split_only_new)[str(metric_micro)]
        self.assertEqual(0, all(result))

        result = metric_mean.perform(split_only_new)[str(metric_mean)]
        self.assertEqual(0, all(result))

        # u1 = [0, 0], u2 = [0, 0]

    def test_perform_w_new_items_macro(self):
        metric_macro = self.metric_macro

        result_macro = metric_macro.perform(split_w_new_items)

        prec_u1 = 2 / 8
        reca_u1 = 2 / 2
        expected_u1 = self.fscore(1, prec_u1, reca_u1)
        result_macro_u1 = float(result_macro.query('user_id == "u1"')[str(metric_macro)])
        self.assertAlmostEqual(expected_u1, result_macro_u1)

        prec_u2 = 3 / 4
        reca_u2 = 3 / 3
        expected_u2 = self.fscore(1, prec_u2, reca_u2)
        result_macro_u2 = float(result_macro.query('user_id == "u2"')[str(metric_macro)])
        self.assertAlmostEqual(expected_u2, result_macro_u2)

        expected_macro_sys = (expected_u1 + expected_u2) / 2
        result_macro_sys = float(result_macro.query('user_id == "sys"')[str(metric_macro)])
        self.assertAlmostEqual(expected_macro_sys, result_macro_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 3
        # u1 = [0, 1, 0, 0, 0, 1, 0, 0], u2 = [1, 0, 1, 1]

    def test_perform_w_new_items_micro(self):
        metric_micro = self.metric_micro

        result_micro = metric_micro.perform(split_w_new_items)

        prec_u1 = 2 / 8
        reca_u1 = 2 / 2
        expected_u1 = self.fscore(1, prec_u1, reca_u1)
        result_micro_u1 = float(result_micro.query('user_id == "u1"')[str(metric_micro)])
        self.assertAlmostEqual(expected_u1, result_micro_u1)

        prec_u2 = 3 / 4
        reca_u2 = 3 / 3
        expected_u2 = self.fscore(1, prec_u2, reca_u2)
        result_micro_u2 = float(result_micro.query('user_id == "u2"')[str(metric_micro)])
        self.assertAlmostEqual(expected_u2, result_micro_u2)

        prec_sys = (2 + 3) / ((2 + 3) + (6 + 1))
        reca_sys = (2 + 3) / ((2 + 3) + (0 + 0))
        expected_micro_sys = self.fscore(1, prec_sys, reca_sys)
        result_micro_sys = float(result_micro.query('user_id == "sys"')[str(metric_micro)])
        self.assertAlmostEqual(expected_micro_sys, result_micro_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 3
        # u1 = [0, 1, 0, 0, 0, 1, 0, 0], u2 = [1, 0, 1, 1]

    def test_perform_w_new_items_mean(self):
        metric_mean = self.metric_mean

        result_mean = metric_mean.perform(split_w_new_items)

        prec_u1 = 2 / 8
        reca_u1 = 2 / 2
        expected_u1 = self.fscore(1, prec_u1, reca_u1)
        result_mean_u1 = float(result_mean.query('user_id == "u1"')[str(metric_mean)])
        self.assertAlmostEqual(expected_u1, result_mean_u1)

        prec_u2 = 1 / 4
        reca_u2 = 1 / 1
        expected_u2 = self.fscore(1, prec_u2, reca_u2)
        result_mean_u2 = float(result_mean.query('user_id == "u2"')[str(metric_mean)])
        self.assertAlmostEqual(expected_u2, result_mean_u2)

        expected_mean_sys = (expected_u1 + expected_u2) / 2
        result_mean_sys = float(result_mean.query('user_id == "sys"')[str(metric_mean)])
        self.assertAlmostEqual(expected_mean_sys, result_mean_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 3
        # u1 = [0, 1, 0, 0, 0, 1, 0, 0], u2 = [1, 0, 1, 1]
        # mean_u1 = 2.2, mean_u2 = 3.33

    def test_perform_only_one_macro(self):
        metric_macro = self.metric_macro

        result_macro = metric_macro.perform(split_only_one)

        prec_u1 = 0 / 1
        reca_u1 = 0 / 2
        expected_u1 = 0
        result_macro_u1 = float(result_macro.query('user_id == "u1"')[str(metric_macro)])
        self.assertAlmostEqual(expected_u1, result_macro_u1)

        prec_u2 = 1 / 1
        reca_u2 = 1 / 3
        expected_u2 = self.fscore(1, prec_u2, reca_u2)
        result_macro_u2 = float(result_macro.query('user_id == "u2"')[str(metric_macro)])
        self.assertAlmostEqual(expected_u2, result_macro_u2)

        expected_macro_sys = (expected_u1 + expected_u2) / 2
        result_macro_sys = float(result_macro.query('user_id == "sys"')[str(metric_macro)])
        self.assertAlmostEqual(expected_macro_sys, result_macro_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 3
        # u1 = [0], u2 = [1]

    def test_perform_only_one_micro(self):
        metric_micro = self.metric_micro

        result_micro = metric_micro.perform(split_only_one)

        prec_u1 = 0 / 1
        reca_u1 = 0 / 2
        expected_u1 = 0
        result_micro_u1 = float(result_micro.query('user_id == "u1"')[str(metric_micro)])
        self.assertAlmostEqual(expected_u1, result_micro_u1)

        prec_u2 = 1 / 1
        reca_u2 = 1 / 3
        expected_u2 = self.fscore(1, prec_u2, reca_u2)
        result_micro_u2 = float(result_micro.query('user_id == "u2"')[str(metric_micro)])
        self.assertAlmostEqual(expected_u2, result_micro_u2)

        prec_sys = (0 + 1) / ((0 + 1) + (1 + 0))
        reca_sys = (0 + 1) / ((0 + 1) + (2 + 2))
        expected_micro_sys = self.fscore(1, prec_sys, reca_sys)
        result_micro_sys = float(result_micro.query('user_id == "sys"')[str(metric_micro)])
        self.assertAlmostEqual(expected_micro_sys, result_micro_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 3
        # u1 = [0], u2 = [1]

    def test_perform_only_one_mean(self):
        metric_mean = self.metric_mean

        result_mean = metric_mean.perform(split_only_one)

        prec_u1 = 0 / 1
        reca_u1 = 0 / 2
        expected_u1 = 0
        result_mean_u1 = float(result_mean.query('user_id == "u1"')[str(metric_mean)])
        self.assertAlmostEqual(expected_u1, result_mean_u1)

        prec_u2 = 0 / 1
        reca_u2 = 0 / 1
        expected_u2 = 0
        result_mean_u2 = float(result_mean.query('user_id == "u2"')[str(metric_mean)])
        self.assertAlmostEqual(expected_u2, result_mean_u2)

        expected_mean_sys = (expected_u1 + expected_u2) / 2
        result_mean_sys = float(result_mean.query('user_id == "sys"')[str(metric_mean)])
        self.assertAlmostEqual(expected_mean_sys, result_mean_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 1
        # u1 = [0], u2 = [0]
        # mean_u1 = 2.2, mean_u2 = 3.33

    def test_perform_f2(self):
        metric_macro = self.metric_f2

        result_macro = metric_macro.perform(split_w_new_items)

        prec_u1 = 2 / 8
        reca_u1 = 2 / 2
        expected_u1 = self.fscore(2, prec_u1, reca_u1)
        result_macro_u1 = float(result_macro.query('user_id == "u1"')[str(metric_macro)])
        self.assertAlmostEqual(expected_u1, result_macro_u1)

        prec_u2 = 3 / 4
        reca_u2 = 3 / 3
        expected_u2 = self.fscore(2, prec_u2, reca_u2)
        result_macro_u2 = float(result_macro.query('user_id == "u2"')[str(metric_macro)])
        self.assertAlmostEqual(expected_u2, result_macro_u2)

        expected_macro_sys = (expected_u1 + expected_u2) / 2
        result_macro_sys = float(result_macro.query('user_id == "sys"')[str(metric_macro)])
        self.assertAlmostEqual(expected_macro_sys, result_macro_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 3
        # u1 = [0, 1, 0, 0, 0, 1, 0, 0], u2 = [1, 0, 1, 1]

    def test_perform_f_real_value(self):
        metric_macro = self.metric_f3_2

        result_macro = metric_macro.perform(split_w_new_items)

        prec_u1 = 2 / 8
        reca_u1 = 2 / 2
        expected_u1 = self.fscore(3.2, prec_u1, reca_u1)
        result_macro_u1 = float(result_macro.query('user_id == "u1"')[str(metric_macro)])
        self.assertAlmostEqual(expected_u1, result_macro_u1)

        prec_u2 = 3 / 4
        reca_u2 = 3 / 3
        expected_u2 = self.fscore(3.2, prec_u2, reca_u2)
        result_macro_u2 = float(result_macro.query('user_id == "u2"')[str(metric_macro)])
        self.assertAlmostEqual(expected_u2, result_macro_u2)

        expected_macro_sys = (expected_u1 + expected_u2) / 2
        result_macro_sys = float(result_macro.query('user_id == "sys"')[str(metric_macro)])
        self.assertAlmostEqual(expected_macro_sys, result_macro_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 3
        # u1 = [0, 1, 0, 0, 0, 1, 0, 0], u2 = [1, 0, 1, 1]


# Only new tests other than those of TestFMeasure, to remove redundancy
class TestFMeasureAtK(TestCase):

    @staticmethod
    def fscore(beta: int, prec: float, reca: float):
        beta_2 = beta ** 2
        num = prec * reca
        den = (beta_2 * prec) + reca

        fbeta = (1 + beta_2) * (num / den)

        fbeta = fbeta

        return fbeta

    def test_perform(self):
        metric = FMeasureAtK(k=3, beta=1, relevant_threshold=3)

        result = metric.perform(split_w_new_items)

        prec_u1 = 1 / 3
        reca_u1 = 1 / 2
        expected_u1 = self.fscore(1, prec_u1, reca_u1)
        result_macro_u1 = float(result.query('user_id == "u1"')[str(metric)])
        self.assertAlmostEqual(expected_u1, result_macro_u1)

        prec_u2 = 2 / 3
        reca_u2 = 2 / 3
        expected_u2 = self.fscore(1, prec_u2, reca_u2)
        result_macro_u2 = float(result.query('user_id == "u2"')[str(metric)])
        self.assertAlmostEqual(expected_u2, result_macro_u2)

        expected_macro_sys = (expected_u1 + expected_u2) / 2
        result_macro_sys = float(result.query('user_id == "sys"')[str(metric)])
        self.assertAlmostEqual(expected_macro_sys, result_macro_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 3
        # u1_total = [0, 1, 0, 0, 0, 1, 0, 0], u2_total = [1, 0, 1, 1]
        # u1@k = [0, 1, 0], u2@k = [1, 0, 1]

    def test_perform_greater_k(self):
        metric = FMeasureAtK(k=999, relevant_threshold=3)

        result = metric.perform(split_w_new_items)
        expected = FMeasure(relevant_threshold=3).perform(split_w_new_items)

        result = np.sort(result, axis=0)
        expected = np.sort(expected, axis=0)

        # If k > than number of rows, then it's a regular FMeasure
        self.assertTrue(np.array_equal(expected, result))

    def test_perform_mean(self):
        metric = FMeasureAtK(k=3)

        result = metric.perform(split_w_new_items)

        prec_u1 = 1 / 3
        reca_u1 = 1 / 2
        expected_u1 = self.fscore(1, prec_u1, reca_u1)
        result_mean_u1 = float(result.query('user_id == "u1"')[str(metric)])
        self.assertAlmostEqual(expected_u1, result_mean_u1)

        prec_u2 = 1 / 3
        reca_u2 = 1
        expected_u2 = self.fscore(1, prec_u2, reca_u2)
        result_mean_u2 = float(result.query('user_id == "u2"')[str(metric)])
        self.assertAlmostEqual(expected_u2, result_mean_u2)

        expected_mean_sys = (expected_u1 + expected_u2) / 2
        result_mean_sys = float(result.query('user_id == "sys"')[str(metric)])
        self.assertAlmostEqual(expected_mean_sys, result_mean_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 1
        # u1_total = [0, 1, 0, 0, 0, 1, 0, 0], u2_total = [0, 0, 1, 0]
        # u1@k = [0, 1, 0], u2@k = [0, 0, 1]
        # mean_u1 = 2.2, mean_u2 = 3.33

    def test_perform_i1_i4_missing(self):
        metric = FMeasureAtK(k=3, relevant_threshold=3)

        result_macro = metric.perform(split_missing)

        prec_u1 = 0 / 3
        reca_u1 = 0 / 2
        expected_u1 = 0
        result_macro_u1 = float(result_macro.query('user_id == "u1"')[str(metric)])
        self.assertAlmostEqual(expected_u1, result_macro_u1)

        prec_u2 = 2 / 3
        reca_u2 = 2 / 3
        expected_u2 = self.fscore(1, prec_u2, reca_u2)
        result_macro_u2 = float(result_macro.query('user_id == "u2"')[str(metric)])
        self.assertAlmostEqual(expected_u2, result_macro_u2)

        expected_macro_sys = (expected_u1 + expected_u2) / 2
        result_macro_sys = float(result_macro.query('user_id == "sys"')[str(metric)])
        self.assertAlmostEqual(expected_macro_sys, result_macro_sys)

        # n_real_relevant_u1 = 2, n_real_relevant_u2 = 3
        # u1_total = [0, 0, 0, 1, 0, 0], u2_total = [0, 1, 1]
        # u1@k = [0, 0, 0], u2@k = [0, 1, 1]


if __name__ == "__main__":
    unittest.main()
