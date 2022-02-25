import os
from unittest import TestCase
import pandas as pd

from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import Rank, Ratings
from orange_cb_recsys.evaluation.metrics.classification_metrics import Precision, Recall

from orange_cb_recsys.evaluation.eval_model import MetricEvaluator
from orange_cb_recsys.evaluation.metrics.plot_metrics import LongTailDistr, PopRecsCorrelation


# Every Metric is tested singularly, so we just check that everything goes smoothly at the
# MetricEvaluator level
class TestMetricCalculator(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        rank1 = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1',
                        'u2', 'u2', 'u2', 'u2', 'u2', 'u2', 'u2',
                        'u3', 'u3', 'u3'],
            'item_id': ['i9', 'i6', 'inew1', 'inew2', 'i6', 'i2', 'i1', 'i8',
                      'i10', 'inew3', 'i2', 'i1', 'i8', 'i4', 'i9',
                      'i3', 'i12', 'i2'],

            'score': [500, 450, 400, 350, 300, 250, 200, 150,
                      400, 300, 200, 100, 50, 25, 10,
                      100, 50, 20]
        })
        rank1 = Rank.from_dataframe(rank1)

        truth1 = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u1', 'u1',
                        'u2', 'u2', 'u2', 'u2', 'u2',
                        'u3', 'u3', 'u3', 'u3', 'u3'],
            'item_id': ['i1', 'i2', 'i6', 'i8', 'i9',
                      'i1', 'i2', 'i4', 'i9', 'i10',
                      'i2', 'i3', 'i12', 'imissing3', 'imissing4'],

            'score': [3, 3, 4, 1, 1,
                      5, 3, 3, 4, 4,
                      4, 2, 3, 3, 3]
        })
        truth1 = Ratings.from_dataframe(truth1)

        rank2 = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u1', 'u1',
                        'u2', 'u2', 'u2', 'u2', 'u2',
                        'u3', 'u3', 'u3'],
            'item_id': ['i10', 'i5', 'i4', 'i3', 'i7',
                      'i70', 'i3', 'i71', 'i8', 'i11',
                      'i10', 'i1', 'i4'],

            'score': [500, 400, 300, 200, 100,
                      400, 300, 200, 100, 50,
                      150, 100, 50]
        })
        rank2 = Rank.from_dataframe(rank2)

        truth2 = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u1', 'u1',
                        'u2', 'u2', 'u2', 'u2', 'u2',
                        'u3', 'u3', 'u3', 'u3', 'u3'],
            'item_id': ['i3', 'i4', 'i5', 'i7', 'i10',
                      'i3', 'i70', 'i71', 'i8', 'i11',
                      'i4', 'i1', 'i10', 'imissing1', 'imissing2'],

            'score': [4, 2, 2, 5, 1,
                      5, 4, 4, 3, 4,
                      2, 3, 1, 1, 1]
        })
        truth2 = Ratings.from_dataframe(truth2)

        cls.rank_pred_list = [rank1, rank2]
        cls.truth_list = [truth1, truth2]

    def test_eval_metrics_empty_dfs(self):
        # test eval_metrics with metrics which returns empty dataframe
        metric_list = [PopRecsCorrelation(), LongTailDistr()]

        sys_result, users_results = MetricEvaluator(self.rank_pred_list, self.truth_list).eval_metrics(metric_list)

        self.assertTrue(len(sys_result) == 0)
        self.assertTrue(len(users_results) == 0)

    def test_eval_metrics_users_missing_truth(self):

        rank_wo_u3 = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1',
                        'u2', 'u2', 'u2', 'u2', 'u2', 'u2', 'u2'],
            'item_id': ['i9', 'i6', 'inew1', 'inew2', 'i2', 'i1', 'i8',
                      'i10', 'inew3', 'i2', 'i1', 'i8', 'i4', 'i9'],

            'score': [500, 450, 400, 350, 300, 200, 150,
                      400, 300, 200, 100, 50, 25, 10]
        })
        rank_wo_u3 = Rank.from_dataframe(rank_wo_u3)

        truth = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u1', 'u1',
                        'u2', 'u2', 'u2', 'u2', 'u2',
                        'u3', 'u3', 'u3', 'u3', 'u3'],
            'item_id': ['i1', 'i2', 'i6', 'i8', 'i9',
                      'i1', 'i2', 'i4', 'i9', 'i10',
                      'i2', 'i3', 'i12', 'imissing3', 'imissing4'],

            'score': [3, 3, 4, 1, 1,
                      5, 3, 3, 4, 4,
                      4, 2, 3, 3, 3]
        })
        truth = Ratings.from_dataframe(truth)

        rank_list = [rank_wo_u3]
        truth_list = [truth]

        sys_result, users_results = MetricEvaluator(rank_list, truth_list).eval_metrics([Precision(), Recall()])

        # check that u3 isn't present in results since we don't have any prediction for it
        self.assertEqual({'u1', 'u2'}, set(users_results.index))

        # the user result frame must contain results for each user of the Precision and Recall
        self.assertEqual(list(users_results.columns), ['Precision - macro', 'Recall - macro'])

        # the sys_result frame must contain result of the system for each fold (1 in this case) + the mean result
        self.assertTrue(len(sys_result) == 2)
        self.assertEqual({'sys - fold1', 'sys - mean'}, set(sys_result.index))

        # the sys result frame must contain results for the system of the Precision and Recall
        self.assertEqual(list(sys_result.columns), ['Precision - macro', 'Recall - macro'])

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove('long_tail_distr_truth.png')
        os.remove('long_tail_distr_truth (1).png')
        os.remove('pop_recs_correlation.png')
        os.remove('pop_recs_correlation (1).png')
        os.remove('pop_recs_correlation_no_zeros.png')
        os.remove('pop_recs_correlation_no_zeros (1).png')
