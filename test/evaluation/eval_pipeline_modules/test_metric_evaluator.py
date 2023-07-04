import os
from unittest import TestCase
import pandas as pd

from clayrs.content_analyzer.ratings_manager.ratings import Rank, Ratings
from clayrs.evaluation import MAP
from clayrs.evaluation.eval_pipeline_modules.metric_evaluator import Split
from clayrs.evaluation.metrics.classification_metrics import Precision, Recall
from clayrs.evaluation.eval_model import MetricEvaluator
from clayrs.evaluation.metrics.plot_metrics import LongTailDistr, PopRecsCorrelation


# Every Metric is tested singularly, so we just check that everything goes smoothly at the
# MetricEvaluator level
class TestMetricEvaluator(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        original_ratings = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1',
                         'u2', 'u2', 'u2', 'u2',
                         'u3', 'u3', 'u3', 'u3', 'u3', 'u3',
                         'u4', 'u4', 'u4', 'u4', 'u4', 'u4', 'u4',
                         'u5', 'u5', 'u5', 'u5', 'u5',
                         'u6', 'u6'],
             'item_id': ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8',
                         'i1', 'i9', 'i10', 'i11',
                         'i1', 'i12', 'i13', 'i3', 'i10', 'i14',
                         'i3', 'i10', 'i15', 'i16', 'i9', 'i17', 'i99',
                         'i10', 'i18', 'i19', 'i20', 'i21',
                         'inew_1', 'inew_2'],
             'score': [5, 4, 4, 1, 2, 3, 3, 1,
                       4, 5, 1, 1,
                       3, 3, 2, 1, 1, 4,
                       4, 4, 5, 5, 3, 3, 3,
                       3, 3, 2, 2, 1,
                       4, 3]})
        original_ratings = Ratings.from_dataframe(original_ratings)

        train = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1',  # removed last 2
                         'u2', 'u2', 'u2',  # removed last 1
                         'u3', 'u3', 'u3', 'u3',  # removed last 2
                         'u4', 'u4', 'u4', 'u4', 'u4',  # removed last 2
                         'u5', 'u5', 'u5', 'u5',  # removed last 1
                         'u6'],  # removed last 1
             'item_id': ['i1', 'i2', 'i3', 'i4', 'i5', 'i6',
                         'i1', 'i9', 'i10',
                         'i1', 'i12', 'i13', 'i3',
                         'i3', 'i10', 'i15', 'i16', 'i9',
                         'i10', 'i18', 'i19', 'i20',
                         'inew_1'],
             'score': [5, 4, 4, 1, 2, 3,
                       4, 5, 1,
                       3, 3, 2, 1,
                       4, 4, 5, 5, 3,
                       3, 3, 2, 2,
                       4]})
        cls.train = Ratings.from_dataframe(train,
                                           user_map=original_ratings.user_map,
                                           item_map=original_ratings.item_map)

        truth = pd.DataFrame(
            {'user_id': ['u1', 'u1',
                         'u2',
                         'u3', 'u3',
                         'u4', 'u4', 'u4',
                         'u5',
                         'u6'],
             'item_id': ['i7', 'i8',
                         'i11',
                         'i10', 'i14',
                         'i9', 'i17', 'i99',
                         'i21',
                         'inew_2'],
             'score': [3, 1,
                       1,
                       1, 4,
                       3, 3, 3,
                       1,
                       3]})
        truth = Ratings.from_dataframe(truth,
                                       user_map=original_ratings.user_map,
                                       item_map=original_ratings.item_map)

        # u6 is missing, just to test DeltaGap in case for some users recs can't be computed
        recs = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1',
                         'u2', 'u2', 'u2', 'u2', 'u2',
                         'u3', 'u3', 'u3', 'u3', 'u3',
                         'u4', 'u4', 'u4', 'u4', 'u4',
                         'u5', 'u5', 'u5', 'u5', 'u5'],
             'item_id': ['i7', 'i10', 'i11', 'i12', 'i13',
                         'i11', 'i20', 'i6', 'i3', 'i4',
                         'i4', 'i5', 'i6', 'i7', 'i10',
                         'i9', 'i2', 'i3', 'i1', 'i5',
                         'i2', 'i3', 'i4', 'i5', 'i6'],
             'score': [500, 400, 300, 200, 100,
                       400, 300, 200, 100, 50,
                       150, 125, 110, 100, 80,
                       390, 380, 360, 320, 200,
                       250, 150, 190, 100, 50]})
        recs = Rank.from_dataframe(recs,
                                   user_map=original_ratings.user_map,
                                   item_map=original_ratings.item_map)

        cls.original_ratings = original_ratings
        cls.rank_pred_list = [recs]
        cls.truth_list = [truth]

    def test_eval_metrics_empty_dfs(self):
        # test eval_metrics with metrics which returns empty dataframe
        metric_list = [PopRecsCorrelation(self.original_ratings), LongTailDistr()]

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
        rank_wo_u3.item_map.append(['inew1', 'inew2', 'inew3'])

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
        truth.item_map.append(['imissing3', 'imissing4'])

        rank_list = [rank_wo_u3]
        truth_list = [truth]

        sys_result, users_results = MetricEvaluator(rank_list, truth_list).eval_metrics([Precision(), Recall(), MAP()])

        # check that u3 isn't present in results since we don't have any prediction for it
        self.assertEqual({'u1', 'u2'}, set(users_results.index))

        # the user result frame must contain results for each user of the Precision, Recall and AP
        self.assertEqual(list(users_results.columns), ['Precision - macro', 'Recall - macro', 'AP'])

        # the sys_result frame must contain result of the system for each fold (1 in this case) + the mean result
        self.assertTrue(len(sys_result) == 2)
        self.assertEqual({'sys - fold1', 'sys - mean'}, set(sys_result.index))

        # the sys result frame must contain results for the system of the Precision, Recall and MAP
        self.assertEqual(list(sys_result.columns), ['Precision - macro', 'Recall - macro', 'MAP'])

    def test_eval_metrics_different_users_same_split(self):
        rank_wo_u3 = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1',
                        'u2', 'u2', 'u2', 'u2', 'u2', 'u2', 'u2'],
            'item_id': ['i9', 'i6', 'inew1', 'inew2', 'i2', 'i1', 'i8',
                        'i10', 'inew3', 'i2', 'i1', 'i8', 'i4', 'i9'],

            'score': [500, 450, 400, 350, 300, 200, 150,
                      400, 300, 200, 100, 50, 25, 10]
        })
        rank_wo_u3 = Rank.from_dataframe(rank_wo_u3)
        rank_wo_u3.item_map.append(['inew1', 'inew2', 'inew3'])

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
        truth.item_map.append(['imissing3', 'imissing4'])

        rank_list = [rank_wo_u3]
        truth_list = [truth]

        sys_result, users_results = MetricEvaluator(rank_list, truth_list).eval_metrics([Precision(), Recall(), MAP()])

        # check that u3 isn't present in results since we don't have any prediction for it
        self.assertEqual({'u1', 'u2'}, set(users_results.index))

        # the user result frame must contain results for each user of the Precision, Recall and AP
        self.assertEqual(list(users_results.columns), ['Precision - macro', 'Recall - macro', 'AP'])

        # the sys_result frame must contain result of the system for each fold (1 in this case) + the mean result
        self.assertTrue(len(sys_result) == 2)
        self.assertEqual({'sys - fold1', 'sys - mean'}, set(sys_result.index))

        # the sys result frame must contain results for the system of the Precision, Recall and MAP
        self.assertEqual(list(sys_result.columns), ['Precision - macro', 'Recall - macro', 'MAP'])

    def test_eval_metrics_different_users_different_split(self):
        rank1 = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1',
                        'u2', 'u2', 'u2', 'u2', 'u2', 'u2', 'u2'],
            'item_id': ['i9', 'i6', 'inew1', 'inew2', 'i2', 'i1', 'i8',
                        'i10', 'inew3', 'i2', 'i1', 'i8', 'i4', 'i9'],

            'score': [500, 450, 400, 350, 300, 200, 150,
                      400, 300, 200, 100, 50, 25, 10]
        })
        rank1 = Rank.from_dataframe(rank1)
        rank1.item_map.append(['inew1', 'inew2', 'inew3'])

        truth1 = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u1', 'u1',
                        'u2', 'u2', 'u2', 'u2', 'u2'],
            'item_id': ['i1', 'i2', 'i6', 'i8', 'i9',
                        'i1', 'i2', 'i4', 'i9', 'i10'],

            'score': [3, 3, 4, 1, 1,
                      5, 3, 3, 4, 4]
        })
        truth1 = Ratings.from_dataframe(truth1)

        rank2 = pd.DataFrame({
            'user_id': ['u2', 'u2', 'u2', 'u2', 'u2', 'u2', 'u2',
                        'u3', 'u3', 'u3', 'u3', 'u3', 'u3', 'u3'],
            'item_id': ['i9', 'i6', 'inew1', 'inew2', 'i2', 'i1', 'i8',
                        'i10', 'inew3', 'i2', 'i1', 'i8', 'i4', 'i9'],

            'score': [500, 450, 400, 350, 300, 200, 150,
                      400, 300, 200, 100, 50, 25, 10]
        })
        rank2 = Rank.from_dataframe(rank2)
        rank2.item_map.append(['inew1', 'inew2', 'inew3'])

        truth2 = pd.DataFrame({
            'user_id': ['u2', 'u2', 'u2', 'u2', 'u2',
                        'u3', 'u3', 'u3', 'u3', 'u3'],
            'item_id': ['i1', 'i2', 'i6', 'i8', 'i9',
                        'i1', 'i2', 'i4', 'i9', 'i10'],

            'score': [3, 3, 4, 1, 1,
                      5, 3, 3, 4, 4]
        })
        truth2 = Ratings.from_dataframe(truth2)

        rank_list = [rank1, rank2]
        truth_list = [truth1, truth2]

        sys_result, users_results = MetricEvaluator(rank_list, truth_list).eval_metrics([Precision(), Recall(), MAP()])
        
        # users in both frames are present
        self.assertEqual({'u1', 'u2', 'u3'}, set(users_results.index))
        self.assertEqual(list(users_results.columns), ['Precision - macro', 'Recall - macro', 'AP'])

        # u2 is the only one that appears in both splits: the final result will contain the mean result for u2 across
        # the two splits
        u2_precision_split1 = float(Precision().perform(Split(rank1, truth1)).query("user_id=='u2'")["Precision - macro"])
        u2_precision_split2 = float(Precision().perform(Split(rank2, truth2)).query("user_id=='u2'")["Precision - macro"])
        u2_precision_expected = (u2_precision_split1 + u2_precision_split2) / 2

        self.assertAlmostEqual(u2_precision_expected, float(users_results.query("user_id=='u2'")["Precision - macro"]))

        u2_recall_split1 = float(Recall().perform(Split(rank1, truth1)).query("user_id=='u2'")["Recall - macro"].values[0])
        u2_recall_split2 = float(Recall().perform(Split(rank2, truth2)).query("user_id=='u2'")["Recall - macro"].values[0])
        u2_recall_expected = (u2_recall_split1 + u2_recall_split2) / 2

        self.assertAlmostEqual(u2_recall_expected, float(users_results.query("user_id=='u2'")["Recall - macro"]))

        u2_ap_split1 = float(MAP().perform(Split(rank1, truth1)).query("user_id=='u2'")["AP"].values[0])
        u2_ap_split2 = float(MAP().perform(Split(rank2, truth2)).query("user_id=='u2'")["AP"].values[0])
        u2_ap_expected = (u2_ap_split1 + u2_ap_split2) / 2

        self.assertAlmostEqual(u2_ap_expected, float(users_results.query("user_id=='u2'")["AP"]))

        # the sys_result frame must contain result of the system for each fold (2 in this case) + the mean result
        self.assertTrue(len(sys_result) == 3)
        self.assertEqual({'sys - fold1', 'sys - fold2', 'sys - mean'}, set(sys_result.index))

        # the sys result frame must contain results for the system of the Precision, Recall and MAP
        self.assertEqual(list(sys_result.columns), ['Precision - macro', 'Recall - macro', 'MAP'])

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove('long_tail_distr_truth.png')
        os.remove('pop_recs_correlation.png')
        os.remove('pop_recs_correlation_no_zeros.png')
