import os
import unittest
from unittest import TestCase

from clayrs.content_analyzer.ratings_manager.ratings import Ratings, Rank
from clayrs.evaluation.metrics.error_metrics import MAE
from clayrs.evaluation.metrics.fairness_metrics import CatalogCoverage, DeltaGap
from clayrs.evaluation.metrics.plot_metrics import LongTailDistr
from clayrs.evaluation.metrics.ranking_metrics import NDCG

from clayrs.evaluation.metrics.classification_metrics import Precision
from clayrs.evaluation.eval_model import EvalModel

import pandas as pd

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

train_1 = pd.DataFrame(
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
train_1 = Ratings.from_dataframe(train_1)

truth_1 = pd.DataFrame(
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
truth_1 = Ratings.from_dataframe(truth_1)

train_2 = pd.DataFrame(
    {'user_id': ['u1', 'u1',  # taken last 2
                 'u2',  # taken last 1
                 'u3', 'u3',  # taken last 2
                 'u4', 'u4', 'u4',  # taken last 2
                 'u5',  # taken last 1
                 'u6'],  # taken last 1
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
train_2 = Ratings.from_dataframe(train_2)

truth_2 = pd.DataFrame(
    {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1',
                 'u2', 'u2', 'u2',
                 'u3', 'u3', 'u3', 'u3',
                 'u4', 'u4', 'u4', 'u4', 'u4',
                 'u5', 'u5', 'u5', 'u5',
                 'u6'],
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
truth_2 = Ratings.from_dataframe(truth_2)

# u6 is missing, just to test case in which for some users recs can't be computed
recs_1 = pd.DataFrame(
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
recs_1 = Rank.from_dataframe(recs_1)

# u6 is missing, just to test case in which for some users recs can't be computed
recs_2 = pd.DataFrame(
    {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1',
                 'u2', 'u2', 'u2', 'u2', 'u2',
                 'u3', 'u3', 'u3', 'u3', 'u3',
                 'u4', 'u4', 'u4', 'u4', 'u4',
                 'u5', 'u5', 'u5', 'u5', 'u5',
                 'u6', 'u6', 'u6', 'u6', 'u6'],
     'item_id': ['i10', 'i11', 'i7', 'i12', 'i13',
                 'i20', 'i11', 'i3', 'i6', 'i4',
                 'i5', 'i4', 'i6', 'i7', 'i10',
                 'i9', 'i3', 'i1', 'i2', 'i5',
                 'i2', 'i4', 'i3', 'i5', 'i6',
                 'i10', 'i11', 'i12', 'i13', 'i14'],
     'score': [500, 400, 300, 200, 100,
               400, 300, 200, 100, 50,
               150, 125, 110, 100, 80,
               390, 380, 360, 320, 200,
               250, 150, 190, 100, 50,
               200, 100, 190, 80, 70]})
recs_2 = Rank.from_dataframe(recs_2)

pred_list = [recs_1, recs_2]
truth_list = [truth_1, truth_2]


class TestEvalModel(TestCase):

    def setUp(self) -> None:
        catalog = set(original_ratings.item_id_column)

        self.metric_list = [
            # one classification metric
            Precision(sys_average='micro'),

            # one ranking metric
            NDCG(),

            # one error metric
            MAE(),

            # two fairness metrics
            CatalogCoverage(catalog),
            DeltaGap({'a': 0.2, 'b': 0.5, 'c': 0.3},
                     user_profiles=[train_1, train_2],
                     original_ratings=original_ratings),

            # one metric which returns a plot
            LongTailDistr()
        ]

    def test_fit(self):
        em = EvalModel(pred_list, truth_list, self.metric_list)
        sys_result, user_results = em.fit()

        self.assertIsInstance(sys_result, pd.DataFrame)
        self.assertIsInstance(user_results, pd.DataFrame)

        # we should have a result for each user of the original rating frame
        self.assertEqual(set(original_ratings.user_id_column), set(user_results.index.map(str)))

        # the user result frame must contain results for each user of the Precision, NDCG, MAE (the first 3 of the
        # metric list). The other metrics do not compute result for each metric so they will not be present as columns
        # in the user_results frame
        self.assertEqual(list(user_results.columns), ['Precision - micro', 'NDCG', 'MAE'])

        # the sys_result frame must contain result of the system for each fold (2 in this case) + the mean result
        self.assertTrue(len(sys_result) == 3)
        self.assertEqual({'sys - fold1', 'sys - fold2', 'sys - mean'}, set(sys_result.index))

        # the sys result frame must contain results for the whole sys of the Precision, NDCG, MAE,
        # Catalog Coverage and DeltaGAP
        self.assertEqual(['Precision - micro', 'NDCG', 'MAE',
                          'CatalogCoverage (PredictionCov)',
                          'DeltaGap | a',
                          'DeltaGap | b',
                          'DeltaGap | c',
                          ],
                         list(sys_result.columns))

        # check that plots are generated and remove them
        self.assertTrue(os.path.isfile('long_tail_distr_truth.png'))
        os.remove('long_tail_distr_truth.png')
        self.assertTrue(os.path.isfile('long_tail_distr_truth (1).png'))
        os.remove('long_tail_distr_truth (1).png')

    def test_fit_user_list(self):
        # we compute evaluation only for a certain users

        # we must also cut user profiles of DeltaGAP by only considering u1, u2, u3
        self.metric_list.pop(4)

        filter_list = ['u1', 'u2', 'u3']
        filter_1 = train_1.user_map.convert_seq_str2int(filter_list)
        filter_2 = train_2.user_map.convert_seq_str2int(filter_list)
        original_filter = original_ratings.user_map.convert_seq_str2int(filter_list)

        self.metric_list.append(DeltaGap({'a': 0.2, 'b': 0.5, 'c': 0.3},
                                         user_profiles=[train_1.filter_ratings(filter_1),
                                                        train_2.filter_ratings(filter_2)],
                                         original_ratings=original_ratings.filter_ratings(original_filter))
                                )

        em = EvalModel(pred_list, truth_list, self.metric_list)
        sys_result, user_results = em.fit(['u1', 'u2', 'u3'])

        self.assertIsInstance(sys_result, pd.DataFrame)
        self.assertIsInstance(user_results, pd.DataFrame)

        self.assertEqual({'u1', 'u2', 'u3'}, set(user_results.index))

        # the user result frame must contain results for each user of the Precision, NDCG, MAE (the first 3 of the
        # metric list). The other metrics do not compute result for each metric so they will not be present as columns
        # in the user_results frame
        self.assertEqual(list(user_results.columns), ['Precision - micro', 'NDCG', 'MAE'])

        # the sys_result frame must contain result of the system for each fold (2 in this case) + the mean result
        self.assertTrue(len(sys_result) == 3)
        self.assertEqual({'sys - fold1', 'sys - fold2', 'sys - mean'}, set(sys_result.index))

        # the sys result frame must contain results for the whole sys of the Precision, NDCG, MAE,
        # Catalog Coverage and DeltaGAP
        self.assertEqual(['Precision - micro', 'NDCG', 'MAE',
                          'CatalogCoverage (PredictionCov)',
                          'DeltaGap | a',
                          'DeltaGap | b',
                          'DeltaGap | c',
                          ],
                         list(sys_result.columns))

        # check that plots are generated and remove them
        self.assertTrue(os.path.isfile('long_tail_distr_truth.png'))
        os.remove('long_tail_distr_truth.png')
        self.assertTrue(os.path.isfile('long_tail_distr_truth (1).png'))
        os.remove('long_tail_distr_truth (1).png')

    def test_fit_error(self):
        # should raise error since pred_list and truth_list must be of equal length
        with self.assertRaises(ValueError):
            pred_list_smaller = [Rank.from_dataframe(pd.DataFrame())]
            pred_list_bigger = [Ratings.from_dataframe(pd.DataFrame()), Ratings.from_dataframe(pd.DataFrame())]

            EvalModel(pred_list_smaller, pred_list_bigger, self.metric_list)


if __name__ == '__main__':
    unittest.main()
