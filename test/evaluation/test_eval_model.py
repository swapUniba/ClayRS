import os
import unittest
from unittest import TestCase

from orange_cb_recsys.content_analyzer.ratings_manager.ratings import Ratings
from orange_cb_recsys.evaluation.metrics.error_metrics import MAE
from orange_cb_recsys.evaluation.metrics.fairness_metrics import CatalogCoverage, DeltaGap
from orange_cb_recsys.evaluation.metrics.plot_metrics import LongTailDistr
from orange_cb_recsys.evaluation.metrics.ranking_metrics import NDCG

from orange_cb_recsys.evaluation.metrics.classification_metrics import Precision
from orange_cb_recsys.evaluation.eval_model import EvalModel

import pandas as pd

from test import dir_test_files

rank_split_1 = pd.read_csv(os.path.join(dir_test_files, 'test_eval', 'rank_split_1.csv'))
rank_split_2 = pd.read_csv(os.path.join(dir_test_files, 'test_eval', 'rank_split_2.csv'))
truth_split_1 = pd.read_csv(os.path.join(dir_test_files, 'test_eval', 'truth_split_1.csv'))
truth_split_2 = pd.read_csv(os.path.join(dir_test_files, 'test_eval', 'truth_split_2.csv'))

pred_list = [Ratings.from_dataframe(rank_split_1), Ratings.from_dataframe(rank_split_2)]
truth_list = [Ratings.from_dataframe(truth_split_1), Ratings.from_dataframe(truth_split_2)]


class TestEvalModel(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        catalog = set(truth_split_1['to_id']).union(set(truth_split_2['to_id']))

        cls.metric_list = [
            # one classification metric
            Precision(sys_average='micro'),

            # one ranking metric
            NDCG(),

            # one error metric
            MAE(),

            # two fairness metrics
            CatalogCoverage(catalog),
            DeltaGap({'first': 0.5, 'second': 0.5}),

            # one metric which returns a plot
            LongTailDistr()
        ]

    def test_fit(self):

        em = EvalModel(pred_list, truth_list, self.metric_list)
        sys_result, user_results = em.fit()

        self.assertIsInstance(sys_result, pd.DataFrame)
        self.assertIsInstance(user_results, pd.DataFrame)

        # we take as reference one split only since same users must be present in both split
        self.assertEqual(set(rank_split_1['from_id'].map(str)), set(user_results.index.map(str)))

        # the user result frame must contain results for each user of the Precision, NDCG, MAE (the first 3 of the
        # metric list). The other metrics do not compute result for each metric so they will not be present as columns
        # in the user_results frame
        self.assertEqual(list(user_results.columns), ['Precision - micro', 'NDCG', 'MAE'])

        # the sys_result frame must contain result of the system for each fold (2 in this case) + the mean result
        self.assertTrue(len(sys_result) == 3)
        self.assertEqual({'sys - fold1', 'sys - fold2', 'sys - mean'}, set(sys_result.index))

        # the sys result frame must contain results for the whole sys of the Precision, NDCG, MAE,
        # Catalog Coverage and Delta Gap
        self.assertEqual(list(sys_result.columns), ['Precision - micro', 'NDCG', 'MAE',
                                                    'CatalogCoverage (PredictionCov)', 'DeltaGap | first',
                                                    'DeltaGap | second'])

    def test_fit_user_list(self):

        # we compute evaluation only for a certain users
        em = EvalModel(pred_list, truth_list, self.metric_list)
        sys_result, user_results = em.fit(['1', '2', '3'])

        self.assertIsInstance(sys_result, pd.DataFrame)
        self.assertIsInstance(user_results, pd.DataFrame)

        self.assertTrue(len(user_results) == len(set(rank_split_1)))
        self.assertEqual({'1', '2', '3'}, set(user_results.index))

        # the user result frame must contain results for each user of the Precision, NDCG, MAE (the first 3 of the
        # metric list). The other metrics do not compute result for each metric so they will not be present as columns
        # in the user_results frame
        self.assertEqual(list(user_results.columns), ['Precision - micro', 'NDCG', 'MAE'])

        # the sys_result frame must contain result of the system for each fold (2 in this case) + the mean result
        self.assertTrue(len(sys_result) == 3)
        self.assertEqual({'sys - fold1', 'sys - fold2', 'sys - mean'}, set(sys_result.index))

        # the sys result frame must contain results for the whole sys of the Precision, NDCG, MAE,
        # Catalog Coverage and Delta Gap
        self.assertEqual(list(sys_result.columns), ['Precision - micro', 'NDCG', 'MAE',
                                                    'CatalogCoverage (PredictionCov)', 'DeltaGap | first',
                                                    'DeltaGap | second'])

    def test_fit_error(self):
        # should raise error since pred_list and truth_list must be of equal length
        with self.assertRaises(ValueError):
            pred_list_smaller = [Ratings.from_dataframe(pd.DataFrame())]
            pred_list_bigger = [Ratings.from_dataframe(pd.DataFrame()), Ratings.from_dataframe(pd.DataFrame())]

            EvalModel(pred_list_smaller, pred_list_bigger, self.metric_list)


if __name__ == '__main__':
    unittest.main()
