from unittest import TestCase
import pandas as pd
import os

from orange_cb_recsys.evaluation.metrics.error_metrics import MAE
from orange_cb_recsys.evaluation.metrics.ranking_metrics import NDCG
from orange_cb_recsys.recsys.content_based_algorithm.classifier.classifiers import SkKNN
from orange_cb_recsys.recsys.content_based_algorithm.regressor.linear_predictor import LinearPredictor
from orange_cb_recsys.recsys.content_based_algorithm.regressor.regressors import SkLinearRegression
from orange_cb_recsys.recsys.graphs.nx_full_graphs import NXFullGraph
from orange_cb_recsys.evaluation.metrics.metrics import RankingNeededMetric, ScoresNeededMetric

from orange_cb_recsys.evaluation.eval_pipeline_modules.prediction_calculator import PredictionCalculator, Split
from orange_cb_recsys.recsys.content_based_algorithm import ClassifierRecommender
from orange_cb_recsys.recsys.graph_based_algorithm.page_rank.nx_page_rank import NXPageRank
from orange_cb_recsys.recsys.recsys import ContentBasedRS, GraphBasedRS
from orange_cb_recsys.utils.const import root_path

contents_path = os.path.join(root_path, 'contents')

movies_dir = os.path.join(contents_path, 'movies_codified/')


class TestPredictionCalculator(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.ratings_original = pd.DataFrame.from_records([
            ("u1", "tt0114576", 5, "54654675"),
            ("u1", "tt0112453", 2, "54654675"),
            ("u1", "tt0113497", 5, "54654675"),
            ("u1", "tt0112896", 2, "54654675"),
            ("u2", "tt0113041", 4, "54654675"),
            ("u2", "tt0112453", 2, "54654675"),
            ("u2", "tt0113497", 4, "54654675"),
            ("u2", "tt0113189", 1, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        train1 = pd.DataFrame.from_records([
            ("u1", "tt0114576", 5, "54654675"),
            ("u1", "tt0112453", 2, "54654675"),
            ("u2", "tt0113041", 4, "54654675"),
            ("u2", "tt0112453", 2, "54654675"), ],
            columns=["from_id", "to_id", "score", "timestamp"])

        test1 = pd.DataFrame.from_records([
            ("u1", "tt0113497", 5, "54654675"),
            ("u1", "tt0112896", 2, "54654675"),
            ("u2", "tt0113497", 4, "54654675"),
            ("u2", "tt0113189", 1, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        train2 = pd.DataFrame.from_records([
            ("u1", "tt0113497", 5, "54654675"),
            ("u1", "tt0112896", 2, "54654675"),
            ("u2", "tt0113497", 4, "54654675"),
            ("u2", "tt0113189", 1, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        test2 = pd.DataFrame.from_records([
            ("u1", "tt0114576", 5, "54654675"),
            ("u1", "tt0112453", 2, "54654675"),
            ("u2", "tt0113041", 4, "54654675"),
            ("u2", "tt0112453", 2, "54654675"), ],
            columns=["from_id", "to_id", "score", "timestamp"])

        cls.split_list = [Split(train1, test1), Split(train2, test2)]

        cls.test_items_list = [test1[['from_id', 'to_id']], test2[['from_id', 'to_id']]]

    def test_calc_rank_content_based(self):

        recsys = ContentBasedRS(
            ClassifierRecommender(
                {'Plot': 'tfidf'},
                SkKNN(),
                threshold=3
            ),
            self.ratings_original,
            movies_dir
        )

        # We just need a Metric of the RankingNeededMetric class to test
        metric_list = [NDCG()]

        valid_metric = PredictionCalculator(self.split_list, recsys).calc_predictions(self.test_items_list, metric_list)
        rank_truth = RankingNeededMetric.rank_truth_list

        # We expect this to be empty, since there are no ScoresNeededMetric in the metric list
        score_truth = ScoresNeededMetric.score_truth_list

        self.assertEqual(valid_metric, metric_list)
        self.assertGreater(len(rank_truth), 0)
        self.assertEqual(len(score_truth), 0)

    def test_calc_scores_content_based(self):
        recsys = ContentBasedRS(
            LinearPredictor(
                {'Plot': 'tfidf'},
                SkLinearRegression()
            ),
            self.ratings_original,
            movies_dir
        )

        # We just need a Metric of the ScoresNeededMetric class to test
        metric_list = [MAE()]

        valid_metric = PredictionCalculator(self.split_list, recsys).calc_predictions(self.test_items_list, metric_list)
        score_truth = ScoresNeededMetric.score_truth_list

        # We expect this to be empty, since there are no RankingNeededMetric in the metric list
        rank_truth = RankingNeededMetric.rank_truth_list

        self.assertEqual(valid_metric, metric_list)
        self.assertGreater(len(score_truth), 0)
        self.assertEqual(len(rank_truth), 0)

    def test_calc_rank_graph_based(self):

        graph = NXFullGraph(self.ratings_original)

        recsys = GraphBasedRS(
            NXPageRank(),
            graph
        )

        # We just need a Metric of the RankingNeededMetric class to test
        metric_list = [NDCG()]

        valid_metric = PredictionCalculator(self.split_list, recsys).calc_predictions(self.test_items_list, metric_list)
        rank_truth = RankingNeededMetric.rank_truth_list

        # We expect this to be empty, since there are no ScoresNeededMetric in the metric list
        score_truth = ScoresNeededMetric.score_truth_list

        self.assertEqual(valid_metric, metric_list)
        self.assertGreater(len(rank_truth), 0)
        self.assertEqual(len(score_truth), 0)

    def test_pop_invalid_metric(self):
        recsys = ContentBasedRS(
            ClassifierRecommender(
                {'Plot': 'tfidf'},
                SkKNN(),
                threshold=3
            ),
            self.ratings_original,
            movies_dir
        )

        # Tries to calc score predictions with a pure ranking algorithm
        metric_list = [MAE()]

        valid_metric = PredictionCalculator(self.split_list, recsys).calc_predictions(self.test_items_list, metric_list)
        score_truth = ScoresNeededMetric.score_truth_list
        rank_truth = RankingNeededMetric.rank_truth_list

        # The metric is excluded from the valid ones and nothing is calculated since
        # there aren't any others
        self.assertEqual(len(valid_metric), 0)
        self.assertEqual(len(score_truth), 0)
        self.assertEqual(len(rank_truth), 0)

        # Tries to calc score predictions with a pure ranking algorithm but there are also
        # other type of metrics
        metric_ranking = NDCG()
        metric_score = MAE()
        metric_list = [metric_score, metric_ranking]

        valid_metric = PredictionCalculator(self.split_list, recsys).calc_predictions(self.test_items_list, metric_list)
        score_truth = ScoresNeededMetric.score_truth_list
        rank_truth = RankingNeededMetric.rank_truth_list

        # The metric MAE is excluded from the valid ones but NDCG is valid so predictions
        # for that metric (RankingNeededMetric) are calculated
        self.assertIn(metric_ranking, valid_metric)
        self.assertNotIn(metric_score, valid_metric)

        self.assertEqual(len(score_truth), 0)
        self.assertGreater(len(rank_truth), 0)

    def doCleanups(self) -> None:
        # We need to clean these class attributes otherwise the tests would overlap and fail
        RankingNeededMetric._clean_pred_truth_list()
        ScoresNeededMetric._clean_pred_truth_list()
