import os
import unittest
from unittest import TestCase

from orange_cb_recsys.evaluation.metrics.error_metrics import MAE, MSE, RMSE
from orange_cb_recsys.evaluation.metrics.fairness_metrics import CatalogCoverage, GiniIndex, DeltaGap
from orange_cb_recsys.evaluation.metrics.metrics import RankingNeededMetric, ScoresNeededMetric
from orange_cb_recsys.evaluation.metrics.ranking_metrics import NDCG, MRR, MRRAtK, NDCGAtK, Correlation
from orange_cb_recsys.recsys.content_based_algorithm.regressor.linear_predictor import LinearPredictor
from orange_cb_recsys.recsys.content_based_algorithm.regressor.regressors import SkLinearRegression
from orange_cb_recsys.utils.load_ratings import load_ratings

from orange_cb_recsys.evaluation.eval_pipeline_modules.methodology import TestItemsMethodology, AllItemsMethodology, \
    TrainingItemsMethodology
from orange_cb_recsys.recsys.graphs.nx_full_graphs import NXFullGraph

from orange_cb_recsys.evaluation.metrics.classification_metrics import Precision, Recall, RPrecision, PrecisionAtK, \
    RecallAtK, FMeasure, FMeasureAtK
from orange_cb_recsys.evaluation.eval_model import EvalModel
from orange_cb_recsys.evaluation.partitioning_techniques.partitioning import KFoldPartitioning
from orange_cb_recsys.recsys.content_based_algorithm.centroid_vector.centroid_vector import CentroidVector
from orange_cb_recsys.recsys.content_based_algorithm.centroid_vector.similarities import CosineSimilarity

import pandas as pd

from orange_cb_recsys.recsys.graph_based_algorithm.page_rank.nx_page_rank import NXPageRank
from orange_cb_recsys.recsys.recsys import GraphBasedRS, ContentBasedRS
from orange_cb_recsys.utils.const import root_path

contents_path = os.path.join(root_path, 'contents')
items_dir = os.path.join(contents_path, 'movies_codified/')

ratings_filename = os.path.join(root_path, 'datasets', 'examples', 'new_ratings.csv')
ratings = pd.read_csv(ratings_filename)
ratings.columns = ['from_id', 'to_id', 'score', 'timestamp']
ratings = ratings.head(1000)


class TestEvalModel(TestCase):

    def test_fit_cb_w_testrating_methodology(self):
        rs = ContentBasedRS(
            CentroidVector(
                {"Plot": "tfidf"},
                CosineSimilarity(),
            ),
            ratings,
            items_dir
        )

        em = EvalModel(rs,
                       KFoldPartitioning(),
                       metric_list=[Precision()])

        sys_result, users_result = em.fit()

        self.assertIsInstance(sys_result, pd.DataFrame)
        self.assertIsInstance(users_result, pd.DataFrame)

    def test_fit_cb_w_testitems_methodology(self):
        rs = ContentBasedRS(
            CentroidVector(
                {"Plot": "tfidf"},
                CosineSimilarity(),
            ),
            ratings,
            items_dir
        )

        em = EvalModel(rs,
                       KFoldPartitioning(),
                       metric_list=[Precision()],
                       methodology=TestItemsMethodology())

        sys_result, users_result = em.fit()

        self.assertIsInstance(sys_result, pd.DataFrame)
        self.assertIsInstance(users_result, pd.DataFrame)

    def test_fit_cb_w_trainingitems_methodology(self):
        rs = ContentBasedRS(
            CentroidVector(
                {"Plot": "tfidf"},
                CosineSimilarity(),
            ),
            ratings,
            items_dir
        )

        em = EvalModel(rs,
                       KFoldPartitioning(),
                       metric_list=[Precision()],
                       methodology=TrainingItemsMethodology())

        sys_result, users_result = em.fit()

        self.assertIsInstance(sys_result, pd.DataFrame)
        self.assertIsInstance(users_result, pd.DataFrame)

    def test_fit_cb_w_allitems_methodology(self):
        rs = ContentBasedRS(
            CentroidVector(
                {"Plot": "tfidf"},
                CosineSimilarity(),
            ),
            ratings,
            items_dir
        )

        items = set([os.path.splitext(f)[0] for f in os.listdir(items_dir)
                     if os.path.isfile(os.path.join(items_dir, f)) and f.endswith('xz')])

        em = EvalModel(rs,
                       KFoldPartitioning(),
                       metric_list=[Precision()],
                       methodology=AllItemsMethodology(items))

        sys_result, users_result = em.fit()

        self.assertIsInstance(sys_result, pd.DataFrame)
        self.assertIsInstance(users_result, pd.DataFrame)

    def test_fit_graph_w_testrating_methodology(self):
        graph = NXFullGraph(ratings)

        rs = GraphBasedRS(
            NXPageRank(),
            graph
        )

        em = EvalModel(rs,
                       KFoldPartitioning(),
                       metric_list=[Precision()])

        sys_result, users_result = em.fit()

        self.assertIsInstance(sys_result, pd.DataFrame)
        self.assertIsInstance(users_result, pd.DataFrame)

    def test_fit_graph_w_testitems_methodology(self):
        graph = NXFullGraph(ratings)

        rs = GraphBasedRS(
            NXPageRank(),
            graph
        )

        em = EvalModel(rs,
                       KFoldPartitioning(),
                       metric_list=[Precision()],
                       methodology=TestItemsMethodology())

        sys_result, users_result = em.fit()

        self.assertIsInstance(sys_result, pd.DataFrame)
        self.assertIsInstance(users_result, pd.DataFrame)

    def test_fit_graph_w_trainitems_methodology(self):
        graph = NXFullGraph(ratings)

        rs = GraphBasedRS(
            NXPageRank(),
            graph
        )

        em = EvalModel(rs,
                       KFoldPartitioning(),
                       metric_list=[Precision()],
                       methodology=TrainingItemsMethodology())

        sys_result, users_result = em.fit()

        self.assertIsInstance(sys_result, pd.DataFrame)
        self.assertIsInstance(users_result, pd.DataFrame)

    def test_fit_graph_w_allitems_methodology(self):
        graph = NXFullGraph(ratings)

        rs = GraphBasedRS(
            NXPageRank(),
            graph
        )

        items = set([os.path.splitext(f)[0] for f in os.listdir(items_dir)
                     if os.path.isfile(os.path.join(items_dir, f)) and f.endswith('xz')])

        em = EvalModel(rs,
                       KFoldPartitioning(),
                       metric_list=[Precision()],
                       methodology=AllItemsMethodology(items))

        sys_result, users_result = em.fit()

        self.assertIsInstance(sys_result, pd.DataFrame)
        self.assertIsInstance(users_result, pd.DataFrame)

    def doCleanups(self) -> None:
        RankingNeededMetric._clean_pred_truth_list()
        ScoresNeededMetric._clean_pred_truth_list()


@unittest.skip("Slow")
class TestEvalModelManyRatings(TestCase):
    def test_all(self):
        ratings_filename = os.path.join(contents_path, '..', 'datasets', 'examples', 'new_ratings.csv')

        ratings_frame = load_ratings(ratings_filename)

        ratings_frame.columns = ['from_id', 'to_id', 'score', 'timestamp']

        ratings_frame["score"] = pd.to_numeric(ratings_frame["score"])

        rs = ContentBasedRS(
            LinearPredictor(
                {"Plot": ['tfidf', 'embedding']},
                SkLinearRegression(),
            ),
            ratings_frame,
            items_dir
        )

        catalog = set([os.path.splitext(f)[0] for f in os.listdir(items_dir)
                       if os.path.isfile(os.path.join(items_dir, f)) and f.endswith('xz')])

        em = EvalModel(rs,
                       KFoldPartitioning(),
                       metric_list=[
                           Precision(sys_average='micro'),
                           PrecisionAtK(1, sys_average='micro'),
                           RPrecision(),
                           Recall(),
                           RecallAtK(3, ),
                           FMeasure(1, sys_average='macro'),
                           FMeasureAtK(2, beta=1, sys_average='micro'),

                           NDCG(),
                           NDCGAtK(3),
                           MRR(),
                           MRRAtK(5, ),
                           Correlation('pearson', top_n=5),
                           Correlation('kendall', top_n=3),
                           Correlation('spearman', top_n=4),

                           MAE(),
                           MSE(),
                           RMSE(),

                           CatalogCoverage(catalog),
                           CatalogCoverage(catalog, k=2),
                           CatalogCoverage(catalog, top_n=3),
                           GiniIndex(),
                           GiniIndex(top_n=3),
                           DeltaGap({'primo': 0.5, 'secondo': 0.5})
                       ],
                       methodology=TestItemsMethodology()
                       )

        result = em.fit()
