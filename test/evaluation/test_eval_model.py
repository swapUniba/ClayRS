import os
from unittest import TestCase

from orange_cb_recsys.evaluation.eval_pipeline_modules.methodology import TestItemsMethodology, AllItemsMethodology, \
    TrainingItemsMethodology
from orange_cb_recsys.recsys.graphs.nx_full_graphs import NXFullGraph

from orange_cb_recsys.evaluation.metrics.classification_metrics import Precision, Recall, RPrecision
from orange_cb_recsys.evaluation.eval_model import EvalModel
from orange_cb_recsys.evaluation.partitioning_techniques.partitioning import KFoldPartitioning
from orange_cb_recsys.recsys.content_based_algorithm import CentroidVector, CosineSimilarity

import pandas as pd

from orange_cb_recsys.recsys.graph_based_algorithm.page_rank import NXPageRank
from orange_cb_recsys.recsys.recsys import GraphBasedRS, ContentBasedRS
from orange_cb_recsys.utils.const import root_path

contents_path = os.path.join(root_path, 'contents')
items_dir = os.path.join(contents_path, 'movies_multiple_repr')

ratings_filename = os.path.join(root_path, 'datasets', 'examples', 'new_ratings.csv')
ratings = pd.read_csv(ratings_filename)
ratings.columns = ['from_id', 'to_id', 'score', 'timestamp']
ratings = ratings.head(1000)


class TestEvalModel(TestCase):

    def test_fit_cb_w_testrating_methodology(self):
        rs = ContentBasedRS(
            CentroidVector(
                {"Plot": "0"},
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
                {"Plot": "0"},
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
                {"Plot": "0"},
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
                {"Plot": "0"},
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

# class TestEvalModelManyRatings(TestCase):
#     def test_all(self):
#         ratings_filename = os.path.join(contents_path, '..', 'datasets', 'examples', 'new_ratings.csv')
#
#         ratings_frame = load_ratings(ratings_filename)
#
#         ratings_frame.columns = ['from_id', 'to_id', 'score', 'timestamp']
#
#         ratings_frame["score"] = pd.to_numeric(ratings_frame["score"])
#
#         rs = ContentBasedRS(
#             LinearPredictor(
#                 {"Plot": "0"},
#                 SkLinearRegression(),
#             ),
#             ratings_frame,
#             items_dir
#         )
#
#         catalog = set([os.path.splitext(f)[0] for f in os.listdir(items_dir)
#                        if os.path.isfile(os.path.join(items_dir, f)) and f.endswith('xz')])
#
#         em = EvalModel(rs,
#                        KFoldPartitioning(),
#                        metric_list=[
#                            Precision(relevant_threshold=3, sys_average='micro'),
#                            PrecisionAtK(1, relevant_threshold=3, sys_average='micro'),
#                            RPrecision(relevant_threshold=3),
#                            Recall(relevant_threshold=3),
#                            RecallAtK(3, relevant_threshold=3),
#                            FMeasure(1, relevant_threshold=3, sys_average='macro'),
#                            FMeasureAtK(2, beta=1, relevant_threshold=3, sys_average='micro'),
#
#                            NDCG(),
#                            NDCGAtK(3),
#                            MRR(relevant_threshold=3),
#                            MRRAtK(5, relevant_threshold=3),
#                            Correlation('pearson', top_n=5),
#                            Correlation('kendall', top_n=3),
#                            Correlation('spearman', top_n=4),
#
#                            MAE(),
#                            MSE(),
#                            RMSE(),
#
#                            CatalogCoverage(catalog),
#                            CatalogCoverage(catalog, k=2),
#                            CatalogCoverage(catalog, top_n=3),
#                            GiniIndex(),
#                            GiniIndex(top_n=3)
#                        ],
#                        )
#
#         result = em.fit()
#
#         print("we")
