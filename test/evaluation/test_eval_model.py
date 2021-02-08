from unittest import TestCase

from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import CSVFile
from orange_cb_recsys.evaluation.classification_metrics import Precision, Recall, FNMeasure, MRR
from orange_cb_recsys.evaluation.eval_model import RankingAlgEvalModel
from orange_cb_recsys.evaluation.partitioning import KFoldPartitioning
from orange_cb_recsys.evaluation.ranking_metrics import NDCG, Correlation
from orange_cb_recsys.recsys import CosineSimilarity
from orange_cb_recsys.recsys.config import RecSysConfig
from orange_cb_recsys.recsys.ranking_algorithms.centroid_vector import CentroidVector


class TestEvalModel(TestCase):
    def test_fit(self):
        ratings_filename = 'datasets/examples/new_ratings.csv'
        users_dir = 'contents/examples/ex_1/users_1600355755.1935306'
        items_dir = 'contents/examples/ex_1/movies_1600355972.49884'
        try:
            open(ratings_filename)
        except FileNotFoundError:
            ratings_filename = '../../datasets/examples/new_ratings.csv'
            users_dir = '../../contents/examples/ex_1/users_1600355755.1935306'
            items_dir = '../../contents/examples/ex_1/movies_1600355972.49884'
            
        t_ratings = RatingsImporter(
                source=CSVFile(ratings_filename),
                rating_configs=[RatingsFieldConfig(
                    field_name='points',
                    processor=NumberNormalizer(min_=1, max_=5))],
                from_field_name='user_id',
                to_field_name='item_id',
                timestamp_field_name='timestamp',
            ).import_ratings()
        print(t_ratings)

        recsys_config = RecSysConfig(
            users_directory='contents/examples/ex_1/users_1600355755.1935306',
            items_directory='contents/examples/ex_1/movies_1600355972.49884',
            score_prediction_algorithm=None,
            ranking_algorithm=CentroidVector(
                item_field='Plot',
                field_representation='0',
                similarity=CosineSimilarity()
            ),
            rating_frame=t_ratings
        )
        RankingAlgEvalModel(config=recsys_config,
                            partitioning=KFoldPartitioning(),
                            metric_list=
                            [
                                Precision(0.4),
                                Recall(0.4),
                                FNMeasure(1, 0.4),
                                MRR(0.4),
                                NDCG({0: (-1, 0), 1: (0, 1)}),
                                Correlation('pearson'),
                                Correlation('kendall'),
                                Correlation('spearman')
                            ]
                            ).fit()
