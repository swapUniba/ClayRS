from unittest import TestCase
import pandas as pd
import numpy as np

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
        """
        item_id_list = [
            'tt0112281',
            'tt0112302',
            'tt0112346',
            'tt0112453',
            'tt0112641',
            'tt0112760',
            'tt0112896',
            'tt0113041',
            'tt0113101',
            'tt0113189',
            'tt0113228',
            'tt0113277',
            'tt0113497',
            'tt0113845',
            'tt0113987',
            'tt0114319',
            'tt0114388',
            'tt0114576',
            'tt0114709',
            'tt0114885',
        ]

        record_list = []
        for i in range(1, 7):
            extract_items = set([x for i, x in enumerate(item_id_list) if np.random.randint(0, 2) == 1 and i < 10])
            for item in extract_items:
                record_list.append((str(i), item, str(np.random.randint(-0, 11) / 10)))

        t_ratings = pd.DataFrame.from_records(record_list, columns=['from_id', 'to_id', 'score'])
        """
        ratings_filename = 'datasets/examples/new_ratings.csv'
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
                field_representation='1',
                similarity=CosineSimilarity()
            ),
            rating_frame=t_ratings
        )
        try:
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
                                ]).fit()
        except TypeError:
            pass
        except ValueError:
            pass
