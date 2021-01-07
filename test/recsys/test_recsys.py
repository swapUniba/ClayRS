from unittest import TestCase
import pandas as pd
from orange_cb_recsys.recsys import RecSys, RecSysConfig, ClassifierRecommender, CentroidVector, CosineSimilarity
import numpy as np


class TestRecSys(TestCase):
    def test_recsys(self):
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
                record_list.append((str(i), item, str(np.random.randint(-10, 11) / 10)))
        t_ratings = pd.DataFrame.from_records(record_list, columns=['from_id', 'to_id', 'score'])
        #print(t_ratings)

        path = '../../contents'
        path = 'contents'
        try:
            RecSysConfig(users_directory='{}/users_test1591814865.8959296'.format(path),
                         items_directory='{}/movielens_test1591885241.5520566'.format(path),
                         rating_frame=t_ratings)
        except ValueError:
            pass
        t_classifier = ClassifierRecommender(item_field='Plot', field_representation='2', classifier="gaussian_process")
        t_config = RecSysConfig(users_directory='{}/users_test1591814865.8959296'.format(path),
                                items_directory='{}/movielens_test1591885241.5520566'.format(path),
                                rating_frame=t_ratings,
                                ranking_algorithm=t_classifier)
        t_recsys = RecSys(config=t_config)
        # t_recsys.fit_predict('1', [x for x in item_id_list if np.random.randint(0, 2) == 1])
        t_recsys.fit_ranking('1', 3)

        user_frame = t_ratings[t_ratings['from_id'] == '1']
        test_set = pd.DataFrame({'to_id': ['tt0112281', 'tt0112302']})
        t_recsys.fit_eval_ranking(user_id='1', user_ratings=user_frame, test_set_items=test_set.to_id.tolist(),
                                  recs_number=len(test_set.to_id.tolist()))
        try:
            t_recsys.fit_ranking('1', 2)
        except ValueError:
            pass

        t_centroid = CentroidVector(item_field='Plot', field_representation='1', similarity=CosineSimilarity())

        t_config = RecSysConfig(users_directory='{}/users_test1591814865.8959296'.format(path),
                                items_directory='{}/movielens_test1591885241.5520566'.format(path),
                                rating_frame=t_ratings,
                                ranking_algorithm=t_centroid)
        t_recsys = RecSys(config=t_config)
        t_recsys.fit_ranking('1', 2)

        t_recsys.fit_eval_ranking(user_id='1', user_ratings=user_frame, test_set_items=test_set.to_id.tolist(), recs_number=len(test_set.to_id.tolist()))

        try:
            t_recsys.fit_predict('1', [])
        except ValueError:
            pass
