import os

import pandas as pd

from orange_cb_recsys.content_analyzer import Ratings
from orange_cb_recsys.recsys import NXFullGraph
from test import dir_test_files
from unittest import TestCase

ratings_filename = os.path.join(dir_test_files, 'new_ratings_small.csv')
movies_dir = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')
users_dir = os.path.join(dir_test_files, 'complex_contents', 'users_codified/')

rat = pd.DataFrame.from_dict({'from_id': ["1", "1", "2", "2", "2", "3", "4", "4"],
                              'to_id': ["tt0112281", "tt0112302", "tt0112281", "tt0112346",
                                        "tt0112453", "tt0112453", "tt0112346", "tt0112453"],
                              'score': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7]})
rat = Ratings.from_dataframe(rat)


class TestGraph(TestCase):
    def setUp(self) -> None:
        # we need to instantiate a sublass of Graph since it's an abstract class to test its methods
        self.g: NXFullGraph = NXFullGraph(rat,
                                          item_contents_dir=movies_dir,
                                          item_exo_properties={'dbpedia': ['film director',
                                                                           'runtime (m)']},

                                          # It's the column in the users .DAT which identifies the gender
                                          user_exo_properties={'local': '1'},
                                          user_contents_dir=users_dir)

    def test_to_ratings(self):
        converted_rat = self.g.to_ratings()

        # check that original ratings and converted ratings are equal

        self.assertEqual(set(rat.user_id_column), set(converted_rat.user_id_column))
        self.assertEqual(set(rat.item_id_column), set(converted_rat.item_id_column))
        self.assertEqual(set(rat.score_column), set(converted_rat.score_column))
        self.assertEqual(set(rat.timestamp_column), set(converted_rat.timestamp_column))

        for user in rat.user_id_column:
            user_rat = rat.get_user_interactions(user)
            user_converted_rat = converted_rat.get_user_interactions(user)

            self.assertCountEqual(user_rat, user_converted_rat)
