import os

import pandas as pd

from clayrs.content_analyzer import Ratings
from clayrs.recsys import NXFullGraph
from test import dir_test_files
from unittest import TestCase

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
        self.assertEqual(set(rat.unique_user_id_column), set(converted_rat.unique_user_id_column))
        self.assertEqual(set(rat.unique_item_id_column), set(converted_rat.unique_item_id_column))
        self.assertEqual(set(rat.score_column), set(converted_rat.score_column))
        self.assertEqual(set(rat.timestamp_column), set(converted_rat.timestamp_column))

        # compare that for each user, we have same user interactions
        for user in rat.user_id_column:
            user_rat = rat.get_user_interactions(user)
            user_converted_rat = converted_rat.get_user_interactions(user)

            self.assertCountEqual(user_rat, user_converted_rat)

        # user map set, so we expected same user map between expected and result
        converted_rat_with_user_map = self.g.to_ratings(user_map=rat.user_map)
        self.assertEqual(list(rat.user_map), list(converted_rat_with_user_map.user_map))

        # item map set, so we expected same item map between expected and result
        converted_rat_with_item_map = self.g.to_ratings(item_map=rat.item_map)
        self.assertEqual(list(rat.item_map), list(converted_rat_with_item_map.item_map))

        # user map and item_map set, so we expected them to be equal between expected and result
        converted_rat_with_user_item_map = self.g.to_ratings(user_map=rat.user_map, item_map=rat.item_map)
        self.assertEqual(list(rat.user_map), list(converted_rat_with_user_item_map.user_map))
        self.assertEqual(list(rat.item_map), list(converted_rat_with_user_item_map.item_map))
