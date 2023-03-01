import pandas as pd
import unittest
import numpy as np
from unittest import TestCase

from clayrs.content_analyzer import Ratings
from clayrs.recsys import TestRatingsMethodology, AllItemsMethodology, ItemNode
from clayrs.recsys import ItemNode, PropertyNode
from clayrs.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from clayrs.recsys.graph_based_algorithm.page_rank.nx_page_rank import NXPageRank
from clayrs.recsys.graphs import NXFullGraph

train_ratings = pd.DataFrame.from_records([
    ("A000", "tt0114576", 1, "54654675"),
    ("A000", "tt0112453", 0.5, "54654675"),
    ("A001", "tt0114576", 0.5, "54654675"),
    ("A001", "tt0112896", 0, "54654675"),
    ("A000", "tt0113041", 0.75, "54654675"),
    ("A002", "tt0112453", 0.5, "54654675"),
    ("A002", "tt0113497", 0.5, "54654675"),
    ("A003", "tt0112453", 0, "54654675")],
    columns=["from_id", "to_id", "score", "timestamp"])

test_ratings = pd.DataFrame.from_records([
    ("A000", "tt0113497", 1, "54654675"),
    ("A001", "tt0112453", 0.5, "54654675"),
    ("A002", "tt0114576", 0.5, "54654675"),
    ("A003", "tt0114576", 0, "54654675")],
    columns=["from_id", "to_id", "score", "timestamp"])

# we create manually the mapping since we want a global mapping containing train and test items
item_map = {}
all_items = train_ratings[["to_id"]].append(test_ratings[["to_id"]])["to_id"]
for item_id in all_items:
    if item_id not in item_map:
        item_map[item_id] = len(item_map)

user_map = {}
all_users = train_ratings[["from_id"]].append(test_ratings[["from_id"]])["from_id"]
for user_id in all_users:
    if user_id not in user_map:
        user_map[user_id] = len(user_map)

train_ratings = Ratings.from_dataframe(train_ratings, user_map=user_map, item_map=item_map)
test_ratings = Ratings.from_dataframe(test_ratings, user_map=user_map, item_map=item_map)


class TestNXPageRank(TestCase):

    def setUp(self) -> None:
        self.graph = NXFullGraph(train_ratings)

    def test_predict(self):
        alg = NXPageRank()

        # Will raise Exception since it's not a Score Prediction Algorithm
        with self.assertRaises(NotPredictionAlg):
            alg.predict(self.graph, train_ratings, test_ratings, {'A000'},
                        TestRatingsMethodology().setup(train_ratings, test_ratings), num_cpus=1)

    def test_rank(self):
        # test not personalized
        alg = NXPageRank()

        # unbound rank with None methodology (all unrated items will be ranked)
        res_all_unrated = alg.rank(self.graph, train_ratings, test_ratings, test_ratings.user_id_column,
                                   recs_number=None,
                                   methodology=AllItemsMethodology().setup(train_ratings, test_ratings),
                                   num_cpus=1)

        # assert that for each user we predict items not in its train set
        for rank_user_uir in res_all_unrated:
            user_id = rank_user_uir[0][0]  # the id for the uir rank is in the first column first cell ([0][0])
            user_idx = train_ratings.user_map[user_id]

            item_rated_ids = train_ratings.item_map[train_ratings.get_user_interactions(user_idx)[:, 1].astype(int)]
            item_rated_set_nodes = set(ItemNode(item_id) for item_id in item_rated_ids)
            item_ranked_set_nodes = set(rank_user_uir[:, 1])

            # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
            rated_in_ranked = item_ranked_set_nodes.intersection(item_rated_set_nodes)
            self.assertEqual(len(rated_in_ranked), 0)

        # rank top-1 with None methodology (all unrated items will be ranked) for only A000
        n_recs = 1
        user_id_list = {"A000"}
        [recs_top_1_a000] = alg.rank(self.graph, train_ratings, test_ratings, user_id_list,
                                     recs_number=n_recs,
                                     methodology=AllItemsMethodology().setup(train_ratings, test_ratings),
                                     num_cpus=1)

        # assert that only A000 is present in the ranking produced
        self.assertEqual(user_id_list, set(recs_top_1_a000[:, 0]))
        # assert that ranking produced is the top-1
        self.assertTrue(len(recs_top_1_a000) == n_recs)

        # test personalized
        alg = NXPageRank(personalized=True)
        result_personalized = alg.rank(self.graph, train_ratings, test_ratings, {'A000'}, recs_number=None,
                                       methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                       num_cpus=1)

        alg = NXPageRank()
        result_not_personalized = alg.rank(self.graph, train_ratings, test_ratings, {'A000'}, recs_number=None,
                                           methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                           num_cpus=1)

        for rank_user_uir_pers, rank_user_uir_normal in zip(result_personalized, result_not_personalized):
            self.assertFalse(np.array_equal(rank_user_uir_pers, rank_user_uir_normal))

        # test with custom parameters
        alg = NXPageRank(alpha=0.9, max_iter=4, tol=1e-3)
        result_custom = alg.rank(self.graph, train_ratings, test_ratings, {'A000'}, recs_number=None,
                                 methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                 num_cpus=1)

        alg = NXPageRank()
        result_not_custom = alg.rank(self.graph, train_ratings, test_ratings, {'A000'}, recs_number=None,
                                     methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                     num_cpus=1)

        for rank_user_uir_custom, rank_user_uir_not_custom in zip(result_custom, result_not_custom):
            self.assertFalse(np.array_equal(rank_user_uir_custom, rank_user_uir_not_custom))

    def test_rank_personalized_weight_schemas(self):

        # result classic pagerank
        alg = NXPageRank(alpha=0)
        [result_not_personalized] = alg.rank(self.graph, train_ratings, test_ratings,
                                             num_cpus=1, user_id_list={'A000'},
                                             methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                             recs_number=None)

        # test personalized 0.8 relevant items, 0.2 other nodes
        alg = NXPageRank(alpha=0, personalized=True)
        [result_personalized_08] = alg.rank(self.graph, train_ratings, test_ratings,
                                            num_cpus=1, user_id_list={'A000'},
                                            methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                            recs_number=None)

        self.assertFalse(np.array_equal(result_personalized_08, result_not_personalized))

        # test personalized 0.4 relevant items, 0.4 relevant properties, 0.2 other nodes
        alg = NXPageRank(alpha=0, personalized=True,
                         rel_items_weight=0.4, rel_items_prop_weight=0.4, default_nodes_weight=0.2)
        [result_personalized_04] = alg.rank(self.graph, train_ratings, test_ratings,
                                            num_cpus=1, user_id_list={'A000'},
                                            methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                            recs_number=None)

        self.assertFalse(np.array_equal(result_personalized_04, result_not_personalized))
        self.assertFalse(np.array_equal(result_personalized_04, result_personalized_08))

    def test_rank_personalized_none_weights(self):

        # add relevant prop for active user to test different weighting schema
        self.graph.add_link(ItemNode("tt0114576"), PropertyNode("Jean-Claude Van Damme"), label="starring")

        # test personalized 0.8 relevant items, none to relevant properties, 0.2 to other nodes
        # this means that relevant properties will be treated as 'other nodes' (and so 0.2 prob of being
        # chosen by the random surfer normalized by the total number of 'other nodes')
        alg = NXPageRank(alpha=0, personalized=True,
                         rel_items_weight=0.8, rel_items_prop_weight=None, default_nodes_weight=0.2)
        [result_personalized] = alg.rank(self.graph, train_ratings, test_ratings,
                                         num_cpus=1, user_id_list={'A000'},
                                         methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                         recs_number=None)

        # test personalized 0.8 relevant items, 0 to relevant properties, 0.2 to other nodes
        # this means that relevant properties will be NOT treated as 'other nodes', but will have 0 prob
        # of being chosen by the random surfer
        alg = NXPageRank(alpha=0, personalized=True,
                         rel_items_weight=0.8, rel_items_prop_weight=0, default_nodes_weight=0.2)
        [result_personalized_strict] = alg.rank(self.graph, train_ratings, test_ratings,
                                                num_cpus=1, user_id_list={'A000'},
                                                methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                                recs_number=None)

        self.assertFalse(np.array_equal(result_personalized, result_personalized_strict))

        # test personalized none relevant items, 0.8 to relevant properties, 0.2 to other nodes
        # this means that relevant items will be treated as 'other nodes' (and so 0.2 prob of being
        # chosen by the random surfer normalized by the total number of 'other nodes')
        alg = NXPageRank(alpha=0, personalized=True,
                         rel_items_weight=None, rel_items_prop_weight=0.8, default_nodes_weight=0.2)
        [result_personalized] = alg.rank(self.graph, train_ratings, test_ratings,
                                         num_cpus=1, user_id_list={'A000'},
                                         methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                         recs_number=None)

        # test personalized 0 relevant items, 0.8 to relevant properties, 0.2 to other nodes
        # this means that relevant items will be NOT treated as 'other nodes', but will have 0 prob
        # of being chosen by the random surfer
        alg = NXPageRank(alpha=0, personalized=True,
                         rel_items_weight=0, rel_items_prop_weight=0.8, default_nodes_weight=0.2)
        [result_personalized_strict] = alg.rank(self.graph, train_ratings, test_ratings,
                                                num_cpus=1, user_id_list={'A000'},
                                                methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                                recs_number=None)

        self.assertFalse(np.array_equal(result_personalized, result_personalized_strict))


if __name__ == "__main__":
    unittest.main()
