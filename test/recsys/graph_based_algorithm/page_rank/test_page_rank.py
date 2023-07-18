import unittest

from clayrs.recsys.graph_based_algorithm.page_rank.page_rank import PageRank


class TestPageRank(unittest.TestCase):

    def test_check_weights(self):
        # check that each probability has remained untouched
        expected_rel_items_weight = 0.5
        expected_rel_items_prop_weight = 0.2
        expected_default_nodes_weight = 0.3

        result_rel_items_weight, result_rel_items_prop_weight, result_default_nodes_weight = PageRank.check_weights(
            expected_rel_items_weight,
            expected_rel_items_prop_weight,
            expected_default_nodes_weight
        )

        self.assertEqual(expected_rel_items_weight, result_rel_items_weight)
        self.assertEqual(expected_rel_items_prop_weight, result_rel_items_prop_weight)
        self.assertEqual(expected_default_nodes_weight, result_default_nodes_weight)

    def test_check_weights_auto_assign(self):

        # check that expected_rel_items_weight has been assigned
        expected_rel_items_weight = None
        expected_rel_items_prop_weight = 0.2
        expected_default_nodes_weight = 0.3

        result_rel_items_weight, result_rel_items_prop_weight, result_default_nodes_weight = PageRank.check_weights(
            expected_rel_items_weight,
            expected_rel_items_prop_weight,
            expected_default_nodes_weight
        )

        expected_rel_items_weight = 1 - expected_rel_items_prop_weight - expected_default_nodes_weight

        self.assertEqual(expected_rel_items_weight, result_rel_items_weight)
        self.assertEqual(expected_rel_items_prop_weight, result_rel_items_prop_weight)
        self.assertEqual(expected_default_nodes_weight, result_default_nodes_weight)

        # check that both expected_rel_items_weight and expected_default_nodes_weight have been assigned
        expected_rel_items_weight = 0.5
        expected_rel_items_prop_weight = None
        expected_default_nodes_weight = None

        result_rel_items_weight, result_rel_items_prop_weight, result_default_nodes_weight = PageRank.check_weights(
            expected_rel_items_weight,
            expected_rel_items_prop_weight,
            expected_default_nodes_weight
        )

        expected_rel_items_prop_weight = (1 - expected_rel_items_weight) / 2
        expected_default_nodes_weight = (1 - expected_rel_items_weight) / 2

        self.assertEqual(expected_rel_items_weight, result_rel_items_weight)
        self.assertAlmostEqual(expected_rel_items_prop_weight, result_rel_items_prop_weight)
        self.assertAlmostEqual(expected_default_nodes_weight, result_default_nodes_weight)

        # check that all probs have been automatically assigned
        expected_rel_items_weight = None
        expected_rel_items_prop_weight = None
        expected_default_nodes_weight = None

        result_rel_items_weight, result_rel_items_prop_weight, result_default_nodes_weight = PageRank.check_weights(
            expected_rel_items_weight,
            expected_rel_items_prop_weight,
            expected_default_nodes_weight
        )

        expected_rel_items_weight = 1 / 3
        expected_rel_items_prop_weight = 1 / 3
        expected_default_nodes_weight = 1 / 3

        self.assertAlmostEqual(expected_rel_items_weight, result_rel_items_weight)
        self.assertAlmostEqual(expected_rel_items_prop_weight, result_rel_items_prop_weight)
        self.assertAlmostEqual(expected_default_nodes_weight, result_default_nodes_weight)

        # check that a prob is assigned to 0 when the sum is already 1
        expected_rel_items_weight = None
        expected_rel_items_prop_weight = 0.5
        expected_default_nodes_weight = 0.5

        result_rel_items_weight, result_rel_items_prop_weight, result_default_nodes_weight = PageRank.check_weights(
            expected_rel_items_weight,
            expected_rel_items_prop_weight,
            expected_default_nodes_weight
        )

        expected_rel_items_weight = 0

        self.assertEqual(expected_rel_items_weight, result_rel_items_weight)
        self.assertEqual(expected_rel_items_prop_weight, result_rel_items_prop_weight)
        self.assertEqual(expected_default_nodes_weight, result_default_nodes_weight)

    def test_check_weights_errors(self):
        # check that one probability is not negative
        rel_items_weight = 0.5
        rel_items_prop_weight = 0.5
        default_nodes_weight = -1

        with self.assertRaises(ValueError):
            PageRank.check_weights(rel_items_weight, rel_items_prop_weight, default_nodes_weight)

        # check that one probability is > 1
        rel_items_weight = 1.5
        rel_items_prop_weight = 0.5
        default_nodes_weight = 0.5

        with self.assertRaises(ValueError):
            PageRank.check_weights(rel_items_weight, rel_items_prop_weight, default_nodes_weight)

        # check that sum of probability is > 1
        rel_items_weight = 0.5
        rel_items_prop_weight = 0.5
        default_nodes_weight = 0.5

        with self.assertRaises(ValueError):
            PageRank.check_weights(rel_items_weight, rel_items_prop_weight, default_nodes_weight)


if __name__ == "__main__":
    unittest.main()
