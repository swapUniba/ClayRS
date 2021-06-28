import unittest
import pandas as pd
import numpy as np

from orange_cb_recsys.evaluation.exceptions import NotEnoughUsers, PercentageError
from orange_cb_recsys.evaluation.metrics.fairness_metrics import PredictionCoverage, CatalogCoverage, GiniIndex, DeltaGap, \
    GroupFairnessMetric, Counter
from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split


# Will be the same for every test
user_truth = pd.DataFrame({'from_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
                           'to_id': ['i1', 'i2', 'i3', 'i4', 'i6', 'i1', 'i8', 'i4'],
                           'score': [3, 2, 3, 1, 2, 4, 2, 3]})


class TestPredictionCoverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = {'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10'}
        cls.metric = PredictionCoverage(cls.catalog)

    def test_perform_only_new(self):
        metric = self.metric

        user_pred_only_new_items = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u2', 'u2'],
             'to_id': ['inew1', 'inew2', 'inew3', 'inew4'],
             'score': [650, 600, 500, 650]})

        split_only_new = Split(user_pred_only_new_items, user_truth)

        expected = 0
        result = float(metric.perform(split_only_new)[str(metric)])

        self.assertEqual(expected, result)

    def test_perform_w_new_items(self):
        metric = self.metric

        user_pred_w_new_items = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2'],
             'to_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i4', 'i6', 'i1', 'i8'],
             'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50]})

        split_w_new_items = Split(user_pred_w_new_items, user_truth)

        result = float(metric.perform(split_w_new_items)[str(metric)])

        self.assertTrue(0 <= result <= 100)

    def test_perform_all_items(self):
        metric = self.metric

        user_pred_all_items = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
             'to_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i4', 'i10', 'i7'],
             'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100]})

        split_all = Split(user_pred_all_items, user_truth)

        expected = 100
        result = float(metric.perform(split_all)[str(metric)])

        self.assertEqual(expected, result)

    def test_perform_all_items_and_plus(self):
        catalog = {'i1'}
        metric = PredictionCoverage(catalog)

        # All catalog plus more
        user_pred_all_items_and_plus = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u1', 'u1'],
             'to_id': ['i2', 'i1', 'i4', 'i5'],
             'score': [650, 600, 500, 400, ]})

        split_all = Split(user_pred_all_items_and_plus, user_truth)

        expected = 100
        result = float(metric.perform(split_all)[str(metric)])

        self.assertEqual(expected, result)


class TestCatalogCoverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = {'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10'}
        cls.metric_top_n = CatalogCoverage(cls.catalog, top_n=2)
        cls.metric_k_sampling = CatalogCoverage(cls.catalog, k=1)

    def test_perform_only_new(self):
        metric_top_n = self.metric_top_n
        metric_k_sampling = self.metric_k_sampling

        user_pred_only_new_items = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u2', 'u2'],
             'to_id': ['inew1', 'inew2', 'inew3', 'inew4'],
             'score': [650, 600, 500, 650]})

        split_only_new = Split(user_pred_only_new_items, user_truth)

        expected = 0

        result_top_n = float(metric_top_n.perform(split_only_new)[str(metric_top_n)])
        self.assertEqual(expected, result_top_n)

        result_k_sampling = float(metric_k_sampling.perform(split_only_new)[str(metric_k_sampling)])
        self.assertEqual(expected, result_k_sampling)

    def test_perform_w_new_items(self):
        metric_top_n = self.metric_top_n
        metric_k_sampling = self.metric_k_sampling

        user_pred_w_new_items = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2'],
             'to_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i4', 'i6', 'i1', 'i8'],
             'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50]})

        split_w_new_items = Split(user_pred_w_new_items, user_truth)

        result_top_n = float(metric_top_n.perform(split_w_new_items)[str(metric_top_n)])
        self.assertTrue(0 <= result_top_n <= 100)

        result_k_sampling = float(metric_k_sampling.perform(split_w_new_items)[str(metric_k_sampling)])
        self.assertTrue(0 <= result_k_sampling <= 100)

    def test_perform_all_items(self):
        catalog = {'i2'}

        metric_top_n = CatalogCoverage(catalog, top_n=1)
        metric_k_sampling = CatalogCoverage(catalog, k=1, top_n=1)

        user_pred_all_items = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u2', 'u2'],
             'to_id': ['i2', 'i1', 'i2', 'i7'],
             'score': [650, 600, 200, 100]})

        split_all = Split(user_pred_all_items, user_truth)

        expected = 100

        result = float(metric_top_n.perform(split_all)[str(metric_top_n)])
        self.assertEqual(expected, result)

        result = float(metric_k_sampling.perform(split_all)[str(metric_k_sampling)])
        self.assertEqual(expected, result)

    def test_perform_all_items_and_plus(self):
        catalog = {'i2'}

        metric_top_n = CatalogCoverage(catalog, top_n=2)
        metric_k_sampling = CatalogCoverage(catalog, k=1)

        user_pred_all_items_and_plus = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u2', 'u2'],
             'to_id': ['i2', 'i1', 'i2', 'i7'],
             'score': [650, 600, 200, 100]})

        split_all = Split(user_pred_all_items_and_plus, user_truth)

        expected = 100

        result = float(metric_top_n.perform(split_all)[str(metric_top_n)])
        self.assertEqual(expected, result)

        result = float(metric_k_sampling.perform(split_all)[str(metric_k_sampling)])
        self.assertEqual(expected, result)


class TestGini(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = GiniIndex()

        cls.metric_top_n = GiniIndex(top_n=2)

    def test_perform_equi(self):
        metric = self.metric

        # i1 and i2 and i3 are recommended in equal ways to users
        user_pred_equi = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
             'to_id': ['i1', 'i2', 'i3', 'i2', 'i3', 'i1'],
             'score': [650, 600, 500, 750, 700, 680]})

        split_pred_equi = Split(user_pred_equi, user_truth)

        expected = 0
        result = float(metric.perform(split_pred_equi)[str(metric)])

        self.assertEqual(expected, result)

    def test_perform_equi_top_n(self):
        metric_top_n = self.metric_top_n

        # i1 and i2 and i3 are recommended in equal ways to users
        user_pred_equi = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
             'to_id': ['i1', 'i2', 'i3', 'i2', 'i3', 'i1'],
             'score': [650, 600, 500, 750, 700, 680]})

        split_pred_equi = Split(user_pred_equi, user_truth)

        result = float(metric_top_n.perform(split_pred_equi)[str(metric_top_n)])

        # In the top 2 i2 is recommended more, so there's no equality
        self.assertTrue(0 < result <= 1)

        metric_top_3 = GiniIndex(top_n=3)

        result = float(metric_top_3.perform(split_pred_equi)[str(metric_top_3)])

        # In the top 3 (total length of rec lists) there's equality
        self.assertEqual(0, result)

    def test_perform_mixed(self):
        metric = self.metric
        metric_top_n = self.metric_top_n

        user_pred_mixed = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2'],
             'to_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i2', 'i6', 'i1', 'i8'],
             'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50]})

        split_mixed = Split(user_pred_mixed, user_truth)

        result = float(metric.perform(split_mixed)[str(metric)])
        self.assertTrue(0 < result <= 1)

        result = float(metric_top_n.perform(split_mixed)[str(metric_top_n)])
        self.assertTrue(0 < result <= 1)

    def test_perform_only_one(self):
        metric = self.metric
        metric_top_n = self.metric_top_n

        user_pred_only_one_item = pd.DataFrame(
            {'from_id': ['u1'],
             'to_id': ['i4'],
             'score': [650]})

        split_only_one = Split(user_pred_only_one_item, user_truth)

        expected = 0
        result = float(metric.perform(split_only_one)[str(metric)])

        self.assertEqual(expected, result)

        # Even if there's only one element and top_n = 2, everything works correctly
        expected = 0
        result = float(metric_top_n.perform(split_only_one)[str(metric_top_n)])

        self.assertEqual(expected, result)


class TestGroupFairnessMetric(unittest.TestCase):
    def test_split_users_in_groups(self):
        user_pred_only_u1 = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u1'],
             'to_id': ['i2', 'i1', 'i4'],
             'score': [650, 600, 500]})

        popular_items = {'i2'}

        # Check error raised when split 1 users in 2 groups (n_users < n_groups)
        with self.assertRaises(NotEnoughUsers):
            GroupFairnessMetric.split_user_in_groups(user_pred_only_u1,
                                                     groups={'a': 0.5, 'b': 0.5},
                                                     pop_items=popular_items)

        # Check error raised a percentage is not valid
        with self.assertRaises(PercentageError):
            GroupFairnessMetric.split_user_in_groups(user_pred_only_u1,
                                                     groups={'a': 1.9},
                                                     pop_items=popular_items)

        # Check default_diverse group when percentage total < 1
        user_pred_4_users = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u3', 'u3', 'u4', 'u4', 'u4', 'u4'],
             'to_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i2', 'i6', 'i1', 'i8'],
             'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50]})
        popular_items = {'i2', 'i1'}

        result = GroupFairnessMetric.split_user_in_groups(user_pred_4_users,
                                                          groups={'a': 0.3, 'b': 0.5},
                                                          pop_items=popular_items)
        self.assertIn('a', result.keys())
        self.assertIn('b', result.keys())
        self.assertIn('default_diverse', result.keys())

        # Check splitted groups in a usual situation
        user_pred_4_users = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u3', 'u3', 'u4', 'u4', 'u4', 'u4'],
             'to_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i2', 'i6', 'i1', 'i8'],
             'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50]})
        popular_items = {'i2', 'i1'}

        result = GroupFairnessMetric.split_user_in_groups(user_pred_4_users,
                                                          groups={'a': 0.5, 'b': 0.3, 'c': 0.2},
                                                          pop_items=popular_items)

        # u1 and u4 are the users that prefer popular items,
        # so they are put into the first group
        self.assertIn('u1', result['a'])
        self.assertIn('u4', result['a'])

        # u2 and u3 have the same popolarity ratio, so we are not sure if they are put
        # into the 2nd group or the third one
        self.assertTrue('u2' in result['b'] or 'u2' in result['c'])
        self.assertTrue('u3' in result['b'] or 'u3' in result['c'])

    def test_get_avg_pop_by_users(self):
        user_pred_4_users = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u3', 'u3', 'u4', 'u4', 'u4', 'u4'],
             'to_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i2', 'i6', 'i1', 'i8'],
             'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50]})

        pop_by_item = Counter(list(user_pred_4_users['to_id']))

        # sum of popularity of every item rated by the user / number of rated items by the user
        # (the popularity of an item is the number of times it is repeated in the frame)
        expected_u1 = 5 / 3
        expected_u2 = 4 / 3
        expected_u3 = 3 / 2
        expected_u4 = 8 / 4

        # Calculate avg_pop for every user of the frame
        result_all = GroupFairnessMetric.get_avg_pop_by_users(user_pred_4_users, pop_by_item)

        # Check that the resuts are corrected
        self.assertAlmostEqual(expected_u1, result_all['u1'])
        self.assertAlmostEqual(expected_u2, result_all['u2'])
        self.assertAlmostEqual(expected_u3, result_all['u3'])
        self.assertAlmostEqual(expected_u4, result_all['u4'])

        # Calculate avg_pop for only users in the 'group' parameter passed
        result_u1_u2 = GroupFairnessMetric.get_avg_pop_by_users(user_pred_4_users, pop_by_item, group={'u1', 'u2'})

        # Check that the results are correct
        self.assertAlmostEqual(expected_u1, result_all['u1'])
        self.assertAlmostEqual(expected_u2, result_all['u2'])
        # Check that 'u3' and 'u4' are not present
        self.assertNotIn('u3', result_u1_u2.keys())
        self.assertNotIn('u4', result_u1_u2.keys())


class TestDeltaGap(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recs = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2'],
             'to_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i4', 'i6', 'i1', 'i8'],
             'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50]})

        cls.split = Split(cls.recs, user_truth)

    def test_calculate_gap(self):
        # This is basically the inner part of the GAP equation, the fraction at the numerator
        # of the GAP formula calculated preemptively for every user.
        # It is set manually for the sake of the test, but it is obtained by the
        # get_avg_pop_by_users() method of the GroupFairnessMetric Class
        avg_pop_by_users = {'u1': 2, 'u2': 1.78, 'u3': 3.5, 'u4': 1.1}

        expected_u1_u3 = (avg_pop_by_users['u1'] + avg_pop_by_users['u3']) / 2
        result_u1_u3 = DeltaGap.calculate_gap({'u1', 'u3'}, avg_pop_by_users)

        self.assertAlmostEqual(expected_u1_u3, result_u1_u3)

        expected_u2_u4 = (avg_pop_by_users['u2'] + avg_pop_by_users['u4']) / 2
        result_u2_u4 = DeltaGap.calculate_gap({'u2', 'u4'}, avg_pop_by_users)

        self.assertAlmostEqual(expected_u2_u4, result_u2_u4)

    def test_calculate_delta_gap(self):
        gap_profile = 2.32
        gap_recs = 3

        result = DeltaGap.calculate_delta_gap(gap_recs, gap_profile)
        expected = (gap_recs - gap_profile) / gap_profile

        self.assertAlmostEqual(expected, result)

    def test_invalid_percentage(self):
        with self.assertRaises(PercentageError):
            DeltaGap(user_groups={'a': 0.5}, pop_percentage=-0.5)
            DeltaGap(user_groups={'a': 0.5}, pop_percentage=0)
            DeltaGap(user_groups={'a': 0.5}, pop_percentage=1.5)

    def test_perform_2_users_2_groups(self):
        metric = DeltaGap(user_groups={'a': 0.5, 'b': 0.5})
        result = metric.perform(self.split)

        pop_by_item_truth = Counter(list(user_truth['to_id']))

        # group_a = { u2 } (since it has higher popular ratio, it is put into the first group)
        # group_b = { u1 }

        # For every user in the group calculate the average popularity of the recommendations.
        # To calculate the avg popularity, pop_by_item_pred is used, since due to the methodology
        # items in the recommendation lists may differ from item in the truth
        RECS_avg_pop_group_a = {'u2': 6 / 4}  # for every user sum_pop_item_rated / n_item_rated
        RECS_avg_pop_group_b = {'u1': 8 / 8}  # for every user sum_pop_item_rated / n_item_rated

        # For every user in the group calculate the average popularity of the profile.
        # To calculate the avg popularity, pop_by_item_truth is used, since due to the methodology
        # items in the truth may differ from item in the recommendation lists
        PROFILE_avg_pop_group_a = {'u2': 5 / 3}  # for every user sum_pop_item_rated / n_item_rated
        PROFILE_avg_pop_group_b = {'u1': 7 / 5}  # for every user sum_pop_item_rated / n_item_rated

        RECS_gap_group_a = (6 / 4) / 1  # sum the RECS_avg_pop of every user of the group_a / n_users in group_a
        RECS_gap_group_b = (8 / 8) / 1  # sum the RECS_avg_pop of every user of the group_b / n_users in group_b

        PROFILE_gap_group_a = (5 / 3) / 1  # sum the PROFILE_avg_pop of every user of the group_a / n_users in group_a
        PROFILE_gap_group_b = (7 / 5) / 1  # sum the PROFILE_avg_pop of every user of the group_b / n_users in group_b

        expected_delta_gap_group_a = (RECS_gap_group_a - PROFILE_gap_group_a) / PROFILE_gap_group_a
        expected_delta_gap_group_b = (RECS_gap_group_b - PROFILE_gap_group_b) / PROFILE_gap_group_b

        result_delta_gap_group_a = float(result["{} | a".format(str(metric))])
        result_delta_gap_group_b = float(result["{} | b".format(str(metric))])

        self.assertAlmostEqual(expected_delta_gap_group_a, result_delta_gap_group_a)
        self.assertAlmostEqual(expected_delta_gap_group_b, result_delta_gap_group_b)

    def test_perform_multiple_users_one_group(self):

        metric = DeltaGap(user_groups={'a': 1})
        result = metric.perform(self.split)

        pop_by_item_truth = Counter(list(user_truth['to_id']))

        # group_a = { u2, u1 }

        # For every user in the group calculate the average popularity of the recommendations.
        # To calculate the avg popularity, pop_by_item_pred is used, since due to the methodology
        # items in the recommendation lists may differ from item in the truth
        RECS_avg_pop_group_a = {'u2': 6 / 4, 'u1': 8 / 8}  # for every user sum_pop_item_rated / n_item_rated

        # For every user in the group calculate the average popularity of the profile.
        # To calculate the avg popularity, pop_by_item_truth is used, since due to the methodology
        # items in the truth may differ from item in the recommendation lists
        PROFILE_avg_pop_group_a = {'u2': 5 / 3, 'u1': 7 / 5}  # for every user sum_pop_item_rated / n_item_rated

        # Sum the RECS_avg_pop of every user of the group_a / n_users in group_a
        RECS_gap_group_a = ((6 / 4) + (8 / 8)) / 2

        # Sum the PROFILE_avg_pop of every user of the group_a / n_users in group_a
        PROFILE_gap_group_a = ((5 / 3) + (7 / 5)) / 2

        expected_delta_gap_group_a = (RECS_gap_group_a - PROFILE_gap_group_a) / PROFILE_gap_group_a

        result_delta_gap_group_a = float(result["{} | a".format(str(metric))])

        self.assertAlmostEqual(expected_delta_gap_group_a, result_delta_gap_group_a)

    def test_perform_0_gap(self):
        # DeltaGap with 2 equals frame should return 0 for every group
        split = Split(user_truth, user_truth)

        metric = DeltaGap(user_groups={'a': 0.5, 'b': 0.5})

        result = metric.perform(split)

        for col in result.columns:
            self.assertTrue(v == 0 for v in result[col])

    def test_perform_top_3(self):

        metric = DeltaGap(user_groups={'a': 1}, top_n=3)
        result = metric.perform(self.split)

        pop_by_item_truth = Counter(list(user_truth['to_id']))

        # group_a = { u2, u1 }

        # For every user in the group calculate the average popularity of the recommendations.
        # To calculate the avg popularity, pop_by_item_pred is used, since due to the methodology
        # items in the recommendation lists may differ from item in the truth
        RECS_avg_pop_group_a = {'u2': 5 / 3, 'u1': 5 / 3}  # for every user sum_pop_item_rated / n_item_rated

        # For every user in the group calculate the average popularity of the profile.
        # To calculate the avg popularity, pop_by_item_truth is used, since due to the methodology
        # items in the truth may differ from item in the recommendation lists
        PROFILE_avg_pop_group_a = {'u2': 5 / 3, 'u1': 7 / 5}  # for every user sum_pop_item_rated / n_item_rated

        # Sum the RECS_avg_pop of every user of the group_a / n_users in group_a
        RECS_gap_group_a = ((5 / 3) + (5 / 3)) / 2

        # Sum the PROFILE_avg_pop of every user of the group_a / n_users in group_a
        PROFILE_gap_group_a = ((5 / 3) + (7 / 5)) / 2

        expected_delta_gap_group_a = (RECS_gap_group_a - PROFILE_gap_group_a) / PROFILE_gap_group_a

        result_delta_gap_group_a = float(result["{} | a".format(str(metric))])

        self.assertAlmostEqual(expected_delta_gap_group_a, result_delta_gap_group_a)

    def test_perform_increased_pop_percentage(self):
        truth = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2',
                         'u3', 'u3', 'u3', 'u3', 'u4', 'u4', 'u4', 'u5', 'u5', 'u5'],
             'to_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i4', 'i6', 'i1', 'i8',
                       'i2', 'i4', 'i3', 'i20', 'i3', 'i1', 'i21', 'i3', 'i5', 'i1'],
             'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50,
                       500, 400, 300, 200, 150, 100, 50, 800, 600, 500]})

        recs = pd.DataFrame(
            {'from_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2', 'u2',
                         'u3', 'u3', 'u3', 'u3', 'u4', 'u4', 'u4', 'u5', 'u5', 'u5', 'u5', 'u5'],
             'to_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i4', 'i6', 'i1', 'i5', 'i35',
                       'i2', 'i4', 'i3', 'i20', 'i3', 'i1', 'i3', 'i5', 'i1', 'i9', 'i36', 'i6'],
             'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50, 25,
                       500, 400, 300, 200, 350, 100, 50, 800, 600, 500, 400, 300]})

        split = Split(recs, truth)

        result_pop_normal = DeltaGap(user_groups={'a': 0.3, 'b': 0.3, 'c': 0.4}).perform(split)
        result_pop_increased = DeltaGap(user_groups={'a': 0.3, 'b': 0.3, 'c': 0.4}, pop_percentage=0.6).perform(split)

        result_pop_normal = np.array(result_pop_normal)
        result_pop_increased = np.array(result_pop_increased)

        result_pop_normal.sort(axis=0)
        result_pop_increased.sort(axis=0)

        # Just check that results with pop_percentage increased are different,
        # since users are put into groups differently
        self.assertFalse(np.array_equal(result_pop_normal, result_pop_increased))
