import unittest
import pandas as pd
import numpy as np

from clayrs.content_analyzer import Ratings, Rank
from clayrs.evaluation.exceptions import NotEnoughUsers
from clayrs.evaluation.metrics.fairness_metrics import PredictionCoverage, CatalogCoverage, GiniIndex, \
    DeltaGap, GroupFairnessMetric, Counter
from clayrs.evaluation.eval_pipeline_modules.metric_evaluator import Split

# Will be the same for every test
truth = pd.DataFrame({'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
                      'item_id': ['i1', 'i2', 'i3', 'i4', 'i6', 'i1', 'i8', 'i4'],
                      'score': [3, 2, 3, 1, 2, 4, 2, 3]})
truth = Ratings.from_dataframe(truth)


class TestPredictionCoverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        catalog = {'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10'}
        cls.metric = PredictionCoverage(catalog)

    def test_perform_only_new(self):
        metric = self.metric

        pred_only_new_items = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u2', 'u2'],
             'item_id': ['inew1', 'inew2', 'inew3', 'inew4'],
             'score': [650, 600, 500, 650]})
        pred_only_new_items = Ratings.from_dataframe(pred_only_new_items)

        split_only_new = Split(pred_only_new_items, truth)

        expected = 0
        result = float(metric.perform(split_only_new)[str(metric)])

        self.assertEqual(expected, result)

    def test_perform_w_new_items(self):
        metric = self.metric

        pred_w_new_items = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2'],
             'item_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i4', 'i6', 'i1', 'i8'],
             'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50]})
        pred_w_new_items = Ratings.from_dataframe(pred_w_new_items)

        split_w_new_items = Split(pred_w_new_items, truth)

        result = float(metric.perform(split_w_new_items)[str(metric)])

        self.assertTrue(0 <= result <= 100)

    def test_perform_all_items(self):
        metric = self.metric

        pred_all_items = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
             'item_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i4', 'i10', 'i7'],
             'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100]})
        pred_all_items = Ratings.from_dataframe(pred_all_items)

        split_all = Split(pred_all_items, truth)

        expected = 100
        result = float(metric.perform(split_all)[str(metric)])

        self.assertEqual(expected, result)

    def test_perform_all_items_and_plus(self):
        catalog = {'i1'}
        metric = PredictionCoverage(catalog)

        # All catalog plus more
        pred_all_items_and_plus = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1', 'u1'],
             'item_id': ['i2', 'i1', 'i4', 'i5'],
             'score': [650, 600, 500, 400, ]})
        pred_all_items_and_plus = Ratings.from_dataframe(pred_all_items_and_plus)

        split_all = Split(pred_all_items_and_plus, truth)

        expected = 100
        result = float(metric.perform(split_all)[str(metric)])

        self.assertEqual(expected, result)


class TestCatalogCoverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        catalog = {'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10'}
        cls.metric_top_n = CatalogCoverage(catalog, top_n=2)
        cls.metric_k_sampling = CatalogCoverage(catalog, k=1)

    def test_perform_only_new(self):
        metric_top_n = self.metric_top_n
        metric_k_sampling = self.metric_k_sampling

        pred_only_new_items = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u2', 'u2'],
             'item_id': ['inew1', 'inew2', 'inew3', 'inew4'],
             'score': [650, 600, 500, 650]})
        pred_only_new_items = Ratings.from_dataframe(pred_only_new_items)

        split_only_new = Split(pred_only_new_items, truth)

        expected = 0

        result_top_n = float(metric_top_n.perform(split_only_new)[str(metric_top_n)])
        self.assertEqual(expected, result_top_n)

        result_k_sampling = float(metric_k_sampling.perform(split_only_new)[str(metric_k_sampling)])
        self.assertEqual(expected, result_k_sampling)

    def test_perform_w_new_items(self):
        metric_top_n = self.metric_top_n
        metric_k_sampling = self.metric_k_sampling

        pred_w_new_items = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2'],
             'item_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i4', 'i6', 'i1', 'i8'],
             'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50]})
        pred_w_new_items = Ratings.from_dataframe(pred_w_new_items)

        split_w_new_items = Split(pred_w_new_items, truth)

        result_top_n = float(metric_top_n.perform(split_w_new_items)[str(metric_top_n)])
        self.assertTrue(0 <= result_top_n <= 100)

        result_k_sampling = float(metric_k_sampling.perform(split_w_new_items)[str(metric_k_sampling)])
        self.assertTrue(0 <= result_k_sampling <= 100)

    def test_perform_all_items(self):
        catalog = {'i2'}

        metric_top_n = CatalogCoverage(catalog, top_n=1)
        metric_k_sampling = CatalogCoverage(catalog, k=1, top_n=1)

        pred_all_items = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u2', 'u2'],
             'item_id': ['i2', 'i1', 'i2', 'i7'],
             'score': [650, 600, 200, 100]})
        pred_all_items = Ratings.from_dataframe(pred_all_items)

        split_all = Split(pred_all_items, truth)

        expected = 100

        result = float(metric_top_n.perform(split_all)[str(metric_top_n)])
        self.assertEqual(expected, result)

        result = float(metric_k_sampling.perform(split_all)[str(metric_k_sampling)])
        self.assertEqual(expected, result)

    def test_perform_all_items_and_plus(self):
        catalog = {'i2'}

        metric_top_n = CatalogCoverage(catalog, top_n=2)
        metric_k_sampling = CatalogCoverage(catalog, k=1)

        pred_all_items_and_plus = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u2', 'u2'],
             'item_id': ['i2', 'i1', 'i2', 'i7'],
             'score': [650, 600, 200, 100]})
        pred_all_items_and_plus = Ratings.from_dataframe(pred_all_items_and_plus)

        split_all = Split(pred_all_items_and_plus, truth)

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
        pred_equi = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
             'item_id': ['i1', 'i2', 'i3', 'i2', 'i3', 'i1'],
             'score': [650, 600, 500, 750, 700, 680]})
        pred_equi = Ratings.from_dataframe(pred_equi)

        split_pred_equi = Split(pred_equi, truth)

        expected = 0
        result = float(metric.perform(split_pred_equi)[str(metric)])

        self.assertEqual(expected, result)

    def test_perform_equi_top_n(self):
        metric_top_n = self.metric_top_n

        # i1 and i2 and i3 are recommended in equal ways to users
        pred_equi = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2'],
             'item_id': ['i1', 'i2', 'i3', 'i2', 'i3', 'i1'],
             'score': [650, 600, 500, 750, 700, 680]})
        pred_equi = Ratings.from_dataframe(pred_equi)

        split_pred_equi = Split(pred_equi, truth)

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

        pred_mixed = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u2'],
             'item_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i2', 'i6', 'i1', 'i8'],
             'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50]})
        pred_mixed = Ratings.from_dataframe(pred_mixed)

        split_mixed = Split(pred_mixed, truth)

        result = float(metric.perform(split_mixed)[str(metric)])
        self.assertTrue(0 < result <= 1)

        result = float(metric_top_n.perform(split_mixed)[str(metric_top_n)])
        self.assertTrue(0 < result <= 1)

    def test_perform_only_one(self):
        metric = self.metric
        metric_top_n = self.metric_top_n

        pred_only_one_item = pd.DataFrame(
            {'user_id': ['u1'],
             'item_id': ['i4'],
             'score': [650]})
        pred_only_one_item = Ratings.from_dataframe(pred_only_one_item)

        split_only_one = Split(pred_only_one_item, truth)

        expected = 0
        result = float(metric.perform(split_only_one)[str(metric)])

        self.assertEqual(expected, result)

        # Even if there's only one element and top_n = 2, everything works correctly
        expected = 0
        result = float(metric_top_n.perform(split_only_one)[str(metric_top_n)])

        self.assertEqual(expected, result)


class TestGroupFairnessMetric(unittest.TestCase):
    def test_split_users_in_groups(self):
        pred_only_u1 = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1'],
             'item_id': ['i2', 'i1', 'i4'],
             'score': [650, 600, 500]})
        popular_items = {'i2'}
        pred_only_u1 = Ratings.from_dataframe(pred_only_u1)

        # Check error raised when split 1 users in 2 groups (n_users < n_groups)
        with self.assertRaises(NotEnoughUsers):
            GroupFairnessMetric.split_user_in_groups(pred_only_u1,
                                                     groups={'a': 0.5, 'b': 0.5},
                                                     pop_items=popular_items)

        # Check error raised a percentage is not valid
        with self.assertRaises(ValueError):
            GroupFairnessMetric.split_user_in_groups(pred_only_u1,
                                                     groups={'a': 1.9},
                                                     pop_items=popular_items)

        # Check error raised when percentage total < 1
        pred_4_users = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u3', 'u3', 'u4', 'u4', 'u4', 'u4'],
             'item_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i2', 'i6', 'i1', 'i8'],
             'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50]})
        popular_items = {'i2', 'i1'}
        pred_4_users = Ratings.from_dataframe(pred_4_users)

        with self.assertRaises(ValueError):
            result = GroupFairnessMetric.split_user_in_groups(pred_4_users,
                                                              groups={'a': 0.3, 'b': 0.5},
                                                              pop_items=popular_items)

        # Check splitted groups in a usual situation
        result = GroupFairnessMetric.split_user_in_groups(pred_4_users,
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
        pred_4_users = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u3', 'u3', 'u4', 'u4', 'u4', 'u4'],
             'item_id': ['i2', 'i1', 'i4', 'i5', 'i6', 'i3', 'i8', 'i9', 'i2', 'i6', 'i1', 'i8'],
             'score': [650, 600, 500, 400, 300, 220, 100, 50, 350, 200, 100, 50]})

        pop_by_item = Counter(list(pred_4_users['item_id']))

        # sum of popularity of every item rated by the user / number of rated items by the user
        # (the popularity of an item is the number of times it is repeated in the frame)
        expected_u1 = 5 / 3
        expected_u2 = 4 / 3
        expected_u3 = 3 / 2
        expected_u4 = 8 / 4

        pred_4_users = Ratings.from_dataframe(pred_4_users)
        # Calculate avg_pop for every user of the frame
        result_all = GroupFairnessMetric.get_avg_pop_by_users(pred_4_users, pop_by_item)

        # Check that the resuts are corrected
        self.assertAlmostEqual(expected_u1, result_all['u1'])
        self.assertAlmostEqual(expected_u2, result_all['u2'])
        self.assertAlmostEqual(expected_u3, result_all['u3'])
        self.assertAlmostEqual(expected_u4, result_all['u4'])

        # Calculate avg_pop for only users in the 'group' parameter passed
        result_u1_u2 = GroupFairnessMetric.get_avg_pop_by_users(pred_4_users, pop_by_item, group={'u1', 'u2'})

        # Check that the results are correct
        self.assertAlmostEqual(expected_u1, result_all['u1'])
        self.assertAlmostEqual(expected_u2, result_all['u2'])
        # Check that 'u3' and 'u4' are not present
        self.assertNotIn('u3', result_u1_u2.keys())
        self.assertNotIn('u4', result_u1_u2.keys())


class TestDeltaGap(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        original_ratings = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1', 'u1',
                         'u2', 'u2', 'u2', 'u2',
                         'u3', 'u3', 'u3', 'u3', 'u3', 'u3',
                         'u4', 'u4', 'u4', 'u4', 'u4', 'u4', 'u4',
                         'u5', 'u5', 'u5', 'u5', 'u5',
                         'u6', 'u6'],
             'item_id': ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8',
                         'i1', 'i9', 'i10', 'i11',
                         'i1', 'i12', 'i13', 'i3', 'i10', 'i14',
                         'i3', 'i10', 'i15', 'i16', 'i9', 'i17', 'i99',
                         'i10', 'i18', 'i19', 'i20', 'i21',
                         'inew_1', 'inew_2'],
             'score': [5, 4, 4, 1, 2, 3, 3, 1,
                       4, 5, 1, 1,
                       3, 3, 2, 1, 1, 4,
                       4, 4, 5, 5, 3, 3, 3,
                       3, 3, 2, 2, 1,
                       4, 3]})
        cls.original_ratings = Ratings.from_dataframe(original_ratings)

        train = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1', 'u1',  # removed last 2
                         'u2', 'u2', 'u2',  # removed last 1
                         'u3', 'u3', 'u3', 'u3',  # removed last 2
                         'u4', 'u4', 'u4', 'u4', 'u4',  # removed last 2
                         'u5', 'u5', 'u5', 'u5',  # removed last 1
                         'u6'],  # removed last 1
             'item_id': ['i1', 'i2', 'i3', 'i4', 'i5', 'i6',
                         'i1', 'i9', 'i10',
                         'i1', 'i12', 'i13', 'i3',
                         'i3', 'i10', 'i15', 'i16', 'i9',
                         'i10', 'i18', 'i19', 'i20',
                         'inew_1'],
             'score': [5, 4, 4, 1, 2, 3,
                       4, 5, 1,
                       3, 3, 2, 1,
                       4, 4, 5, 5, 3,
                       3, 3, 2, 2,
                       4]})
        cls.train = Ratings.from_dataframe(train)

        truth = pd.DataFrame(
            {'user_id': ['u1', 'u1',
                         'u2',
                         'u3', 'u3',
                         'u4', 'u4', 'u4',
                         'u5',
                         'u6'],
             'item_id': ['i7', 'i8',
                         'i11',
                         'i10', 'i14',
                         'i9', 'i17', 'i99',
                         'i21',
                         'inew_2'],
             'score': [3, 1,
                       1,
                       1, 4,
                       3, 3, 3,
                       1,
                       3]})
        truth = Ratings.from_dataframe(truth)

        # u6 is missing, just to test DeltaGap in case for some users recs can't be computed
        recs = pd.DataFrame(
            {'user_id': ['u1', 'u1', 'u1', 'u1', 'u1',
                         'u2', 'u2', 'u2', 'u2', 'u2',
                         'u3', 'u3', 'u3', 'u3', 'u3',
                         'u4', 'u4', 'u4', 'u4', 'u4',
                         'u5', 'u5', 'u5', 'u5', 'u5'],
             'item_id': ['i7', 'i10', 'i11', 'i12', 'i13',
                         'i11', 'i20', 'i6', 'i3', 'i4',
                         'i4', 'i5', 'i6', 'i7', 'i10',
                         'i9', 'i2', 'i3', 'i1', 'i5',
                         'i2', 'i3', 'i4', 'i5', 'i6'],
             'score': [500, 400, 300, 200, 100,
                       400, 300, 200, 100, 50,
                       150, 125, 110, 100, 80,
                       390, 380, 360, 320, 200,
                       250, 150, 190, 100, 50]})
        recs = Rank.from_dataframe(recs)

        cls.split = Split(recs, truth)

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
        with self.assertRaises(ValueError):
            DeltaGap(user_groups={'a': 0.5}, pop_percentage=-0.5, user_profiles=self.train,
                     original_ratings=self.original_ratings)
            DeltaGap(user_groups={'a': 0.5}, pop_percentage=0, user_profiles=self.train,
                     original_ratings=self.original_ratings)
            DeltaGap(user_groups={'a': 0.5}, pop_percentage=1.5, user_profiles=self.train,
                     original_ratings=self.original_ratings)

    def _compute_deltagap(self, valid_groups_splitted: dict, top_n: int = None):

        n_users = len(set(self.original_ratings.user_id_column))  # { u1, u2, u3, u4, u5, u6 } so 6 users
        pop_by_item = {item_id: count / n_users
                       for item_id, count in Counter(self.original_ratings.item_id_column).items()}

        expected_result_list = []

        for group_name, valid_group in valid_groups_splitted.items():
            valid_group_idx = self.original_ratings.user_map.convert_seq_str2int(list(valid_group))

            # *******************************************************************************************
            # *** For every user in the group calculate the average popularity of the recommendations ***
            # *******************************************************************************************

            # compute for each user of the group the popularity sum in the recommendations
            # (cut the ranking list if top_n != None)
            RECS_sum_pop_group = {
                user_idx: sum([pop_by_item.get(self.split.pred.item_id_column[interaction_idx])
                               for interaction_idx in self.split.pred.get_user_interactions(user_idx,
                                                                                            as_indices=True)][:top_n])

                for user_idx in valid_group_idx
            }

            # compute for each user of the group the average popularity in the recommendations
            # (sum_pop_item_recommended / n_item_recommended)
            # (cut the ranking list if top_n != None)
            RECS_avg_pop_group = {user_idx: sum_pop / len(self.split.pred.get_user_interactions(user_idx)[:top_n])
                                  for user_idx, sum_pop in RECS_sum_pop_group.items()}

            # ************************************************************************************
            # *** For every user in the group calculate the average popularity of the profiles ***
            # ************************************************************************************

            # compute for each user of the group the popularity sum in the recommendations
            PROFILE_sum_pop_group = {
                user_idx: sum([pop_by_item.get(self.train.item_id_column[interaction_idx])
                               for interaction_idx in self.train.get_user_interactions(user_idx, as_indices=True)])

                for user_idx in valid_group_idx
            }

            # compute for each user of the group the average popularity in the recommendations
            # (sum_pop_item_recommended / n_item_recommended)
            PROFILE_avg_pop_group = {user_idx: sum_pop / len(self.train.get_user_interactions(user_idx))
                                     for user_idx, sum_pop in PROFILE_sum_pop_group.items()}

            # ************************
            # *** Compute DeltaGAP ***
            # ************************

            # sum the RECS_avg_pop of every user of the group / n_users in group
            RECS_gap_group = sum(RECS_avg_pop_group.values()) / len(valid_group)

            # sum the PROFILE_avg_pop of every user of the group / n_users in group
            PROFILE_gap_group = sum(PROFILE_avg_pop_group.values()) / len(valid_group)

            expected_delta_gap_group = (RECS_gap_group - PROFILE_gap_group) / PROFILE_gap_group

            expected_result_list.append(expected_delta_gap_group)

        return expected_result_list

    def test_perform_1_group(self):
        metric = DeltaGap(user_groups={'a': 1},
                          user_profiles=self.train,
                          original_ratings=self.original_ratings)
        result = metric.perform(self.split)

        # since one group, all users will belong to this group
        # except u6 for which we don't have any recs
        valid_group_a = {'u1', 'u2', 'u3', 'u4', 'u4', 'u5'}

        valid_groups_splitted = {'a': valid_group_a}

        [expected_delta_gap_group_a] = self._compute_deltagap(valid_groups_splitted)

        result_delta_gap_group_a = float(result["{} | a".format(str(metric))])

        self.assertAlmostEqual(expected_delta_gap_group_a, result_delta_gap_group_a)

    def test_perform_2_groups(self):
        metric = DeltaGap(user_groups={'a': 0.4, 'b': 0.6},
                          user_profiles=self.train,
                          original_ratings=self.original_ratings)
        result = metric.perform(self.split)

        # since u2 and u4 have higher popular ratio, it is put into the first group)
        valid_group_a = {'u2', 'u4'}

        # u6 won't be considered in computation since we don't have recs for it,
        # but it should belong to group b
        valid_group_b = {'u3', 'u1', 'u5'}

        valid_groups_splitted = {'a': valid_group_a, 'b': valid_group_b}

        [expected_delta_gap_group_a, expected_delta_gap_group_b] = self._compute_deltagap(valid_groups_splitted)

        result_delta_gap_group_a = float(result["{} | a".format(str(metric))])
        result_delta_gap_group_b = float(result["{} | b".format(str(metric))])

        self.assertAlmostEqual(expected_delta_gap_group_a, result_delta_gap_group_a)
        self.assertAlmostEqual(expected_delta_gap_group_b, result_delta_gap_group_b)

    def test_perform_3_group_without_recs(self):
        metric = DeltaGap(user_groups={'a': 0.2, 'b': 0.6, 'c': 0.2},
                          user_profiles=self.train,
                          original_ratings=self.original_ratings)
        result = metric.perform(self.split)

        # no 'c' group since u6 belongs to this group but we don't have any recs for it,
        # so it won't be considered and a warning is printed
        valid_groups_splitted = {'a': {'u2'},
                                 'b': {'u4', 'u5', 'u1', 'u3'}}

        [expected_delta_gap_group_a,
         expected_delta_gap_group_b] = self._compute_deltagap(valid_groups_splitted)

        result_delta_gap_group_a = float(result["{} | a".format(str(metric))])
        result_delta_gap_group_b = float(result["{} | b".format(str(metric))])

        self.assertAlmostEqual(expected_delta_gap_group_a, result_delta_gap_group_a)
        self.assertAlmostEqual(expected_delta_gap_group_b, result_delta_gap_group_b)

    def test_perform_0_gap(self):
        # DeltaGap with 2 equals frame should return 0 for every group
        equal_split = Split(self.split.pred, self.split.pred)

        metric = DeltaGap(user_groups={'a': 0.5, 'b': 0.5},
                          user_profiles=self.train,
                          original_ratings=self.original_ratings)

        result = metric.perform(equal_split)

        for col in result.columns:
            self.assertTrue(v == 0 for v in result[col])

    def test_perform_top_3(self):

        top_n = 3

        metric = DeltaGap(user_groups={'a': 0.4, 'b': 0.6},
                          user_profiles=self.train,
                          original_ratings=self.original_ratings,
                          top_n=top_n)
        result = metric.perform(self.split)

        # since u2 and u4 have higher popular ratio, it is put into the first group)
        valid_group_a = {'u2', 'u4'}

        # u6 won't be considered in computation since we don't have recs for it,
        # but it should belong to group b
        valid_group_b = {'u3', 'u1', 'u5'}

        valid_groups_splitted = {'a': valid_group_a, 'b': valid_group_b}

        [expected_delta_gap_group_a, expected_delta_gap_group_b] = self._compute_deltagap(valid_groups_splitted, top_n)

        result_delta_gap_group_a = float(result["{} | a".format(str(metric))])
        result_delta_gap_group_b = float(result["{} | b".format(str(metric))])

        self.assertAlmostEqual(expected_delta_gap_group_a, result_delta_gap_group_a)
        self.assertAlmostEqual(expected_delta_gap_group_b, result_delta_gap_group_b)

    def test_perform_increased_pop_percentage(self):

        result_pop_normal = DeltaGap(user_groups={'a': 0.4, 'b': 0.6},
                                     user_profiles=self.train,
                                     original_ratings=self.original_ratings).perform(self.split)

        result_pop_increased = DeltaGap(user_groups={'a': 0.4, 'b': 0.6},
                                        user_profiles=self.train,
                                        original_ratings=self.original_ratings,
                                        pop_percentage=0.6).perform(self.split)

        result_pop_normal = np.array(result_pop_normal)
        result_pop_increased = np.array(result_pop_increased)

        result_pop_normal.sort(axis=0)
        result_pop_increased.sort(axis=0)

        # Just check that results with pop_percentage increased are different,
        # since users are put into groups differently
        self.assertFalse(np.array_equal(result_pop_normal, result_pop_increased))

    def test_perform_2_groups_2_splits(self):
        metric = DeltaGap(user_groups={'a': 0.4, 'b': 0.6},
                          user_profiles=[self.train, self.train],
                          original_ratings=self.original_ratings)

        # we are simulating two splits
        for _ in range(2):
            result = metric.perform(self.split)

            # since u2 and u4 have higher popular ratio, it is put into the first group)
            valid_group_a = {'u2', 'u4'}

            # u6 won't be considered in computation since we don't have recs for it,
            # but it should belong to group b
            valid_group_b = {'u3', 'u1', 'u5'}

            valid_groups_splitted = {'a': valid_group_a, 'b': valid_group_b}

            [expected_delta_gap_group_a, expected_delta_gap_group_b] = self._compute_deltagap(valid_groups_splitted)

            result_delta_gap_group_a = float(result["{} | a".format(str(metric))])
            result_delta_gap_group_b = float(result["{} | b".format(str(metric))])

            self.assertAlmostEqual(expected_delta_gap_group_a, result_delta_gap_group_a)
            self.assertAlmostEqual(expected_delta_gap_group_b, result_delta_gap_group_b)

    def test_perform_repeated(self):
        metric = DeltaGap(user_groups={'a': 0.4, 'b': 0.6},
                          user_profiles=self.train,
                          original_ratings=self.original_ratings)

        # we are simulating two splits but only one user profile metric given to the metric
        metric.perform(self.split)
        metric.perform(self.split)


if __name__ == '__main__':
    unittest.main()
