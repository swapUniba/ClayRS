import unittest
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ranksums

from clayrs.evaluation.statistical_test import StatisticalTest, Ttest, Wilcoxon, PairedTest


class TestStatisticalTest(unittest.TestCase):
    def test__common_users(self):
        # all_users are in common
        equal_users_sys1 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                         'NDCG': [0.5, 0.7, 0.8, 0.2],
                                         'Precision - micro': [0.998, 0.123, 0.556, 0.887]})

        equal_users_sys2 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                         'NDCG': [0.8, 0.2, 0.2, 0.4],
                                         'Precision - micro': [0.45, 0.23, 0.112, 0.776]})

        result = StatisticalTest._common_users(equal_users_sys1, "user_id", equal_users_sys2, "user_id",
                                               column_list=['NDCG'])

        expected_dict = {'NDCG_x': [0.5, 0.7, 0.8, 0.2],
                         'user_id': ['u1', 'u2', 'u3', 'u4'],
                         'NDCG_y': [0.8, 0.2, 0.2, 0.4]}
        result_dict = result.to_dict(orient='list')

        self.assertDictEqual(expected_dict, result_dict)

        # not all user are in common
        equal_users_sys1 = pd.DataFrame({'user_id': ['u5', 'u6', 'u3', 'u4'],
                                         'NDCG': [0.5, 0.7, 0.8, 0.2],
                                         'Precision - micro': [0.998, 0.123, 0.556, 0.887]})

        equal_users_sys2 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                         'NDCG': [0.8, 0.2, 0.2, 0.4],
                                         'Precision - micro': [0.45, 0.23, 0.112, 0.776]})

        result = StatisticalTest._common_users(equal_users_sys1, "user_id", equal_users_sys2,
                                               "user_id", column_list=['NDCG'])

        expected_dict = {'NDCG_x': [0.8, 0.2],
                         'user_id': ['u3', 'u4'],
                         'NDCG_y': [0.2, 0.4]}
        result_dict = result.to_dict(orient='list')

        self.assertDictEqual(expected_dict, result_dict)

        # no metric in common
        equal_users_sys1 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                         'MRR': [0.5, 0.7, 0.8, 0.2],
                                         'Precision - macro': [0.998, 0.123, 0.556, 0.887]})

        equal_users_sys2 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                         'NDCG': [0.8, 0.2, 0.2, 0.4],
                                         'Precision - micro': [0.45, 0.23, 0.112, 0.776]})

        result = StatisticalTest._common_users(equal_users_sys1, "user_id", equal_users_sys2, "user_id",
                                               column_list=[])

        expected_dict = {'user_id': ['u1', 'u2', 'u3', 'u4']}
        result_dict = result.to_dict(orient='list')

        self.assertDictEqual(expected_dict, result_dict)

        # no metric in common and no user in common
        equal_users_sys1 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                         'MRR': [0.5, 0.7, 0.8, 0.2],
                                         'Precision - macro': [0.998, 0.123, 0.556, 0.887]})

        equal_users_sys2 = pd.DataFrame({'user_id': ['u5', 'u6', 'u7', 'u8'],
                                         'NDCG': [0.8, 0.2, 0.2, 0.4],
                                         'Precision - micro': [0.45, 0.23, 0.112, 0.776]})

        result = StatisticalTest._common_users(equal_users_sys1, "user_id", equal_users_sys2, "user_id",
                                               column_list=[])

        expected_dict = {'user_id': []}
        result_dict = result.to_dict(orient='list')

        self.assertDictEqual(expected_dict, result_dict)


class TestPairedTest(unittest.TestCase):
    """
    Contains all methods on which a paired test must be tested.
    The class_to_test parameter is basically the class of the framework deriving PairedTest that
    we want to test
    The 'external_function' parameter is the function used to implement the class, usually taken from an external
    library
    EXAMPLE:
            class_to_test = Ttest()
            external_function = scipy.stats.ttest_ind
    """

    def perform_all_in_common(self, class_to_test, external_function):
        # perform all user and all columns in common
        users_metrics_result1 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                              'NDCG': [0.5, 0.7, 0.8, 0.2],
                                              'Precision - micro': [0.998, 0.123, 0.556, 0.887]})

        users_metrics_result2 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                              'NDCG': [0.8, 0.2, 0.2, 0.4],
                                              'Precision - micro': [0.45, 0.23, 0.112, 0.776]})

        result_df = class_to_test.perform([users_metrics_result1, users_metrics_result2])

        self.assertCountEqual([('NDCG', 'statistic'), ('NDCG', 'pvalue'),
                               ('Precision - micro', 'statistic'), ('Precision - micro', 'pvalue')],

                         list(result_df.columns))

        result_ndcg = tuple(result_df['NDCG'].values[0])

        result_precision = tuple(result_df['Precision - micro'].values[0])

        expected_ndcg = external_function([0.5, 0.7, 0.8, 0.2], [0.8, 0.2, 0.2, 0.4])
        expected_precision = external_function([0.998, 0.123, 0.556, 0.887], [0.45, 0.23, 0.112, 0.776])

        self.assertEqual(expected_ndcg, result_ndcg)
        self.assertEqual(expected_precision, result_precision)

    def perform_3_systems(self, class_to_test, external_function):
        # perform all user and all columns in common
        users_metrics_result1 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                              'NDCG': [0.5, 0.7, 0.8, 0.2],
                                              'Precision - micro': [0.998, 0.123, 0.556, 0.887]})

        users_metrics_result2 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                              'NDCG': [0.8, 0.2, 0.2, 0.4],
                                              'Precision - micro': [0.45, 0.23, 0.112, 0.776]})

        users_metrics_result3 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                              'NDCG': [0.9, 0.3, 0.2, 0.5],
                                              'Precision - micro': [0.44, 0.88, 0.21, 0.56]})

        result_df = class_to_test.perform([users_metrics_result1, users_metrics_result2, users_metrics_result3])

        self.assertCountEqual([('NDCG', 'statistic'), ('NDCG', 'pvalue'),
                               ('Precision - micro', 'statistic'), ('Precision - micro', 'pvalue')],

                              list(result_df.columns))

        # we expect sys1/sys2, sys1/sys3, sys2/sys3
        self.assertTrue(len(result_df) == 3)

        result_ndcg_12 = tuple(result_df['NDCG'].values[0])
        result_ndcg_13 = tuple(result_df['NDCG'].values[1])
        result_ndcg_23 = tuple(result_df['NDCG'].values[2])

        result_precision_12 = tuple(result_df['Precision - micro'].values[0])
        result_precision_13 = tuple(result_df['Precision - micro'].values[1])
        result_precision_23 = tuple(result_df['Precision - micro'].values[2])

        expected_ndcg_12 = external_function([0.5, 0.7, 0.8, 0.2], [0.8, 0.2, 0.2, 0.4])
        expected_ndcg_13 = external_function([0.5, 0.7, 0.8, 0.2], [0.9, 0.3, 0.2, 0.5])
        expected_ndcg_23 = external_function([0.8, 0.2, 0.2, 0.4], [0.9, 0.3, 0.2, 0.5])
        expected_precision_12 = external_function([0.998, 0.123, 0.556, 0.887], [0.45, 0.23, 0.112, 0.776])
        expected_precision_13 = external_function([0.998, 0.123, 0.556, 0.887], [0.44, 0.88, 0.21, 0.56])
        expected_precision_23 = external_function([0.45, 0.23, 0.112, 0.776], [0.44, 0.88, 0.21, 0.56])

        self.assertEqual(expected_ndcg_12, result_ndcg_12)
        self.assertEqual(expected_ndcg_13, result_ndcg_13)
        self.assertEqual(expected_ndcg_23, result_ndcg_23)
        self.assertEqual(expected_precision_12, result_precision_12)
        self.assertEqual(expected_precision_13, result_precision_13)
        self.assertEqual(expected_precision_23, result_precision_23)

        # perform all users in common but not all metrics
        users_metrics_result1 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                              'NDCG': [0.5, 0.7, 0.8, 0.2],
                                              'Precision - micro': [0.998, 0.123, 0.556, 0.887]})

        users_metrics_result2 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                              'NDCG': [0.8, 0.2, 0.2, 0.4],
                                              'Recall - micro': [0.45, 0.23, 0.112, 0.776]})

        users_metrics_result3 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                              'MRR': [0.9, 0.3, 0.2, 0.5],
                                              'Recall - micro': [0.44, 0.88, 0.21, 0.56]})

        result_df = class_to_test.perform([users_metrics_result1, users_metrics_result2, users_metrics_result3])

        self.assertCountEqual([('NDCG', 'statistic'), ('NDCG', 'pvalue'),
                               ('Recall - micro', 'statistic'), ('Recall - micro', 'pvalue')],

                              list(result_df.columns))

        # we expect sys1/sys2, sys2/sys3 (sys1/sys3 missing since no metric is in common)
        self.assertTrue(len(result_df) == 2)

        # sys1/sys2 do not have in common recall (so nan value is expected)
        self.assertTrue(np.isnan(result_df['Recall - micro']["statistic"].values[0]))
        self.assertTrue(np.isnan(result_df['Recall - micro']["pvalue"].values[0]))

        # sys2/sys3 do not have in common ndcg (so nan value is expected)
        self.assertTrue(np.isnan(result_df["NDCG"]["statistic"].values[1]))
        self.assertTrue(np.isnan(result_df["NDCG"]["pvalue"].values[1]))

    def perform_only_some_in_common(self, class_to_test, external_function):
        # perform only some users in common
        users_metrics_result1 = pd.DataFrame({'user_id': ['u5', 'u6', 'u3', 'u4'],
                                              'NDCG': [0.5, 0.7, 0.8, 0.2],
                                              'Precision - micro': [0.998, 0.123, 0.556, 0.887]})

        users_metrics_result2 = pd.DataFrame({'user_id': ['u1', 'u3', 'u2', 'u4'],
                                              'NDCG': [0.8, 0.2, 0.2, 0.4],
                                              'Precision - micro': [0.45, 0.23, 0.112, 0.776]})

        result_df = class_to_test.perform([users_metrics_result1, users_metrics_result2])

        self.assertCountEqual([('NDCG', 'statistic'), ('NDCG', 'pvalue'),
                               ('Precision - micro', 'statistic'), ('Precision - micro', 'pvalue')],

                         list(result_df.columns))

        result_ndcg = tuple(result_df['NDCG'].values[0])
        result_precision = tuple(result_df['Precision - micro'].values[0])

        # we consider values only from user in common
        expected_ndcg = external_function([0.8, 0.2], [0.2, 0.4])
        expected_precision = external_function([0.556, 0.887], [0.23, 0.776])

        self.assertEqual(expected_ndcg, result_ndcg)
        self.assertEqual(expected_precision, result_precision)

        # perform only some users and only some columns in common
        users_metrics_result1 = pd.DataFrame({'user_id': ['u5', 'u6', 'u3', 'u4'],
                                              'NDCG': [0.5, 0.7, 0.8, 0.2],
                                              'Precision - micro': [0.998, 0.123, 0.556, 0.887]})

        users_metrics_result2 = pd.DataFrame({'user_id': ['u1', 'u3', 'u2', 'u4'],
                                              'MRR': [0.8, 0.2, 0.2, 0.4],
                                              'NDCG': [0.45, 0.23, 0.112, 0.776]})

        result_df = class_to_test.perform([users_metrics_result1, users_metrics_result2])

        self.assertCountEqual([('NDCG', 'statistic'), ('NDCG', 'pvalue')],

                         list(result_df.columns))

        result_ndcg = tuple(result_df['NDCG'].values[0])

        expected_ndcg = external_function([0.8, 0.2], [0.23, 0.776])

        self.assertEqual(expected_ndcg, result_ndcg)
        self.assertEqual(expected_precision, result_precision)

    def perform_nothing_in_common(self, class_to_test, external_function):
        # perform columns in common but no user in common
        users_metrics_result1 = pd.DataFrame({'user_id': ['u5', 'u6', 'u8', 'u9'],
                                              'NDCG': [0.5, 0.7, 0.8, 0.2],
                                              'Precision - micro': [0.998, 0.123, 0.556, 0.887]})

        users_metrics_result2 = pd.DataFrame({'user_id': ['u1', 'u3', 'u2', 'u4'],
                                              'NDCG': [0.8, 0.2, 0.2, 0.4],
                                              'Precision - micro': [0.45, 0.23, 0.112, 0.776]})

        result_df = class_to_test.perform([users_metrics_result1, users_metrics_result2])

        self.assertCountEqual([('NDCG', 'statistic'), ('NDCG', 'pvalue'),
                               ('Precision - micro', 'statistic'), ('Precision - micro', 'pvalue')],

                         list(result_df.columns))

        result_ndcg = tuple(result_df['NDCG'].values[0])
        result_precision = tuple(result_df['Precision - micro'].values[0])

        self.assertTrue(np.isnan(result_ndcg[0]))
        self.assertTrue(np.isnan(result_ndcg[1]))
        self.assertTrue(np.isnan(result_precision[0]))
        self.assertTrue(np.isnan(result_precision[1]))

        # perform users in common but not columns in common
        users_metrics_result1 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                              'NDCG': [0.5, 0.7, 0.8, 0.2],
                                              'Precision - micro': [0.998, 0.123, 0.556, 0.887]})

        users_metrics_result2 = pd.DataFrame({'user_id': ['u1', 'u3', 'u2', 'u4'],
                                              'MRR': [0.8, 0.2, 0.2, 0.4],
                                              'Recall': [0.45, 0.23, 0.112, 0.776]})

        result_df = class_to_test.perform([users_metrics_result1, users_metrics_result2])

        self.assertTrue(result_df.empty)

    def perform_nan_present(self, class_to_test, external_function):
        # u1 has nan values in NDCG column
        users_metrics_result1 = pd.DataFrame({'user_id': ['u1', 'u3', 'u2', 'u4'],
                                              'NDCG': [np.nan, 0.7, 0.8, 0.2],
                                              'Precision - micro': [0.998, 0.123, 0.556, 0.887]})

        users_metrics_result2 = pd.DataFrame({'user_id': ['u1', 'u3', 'u2', 'u4'],
                                              'NDCG': [np.nan, 0.2, 0.2, 0.4],
                                              'Precision - micro': [0.45, 0.23, 0.112, 0.776]})

        result_df = class_to_test.perform([users_metrics_result1, users_metrics_result2])

        self.assertCountEqual([('NDCG', 'statistic'), ('NDCG', 'pvalue'),
                               ('Precision - micro', 'statistic'), ('Precision - micro', 'pvalue')],

                         list(result_df.columns))

        # ndcg column has one nan values, we don't use it
        expected_ndcg = external_function([0.7, 0.8, 0.2], [0.2, 0.2, 0.4])
        # other metric columns are unaffected
        expected_precision = external_function([0.998, 0.123, 0.556, 0.887], [0.45, 0.23, 0.112, 0.776])

        result_ndcg = tuple(result_df['NDCG'].values[0])
        result_precision = tuple(result_df['Precision - micro'].values[0])

        self.assertEqual(expected_ndcg, result_ndcg)
        self.assertEqual(expected_precision, result_precision)

        # all users have nan values in NDCG column
        users_metrics_result1 = pd.DataFrame({'user_id': ['u1', 'u3', 'u2', 'u4'],
                                              'NDCG': [np.nan, np.nan, np.nan, np.nan],
                                              'Precision - micro': [0.998, 0.123, 0.556, 0.887]})

        users_metrics_result2 = pd.DataFrame({'user_id': ['u1', 'u3', 'u2', 'u4'],
                                              'NDCG': [np.nan, np.nan, np.nan, np.nan],
                                              'Precision - micro': [0.45, 0.23, 0.112, 0.776]})

        result_df = class_to_test.perform([users_metrics_result1, users_metrics_result2])

        self.assertCountEqual([('NDCG', 'statistic'), ('NDCG', 'pvalue'),
                               ('Precision - micro', 'statistic'), ('Precision - micro', 'pvalue')],

                         list(result_df.columns))

        # NDCG columns has all nan values, so we can't compute the statistic
        result_ndcg = tuple(result_df['NDCG'].values[0])

        self.assertTrue(np.isnan(result_ndcg[0]))
        self.assertTrue(np.isnan(result_ndcg[1]))

        # other metric columns are unaffected
        expected_precision = external_function([0.998, 0.123, 0.556, 0.887], [0.45, 0.23, 0.112, 0.776])
        result_precision = tuple(result_df['Precision - micro'].values[0])

        self.assertEqual(expected_precision, result_precision)

    def test_check_col_specified(self):

        # perform by default column name is "user_id", so no necessary to specify it
        stat_test = Ttest()
        stat_test_col_specified = Ttest("user_id")

        users_metrics_result1 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                              'NDCG': [0.5, 0.7, 0.8, 0.2],
                                              'Precision - micro': [0.998, 0.123, 0.556, 0.887]})

        users_metrics_result2 = pd.DataFrame({'user_id': ['u1', 'u2', 'u3', 'u4'],
                                              'NDCG': [0.8, 0.2, 0.2, 0.4],
                                              'Precision - micro': [0.45, 0.23, 0.112, 0.776]})

        expected = stat_test.perform([users_metrics_result1, users_metrics_result2])
        result = stat_test_col_specified.perform([users_metrics_result1, users_metrics_result2])

        self.assertEqual(tuple(expected["NDCG"].values[0]), tuple(result["NDCG"].values[0]))
        self.assertEqual(tuple(expected["Precision - micro"].values[0]),
                         tuple(result["Precision - micro"].values[0]))

        # the two dataframes have different column name for "user_id"

        users_metrics_result2.columns = ["user_idx", "NDCG", "Precision - micro"]

        with self.assertRaises(KeyError):
            stat_test.perform([users_metrics_result1, users_metrics_result2])

        # but if we specify it we are safe
        stat_test_col_specified = Ttest("user_id", "user_idx")

        result = stat_test_col_specified.perform([users_metrics_result1, users_metrics_result2])

        self.assertEqual(tuple(expected["NDCG"].values[0]), tuple(result["NDCG"].values[0]))
        self.assertEqual(tuple(expected["Precision - micro"].values[0]),
                         tuple(result["Precision - micro"].values[0]))

        # case in which the user columns specified do not match len of df_list
        stat_test_col_specified = Ttest("user_id", "user_idx", "user_idxx")

        with self.assertRaises(ValueError):
            stat_test_col_specified.perform([users_metrics_result1, users_metrics_result2])



class TestTtest(TestPairedTest):

    def test_perform_all_in_common(self):
        self.perform_all_in_common(Ttest(), ttest_ind)

    def test_perform_only_some_in_common(self):
        self.perform_only_some_in_common(Ttest(), ttest_ind)

    def test_perform_nothing_in_common(self):
        self.perform_nothing_in_common(Ttest(), ttest_ind)

    def test_perform_3_systems(self):
        self.perform_3_systems(Ttest(), ttest_ind)

    def test_perform_nan_present(self):
        self.perform_nan_present(Ttest(), ttest_ind)


class TestWilcoxon(TestPairedTest):

    def test_perform_all_in_common(self):
        self.perform_all_in_common(Wilcoxon(), ranksums)

    def test_perform_only_some_in_common(self):
        self.perform_only_some_in_common(Wilcoxon(), ranksums)

    def test_perform_nothing_in_common(self):
        self.perform_nothing_in_common(Wilcoxon(), ranksums)

    def test_perform_3_systems(self):
        self.perform_3_systems(Wilcoxon(), ranksums)

    def test_perform_nan_present(self):
        self.perform_nan_present(Wilcoxon(), ranksums)


if __name__ == '__main__':
    unittest.main()
