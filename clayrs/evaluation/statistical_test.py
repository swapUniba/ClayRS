from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ranksums


class StatisticalTest(ABC):
    """
    Abstract class for Statistical Test.

    Every statistical test have to identify common users if the user
    chooses to pass us a df. The method stat_test_results is implemented
    differently for each statistical test you decide to do.
    """

    def __init__(self, *user_id_column):
        self.user_id_column = list(user_id_column) if user_id_column else None

    @staticmethod
    def _common_users(df1, user_id_df1, df2, user_id_df2, column_list) -> pd.DataFrame:
        """
        Method called by the statistical test in case of use of df.
        Common users are searched for meaningful comparison.
        """

        column_list_df1 = set(column_list)
        column_list_df1.add(user_id_df1)
        column_list_df2 = set(column_list)
        column_list_df2.add(user_id_df2)

        df1 = df1[column_list_df1]
        df2 = df2[column_list_df2]

        common_rows = pd.merge(df1, df2, how="inner", left_on=user_id_df1, right_on=user_id_df2)

        return common_rows

    @abstractmethod
    def perform(self, users_metric_results: list) -> pd.DataFrame:
        """
        Abstract method in which must be specified how to calculate Statistical test
        """
        pass


class PairedTest(StatisticalTest):

    def perform(self, df_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Method which performs the chosen paired statistical test.

        Since it's a paired test, the final result is a pandas DataFrame which contains learning
        schemas compared in pair.
        For example if you call the `perform()` method by passing a list containing three different DataFrames, one for
        each learning schema to compare:

        ```python
        # Ttest as example since it's a Paired Test
        Ttest().perform([user_df1, user_df2, user_df3])
        ```

        You will obtain a DataFrame comparing all different combinations:

        * (system1, system2)
        * (system1, system3)
        * (system2, system3)

        The first value of each cell is the ***statistic***, the second is the ***p-value***

        Args:
            df_list: List containing DataFrames with several metrics to compare, preferably metrics computed for each
                user. One DataFrame corresponds to one learning schema

        Returns:
            A Pandas DataFrame where each combination of learning schemas are compared in pair. The first value of each
                cell is the ***statistic***, the second is the ***p-value***
        """

        user_id_column = self.user_id_column
        if user_id_column is None:
            user_id_column = ["user_id" for _ in range(len(df_list))]
        elif len(user_id_column) == 1:
            user_id_column = [self.user_id_column[0] for _ in range(len(df_list))]
        elif len(user_id_column) != len(df_list):
            raise ValueError("You must either specify a single user_id column for all dataframes or a different "
                             "user_id column for each DataFrame passed!")

        # if the user id column is the index we consider it as an additional column
        df_list_reset = []
        for user_id_col, df in zip(user_id_column, df_list):
            if user_id_col not in df.columns:
                if user_id_col == df.index.name:
                    df = df.reset_index()
                else:
                    raise KeyError(f"Column {user_id_col} not present neither in the columns nor as index!")

            df_list_reset.append(df)

        data = defaultdict(lambda: defaultdict(dict))

        # this will contain metrics that are in common at least by two system
        global_metrics = set()

        n_system_evaluated = 1
        while len(df_list_reset) != 0:
            df1 = df_list_reset.pop(0)
            user_id_col1 = user_id_column.pop(0)
            for i, (user_id_col2, other_df) in enumerate(zip(user_id_column, df_list_reset),
                                                         start=n_system_evaluated + 1):

                common_metrics = [column for column in df1.columns
                                  if column != user_id_col1 and column in other_df.columns]

                common_rows = self._common_users(df1, user_id_col1, other_df, user_id_col2, common_metrics)

                for metric in common_metrics:

                    global_metrics.add(metric)
                    # drop nan values since otherwise test may behave unexpectedly
                    metric_rows = common_rows[[f"{metric}_x", f"{metric}_y"]].dropna()

                    score_system1 = metric_rows[f"{metric}_x"]
                    score_system2 = metric_rows[f"{metric}_y"]

                    statistic, pvalue = self._perform_test(score_system1, score_system2)

                    data[(f"system_{n_system_evaluated}", f"system_{i}")][str(metric)]["statistic"] = statistic
                    data[(f"system_{n_system_evaluated}", f"system_{i}")][str(metric)]["pvalue"] = pvalue

            n_system_evaluated += 1

        # index will contain (sys1, sys2), (sys1, sys3), ... tuples
        # formatted_data will be in the form
        # {(NDCG, "statistic"): [..., ..., ..., ...], (NDCG, "pvalue"): [..., ..., ..., ...],
        #  (Precision, "statistic"): [..., ..., ..., ...], (Precision, "pvalue"): [..., ..., ..., ...],
        #  ...}
        index = []
        formatted_data = defaultdict(list)
        for df_index_row, df_data_row in data.items():
            index.append(df_index_row)

            for metric_column in global_metrics:
                stat, pval = np.nan, np.nan

                stat_pval_dict = df_data_row.get(metric_column)
                if stat_pval_dict is not None:
                    stat = stat_pval_dict["statistic"]
                    pval = stat_pval_dict["pvalue"]

                formatted_data[(metric_column, "statistic")].append(stat)
                formatted_data[(metric_column, "pvalue")].append(pval)

        res = pd.DataFrame(formatted_data, index=index)
        res.index.rename("sys_pair", inplace=True)

        return res

    @abstractmethod
    def _perform_test(self, score_metric_system1: list, score_metric_system2: list):
        """
        Abstract method in which must be specified how to calculate the paired statistical test
        """
        raise NotImplementedError


class Ttest(PairedTest):
    """
    Calculate the T-test for the means of *two independent* samples of scores.

    This is a two-sided test for the null hypothesis that 2 independent samples
    have identical average (expected) values. This test assumes that the
    populations have identical variances by default.
    """
    def _perform_test(self, score_metric_system1: list, score_metric_system2: list):
        return ttest_ind(score_metric_system1, score_metric_system2)


class Wilcoxon(PairedTest):
    """
    Compute the Wilcoxon rank-sum statistic for two samples.

    The Wilcoxon rank-sum test tests the null hypothesis that two sets
    of measurements are drawn from the same distribution. The alternative
    hypothesis is that values in one sample are more likely to be
    larger than the values in the other sample.
    """
    def _perform_test(self, score_metric_system1: list, score_metric_system2: list):
        return ranksums(score_metric_system1, score_metric_system2)
