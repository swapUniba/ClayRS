from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List

import pandas as pd
from scipy.stats import ttest_ind, ranksums


class StatisticalTest(ABC):
    """
        Abstract class for Statistical Test.

        Every statistical test have to identify common users if the user
        chooses to pass us a df. The method stat_test_results is implemented
        differently for each statistical test you decide to do.

    """
    @staticmethod
    def _common_users(df1, df2, column_list) -> pd.DataFrame:
        """
        Method called by the statistical test in case of use of df.
        Common users are searched for meaningful comparison.
        """
        # we need to also extract the user_id column, that's why we append 'user_id'
        column_list.append('user_id')

        df1 = df1[column_list]
        df2 = df2[column_list]

        common_rows = pd.merge(df1, df2, how="inner", on=['user_id'])

        return common_rows

    @abstractmethod
    def perform(self, users_metric_results: list):
        """
        Abstract method in which must be specified how to calculate Statistical test
        """
        pass


class PairedTest(StatisticalTest):

    def perform(self, df_list: List[pd.DataFrame]):

        final_result = defaultdict(list)

        n_system_evaluated = 1
        while len(df_list) != 0:
            df1 = df_list.pop(0)
            for i, other_df in enumerate(df_list, start=n_system_evaluated + 1):
                common_metrics = [column for column in df1.columns
                                  if column != 'user_id' and column in other_df.columns]

                common_rows = self._common_users(df1, other_df, list(common_metrics))

                final_result["Systems evaluated"].append((f"system_{n_system_evaluated}", f"system_{i}"))
                for metric in common_metrics:

                    # drop nan values since otherwise test may behave unexpectedly
                    metric_rows = common_rows[[f"{metric}_x", f"{metric}_y"]].dropna()

                    score_system1 = metric_rows[f"{metric}_x"]
                    score_system2 = metric_rows[f"{metric}_y"]

                    single_metric_result = self._perform_test(score_system1, score_system2)
                    final_result[metric].append(single_metric_result)

            n_system_evaluated += 1

        return pd.DataFrame(final_result).set_index("Systems evaluated")

    @abstractmethod
    def _perform_test(self, score_metric_system1: list, score_metric_system2: list):
        """
        Abstract method in which must be specified how to calculate the paired statistical test
        """
        raise NotImplementedError


class Ttest(PairedTest):

    def _perform_test(self, score_metric_system1: list, score_metric_system2: list):
        return ttest_ind(score_metric_system1, score_metric_system2)


class Wilcoxon(PairedTest):

    def _perform_test(self, score_metric_system1: list, score_metric_system2: list):
        return ranksums(score_metric_system1, score_metric_system2)
