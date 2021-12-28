from abc import ABC, abstractmethod
import pandas as pd
from scipy.stats import ttest_ind, ranksums

class StatisticalTest(ABC):
    """
        Abstract class for Statistical Test.

        Every statistical test have to identify common users if the user
        chooses to pass us a df. The method stat_test_results is implemented
        differently for each statistical test you decide to do.

    """

    def _common_users(value_1, value_2, columns1, columns2):
        """
        Method called by the statistical test in case of use of df.
        Common users are searched for meaningful comparison.

        Args:

            value_1 value_2: dataframe on which to identify common users

        Returns: List of value's metric for common users

        """
        dict_user1 = dict(zip(value_1['from_id'], value_1[columns1]))
        dict_user2 = dict(zip(value_2['from_id'], value_2[columns2]))
        common_users = list(set(dict_user1.keys()) & set(dict_user2.keys()))
        common_value_list1 = list(map(dict_user1.get, common_users))
        common_value_list2 = list(map(dict_user2.get, common_users))
        return common_value_list1, common_value_list2

    @abstractmethod

    def stat_test_results(user_score_list:list):
        """
                Abstract method in which must be specified how to calculate Statistical test
        """
        pass

class PairedTtest(StatisticalTest):

    @abstractmethod
    def stat_test_results(user_score_list:list):
        """
                  Abstract method in which must be specified how to calculate Statistical test
        """
        pass

class Ttest(PairedTtest):

    def stat_test_results(user_score_list:list):
        """
        The method performs the Ttest on every possible pair of df or lists.
        In the case of dataframe, it identifies the pairs of dataframe that have the same
        metrics selected, and carries out the tests for each equal metric.

        Args:

            user_score_list: list of numbers or list of dataframe on which to permorm statistical tests

        """
        findSameMetrics=False
        if (isinstance(user_score_list[0], pd.DataFrame)):
            for value_1 in range(0, len(user_score_list)):
                for value_2 in range(value_1 + 1, len(user_score_list)):
                    for columns1 in user_score_list[value_1].columns:
                        for columns2 in user_score_list[value_2].columns:
                            if (columns1 == columns2) and (columns1 != 'from_id') and (columns2 != 'from_id'):
                                findSameMetrics=True
                                first_values, second_values = StatisticalTest._common_users(user_score_list[value_1],
                                                                                            user_score_list[value_2], columns1, columns2)
                                print(
                                    "System:" + repr(value_1+1) + "   System:" + repr(value_2+1) + "  Metric: " + columns1)
                                print(ttest_ind(first_values, second_values))
            if findSameMetrics==False:
                raise TypeError("Method requires same metric to compare")

        elif (isinstance(user_score_list[0], list)):
            for value_1 in range (0, len(user_score_list)):
                for value_2 in range (value_1+1, len(user_score_list)):
                    print("Lista:" + repr(value_1+1) + "   Lista:" + repr(value_2+1) + "  " + repr(
                        ttest_ind(user_score_list[value_2], user_score_list[value_1])))
        else:
            raise TypeError("Method requires list of numbers or dataframe")


class WilconxonTest(PairedTtest):

    def stat_test_results(user_score_list):
        """
            The method performs the Wilcoxon test on every possible pair of df or lists.
            In the case of dataframe, it identifies the pairs of dataframe that have the same
            metrics selected, and carries out the tests for each equal metric.

         Args:

        user_score_list: list of numbers or list of dataframe on which to permorm statistical tests

        """
        findSameMetrics = False
        if (isinstance(user_score_list[0], pd.DataFrame)):
            for value_1 in range(0, len(user_score_list)):
                for value_2 in range(value_1 + 1, len(user_score_list)):
                    for columns1 in user_score_list[value_1].columns:
                        for columns2 in user_score_list[value_2].columns:
                            if (columns1 == columns2) and (columns1 != 'from_id') and (columns2 != 'from_id'):
                                findSameMetrics = True
                                first_values, second_values = StatisticalTest._common_users(user_score_list[value_1],
                                                                                            user_score_list[value_2],
                                                                                            columns1, columns2)
                                print(
                                    "System:" + repr(value_1 + 1) + "   System:" + repr(
                                        value_2 + 1) + "  Metric: " + columns1)
                                print(ttest_ind(first_values, second_values))
            if findSameMetrics == False:
                raise TypeError("Method requires same metric to compare")

        elif (isinstance(user_score_list[0], list)):
            for value_1 in range(0, len(user_score_list)):
                for value_2 in range(value_1 + 1, len(user_score_list)):
                    print("Lista:" + repr(value_1 + 1) + "   Lista:" + repr(value_2 + 1) + "  " + repr(
                    WilconxonTest(user_score_list[value_2], user_score_list[value_1])))

        else:
            raise TypeError("Method requires list of numbers or dataframe")
