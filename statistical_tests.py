from scipy.stats import ttest_ind, ranksums
import numpy as np
import pandas as pd


class pairedtTest:
    def t_test(score_list1, score_list2):
        """
                T-student

                Calculates the T-test for the means of TWO INDEPENDENT samples of scores.

                This is a two-sided test for the null hypothesis that 2 independent samples have identical
                average (expected) values

                This test assumes that the populations have identical variances.
                """
        t, p = ttest_ind(score_list1, score_list2)
        print("=== T- Student Analysis ===")
        print("The calculated t-statistic: " + str(t))
        print("The two-tailed p-value: " + str(p) + "\n")

    #@staticmethod
    def get_score(first_prediction: pd.DataFrame, second_prediction: pd.DataFrame):
        """
        The get_score method takes two dataframes from two different recommender systems
        and creates two lists of predictions to compare with the paired t-test
        """
        score_list1 = []
        score_list2 = []

        score_list1 = list(first_prediction["score"])
        score_list2 = list(second_prediction["score"])
        print(score_list1)
        print(score_list2)

        pairedtTest.t_test(score_list1, score_list2)


class wilcoxonTest:
    def wilcoxon(score_list1, score_list2):
        """
                Wilcoxon

                The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from
                the same distribution. In particular, it tests whether the distribution of the differences x - y
                is symmetric about zero. It is a non-parametric version of the paired T-test.
                """

        t, p = ranksums(score_list1, score_list2)
        print("=== Wilcoxon Analysis ===")
        print("The calculated t-statistic: " + str(t))
        print("The two-tailed p-value: " + str(p) + "\n")

    @staticmethod
    def get_score(first_prediction: pd.DataFrame, second_prediction: pd.DataFrame):
        """
              The get_score method takes two dataframes from two different recommender systems
              and creates two lists of predictions to compare with the Wilcoxon test
              """
        score_list1 = list(first_prediction["score"])
        score_list2 = list(second_prediction["score"])
        print(score_list1)
        print(score_list2)
        wilcoxonTest.wilcoxon(score_list1, score_list2)
