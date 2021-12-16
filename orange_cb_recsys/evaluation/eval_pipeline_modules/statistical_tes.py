"""
      Contains Statical functions for recommender systems.
    - T-test
    - Wilcoxon
"""

from scipy.stats import ttest_ind, ranksums
import numpy as np


class StatisticalTest(object):

    def __init__(self, sample1, sample2):
        """
        Compares 2 list of sample and generate a statical analyse
        :param sample1: List of results of a recommender 1
        :param sample2: List of results of a recommender 2
        """
        self.sample1 = np.array(sample1)
        self.sample2 = np.array(sample2)


    def ttest(self):
        """
        T-student

        Calculates the T-test for the means of TWO INDEPENDENT samples of scores.

        This is a two-sided test for the null hypothesis that 2 independent samples have identical
        average (expected) values

        """

        t, p = ttest_ind(self.sample1, self.sample2)
        print("=== T- Student Analysis ===")
        print("The calculated t-statistic: " + str(t))
        print("The two-tailed p-value: " + str(p) + "\n")

    def wilcoxon(self):
        """
        Wilcoxon

        The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from
        the same distribution. In particular, it tests whether the distribution of the differences x - y
        is symmetric about zero. It is a non-parametric version of the paired T-test.
        """

        t, p = ranksums(self.sample1, self.sample2)
        print("=== Wilcoxon Analysis ===")
        print("The calculated t-statistic: " + str(t))
        print("The two-tailed p-value: " + str(p) + "\n")