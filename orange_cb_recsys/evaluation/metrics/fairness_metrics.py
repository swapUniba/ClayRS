import itertools
import random
from abc import abstractmethod
from typing import List, Dict

import pandas as pd
import numpy as np

from orange_cb_recsys.content_analyzer import Ratings
from orange_cb_recsys.evaluation.metrics.metrics import Metric
from orange_cb_recsys.recsys.partitioning import Split
from orange_cb_recsys.evaluation.exceptions import NotEnoughUsers
from orange_cb_recsys.evaluation.utils import *

from orange_cb_recsys.utils.const import logger


class FairnessMetric(Metric):
    """
    Abstract class that generalize fairness metrics
    """

    @abstractmethod
    def perform(self, split: Split):
        raise NotImplementedError


class GroupFairnessMetric(FairnessMetric):
    """
    Abstract class for fairness metrics based on user groups

    It has some concrete methods useful for group divisions, since every subclass needs to split users into groups:

    Users are splitted into groups based on the *user_groups* parameter, which contains names of the groups as keys,
    and percentage of how many user must contain a group as values. For example::

        user_groups = {'popular_users': 0.3, 'medium_popular_users': 0.2, 'low_popular_users': 0.5}

    Every user will be inserted in a group based on how many popular items the user has rated (in relation to the
    percentage of users we specified as value in the dictionary):
    users with many popular items will be inserted into the first group, users with niche items rated will be inserted
    into one of the last groups

    Args:
        user_groups (Dict<str, float>): Dict containing group names as keys and percentage of users as value, used to
            split users in groups. Users with more popular items rated are grouped into the first group, users with
            slightly less popular items rated are grouped into the second one, etc.
    """

    def __init__(self, user_groups: Dict[str, float]):
        self.__user_groups = user_groups

    @property
    def user_groups(self):
        return self.__user_groups

    @abstractmethod
    def perform(self, split: Split):
        raise NotImplementedError

    @staticmethod
    def get_avg_pop_by_users(data: Ratings, pop_by_items: Counter,
                             group: Set[str] = None) -> Dict[str, float]:
        """
        Get the average popularity for each user in the DataFrame

        Args:
            data (pd.DataFrame): a pandas dataframe with columns = ['user_id', 'to_id', 'rating']
            pop_by_items (Dict<str, object>): popularity for each label ('label', 'popularity')
            group (Set<str>): (optional) the set of users (user_id)

        Returns:
            avg_pop_by_users (Dict<str, float>): average popularity by user
        """
        if group is None:
            group = set(data.user_id_column)

        list_by_user = {
            user: [interaction.item_id for interaction in data.get_user_interactions(user)]
            for user in group
        }
        avg_pop_by_users = {
            user: get_avg_pop(list_by_user[user], pop_by_items)
            for user in group
        }

        return avg_pop_by_users

    @staticmethod
    def split_user_in_groups(score_frame: Ratings, groups: Dict[str, float], pop_items: Set[str]
                             ) -> Dict[str, Set[str]]:
        """
        Splits the DataFrames in 3 different Sets, based on the recommendation popularity of each user

        Args:
            score_frame (pd.DataFrame): DataFrame with columns = ['user_id', 'to_id', 'rating']
            groups (Dict[str, float]): each key contains the name of the group and each value contains the
            percentage of the specified group. If the groups don't cover the entire user collection,
            the rest of the users are considered in a 'default_diverse' group
            pop_items (Set[str]): set of most popular 'to_id' labels

        Returns:
            groups_dict (Dict<str, Set<str>>): key = group_name, value = Set of 'user_id' labels
        """
        num_of_users = len(set(score_frame.user_id_column))
        if num_of_users < len(groups):
            raise NotEnoughUsers("You can't split in {} groups {} users! "
                                 "Try reducing number of groups".format(len(groups), num_of_users))

        for percentage_chosen in groups.values():
            if not 0 < percentage_chosen <= 1:
                raise ValueError('Incorrect percentage! Valid percentage range: 0 < percentage <= 1')
        total = sum(groups.values())
        if total > 1:
            raise ValueError("Incorrect percentage! Sum of percentage is > than 1")
        elif total < 1:
            remaining = round(1 - total, 10)  # rounded at the 10th digit
            logger.warning("Sum of percentage is < than 1, "
                           f"the {remaining} percentage of users will be inserted into the "
                           "'default_diverse' group")

        pop_ratio_by_users = pop_ratio_by_user(score_frame, most_pop_items=pop_items)
        pop_ratio_by_users = sorted(pop_ratio_by_users, key=pop_ratio_by_users.get, reverse=True)

        groups_dict: Dict[str, Set[str]] = {}
        last_index = 0
        percentage = 0.0
        for group_name in groups:
            percentage += groups[group_name]
            group_index = round(num_of_users * percentage)
            if group_index == 0:
                logger.warning('Not enough rows for group {}! It will be discarded'.format(group_name))
            else:
                groups_dict[group_name] = set(pop_ratio_by_users[last_index:group_index])
                last_index = group_index
        if percentage < 1:
            group_index = round(num_of_users)
            groups_dict['default_diverse'] = set(pop_ratio_by_users[last_index:group_index])
        return groups_dict


class GiniIndex(FairnessMetric):
    """
    The Gini Index metric measures inequality in recommendation lists. It's a system wide metric, so only its
    result it will be returned and not those of every user.
    The metric is calculated as such:

    .. math:: Gini_sys = \\frac{\sum_i(2i - n - 1)x_i}{n\cdot\sum_i x_i}
    |
    Where:

    - :math:`n` is the total number of distinct items that are being recommended
    - :math:`x_i` is the number of times that the item :math:`i` has been recommended

    A perfectly equal recommender system should recommend every item the same number of times, in which case the Gini
    index would be equal to 0. The more the recsys is "disegual", the more the Gini Index is closer to 1

    If the 'top_n' parameter is specified, then the Gini index will measure inequality considering only the first
    *n* items of every recommendation list of all users

    Args:
        top_n (int): it's a cutoff parameter, if specified the Gini index will be calculated considering only ther first
            'n' items of every recommendation list of all users. Default is None
    """

    def __init__(self, top_n: int = None):
        self.__top_n = top_n

    def __str__(self):
        name = "Gini"
        if self.__top_n:
            name += " - Top {}".format(self.__top_n)

        return name

    def perform(self, split: Split):
        def gini(x: List):
            """
            Inner method which given a list of values, calculates the gini index

            Args:
                x (list): list of values of which we want to measure inequality
            """
            # The rest of the code requires numpy arrays.
            x = np.asarray(x)
            sorted_x = np.sort(x)
            n = len(x)
            cumx = np.cumsum(sorted_x, dtype=float)
            # The above formula, with all weights equal to 1 simplifies to:
            return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

        predictions = split.pred

        score_dict = {'user_id': [], str(self): []}

        if self.__top_n is not None:
            predictions = itertools.chain.from_iterable([predictions.get_user_interactions(user_id, self.__top_n)
                                                         for user_id in set(predictions.user_id_column)])

        coun = Counter([prediction.item_id for prediction in predictions])

        result = gini(list(coun.values()))

        score_dict['user_id'].append('sys')
        score_dict[str(self)].append(result)

        return pd.DataFrame(score_dict)


class PredictionCoverage(FairnessMetric):
    """
    The Prediction Coverage metric measures in percentage how many distinct items are being recommended in relation
    to all available items. It's a system wide metric, so only its result it will be returned and not those of every
    user.
    The metric is calculated as such:

    .. math:: Prediction Coverage_sys = (\\frac{|I_p|}{|I|})\cdot100
    |
    Where:

    - :math:`I` is the set of all available items
    - :math:`I_p` is the set of recommended items

    The :math:`I` must be specified through the 'catalog' parameter

    Check the 'Beyond Accuracy: Evaluating Recommender Systems  by Coverage and Serendipity' paper for more
    """

    def __init__(self, catalog: Set[str]):
        self.__catalog = catalog

    def __str__(self):
        return "PredictionCoverage"

    @property
    def catalog(self):
        return self.__catalog

    def _get_covered(self, pred: Ratings):
        """
        Private function which calculates all recommended items given a catalog of all available items (specified in
        the constructor)

        Args:
            pred (pd.DataFrame): DataFrame containing recommendation lists of all users

        Returns:
            Set of distinct items that have been recommended that also appear in the catalog
        """
        catalog = self.catalog
        pred_items = set(pred.item_id_column)
        return pred_items.intersection(catalog)

    def perform(self, split: Split) -> pd.DataFrame:
        prediction = {'user_id': [], str(self): []}
        catalog = self.__catalog

        pred = split.pred

        covered_items = self._get_covered(pred)

        percentage = (len(covered_items) / len(catalog)) * 100
        coverage_percentage = np.round(percentage, 2)

        prediction['user_id'].append('sys')
        prediction[str(self)].append(coverage_percentage)

        return pd.DataFrame(prediction)


class CatalogCoverage(PredictionCoverage):
    """
    The Catalog Coverage metric measures in percentage how many distinct items are being recommended in relation
    to all available items. It's a system wide metric, so only its result it will be returned and not those of every
    user. It differs from the Prediction Coverage since it allows for different parameters to come into play. If no
    parameter is passed then it's a simple Prediction Coverage.
    The metric is calculated as such:

    .. math:: Catalog Coverage_sys = (\\frac{|\\bigcup_{j=1...N}reclist(u_j)|}{|I|})\cdot100
    |
    Where:

    - :math:`N` is the total number of users
    - :math:`reclist(u_j)` is the set of items contained in the recommendation list of user :math:`j`
    - :math:`I` is the set of all available items

    The :math:`I` must be specified through the 'catalog' parameter

    The recommendation list of every user (:math:`reclist(u_j)`) can be reduced to the first *n* parameter with the
    top-n parameter, so that catalog coverage is measured considering only the most highest ranked items.

    With the 'k' parameter one could specify the number of users that will be used to calculate catalog coverage:
    k users will be randomly sampled and their recommendation lists will be used. The formula above becomes:

    .. math:: Catalog Coverage_sys = (\\frac{|\\bigcup_{j=1...k}reclist(u_j)|}{|I|})\cdot100
    |
    Where:

    - :math:`k` is the parameter specified

    Obviously 'k' < N, else simply recommendation lists of all users will be used

    Check the 'Beyond Accuracy: Evaluating Recommender Systems  by Coverage and Serendipity' paper and
    page 13 of the 'Comparison of group recommendation algorithms' paper for more
    """

    def __init__(self, catalog: Set[str], top_n: int = None, k: int = None):
        super().__init__(catalog)
        self.__top_n = top_n
        self.__k = k

    def __str__(self):
        # If none of the parameter is passed, then it's simply a PredictionCoverage
        name = "CatalogCoverage (PredictionCov)"

        if self.__top_n:
            name = "CatalogCoverage"
            name += " - Top {}".format(self.__top_n)
        if self.__k:
            name = "CatalogCoverage"
            name += " - {} sampled users".format(self.__k)

        return name

    def _get_covered(self, pred: Ratings):
        catalog = self.catalog

        if self.__top_n is not None:
            pred = list(itertools.chain.from_iterable([pred.get_user_interactions(user_id, self.__top_n)
                                                       for user_id in set(pred.user_id_column)]))

        # IF k is passed, then we choose randomly k users and calc catalog coverage
        # based on their predictions. We check that k is < n_user since if it's the equal
        # or it's greater, then all predictions generated for all user must be used
        if self.__k is not None and self.__k < len(pred):
            user_list = list(set([interaction_pred.user_id for interaction_pred in pred]))

            sampling = random.choices(user_list, k=self.__k)
            predicted_items = set([interaction_pred.item_id
                                   for interaction_pred in pred if interaction_pred.user_id in sampling])
            covered_items = predicted_items.intersection(catalog)
        else:
            predicted_items = set([interaction_pred.item_id for interaction_pred in pred])
            covered_items = predicted_items.intersection(catalog)

        return covered_items


class DeltaGap(GroupFairnessMetric):
    """
    The Delta GAP (Group Average popularity) metric lets you compare the average popularity "requested" by one or
    multiple groups of users and the average popularity "obtained" with the recommendation given by the recsys.
    It's a system wide metric and results of every group will be returned.

    It is calculated as such:

    .. math:: \Delta GAP = \\frac{recs_GAP - profile_GAP}{profile_GAP}

    Users are splitted into groups based on the *user_groups* parameter, which contains names of the groups as keys,
    and percentage of how many user must contain a group as values. For example::

        user_groups = {'popular_users': 0.3, 'medium_popular_users': 0.2, 'low_popular_users': 0.5}

    Every user will be inserted in a group based on how many popular items the user has rated (in relation to the
    percentage of users we specified as value in the dictionary):
    users with many popular items will be inserted into the first group, users with niche items rated will be inserted
    into one of the last groups

    If the 'top_n' parameter is specified, then the Delta GAP will be calculated considering only the first
    *n* items of every recommendation list of all users



    Args:
        user_groups (Dict<str, float>): Dict containing group names as keys and percentage of users as value, used to
            split users in groups. Users with more popular items rated are grouped into the first group, users with
            slightly less popular items rated are grouped into the second one, etc.
        top_n (int): it's a cutoff parameter, if specified the Gini index will be calculated considering only ther first
            'n' items of every recommendation list of all users. Default is None
        pop_percentage (float): How many (in percentage) 'most popular items' must be considered. Default is 0.2
    """

    def __init__(self, user_groups: Dict[str, float], top_n: int = None, pop_percentage: float = 0.2):
        if not 0 < pop_percentage <= 1:
            raise ValueError('Incorrect percentage! Valid percentage range: 0 < percentage <= 1')

        self.__pop_percentage = pop_percentage
        self.__top_n = top_n
        super().__init__(user_groups)

    def __str__(self):
        name = "DeltaGap"
        if self.__top_n:
            name += " - Top {}".format(self.__top_n)
        return name

    @staticmethod
    def calculate_gap(group: Set[str], avg_pop_by_users: Dict[str, object]) -> float:
        """
        Compute the GAP (Group Average Popularity) formula


        .. math:: GAP = \\frac{\sum_{u \in U}\cdot \\frac{\sum_{i \in iu} pop_i}{|iu|}}{|G|}

        Where:

        - G is the set of users
        - iu is the set of items rated by user u
        - pop_i is the popularity of item i

        Args:
            group (Set<str>): the set of users (user_id)
            avg_pop_by_users (Dict<str, object>): average popularity by user

        Returns:
            score (float): gap score
        """
        total_pop = 0
        for element in group:
            if avg_pop_by_users.get(element):
                total_pop += avg_pop_by_users[element]
        return total_pop / len(group)

    @staticmethod
    def calculate_delta_gap(recs_gap: float, profile_gap: float) -> float:
        """
        Compute the ratio between the recommendation gap and the user profiles gap

        Args:
            recs_gap (float): recommendation gap
            profile_gap: user profiles gap

        Returns:
            score (float): delta gap measure
        """
        result = 0
        if profile_gap != 0.0:
            result = (recs_gap - profile_gap) / profile_gap

        return result

    def perform(self, split: Split) -> pd.DataFrame:
        predictions = split.pred
        truth = split.truth

        if self.__top_n:
            predictions = predictions.take_head_all(self.__top_n)

        most_popular_items = popular_items(score_frame=truth, pop_percentage=self.__pop_percentage)
        user_groups = self.split_user_in_groups(score_frame=predictions, groups=self.user_groups,
                                                pop_items=most_popular_items)

        split_result = {"{} | {}".format(str(self), group): []
                        for group in user_groups}
        split_result['user_id'] = ['sys']

        pop_by_items = Counter(list(truth.item_id_column))

        for group_name in user_groups:
            # Computing avg pop by users recs for delta gap
            avg_pop_by_users_recs = self.get_avg_pop_by_users(predictions, pop_by_items, user_groups[group_name])
            # Computing avg pop by users profiles for delta gap
            avg_pop_by_users_profiles = self.get_avg_pop_by_users(truth, pop_by_items, user_groups[group_name])

            # Computing delta gap for every group
            recs_gap = self.calculate_gap(group=user_groups[group_name], avg_pop_by_users=avg_pop_by_users_recs)
            profile_gap = self.calculate_gap(group=user_groups[group_name], avg_pop_by_users=avg_pop_by_users_profiles)
            group_delta_gap = self.calculate_delta_gap(recs_gap=recs_gap, profile_gap=profile_gap)

            split_result['{} | {}'.format(str(self), group_name)].append(group_delta_gap)

        return pd.DataFrame(split_result)
