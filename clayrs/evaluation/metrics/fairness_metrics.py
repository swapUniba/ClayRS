from __future__ import annotations
import itertools
import random
from abc import abstractmethod
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from typing import Dict, Set, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from clayrs.content_analyzer import Ratings
    from clayrs.recsys.partitioning import Split

from clayrs.evaluation.metrics.metrics import Metric
from clayrs.evaluation.utils import get_avg_pop, pop_ratio_by_user, get_item_popularity, get_most_popular_items
from clayrs.evaluation.exceptions import NotEnoughUsers
from clayrs.utils.const import logger


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

    It has some concrete methods useful for group divisions, since every subclass needs to split users into groups.

    Args:
        user_groups: Dict containing group names as keys and percentage of users as value, used to
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
    def get_avg_pop_by_users(data: Ratings, pop_by_items: Dict, group: Set[str] = None) -> Dict[str, float]:
        r"""
        Get the average popularity for each user in the `data` parameter.

        Average popularity of a single user $u$ is defined as:

        $$
        avg\_pop_u = \frac{\sum_{i \in i_u} pop_i}{|i_u|}
        $$

        Args:
            data: The `Ratings` object that will be used to compute average popularity of each user
            pop_by_items: popularity for each label ('label', 'popularity')
            group: (optional) the set of users (user_id)

        Returns:
            Python dictionary containing as keys each user id and as values the average popularity of each user
        """
        if group is None:
            group = data.unique_user_id_column
            group_int = data.unique_user_idx_column
        else:
            group_int = data.user_map.convert_seq_str2int(list(group))

        avg_pop_by_users = []

        for user_idx in group_int:
            user_interactions_rows = data.get_user_interactions(user_idx, as_indices=True)
            user_items = data.item_id_column[user_interactions_rows]

            avg_pop_by_users.append(get_avg_pop(user_items, pop_by_items))

        avg_pop_by_users = dict(zip(group, avg_pop_by_users))

        return avg_pop_by_users

    @staticmethod
    def split_user_in_groups(score_frame: Ratings, groups: Dict[str, float],
                             pop_items: Set[str]) -> Dict[str, Set[str]]:
        r"""
        Users are split into groups based on the *groups* parameter, which contains names of the groups as keys,
        and percentage of how many user must contain a group as values. For example:

            groups = {'popular_users': 0.3, 'medium_popular_users': 0.2, 'low_popular_users': 0.5}

        Every user will be inserted in a group based on how many popular items the user has rated (in relation to the
        percentage of users we specified as value in the dictionary):

        * users with many popular items will be inserted into the first group
        * users with niche items rated will be inserted into one of the last groups.

        In general users are grouped by $Popularity\_ratio$ in a descending order. $Popularity\_ratio$ for a
        single user $u$ is defined as:

        $$
        Popularity\_ratio_u = n\_most\_popular\_items\_rated_u / n\_items\_rated_u
        $$

        The *most popular items* are the first `pop_percentage`% items of all items ordered in a descending order by
        popularity.

        The popularity of an item is defined as the number of times it is rated in the `original_ratings` parameter
        divided by the total number of users in the `original_ratings`.

        Args:
            score_frame: the Ratings object
            groups: each key contains the name of the group and each value contains the
                percentage of the specified group. If the groups don't cover the entire user collection,
                the rest of the users are considered in a 'default_diverse' group
            pop_items: set of most popular *item_id* labels

        Returns:
            A python dictionary containing as keys each group name and as values the set of *user_id* belonging to
                the particular group.
        """
        num_of_users = len(score_frame.unique_user_id_column)
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
            raise ValueError("Sum of percentage is < than 1! Please add another group or redistribute percentages "
                             "among already defined group to reach a total of 1!")

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
        return groups_dict


class GiniIndex(FairnessMetric):
    r"""
    The Gini Index metric measures inequality in recommendation lists. It's a system wide metric, so only its
    result it will be returned and not those of every user.
    The metric is calculated as such:

    $$
    Gini_{sys} = \frac{\sum_i(2i - n - 1)x_i}{n\cdot\sum_i x_i}
    $$

    Where:

    - $n$ is the total number of distinct items that are being recommended
    - $x_i$ is the number of times that the item $i$ has been recommended

    A perfectly equal recommender system should recommend every item the same number of times, in which case the Gini
    index would be equal to 0. The more the recsys is "disegual", the more the Gini Index is closer to 1

    If the 'top_n' parameter is specified, then the Gini index will measure inequality considering only the first
    *n* items of every recommendation list of all users

    Args:
        top_n: it's a cutoff parameter, if specified the Gini index will be calculated considering only the first
            'n' items of every recommendation list of all users. Default is None
    """

    def __init__(self, top_n: int = None):
        self.__top_n = top_n

    def __str__(self):
        name = "Gini"
        if self.__top_n:
            name += " - Top {}".format(self.__top_n)

        return name

    def __repr__(self):
        return f'GiniIndex(top_n={self.__top_n})'

    def perform(self, split: Split):
        def gini(x: List):
            """
            Inner method which given a list of values, calculates the gini index

            Args:
                x: list of values of which we want to measure inequality
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

        prediction_items = predictions.item_id_column
        if self.__top_n is not None:

            prediction_items = []

            for user_idx in predictions.unique_user_idx_column:
                user_interactions_indices = predictions.get_user_interactions(user_idx,
                                                                              head=self.__top_n,
                                                                              as_indices=True)

                user_items = predictions.item_id_column[user_interactions_indices]
                prediction_items.append(user_items)

            prediction_items = itertools.chain.from_iterable(prediction_items)

        coun = Counter(prediction_items)

        result = gini(list(coun.values()))

        score_dict['user_id'].append('sys')
        score_dict[str(self)].append(result)

        return pd.DataFrame(score_dict)


class PredictionCoverage(FairnessMetric):
    r"""
    The Prediction Coverage metric measures in percentage how many distinct items are being recommended in relation
    to all available items. It's a system wise metric, so only its result it will be returned and not those of every
    user.
    The metric is calculated as such:

    $$
    Prediction Coverage_{sys} = (\frac{|I_p|}{|I|})\cdot100
    $$

    Where:

    - $I$ is the set of all available items
    - $I_p$ is the set of recommended items

    The $I$ must be specified through the 'catalog' parameter

    Check the 'Beyond Accuracy: Evaluating Recommender Systems by Coverage and Serendipity' paper for more

    Args:
        catalog: set of item id of the catalog on which the prediction coverage must be computed
    """

    def __init__(self, catalog: Set[str]):
        self.__catalog = set(str(item_id) for item_id in catalog)

    def __str__(self):
        return "PredictionCoverage"

    def __repr__(self):
        return f'PredictionCoverage(catalog={self.__catalog})'

    @property
    def catalog(self):
        return self.__catalog

    def _get_covered(self, pred: Ratings):
        """
        Private function which calculates all recommended items given a catalog of all available items (specified in
        the constructor)

        Args:
            pred: Ratings object containing recommendation lists of all users

        Returns:
            Set of distinct items that have been recommended that also appear in the catalog
        """
        pred_items = set(pred.unique_item_id_column)
        return pred_items.intersection(self.catalog)

    def perform(self, split: Split) -> pd.DataFrame:
        prediction = {'user_id': [], str(self): []}

        pred = split.pred

        covered_items = self._get_covered(pred)

        percentage = (len(covered_items) / len(self.__catalog)) * 100
        coverage_percentage = np.round(percentage, 2)

        prediction['user_id'].append('sys')
        prediction[str(self)].append(coverage_percentage)

        return pd.DataFrame(prediction)


class CatalogCoverage(PredictionCoverage):
    r"""
    The Catalog Coverage metric measures in percentage how many distinct items are being recommended in relation
    to all available items. It's a system wide metric, so only its result it will be returned and not those of every
    user. It differs from the Prediction Coverage since it allows for different parameters to come into play. If no
    parameter is passed then it's a simple Prediction Coverage.
    The metric is calculated as such:

    $$
    Catalog Coverage_{sys} = (\frac{|\bigcup_{j=1...N}reclist(u_j)|}{|I|})\cdot100
    $$

    Where:

    - $N$ is the total number of users
    - $reclist(u_j)$ is the set of items contained in the recommendation list of user $j$
    - $I$ is the set of all available items

    The $I$ must be specified through the 'catalog' parameter

    The recommendation list of every user ($reclist(u_j)$) can be reduced to the first *n* parameter with the
    top-n parameter, so that catalog coverage is measured considering only the most highest ranked items.

    With the 'k' parameter one could specify the number of users that will be used to calculate catalog coverage:
    k users will be randomly sampled and their recommendation lists will be used. The formula above becomes:

    $$
    Catalog Coverage_{sys} = (\frac{|\bigcup_{j=1...k}reclist(u_j)|}{|I|})\cdot100
    $$

    Where:

    - $k$ is the parameter specified

    Obviously 'k' < N, else simply recommendation lists of all users will be used

    Check the 'Beyond Accuracy: Evaluating Recommender Systems  by Coverage and Serendipity' paper and
    page 13 of the 'Comparison of group recommendation algorithms' paper for more

    Args:
        catalog: set of item id of the catalog on which the prediction coverage must be computed
        top_n: it's a cutoff parameter, if specified the Catalog Coverage will be calculated considering only the first
            'n' items of every recommendation list of all users. Default is None
        k: number of users randomly sampled. If specified, k users will be randomly sampled across all users and only
            their recommendation lists will be used to compute the CatalogCoverage
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

    def __repr__(self):
        return f'CatalogCoverage(catalog={self.catalog}, top_n={self.__top_n}, k={self.__k})'

    def _get_covered(self, pred: Ratings):

        # IF k is passed, then we choose randomly k users and calc catalog coverage
        # based on their predictions. We check that k is < n_user since if it's the equal
        # or it's greater, then all predictions generated for all user must be used
        user_list = pred.unique_user_idx_column
        if self.__k is not None and self.__k < len(pred.unique_user_id_column):

            user_list = random.choices(user_list, k=self.__k)

        prediction_items = []

        for user_idx in user_list:
            user_interactions_indices = pred.get_user_interactions(user_idx,
                                                                   head=self.__top_n,
                                                                   as_indices=True)

            user_items = pred.item_id_column[user_interactions_indices]
            prediction_items.append(user_items)

        prediction_items = list(itertools.chain.from_iterable(prediction_items))
        covered_items = set(prediction_items).intersection(self.catalog)

        return covered_items


class DeltaGap(GroupFairnessMetric):
    r"""
    The Delta GAP (Group Average popularity) metric lets you compare the average popularity "requested" by one or
    multiple groups of users and the average popularity "obtained" with the recommendation given by the recsys.
    It's a system wise metric and results of every group will be returned.

    It is calculated as such:

    $$
    \Delta GAP = \frac{recs_GAP - profile_GAP}{profile_GAP}
    $$

    Users are split into groups based on the *user_groups* parameter, which contains names of the groups as keys,
    and percentage of how many user must contain a group as values. For example:

        user_groups = {'popular_users': 0.3, 'medium_popular_users': 0.2, 'low_popular_users': 0.5}

    Every user will be inserted in a group based on how many popular items the user has rated (in relation to the
    percentage of users we specified as value in the dictionary):

    * users with many popular items will be inserted into the first group
    * users with niche items rated will be inserted into one of the last groups.

    In general users are grouped by $Popularity\_ratio$ in a descending order. $Popularity\_ratio$ for a single user $u$
    is defined as:

    $$
    Popularity\_ratio_u = n\_most\_popular\_items\_rated_u / n\_items\_rated_u
    $$

    The *most popular items* are the first `pop_percentage`% items of all items ordered in a descending order by
    popularity.

    The popularity of an item is defined as the number of times it is rated in the `original_ratings` parameter
    divided by the total number of users in the `original_ratings`.

    It can happen that for a particular user of a group no recommendation are available: in that case it will be skipped
    and it won't be considered in the $\Delta GAP$ computation of its group. In case no user of a group has recs
    available, a warning will be printed and the whole group won't be considered.

    If the 'top_n' parameter is specified, then the $\Delta GAP$ will be calculated considering only the first
    *n* items of every recommendation list of all users

    Args:
        user_groups: Dict containing group names as keys and percentage of users as value, used to
            split users in groups. Users with more popular items rated are grouped into the first group, users with
            slightly less popular items rated are grouped into the second one, etc.
        user_profiles: one or more `Ratings` objects containing interactions of the profile of each user
            (e.g. the **train set**). It should be one for each split to evaluate!
        original_ratings: `Ratings` object containing original interactions of the dataset that will be used to
            compute the popularity of each item (i.e. the number of times it is rated divided by the total number of
            users)
        top_n: it's a cutoff parameter, if specified the Gini index will be calculated considering only their first
            'n' items of every recommendation list of all users. Default is None
        pop_percentage: How many (in percentage) *most popular items* must be considered. Default is 0.2
    """

    def __init__(self, user_groups: Dict[str, float], user_profiles: Union[list, Ratings], original_ratings: Ratings,
                 top_n: int = None, pop_percentage: float = 0.2):
        if not 0 < pop_percentage <= 1:
            raise ValueError('Incorrect percentage! Valid percentage range: 0 < percentage <= 1')

        super().__init__(user_groups)
        self._pop_by_item = get_item_popularity(original_ratings)

        if not isinstance(user_profiles, list):
            user_profiles = [user_profiles]
        self._user_profiles = user_profiles
        self.__top_n = top_n
        self._pop_percentage = pop_percentage

    def __str__(self):
        name = "DeltaGap"
        if self.__top_n:
            name += " - Top {}".format(self.__top_n)
        return name

    # not a complete repr, better understand how to manage cases with 'ratings' repr
    def __repr__(self):
        return f"DeltaGap(user_groups={self.user_groups}, top_n={self.__top_n}, pop_percentage={self._pop_percentage})"

    @staticmethod
    def calculate_gap(group: Set[str], avg_pop_by_users: Dict[str, object]) -> float:
        r"""
        Compute the GAP (Group Average Popularity) formula

        $$
        GAP = \frac{\sum_{u \in U}\cdot \frac{\sum_{i \in i_u} pop_i}{|i_u|}}{|G|}
        $$

        Where:

        - $G$ is the set of users
        - $i_u$ is the set of items rated/recommended by/to user $u$
        - $pop_i$ is the popularity of item i

        Args:
            group: the set of users (user_id)
            avg_pop_by_users: average popularity by user

        Returns:
            score (float): gap score
        """
        total_pop = 0
        for user in group:
            if avg_pop_by_users.get(user):
                total_pop += avg_pop_by_users[user]
        return total_pop / len(group)

    @staticmethod
    def calculate_delta_gap(recs_gap: float, profile_gap: float) -> float:
        """
        Compute the ratio between the recommendation gap and the user profiles gap

        Args:
            recs_gap: recommendation gap
            profile_gap: user profiles gap

        Returns:
            score: delta gap measure
        """
        result = 0
        if profile_gap != 0.0:
            result = (recs_gap - profile_gap) / profile_gap

        return result

    def perform(self, split: Split) -> pd.DataFrame:

        # in order to point to the right `user_profile` set each time the
        # `perform()` method is called, we pop the list but add the `user_profile` set
        # back at the end so that DeltaGap is ready for another evaluation without
        # need to instantiate it again
        split_user_profile = self._user_profiles.pop(0)
        self._user_profiles.append(split_user_profile)

        predictions = split.pred

        if self.__top_n:
            predictions = predictions.take_head_all(self.__top_n)

        most_pop_items = get_most_popular_items(self._pop_by_item, self._pop_percentage)
        splitted_user_groups = self.split_user_in_groups(score_frame=split_user_profile, groups=self.user_groups,
                                                         pop_items=most_pop_items)

        split_result = defaultdict(list)
        split_result['user_id'] = ['sys']

        for group_name in splitted_user_groups:

            # we don't consider users of the group for which we do not have any recommendation
            valid_group = splitted_user_groups[group_name].intersection(set(predictions.unique_user_id_column))

            if len(valid_group) == 0:
                logger.warning(f"Group {group_name} won't be considered in the DeltaGap since no recs is available "
                               f"for any user of said group!")
                continue

            # Computing avg pop by users recs for delta gap
            avg_pop_by_users_recs = self.get_avg_pop_by_users(predictions, self._pop_by_item, valid_group)
            # Computing avg pop by users profiles for delta gap
            avg_pop_by_users_profiles = self.get_avg_pop_by_users(split_user_profile, self._pop_by_item, valid_group)

            # Computing delta gap for every group
            recs_gap = self.calculate_gap(group=valid_group, avg_pop_by_users=avg_pop_by_users_recs)
            profile_gap = self.calculate_gap(group=valid_group, avg_pop_by_users=avg_pop_by_users_profiles)
            group_delta_gap = self.calculate_delta_gap(recs_gap=recs_gap, profile_gap=profile_gap)

            split_result['{} | {}'.format(str(self), group_name)].append(group_delta_gap)

        return pd.DataFrame(split_result)
