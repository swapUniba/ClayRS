import random
from abc import abstractmethod
from typing import List, Dict

import pandas as pd
import numpy as np

from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split
from orange_cb_recsys.evaluation.exceptions import NotEnoughUsers, PercentageError
from orange_cb_recsys.evaluation.metrics.metrics import RankingNeededMetric
from orange_cb_recsys.evaluation.utils import *

from orange_cb_recsys.utils.const import logger


class FairnessMetric(RankingNeededMetric):
    """
    Abstract class that generalize fairness metrics.

    Args:
        file_name (str): name of the file that the metrics will serialize
        out_dir (str): directory in which the file will be serialized
    """

    @abstractmethod
    def perform(self, split: Split):
        """
        Method that execute the fairness metric computation

        Args:
              truth (pd.DataFrame): original rating frame used for recsys config
              predictions (pd.DataFrame): dataframe with recommendations for multiple users
        """
        raise NotImplementedError


class GroupFairnessMetric(FairnessMetric):
    """
    Fairness metrics based on user groups

    Args:
        user_groups (dict<str, float>): specify how to divide user in groups, so
            specify for each group specify:
            - name
            - percentage of users
        file_name (str): name of the file that the metrics will serialize
        out_dir (str): directory in which the file will be serialized
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
    def get_avg_pop_by_users(data: pd.DataFrame, pop_by_items: Counter,
                             group: Set[str] = None) -> Dict[str, float]:
        """
        Get the average popularity for each user in the DataFrame

        Args:
            data (pd.DataFrame): a pandas dataframe with columns = ['from_id', 'to_id', 'rating']
            pop_by_items (Dict<str, object>): popularity for each label ('label', 'popularity')
            group (Set<str>): (optional) the set of users (from_id)

        Returns:
            avg_pop_by_users (Dict<str, float>): average popularity by user
        """

        def show_progress(coll, milestones=10):
            processed = 0
            for element in coll:
                yield element
                processed += 1
                if processed % milestones == 0:
                    logger.info('Processed %s user in the group', processed)

        if group is None:
            group = set(data['from_id'])
        logger.info("Group length: %d", len(group))
        series_by_user = {
            user: data[data.from_id == user].to_id.values.flatten()
            for user in show_progress(group)
        }
        avg_pop_by_users = {
            user: get_avg_pop(series_by_user[user], pop_by_items)
            for user in show_progress(group)
        }

        return avg_pop_by_users

    @staticmethod
    def split_user_in_groups(score_frame: pd.DataFrame, groups: Dict[str, float], pop_items: Set[str]
                             ) -> Dict[str, Set[str]]:
        """
        Splits the DataFrames in 3 different Sets, based on the recommendation popularity of each user

        Args:
            score_frame (pd.DataFrame): DataFrame with columns = ['from_id', 'to_id', 'rating']
            groups (Dict[str, float]): each key contains the name of the group and each value contains the
            percentage of the specified group. If the groups don't cover the entire user collection,
            the rest of the users are considered in a 'default_diverse' group
            pop_items (Set[str]): set of most popular 'to_id' labels

        Returns:
            groups_dict (Dict<str, Set<str>>): key = group_name, value = Set of 'from_id' labels
        """
        num_of_users = len(set(score_frame['from_id']))
        if num_of_users < len(groups):
            raise NotEnoughUsers("You can't split in {} groups {} users! "
                                 "Try reducing number of groups".format(len(groups), num_of_users))

        for percentage_chosen in groups.values():
            if not 0 < percentage_chosen <= 1:
                raise PercentageError('Incorrect percentage! Valid percentage range: 0 < percentage <= 1')
        total = sum(groups.values())
        if total > 1:
            raise PercentageError("Incorrect percentage! Sum of percentage is > than 1")
        elif total < 1:
            logger.warning("Sum of percentage is < than 1, "
                           "the {} percentage of users will be inserted into the "
                           "'default_diverse' group".format(1 - total))

        pop_ratio_by_users = pop_ratio_by_user(score_frame, most_pop_items=pop_items)
        pop_ratio_by_users.sort_values(['popularity_ratio'], inplace=True, ascending=False)

        groups_dict: Dict[str, Set[str]] = {}
        last_index = 0
        percentage = 0.0
        for group_name in groups:
            percentage += groups[group_name]
            group_index = round(num_of_users * percentage)
            if group_index == 0:
                logger.warning('Not enough rows for group {}! It will be discarded'.format(group_name))
            else:
                groups_dict[group_name] = set(pop_ratio_by_users['from_id'][last_index:group_index])
                last_index = group_index
        if percentage < 1:
            group_index = round(num_of_users)
            groups_dict['default_diverse'] = set(pop_ratio_by_users['from_id'][last_index:group_index])
        return groups_dict


class GiniIndex(FairnessMetric):
    """
    Gini index
    
    .. image:: metrics_img/gini.png
    \n\n
    Where:
    - n is the size of the user or item set
    - elem(i) is the user or the item in the i-th position in the sorted frame by user or item
    """

    def __init__(self, top_n: int = None):
        self.__top_n = top_n

    def __str__(self):
        name = "Gini"
        if self.__top_n:
            name += " - Top {}".format(self.__top_n)

        return name

    def perform(self, split: Split):
        """
        Calculate Gini index score for each user or item in the DataFrame

        Args:
              truth (pd.DataFrame): original rating frame used for recsys config
              predictions (pd.DataFrame): dataframe with recommendations for multiple users

        Returns:
            results (pd.DataFrame): each row contains the 'gini_index' for each user or item
        """

        def gini(x: List):
            """
            https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
            """
            # The rest of the code requires numpy arrays.
            x = np.asarray(x)
            sorted_x = np.sort(x)
            n = len(x)
            cumx = np.cumsum(sorted_x, dtype=float)
            # The above formula, with all weights equal to 1 simplifies to:
            return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

        predictions = split.pred

        score_dict = {'from_id': [], str(self): []}

        if self.__top_n is not None:
            predictions = predictions.groupby('from_id').head(self.__top_n)

        coun = Counter(predictions['to_id'])

        result = gini(list(coun.values()))

        score_dict['from_id'].append('sys')
        score_dict[str(self)].append(result)

        return pd.DataFrame(score_dict)


class PredictionCoverage(FairnessMetric):
    """
    Prediction Coverage
    https://www.researchgate.net/publication/221140976_Beyond_accuracy_Evaluating_recommender_systems_by_coverage_and_serendipity

    .. image:: metrics_img/cat_coverage.png
    \n\n
    """

    def __init__(self, catalog: Set[str]):
        self.__catalog = catalog

    def __str__(self):
        return "PredictionCoverage"

    @property
    def catalog(self):
        return self.__catalog

    def _get_covered(self, pred: pd.DataFrame):
        catalog = self.catalog
        return set(pred.query('to_id in @catalog')['to_id'])

    def perform(self, split: Split) -> pd.DataFrame:
        """
        Calculates the catalog coverage

        Args:
              truth (pd.DataFrame): original rating frame used for recsys config
              predictions (pd.DataFrame): dataframe with recommendations for multiple users

        Returns:
            score (float): coverage percentage
        """
        logger.info("Computing catalog coverage")

        prediction = {'from_id': [], str(self): []}
        catalog = self.__catalog

        pred = split.pred

        covered_items = self._get_covered(pred)

        percentage = (len(covered_items) / len(catalog)) * 100
        coverage_percentage = np.round(percentage, 2)

        prediction['from_id'].append('sys')
        prediction[str(self)].append(coverage_percentage)

        return pd.DataFrame(prediction)


class CatalogCoverage(PredictionCoverage):
    """
    https://www.researchgate.net/publication/221140976_Beyond_accuracy_Evaluating_recommender_systems_by_coverage_and_serendipity
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

    def _get_covered(self, pred: pd.DataFrame):
        catalog = self.catalog

        if self.__top_n is not None:
            pred = pred.groupby('from_id').head(self.__top_n)

        # IF k is passed, then we choose randomly k users and calc catalog coverage
        # based on their predictions. We check that k is < n_user since if it's the equal
        # or it's greater, then all predictions generated for all user must be used
        if self.__k is not None and self.__k < len(pred.from_id):
            user_list = list(set(pred.from_id))

            sampling = random.choices(user_list, k=self.__k)
            covered_items = set(pred.query('(from_id in @sampling) and (to_id in @catalog)')['to_id'])
        else:
            covered_items = set(pred.query('to_id in @catalog')['to_id'])

        return covered_items


class DeltaGap(GroupFairnessMetric):
    """
    DeltaGap

    .. image:: metrics_img/d_gap.png
    \n\n
    Args:
        user_groups (dict<str, float>): specify how to divide user in groups, so
            specify for each group:
            - name
            - percentage of users
    """

    def __init__(self, user_groups: Dict[str, float], top_n: int = None, pop_percentage: float = 0.2):
        if not 0 < pop_percentage <= 1:
            raise PercentageError('Incorrect percentage! Valid percentage range: 0 < percentage <= 1')

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


        .. image:: metrics_img/gap.png


        Where:
          • G is the set of users
          • iu is the set of items rated by user u
          • pop_i is the popularity of item i

        Args:
            group (Set<str>): the set of users (from_id)
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
        """
        Compute the Delta - GAP (Group Average Popularity) metric

        Args:
              truth (pd.DataFrame): original rating frame used for recsys config
              predictions (pd.DataFrame): dataframe with recommendations for multiple users

        Returns:
            results (pd.DataFrame): each row contains ('from_id', 'delta-gap')
        """
        predictions = split.pred
        truth = split.truth

        if self.__top_n:
            predictions = predictions.groupby('from_id').head(self.__top_n)

        most_popular_items = popular_items(score_frame=truth, pop_percentage=self.__pop_percentage)
        user_groups = self.split_user_in_groups(score_frame=predictions, groups=self.user_groups,
                                                pop_items=most_popular_items)

        split_result = {"{} | {}".format(str(self), group): []
                        for group in user_groups}
        split_result['from_id'] = ['sys']

        pop_by_items = Counter(list(truth.to_id))

        for group_name in user_groups:
            avg_pop_by_users_recs = self.get_avg_pop_by_users(predictions, pop_by_items, user_groups[group_name])
            logger.info("Computing avg pop by users profiles for delta gap")
            avg_pop_by_users_profiles = self.get_avg_pop_by_users(truth, pop_by_items, user_groups[group_name])
            logger.info("Computing delta gap for group: %s" % group_name)

            recs_gap = self.calculate_gap(group=user_groups[group_name], avg_pop_by_users=avg_pop_by_users_recs)
            profile_gap = self.calculate_gap(group=user_groups[group_name], avg_pop_by_users=avg_pop_by_users_profiles)
            group_delta_gap = self.calculate_delta_gap(recs_gap=recs_gap, profile_gap=profile_gap)

            split_result['{} | {}'.format(str(self), group_name)].append(group_delta_gap)

        return pd.DataFrame(split_result)
