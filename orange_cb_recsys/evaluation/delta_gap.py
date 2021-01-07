from typing import Dict, Set

import pandas as pd

from orange_cb_recsys.utils.const import logger


def get_avg_pop(items: pd.Series, pop_by_items: Dict[str, object]) -> float:
    """
    Get the average popularity of the given items Series

    Args:
        items (pd.Series): a pandas Series that contains string labels ('label')
        pop_by_items (Dict<str, object>): popularity for each label ('label', 'popularity')

    Returns:
        score (float): average popularity
    """

    popularities = [pop_by_items[item] for item in items]

    return sum(popularities) / len(items)


def get_avg_pop_by_users(data: pd.DataFrame, pop_by_items: Dict[str, object],
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
        group = data[['from_id']].values.flatten()
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


# pop_by_items = Counter(group['item_id'].to_numpy())
# It calculates the Group Average Popularity(GAP)
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
        try:
            total_pop += avg_pop_by_users[element]
        except KeyError:
            pass
    return total_pop / len(group)


def calculate_delta_gap(recs_gap: float, profile_gap: float) -> float:
    """
    Compute the ratio between the recommendation gap and the user profiles gap

    Args:
        recs_gap (float): recommendation gap
        profile_gap: user profiles gap

    Returns:
        score (float): delta gap measure
    """
    if profile_gap == 0.0:
        return 0.0
    return (recs_gap - profile_gap) / profile_gap
