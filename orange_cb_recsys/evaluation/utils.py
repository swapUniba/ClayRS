from typing import Set
import pandas as pd
from collections import Counter


def popular_items(score_frame: pd.DataFrame, pop_percentage: float = 0.2) -> Set[str]:
    """
    Find a set of most popular items ('to_id's)

    Args:
        score_frame (pd.DataFrame): each row contains index(the rank position), label, value predicted

    Returns:
        Set<str>: set of most popular labels
    """
    items = score_frame[['to_id']].values.flatten()

    ratings_counter = Counter(items)

    num_of_items = len(ratings_counter.keys())
    top_n_percentage = pop_percentage
    top_n_index = round(num_of_items * top_n_percentage)

    most_common = ratings_counter.most_common(top_n_index)

    # removing counts from most_common
    return set(map(lambda x: x[0], most_common))


def pop_ratio_by_user(score_frame: pd.DataFrame, most_pop_items: Set[str]) -> pd.DataFrame:
    """
    Perform the popularity ratio for each user
    Args:
        score_frame (pd.DataFrame): each row contains index(the rank position), label, value predicted
        most_pop_items (Set[str]): set of most popular 'to_id' labels

    Returns:
        (pd.DataFrame): contains the 'popularity_ratio' for each 'from_id' (user)
    """
    # Splitting users by popularity
    users = set(score_frame[['from_id']].values.flatten())

    popularity_ratio_by_user = {}

    for user in users:
        # filters by the current user and returns all the items he has rated
        rated_items = set(score_frame.query('from_id == @user')[['to_id']].values.flatten())
        # intersects rated_items with popular_items
        popular_rated_items = rated_items.intersection(most_pop_items)
        popularity_ratio = len(popular_rated_items) / len(rated_items)

        popularity_ratio_by_user[user] = popularity_ratio
    return pd.DataFrame.from_dict({'from_id': list(popularity_ratio_by_user.keys()),
                                   'popularity_ratio': list(popularity_ratio_by_user.values())})


def get_avg_pop(items: pd.Series, pop_by_items: Counter) -> float:
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
