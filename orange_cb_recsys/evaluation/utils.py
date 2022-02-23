from typing import Set, Dict, List
import pandas as pd
from collections import Counter

from orange_cb_recsys.content_analyzer import Ratings


def popular_items(score_frame: Ratings, pop_percentage: float = 0.2) -> Set[str]:
    """
    Find a set of most popular items ('to_id's)

    Args:
        score_frame (pd.DataFrame): each row contains index(the rank position), label, value predicted
        pop_percentage (float): percentage of how many 'most popular items' must be returned

    Returns:
        Set<str>: set of most popular labels
    """
    items = score_frame.item_id_column

    ratings_counter = Counter(items)

    num_of_items = len(ratings_counter.keys())
    top_n_percentage = pop_percentage
    top_n_index = round(num_of_items * top_n_percentage)

    most_common = ratings_counter.most_common(top_n_index)

    # removing counts from most_common
    return set(map(lambda x: x[0], most_common))


def pop_ratio_by_user(score_frame: Ratings, most_pop_items: Set[str]) -> Dict:
    """
    Perform the popularity ratio for each user
    Args:
        score_frame (pd.DataFrame): each row contains index(the rank position), label, value predicted
        most_pop_items (Set[str]): set of most popular 'to_id' labels

    Returns:
        (pd.DataFrame): contains the 'popularity_ratio' for each 'from_id' (user)
    """
    # Splitting users by popularity
    users = set(score_frame.user_id_column)

    popularity_ratio_by_user = {}

    for user in users:
        # filters by the current user and returns all the items he has rated
        user_ratings = score_frame.get_user_interactions(user)
        rated_items = set([user_interaction.item_id for user_interaction in user_ratings])
        # intersects rated_items with popular_items
        popular_rated_items = rated_items.intersection(most_pop_items)
        popularity_ratio = len(popular_rated_items) / len(rated_items)

        popularity_ratio_by_user[user] = popularity_ratio

    return popularity_ratio_by_user


def get_avg_pop(items: List, pop_by_items: Counter) -> float:
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
