from __future__ import annotations
import itertools
from typing import Set, Dict, List, TYPE_CHECKING
from collections import Counter

if TYPE_CHECKING:
    from clayrs.content_analyzer import Ratings


def get_item_popularity(original_ratings: Ratings) -> Dict:
    """
    Compute item popularity for each item_id in the `original_ratings` parameter

    The popularity of an item is defined as the number of times it is rated in the `original_ratings` parameter
    divided by the total number of users in the `original_ratings`.

    Args:
        original_ratings: `Ratings` object used to compute popularity for each item

    Returns:
        Python dictionary containing popularity computed for each item in the `original_ratings` parameter
    """
    n_users = len(original_ratings.unique_user_id_column)

    pop_by_item = {item_id: count / n_users for item_id, count in Counter(original_ratings.item_id_column).items()}

    return pop_by_item


def get_most_popular_items(pop_by_item: Dict, top_n_percentage: float = 0.2) -> Set[str]:
    """
    Find the set of *most popular items*, which are the first `top_n_percentage`% items of all items ordered in a
    descending order by popularity.

    Args:
        pop_by_item: Python dictionary containing popularity computed for each item in the `original_ratings` parameter
        top_n_percentage: How many (in percentage) *most popular items* must be considered. Default is 0.2

    Returns:
        Set of most popular item_ids
    """
    num_of_items = len(pop_by_item)
    top_n_index = round(num_of_items * top_n_percentage)

    sorted_popularities = dict(sorted(pop_by_item.items(), key=lambda item: item[1], reverse=True))
    most_common = dict(itertools.islice(sorted_popularities.items(), top_n_index))

    # only item ids
    most_popular_items = set(most_common.keys())

    return most_popular_items


def pop_ratio_by_user(score_frame: Ratings, most_pop_items: Set[str]) -> Dict:
    r"""
    Compute popularity ratio for each user. $Popularity\_ratio$ for a single user $u$ is defined as:

    $$
    Popularity\_ratio_u = n\_most\_popular\_items\_rated_u / n\_items\_rated_u
    $$

    Args:
        score_frame: `Ratings` object used to compute popularity ratio for each user
        most_pop_items: set of most popular 'item_id' labels

    Returns:
        Python dictionary containing as keys each user id and as value the popularity ratio of each user
    """
    # Splitting users by popularity
    users_idxs = score_frame.unique_user_idx_column

    popularity_ratios = []

    for user_idx in users_idxs:
        # filters by the current user and returns all the items he has rated
        user_ratings_idxs = score_frame.get_user_interactions(user_idx, as_indices=True)
        rated_items = set(score_frame.item_id_column[user_ratings_idxs])
        # intersects rated_items with popular_items
        popular_rated_items = rated_items.intersection(most_pop_items)
        popularity_ratio = len(popular_rated_items) / len(rated_items)

        popularity_ratios.append(popularity_ratio)

    popularity_ratio_by_user = dict(zip(score_frame.unique_user_id_column, popularity_ratios))

    return popularity_ratio_by_user


def get_avg_pop(items: List, pop_by_item: Dict) -> float:
    """
    Get the average popularity of the given items list

    Args:
        items: A list containing *item ids*
        pop_by_item: Python dictionary containing popularity computed for each item in the `original_ratings` parameter

    Returns:
        The average popularity of the input `items` list
    """

    popularities = [pop_by_item.get(item, 0.0) for item in items]

    return sum(popularities) / len(items)
