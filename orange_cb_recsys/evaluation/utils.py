from typing import Set, Dict
import pandas as pd
from collections import Counter
import numpy as np


def popular_items(score_frame: pd.DataFrame) -> Set[str]:
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
    top_n_percentage = 0.2
    top_n_index = round(num_of_items * top_n_percentage)

    # a plot could be produced
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
        # interesects rated_items with popular_items
        popular_rated_items = rated_items.intersection(most_pop_items)
        popularity_ratio = len(popular_rated_items) / len(rated_items)

        popularity_ratio_by_user[user] = popularity_ratio
    return pd.DataFrame.from_dict({'from_id': list(popularity_ratio_by_user.keys()),
                                   'popularity_ratio': list(popularity_ratio_by_user.values())})


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

    pop_ratio_by_users = pop_ratio_by_user(score_frame, most_pop_items=pop_items)
    pop_ratio_by_users.sort_values(['popularity_ratio'], inplace=True, ascending=False)
    num_of_users = len(pop_ratio_by_users)
    groups_dict: Dict[str, Set[str]] = {}
    first_index = 0
    last_index = first_index
    percentage = 0.0
    for group_name in groups:
        percentage += groups[group_name]
        group_index = round(num_of_users * percentage)
        groups_dict[group_name] = set(pop_ratio_by_users['from_id'][last_index:group_index])
        last_index = group_index
    if percentage < 1.0:
        group_index = round(num_of_users)
        groups_dict['default_diverse'] = set(pop_ratio_by_users['from_id'][last_index:group_index])
    return groups_dict


def get_profile_avg_pop_ratio(users: Set[str], pop_ratio_by_users: pd.DataFrame) -> np.ndarray:
    """
    Calculates the average profile popularity ratio

    Args:
        users (Set<str>): set of 'from_id' labels
        pop_ratio_by_users (pd.DataFrame): contains the 'popularity_ratio' for each 'from_id' (user)

    Returns:
        (float): average profile popularity ratio
    """
    profile_pop_ratios = np.array([])
    for user in users:
        user_pop_ratio = pop_ratio_by_users.query('from_id == @user')[['popularity_ratio']].values.flatten()[0]
        profile_pop_ratios = np.append(profile_pop_ratios, user_pop_ratio)
    return profile_pop_ratios


def get_recs_avg_pop_ratio(users: Set[str], recommendations: pd.DataFrame, most_popular_items: Set[str]) -> np.ndarray:
    """
    Calculates the popularity ratio
    Args:
        users (Set[str]): set of 'from_id' labels
        recommendations (pd.DataFrame): DataFrame with columns = ['from_id', 'to_id', 'rating']
        most_popular_items (Set[str]): set of most popular 'to_id' labels

    Returns:
        score (float): avg popularity ratio for recommendations
    """
    pop_ratios = np.array([])
    for user in users:
        recommended_items = recommendations.query('from_id == @user')[['to_id']].values.flatten()

        if len(recommended_items) > 0:
            pop_items_count = 0
            for item in recommended_items:
                if item in most_popular_items:
                    pop_items_count += 1

            pop_ratios = np.append(pop_ratios, pop_items_count / len(recommended_items))
    return pop_ratios
