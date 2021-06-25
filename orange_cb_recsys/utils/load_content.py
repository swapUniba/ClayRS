import lzma
import os
import pickle
import re
from typing import List

from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.utils.const import logger, progbar


def load_content_instance(directory: str, content_id: str) -> Content:
    """
    Loads a serialized content
    Args:
        directory (str): Path to the directory in which the content is stored
        content_id (str): Id of the content to load

    Returns:
        content (Content)
    """
    try:
        content_filename = os.path.join(directory, '{}.xz'.format(content_id))
        with lzma.open(content_filename, "rb") as content_file:
            content = pickle.load(content_file)
    except FileNotFoundError:
        content = None

    return content


def get_unrated_items(items_directory: str, ratings) -> List[Content]:
    """
    Gets the items that a user has not rated

    Args:
        items_directory (str): Path to the items directory
        ratings (pd.DataFrame): Ratings of a user

    Returns:
        unrated_items (List<Content>): List of items that the user has not rated
    """
    directory_filename_list = [os.path.splitext(filename)[0]
                               for filename in os.listdir(items_directory)
                               if filename != 'search_index']

    # logger.info("Getting filenames from IDs")
    # list of id of item without rating
    rated_items_filename_list = set([re.sub(r'[^\w\s]', '', item_id) for item_id in ratings.to_id])

    # logger.info("Checking if unrated")
    filename_list = [item_id for item_id in directory_filename_list if
                     item_id not in rated_items_filename_list]

    intersection = [x for x in filename_list if x in directory_filename_list]
    filename_list = intersection

    logger.info("Loading unrated items")
    unrated_items = [
        load_content_instance(items_directory, item_id)
        for item_id in progbar(filename_list, prefix="Loading unrated items:")]

    return unrated_items


def get_rated_items(items_directory, ratings) -> List[Content]:
    """
    Gets the items that a user not rated

    Args:
        items_directory (str): Path to the items directory
        ratings (pd.DataFrame): Ratings of the user

    Returns:
        unrated_items (List<Content>): List of items that the user has rated
    """
    directory_filename_list = [os.path.splitext(filename)[0]
                               for filename in os.listdir(items_directory)
                               if filename != 'search_index']

    # logger.info("Getting filenames from IDs")
    # list of id of item without rating
    rated_items_filename_list = set([re.sub(r'[^\w\s]', '', item_id) for item_id in ratings.to_id])

    # logger.info("Checking if rated")
    filename_list = [item_id for item_id in directory_filename_list if
                     item_id in rated_items_filename_list]

    intersection = [x for x in filename_list if x in directory_filename_list]
    filename_list = intersection

    filename_list.sort()

    rated_items = [
        load_content_instance(items_directory, item_id)
        for item_id in progbar(filename_list, prefix="Loading rated items:")]

    return rated_items


def remove_not_existent_items(ratings, items_directory: str):
    """
    Sometimes a dataset can contain ratings about an item which is not in the dataset. This
    function locates these items nd removes them from the ratings frame

    Args:
        ratings (pd.DataFrame): Ratings of the user
        items_directory (str): Path to the directory in which the items are stored
    """
    directory_filename_list = [os.path.splitext(filename)[0]
                               for filename in os.listdir(items_directory)
                               if filename != 'search_index']

    rated_items_filename_list = set([re.sub(r'[^\w\s]', '', item_id) for item_id in ratings.to_id])

    intersection = [x for x in rated_items_filename_list if x in directory_filename_list]
    ratings = ratings[ratings["to_id"].isin(intersection)]

    return ratings


def remove_not_existent_items_list(items: list, items_dir: str):
    """
    Sometimes a dataset can contain ratings about an item which is not in the dataset. This
    function locates these items nd removes them from the ratings frame

    Args:
        ratings (pd.DataFrame): Ratings of the user
        items_directory (str): Path to the directory in which the items are stored
    """

    items_locally = set([os.path.splitext(f)[0] for f in os.listdir(items_dir)
                         if os.path.isfile(os.path.join(items_dir, f)) and f.endswith('xz')])

    intersection = [x for x in items if x in items_locally]

    return intersection
