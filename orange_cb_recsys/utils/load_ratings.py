import os
import pandas as pd
from orange_cb_recsys.utils.const import home_path, DEVELOPING


def load_ratings(filename: str):
    """
    Loads the ratings from the directory in which they are stored and puts them in a DataFrame
    Args:
        filename (str): Name of the file that contains the ratings

    Returns:
        (pd.DataFrame): Ratings
    """

    return pd.read_csv(filename, dtype=str)
