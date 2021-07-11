from orange_cb_recsys.utils.const import recsys_logger
import pandas as pd

from functools import wraps



class OnlyPositiveItems(Exception):
    """
    Exception to raise when there's only positive items available locally for the user
    """
    pass


class OnlyNegativeItems(Exception):
    """
    Exception to raise when there's only positive items available locally for the user
    """
    pass


class NoRatedItems(Exception):
    """
    Exception to raise when there's no item available locally for the user
    """
    pass


class NotRankingAlg(Exception):
    """
    Exception to raise when the algorithm is not a ranking algorithm, but it is asked to rank
    """
    pass


class NotPredictionAlg(Exception):
    """
    Exception to raise when the algorithm is not a prediction algorithm, but it is asked to predict
    """
    pass


class EmptyUserRatings(Exception):
    """
    Exception to raise when the user ratings is empty
    """
    pass


def Handler_EmptyFrame(func):
    """
    Handler that catches the above exceptions.

    Tries to run the functions normally, if one of the above exceptions is caught then it must return
    an empty frame for the user since predictions can't be calculated for it.
    """
    @wraps(func)
    def Inner_Function(*args, **kwargs):
        try:
            frame = func(*args, **kwargs)
        except (OnlyNegativeItems, OnlyPositiveItems, NoRatedItems, EmptyUserRatings) as e:
            warning_message = str(e) + "\nThe score frame will be empty for the user"
            recsys_logger.warning(warning_message)
            columns = ["to_id", "score"]
            frame = pd.DataFrame(columns=columns)
        return frame

    return Inner_Function