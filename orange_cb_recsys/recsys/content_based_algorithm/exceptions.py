from orange_cb_recsys.utils.const import logger
import pandas as pd


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


def Handler_EmptyFrame(func):
    """
    Handler that catches the above exceptions.

    Tries to run the functions normally, if one of the above exceptions is caught then it must return
    an empty frame for the user since predictions can't be calculated for it.
    """
    def Inner_Function(*args, **kwargs):
        try:
            frame = func(*args, **kwargs)
        except (OnlyNegativeItems, OnlyPositiveItems, NoRatedItems) as e:
            logger.warning(e)
            columns = ["to_id", "rating"]
            frame = pd.DataFrame(columns=columns)
        return frame

    return Inner_Function