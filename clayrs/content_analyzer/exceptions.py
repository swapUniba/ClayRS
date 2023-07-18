from functools import wraps

import numpy as np


def handler_score_not_float(func):
    """
    Handler that catches the above exception.

    Tries to run the functions normally, if one of the above exceptions is caught then it must return
    an empty frame for the user since predictions can't be calculated for it.
    """
    @wraps(func)
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError:
            raise ValueError("The 'score' and 'timestamp' columns must contains numbers!\n"
                             "Try to apply a score processor or change columns!") from None

    return inner_function


def handler_empty_matrix(dtype):

    def handler_for_function(func):
        """
        Handler that catches the above exception.

        Tries to run the functions normally, if one of the above exceptions is caught then it must return
        an empty frame for the user since predictions can't be calculated for it.
        """
        @wraps(func)
        def inner_function(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except IndexError:
                return np.array([], dtype=dtype)

        return inner_function

    return handler_for_function


class UserNone(Exception):
    pass


class ItemNone(Exception):
    pass
