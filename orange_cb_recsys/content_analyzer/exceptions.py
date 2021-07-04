def Handler_ScoreNotFloat(func):
    """
    Handler that catches the above exceptions.

    Tries to run the functions normally, if one of the above exceptions is caught then it must return
    an empty frame for the user since predictions can't be calculated for it.
    """
    def Inner_Function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError:
            raise ValueError("The 'score' column must contains numbers!\n"
                             "Try the same column with a score processor or change column!")

    return Inner_Function