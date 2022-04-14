class UserSkipAlgFit(Exception):
    """
    Super class for exception related to the fit of a single user. If one of the exception happens, the algorithm
    can't be fitted for the user, therefore will be skipped
    """
    pass


class OnlyPositiveItems(UserSkipAlgFit):
    """
    Exception to raise when there's only positive items available locally for the user
    """
    pass


class OnlyNegativeItems(UserSkipAlgFit):
    """
    Exception to raise when there's only negative items available locally for the user
    """
    pass


class NoRatedItems(UserSkipAlgFit):
    """
    Exception to raise when there's no item available locally for the user
    """
    pass


class EmptyUserRatings(UserSkipAlgFit):
    """
    Exception to raise when the user ratings is empty
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


class NotFittedAlg(Exception):
    """
    Exception to raise when the algorithm has not been fitted
    """
    pass
