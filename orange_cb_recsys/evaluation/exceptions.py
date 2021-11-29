class AlreadyFittedRecSys(Exception):
    """
    Exception to raise when rankings/score predictions are already calculated for the users
    """
    pass

class KError(Exception):
    """
    Exception to raise when k passed is not valid
    """
    pass


class StringNotSupported(Exception):
    """
    Exception to raise when average method is not valid
    """
    pass


class NotEnoughUsers(Exception):
    """
    Exception to raise when DeltaGap tries to split n_users in n_groups but n_users < n_groups
    """


class PercentageError(Exception):
    """
    Exception to raise when there's something wrong with percentages
    """
