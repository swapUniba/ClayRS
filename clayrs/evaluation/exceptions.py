class NotEnoughUsers(Exception):
    """
    Exception to raise when DeltaGap tries to split n_users in n_groups but n_users < n_groups
    """
    pass
