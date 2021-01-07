from typing import List


def avg(score_list: List[float]) -> float:
    """
    Compute the average score
    """
    return sum(score_list) / len(score_list)


def mode(score_list: List[float]) -> float:
    """
    Return the mode between the ratings
    """
    return max(set(score_list), key=score_list.count)


class ScoreCombiner:
    """
    Class that combines the scores given by a user

    Args:
        function (str): Name of the method to use to combine the ratings
    """
    def __init__(self, function: str):
        self.__function = eval(function)

    def combine(self, score_list: List[float], **kwargs) -> float:
        return self.__function(score_list, **kwargs)
