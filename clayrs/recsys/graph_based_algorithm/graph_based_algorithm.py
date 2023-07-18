from __future__ import annotations
import abc
from typing import Dict, List, Set, Union, TYPE_CHECKING, Optional, Iterable

import numpy as np

if TYPE_CHECKING:
    from clayrs.content_analyzer.ratings_manager.ratings import Ratings
    from clayrs.recsys.graphs.graph import UserNode, Node, Graph, BipartiteDiGraph
    from clayrs.recsys.methodology import Methodology

from clayrs.recsys.graphs.graph import ItemNode
from clayrs.recsys.algorithm import Algorithm


class GraphBasedAlgorithm(Algorithm):
    """
    Abstract class for the graph-based algorithms
    """

    def __init__(self):
        # FUTURE WORK: this can be expanded in making the page rank keeping also PropertyNodes, etc.
        self._nodes_to_keep = {ItemNode}

    def filter_result(self, graph: BipartiteDiGraph, result: Dict, filter_list: Union[Iterable[Node], None],
                      user_node: UserNode) -> Dict:
        """
        Method which filters and cleans the result dict based on the parameters passed

        If `filter_list` parameter is not None, then the final dict will contain items and score only for items
        in said `filter_list`

        Otherwise, if no filter list is specified, then all unrated items by the user will be returned in the final
        dict

        Args:
            graph: Directed graph which models interactions between users and items
            result: dictionary representing the result (keys are nodes and values are their score)
            filter_list: list of the items for which a ranking score/score prediction must be computed.
                If None all unrated items for the user will be ranked/score predicted.
            user_node: Node of the particular user for which we want to compute rank/score prediction
        """

        def must_keep(node: object, user_profile):
            must_be_kept = True
            if node in user_profile or type(node) not in self._nodes_to_keep:
                must_be_kept = False

            return must_be_kept

        if filter_list is not None:
            filtered_keys = result.keys() & set(filter_list)
            filtered_result = {k: result[k] for k in filtered_keys}
        else:
            extracted_profile = set(graph.get_successors(user_node))
            filtered_result = {k: result[k] for k in result.keys() if must_keep(k, extracted_profile)}

        return filtered_result

    @abc.abstractmethod
    def predict(self, graph: Graph, train_set: Ratings, test_set: Ratings, user_id_list: Set[str],
                methodology: Methodology, num_cpus: int) -> List[np.ndarray]:
        """
        Abstract method that predicts how much a user will like unrated items.
        If the algorithm is not a PredictionScore Algorithm, implement this method like this:

        ```python
        def predict():
            raise NotPredictionAlg
        ```

        Args:
            graph: A graph which models interactions of users and items
            train_set: a Ratings object containing interactions between users and items
            test_set: Ratings object which represents the ground truth of the split considered
            user_id_list: Set of user id for which a recommendation list must be generated. Users should be represented
                as strings rather than with their mapped integer
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method

        Returns:
            List of uir matrices for each user, where each uir contains predicted interactions between users and unseen
                items
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rank(self, graph: Graph, train_set: Ratings, test_set: Ratings, user_id_list: Set[str],
             recs_number: Optional[int], methodology: Methodology, num_cpus: int) -> List[np.ndarray]:
        """
        Rank the top-n recommended items for the user. If the recs_number parameter isn't specified,
        All unrated items for the user will be ranked (or only items in the filter list, if specified).

        ```python
        def rank():
            raise NotRankingAlg
        ```

        Args:
            graph: A graph which models interactions of users and items
            train_set: a Ratings object containing interactions between users and items
            test_set: Ratings object which represents the ground truth of the split considered
            recs_number: number of the top ranked items to return, if None all ranked items will be returned
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method

        Returns:
            List of uir matrices for each user, where each uir contains predicted interactions between users and unseen
                items sorted in a descending way w.r.t. the third dimension which is the ranked score
        """
        raise NotImplementedError
