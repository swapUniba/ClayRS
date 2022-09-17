import abc
from typing import Dict, List, Set, Union

from clayrs.content_analyzer.ratings_manager.ratings import Interaction, Ratings
from clayrs.recsys.algorithm import Algorithm

from clayrs.recsys.graphs.graph import UserNode, Node, Graph, ItemNode, BipartiteDiGraph
from clayrs.recsys.methodology import Methodology, TestRatingsMethodology


class GraphBasedAlgorithm(Algorithm):
    """
    Abstract class for the graph-based algorithms
    """

    def __init__(self):
        # FUTURE WORK: this can be expanded in making the page rank keeping also PropertyNodes, etc.
        self._nodes_to_keep = {ItemNode}

    def filter_result(self, graph: BipartiteDiGraph, result: Dict, filter_list: Union[List[Node], None],
                      user_node: UserNode) -> Dict:
        """
        Method which filters and cleans the result dict based on the parameters passed

        If `filter_list` parameter is not None, then the final dict will contains items and score only for items
        in said `filter_list`

        Otherwise if no filter list is specified, then all unrated items by the user will be returned in the final
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
    def predict(self, all_users: Set[str], graph: Graph, test_set: Ratings,
                methodology: Union[Methodology, None] = TestRatingsMethodology(),
                num_cpus: int = 0) -> List[Interaction]:
        """
        Abstract method that predicts how much a user will like unrated items.
        If the algorithm is not a PredictionScore Algorithm, implement this method like this:

        ```python
        def predict():
            raise NotPredictionAlg
        ```

        One can specify on which items score prediction must be performed for each user with the `filter_dict`
        parameter, in this case every user is mapped with a list of items for which a prediction score must be computed.
        Otherwise, for **ALL** unrated items a prediction score will be computed for each user.

        Args:
            all_users: Set of user id for which a recommendation list must be generated
            graph: A graph previously instantiated
            test_set: Ratings object which represents the ground truth of the split considered
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method. Default is 0, meaning that
                the number of cpus will be automatically detected.

        Returns:
            List of Interactions object where the 'score' attribute is the rating predicted by the algorithm
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rank(self, all_users: Set[str], graph: Graph, test_set: Ratings,
             recs_number: int = None, methodology: Union[Methodology, None] = TestRatingsMethodology(),
             num_cpus: int = 0) -> List[Interaction]:
        """
        Rank the top-n recommended items for the user. If the recs_number parameter isn't specified,
        All unrated items for the user will be ranked (or only items in the filter list, if specified).

        ```python
        def rank():
            raise NotRankingAlg
        ```

        One can specify which items must be ranked for each user with the `filter_dict` parameter,
        in this case every user is mapped with a list of items for which a ranking score must be computed.
        Otherwise, **ALL** unrated items will be ranked for each user.

        Args:
            all_users: Set of user id for which a recommendation list must be generated
            graph: A graph previously instantiated
            test_set: Ratings object which represents the ground truth of the split considered
            recs_number: number of the top ranked items to return, if None all ranked items will be returned
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method. Default is 0, meaning that
                the number of cpus will be automatically detected.

        Returns:
            List of Interactions object in a descending order w.r.t the 'score' attribute, representing the ranking for
                a single user
        """
        raise NotImplementedError
