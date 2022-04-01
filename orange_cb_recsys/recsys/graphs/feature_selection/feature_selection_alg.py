import networkx as nx
from abc import ABC, abstractmethod
from typing import List, Union, Any

from orange_cb_recsys.recsys.graphs.feature_selection.exceptions import FeatureSelectionException
from orange_cb_recsys.recsys.graphs.graph import FullDiGraph, PropertyNode, Node, UserNode, ItemNode


class FeatureSelectionAlgorithm(ABC):
    """
    Feature Selection algorithm class. It contains all the methods to use Feature Selection on a Full Graph.
    The main method to do so is the perform method.

    All the Feature Selection algorithms will create a new graph based on the one passed by the user. This new graph
    will contain only the Item or the User nodes from the original graph but the Property nodes will be different.
    Instead of having the property instance (for example: http://dbpedia.org/resource/Phil_Collins) the Property nodes
    in the new graph will have the property names associated to the property instance (for example: 'starring').
    This is retrieved because the label of the edge connecting a User or Item node to a Property node will be the
    Property name. So, for example:

        Item Node someitem -- (label: starring) --> Property Node http://dbpedia.org/resource/Phil_Collins

        will be transformed to

        Item Node someitem ----> Property Node starring

    This is done so that only the instances related to the most important properties will be kept in the original graph

    After creating the new graph, an algorithm which can compute the 'importance' scores for each node in the new graph
    is used (for example, PageRank) and the Property nodes (in this case being the property labels of the original
    graph) with the best scores will be returned

    Args:
        additional_arguments (dict): it's possible to pass the parameters one would normally pass to the scores
            computation algorithms (for example, if the PageRank from networkx is used to compute the scores for
            each node, this parameter can contain the additional parameters for said method)
    """

    def perform(self, graph: FullDiGraph, target_nodes: Union[List[UserNode], List[ItemNode]],
                mode: str = 'to_keep') -> list:
        """
        Main method that handles the entire Feature Selection process. After every step of the process is completed, it
        will return a list containing the most important property labels of the original graph.

        Args:
            graph (FullGraph): the original graph to apply Feature Selection on
            target_nodes (list[Node]): list of nodes from the original graph to consider for the creation
                of the new graph (for example, only items recommendable to the active user in a ranking algorithm should
                be considered)

        Returns:
            list of properties to keep of the original graph
        """
        raise NotImplementedError

    @staticmethod
    def create_new_graph(graph: FullDiGraph, target_nodes: Union[List[UserNode], List[ItemNode]]) -> nx.DiGraph:
        """
        Creates a NetworkX directed graph from the original graph and the target nodes list passed as argument

        Args:
            graph (FullGraph): the original graph to apply Feature Selection on
            target_nodes (list[object]): list of user or item nodes from the original graph to consider for the creation
                of the new graph (for example, only items recommendable to the active user in a ranking algorithm should
                be considered)

        Returns:
            mew_graph (DiGraph): new graph that will be created from the original graph and the list of its nodes to
                consider
        """
        new_graph = nx.DiGraph()

        # for each node in the target list, if it has any property node in the original graph, the target node will
        # be added together with all the property labels it is connected with in the graph. The property labels will be
        # turned into Property nodes in the new graph. A link between the target node and each new Property node will
        # be instantiated
        links_to_add = []
        for node in target_nodes:
            for successor_node in graph.get_successors(node):
                if isinstance(successor_node, PropertyNode):
                    link_data = graph.get_link_data(node, successor_node)
                    if link_data.get('weight') is None:
                        link = (node, PropertyNode(link_data['label']))
                    else:
                        link = (node, PropertyNode(link_data['label']), {'weight': link_data['weight']})

                    links_to_add.append(link)

        new_graph.add_edges_from(links_to_add)

        return new_graph


class TopKFeatureSelection(FeatureSelectionAlgorithm):
    """
    Feature Selection class that uses a top k strategy (meaning that the k properties with the highest score in the
    ranking will be considered as the most important properties)

    Args:
        k (int): number of properties to keep
    """

    def __init__(self, k: int):
        if k < 0:
            raise ValueError("The number of properties to keep passed (k parameter) can't be negative!")
        self._k = k

    @abstractmethod
    def create_rank(self, graph: nx.DiGraph) -> dict:
        """
        Method that calculates the rank used for feature selection. The method will return a dictionary where the keys
        will be the nodes in the graph passed as argument and the values will be the 'importance' score for each
        node. This rank will be used by the Feature Selection process to select which nodes to keep and which to discard

        Args:
            graph (DiGraph): graph from which the rank for the nodes will be calculated

        Returns:
            rank (dict): rank in the following form:

                {Node: rank_score, Node: rank_score, ...}
        """
        raise NotImplementedError

    def perform(self, graph: FullDiGraph, target_nodes: Union[List[UserNode], List[ItemNode]],
                mode: str = 'to_keep') -> list:

        if mode != 'to_keep' and mode != 'to_remove':
            raise TypeError("The only valid modes are 'to_keep' and 'to_remove'!")

        new_graph = self.create_new_graph(graph, target_nodes)

        # if the algorithm is not able to converge, a FeatureSelectionException is thrown
        try:
            rank = self.create_rank(new_graph)
        except nx.NetworkXException as e:
            raise FeatureSelectionException(str(e))

        # only property nodes of the new graph are kept in the ranking
        rank = {node.value: rank[node] for node in rank if isinstance(node, PropertyNode)}

        # the ranking produced by the Feature Selection algorithm is sorted
        rank = dict(sorted(rank.items(), key=lambda item: item[1], reverse=True))

        if mode == 'to_keep':
            # all the nodes up to the k-th node in the rank will be returned
            # in a list. These need to be kept
            nodes_list = list(rank.keys())[:self._k]
        else:
            # all the nodes from the k-th node in the rank onwards will be returned
            # in a list. These need to be removed
            nodes_list = list(rank.keys())[self._k:]

        return nodes_list


class TopKPageRank(TopKFeatureSelection):
    """
    Computes the PageRank for the new graph used by Feature Selection passed as argument. It's also possible to pass
    the parameters that one could normally pass to the NetworkX pagerank method
    """

    def __init__(self, k: int = 10, alpha: Any = 0.85, personalization: Any = None, max_iter: Any = 100,
                 tol: Any = 1.0e-6, nstart: Any = None, weight: bool = True, dangling: Any = None):
        super().__init__(k)

        self.alpha = alpha
        self.personalization = personalization
        self.max_iter = max_iter
        self.tol = tol
        self.nstart = nstart
        self.weight = 'weight' if weight is True else None
        self.dangling = dangling

    def create_rank(self, graph: nx.DiGraph) -> dict:
        return nx.pagerank(graph, alpha=self.alpha, personalization=self.personalization,
                           max_iter=self.max_iter, tol=self.tol, nstart=self.nstart, weight=self.weight,
                           dangling=self.dangling)


class TopKEigenVectorCentrality(TopKFeatureSelection):
    """
    Computes the EigenVector Centrality for the new graph used by Feature Selection passed as argument. It's also
    possible to pass the parameters that one could normally pass to the NetworkX eigenvector_centrality method
    """

    def __init__(self, k: int = 10, max_iter: Any = 100, tol: Any = 1.0e-6, nstart: Any = None, weight: bool = False):
        super().__init__(k)

        self.max_iter = max_iter
        self.tol = tol
        self.nstart = nstart
        self.weight = 'weight' if weight is True else None

    def create_rank(self, graph: nx.DiGraph) -> dict:
        return nx.eigenvector_centrality(graph, max_iter=self.max_iter, tol=self.tol,
                                         nstart=self.nstart, weight=self.weight)


class TopKDegreeCentrality(TopKFeatureSelection):
    """
    Computes the Degree Centrality for the new graph used by Feature Selection passed as argument.
    """

    def __init__(self, k: int = 10):
        super().__init__(k)

    def create_rank(self, graph: nx.DiGraph) -> dict:
        return nx.degree_centrality(graph)
