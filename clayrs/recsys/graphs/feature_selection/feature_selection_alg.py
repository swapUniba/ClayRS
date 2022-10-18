from __future__ import annotations
import networkx as nx
from abc import ABC, abstractmethod
from typing import List, Union, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from clayrs.recsys.graphs.graph import FullDiGraph
    from clayrs.recsys.graphs.graph import UserNode, ItemNode

from clayrs.recsys.graphs.feature_selection.exceptions import FeatureSelectionException
from clayrs.recsys.graphs.graph import PropertyNode


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

    ```
    Item Node someitem -- (label: starring) --> Property Node http://dbpedia.org/resource/Phil_Collins
    ```

    will be transformed to:
    ```
    Item Node someitem ----> Property Node starring
    ```

    This is done so that only the instances related to the most important properties will be kept in the original graph

    After creating the new graph, an algorithm which can compute the 'importance' scores for each node in the new graph
    is used (for example, PageRank) and the Property nodes (in this case being the property labels of the original
    graph) with the best scores will be returned
    """

    def perform(self, graph: FullDiGraph, target_nodes: Union[List[UserNode], List[ItemNode]],
                mode: str = 'to_keep') -> list:
        """
        Main method that handles the entire Feature Selection process. After every step of the process is completed, it
        will return a list containing the most important property labels of the original graph if `mode='to_keep'` or
        it will return a list containing the least important property labels of the original graph if `mode='to_remove'`

        Args:
            graph: The original graph to apply Feature Selection on
            target_nodes: list of nodes from the original graph to consider for the creation
                of the new graph
            mode: String which governs what the method will return. With `mode='to_keep'` the method will return the
                the most important property labels, with `mode='to_remove'` the method will return the
                the least important property labels

        Returns:
            List of properties to keep or to remove of the original graph
        """
        raise NotImplementedError

    @staticmethod
    def create_new_graph(graph: FullDiGraph, target_nodes: Union[List[UserNode], List[ItemNode]]) -> nx.DiGraph:
        """
        Creates a NetworkX directed graph from the original graph and the target nodes list passed as argument

        Given links of this kind in the original graph:

        ```
        Item Node someitem -- (label: starring) --> Property Node http://dbpedia.org/resource/Phil_Collins
        ```

        This method will create a new graph where links will be transformed to:

        ```
        Item Node someitem ----> Property Node starring
        ```

        Args:
            graph (FullGraph): the original graph to apply Feature Selection on
            target_nodes (list[object]): list of user or item nodes from the original graph to consider for the creation
                of the new graph

        Returns:
            A new graph that will be created from the original graph and the list of its nodes to consider
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
        k: Number of properties to keep
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
            Dictionary containing the 'importance' score for each node in the graph
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
    Computes the PageRank as FeatureSelection algorithm. Property labels of the original graph will be scored with their
    page rank score and only the top-k labels will be kept in the *feature selected graph*,
    while discarding the others

    Args:
        k: Top-k property labels to keep in the *feature selected graph*

        alpha: Damping parameter for PageRank, default=0.85.

        personalization: The "personalization vector" consisting of a dictionary with a key some subset of graph nodes
            and personalization value each of those. At least one personalization value must be non-zero.
            If not specfiied, a nodes personalization value will be zero.
            By default, a uniform distribution is used.

        max_iter: Maximum number of iterations in power method eigenvalue solver.

        tol: Error tolerance used to check convergence in power method solver.

        nstart: Starting value of PageRank iteration for each node.

        weight: Edge data key to use as weight.  If None weights are set to 1.

        dangling: The outedges to be assigned to any "dangling" nodes, i.e., nodes without any outedges.
            The dict key is the node the outedge points to and the dict value is the weight of that outedge.
            By default, dangling nodes are given outedges according to the personalization vector (uniform if not
            specified). This must be selected to result in an irreducible transition matrix
            (see notes under google_matrix). It may be common to have the dangling dict to be the same as the
            personalization dict.
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
    Computes the Eigen Vector Centrality as FeatureSelection algorithm. Property labels of the original graph will be
    scored with their eigen vector centrality score and only the top-k labels will be kept in the
    *feature selected graph*, while discarding the others

    Args:
        k: Top-k property labels to keep in the *feature selected graph*
        max_iter: Maximum number of iterations in power method.
        tol: Error tolerance used to check convergence in power method iteration.
        nstart: Starting value of eigenvector iteration for each node.
        weight: Boolean value which tells the algorithm if weight of the edges must be considered or not.
            Default is True
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
    Computes the Degree Centrality as FeatureSelection algorithm. Property labels of the original graph will be
    scored with their degree centrality score and only the top-k labels will be kept in the *feature selected graph*,
    while discarding the others
    """

    def __init__(self, k: int = 10):
        super().__init__(k)

    def create_rank(self, graph: nx.DiGraph) -> dict:
        return nx.degree_centrality(graph)
