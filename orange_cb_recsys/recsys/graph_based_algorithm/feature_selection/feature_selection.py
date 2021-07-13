import networkx as nx
from abc import ABC, abstractmethod
from typing import List, Union

from orange_cb_recsys.recsys.graph_based_algorithm.feature_selection.exceptions import FeatureSelectionException
from orange_cb_recsys.recsys.graphs.graph import FullGraph, PropertyNode, Node, UserNode, ItemNode


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

    def __init__(self, additional_arguments: dict = None):
        if additional_arguments is None:
            additional_arguments = {}
        self.__additional_arguments = additional_arguments

    @property
    def additional_arguments(self):
        return self.__additional_arguments

    @abstractmethod
    def perform_feature_selection(self, graph: FullGraph, target_nodes: Union[List[UserNode], List[ItemNode]]) -> dict:
        """
        This method contains the feature selection process, it will return a dictionary containing the 'importance'
        scores calculated for each node inside the new graph.

        Args:
            graph (FullGraph): the original graph to apply Feature Selection on
            target_nodes (list[Node]): list of user or item nodes from the original graph to consider for the creation
                of the new graph (for example, only items recommendable to the active user in a ranking algorithm should
                be considered)

        Returns:
            rank (dict): dictionary containing the nodes inside the new graph as keys (so the property labels of the
                original graph) and the 'importance' score associated with each of them as values
        """
        raise NotImplementedError

    @abstractmethod
    def get_properties_to_keep(self, rank: dict) -> list:
        """
        Implements the strategy that will be used to decide which properties should be kept from the ranking generated
        by the feature selection algorithm

        Args:
            rank (dict): dictionary containing the property nodes inside the new graph as keys and the
                'importance' score associated with each of them as values

        Returns:
            list containing the properties to keep
        """
        raise NotImplementedError

    def perform(self, graph: FullGraph, target_nodes: Union[List[UserNode], List[ItemNode]]) -> list:
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
        if len(target_nodes) == 0:
            raise FeatureSelectionException("No target nodes defined for Feature Selection")

        try:
            rank = self.perform_feature_selection(graph, target_nodes)
        except FeatureSelectionException as e:
            # in any scenario where the feature selection algorithm isn't capable of computing the feature selection,
            # an exception is thrown
            raise FeatureSelectionException(str(e))

        # in case multiple representations are defined in the graph, they need to be fused into a single one
        if len(rank) != 0 and '#' in list(rank.keys())[0]:
            rank = self.__fuse_multiple_representations(rank)

        # the ranking produced by the Feature Selection algorithm is sorted
        rank = dict(sorted(rank.items(), key=lambda item: item[1], reverse=True))

        return self.get_properties_to_keep(rank)

    @staticmethod
    def __fuse_multiple_representations(rank: dict) -> dict:
        """
        In case multiple representations are considered, the ranking containing multiple representations
        will be transformed into a ranking containing a single one.

            Example: {'starring#0': 0.03, 'starring#1': 0.1, 'something_else#0': some_value, ...}
            Will be transformed into: {'starring': 0.13, ...}

        Args:
            rank (dict): the ranking generated by the feature selection algorithm which is in the same form as the
                one in the example (so the keys should be either in this form 'starring#0' or this form
                'starring#0#something' to match the ones in the actual graph)

        Returns:
            rank modified with the new fused property names
        """
        # retrieves and filters the properties with # (meaning multiple representations)
        rank_properties = list(rank.keys())
        rank_properties = list(property_name.split('#')[0] for property_name in rank_properties)

        new_rank = {}

        for property_name in rank_properties:
            properties_labels = [key for key in rank.keys() if property_name in key]
            for property_label in properties_labels:
                if property_name in new_rank.keys():
                    new_rank[property_name] += rank[property_label]
                else:
                    new_rank[property_name] = rank[property_label]

        return new_rank


class NXFeatureSelectionAlgorithm(FeatureSelectionAlgorithm):
    """
    Feature Selection class where the new graph that will be created will be a NetworkX DiGraph()
    """

    def __init__(self, additional_arguments: dict = None):
        super().__init__(additional_arguments)

    @abstractmethod
    def perform_feature_selection(self, graph: FullGraph, target_nodes: Union[List[UserNode], List[ItemNode]]) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_properties_to_keep(self, rank: dict) -> list:
        raise NotImplementedError

    @staticmethod
    def create_new_graph_nx(graph: FullGraph, target_nodes: Union[List[UserNode], List[ItemNode]]) -> nx.DiGraph:
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
        for node in target_nodes:
            for successor_node in graph.get_successors(node):
                if graph.is_property_node(successor_node):
                    link_data = graph.get_link_data(node, successor_node)
                    new_graph.add_edge(node, PropertyNode(link_data['label']), weight=link_data['weight'])
        return new_graph


class NXTopKFeatureSelection(NXFeatureSelectionAlgorithm):
    """
    Feature Selection class that uses a top k strategy (meaning that the k properties with the highest score in the
    ranking will be considered as the most important properties)

    Args:
        k (int): number of properties to keep
    """

    def __init__(self, k: int, additional_arguments: dict = None):
        super().__init__(additional_arguments)
        self.__k = k

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

    def perform_feature_selection(self, graph: FullGraph, target_nodes: Union[List[UserNode], List[ItemNode]]) -> dict:

        if self.__k <= 0:
            return {}

        new_graph = self.create_new_graph_nx(graph, target_nodes)

        # if the algorithm is not able to converge, a FeatureSelectionException is thrown
        try:
            rank = self.create_rank(new_graph)
        except nx.PowerIterationFailedConvergence as e:
            raise FeatureSelectionException(str(e))

        # only property nodes of the new graph are kept in the ranking
        rank = {node.value: rank[node] for node in rank if isinstance(node, PropertyNode)} if len(rank) != 0 else rank

        return rank

    def get_properties_to_keep(self, rank: dict) -> list:
        # the first k properties in the rank will be returned in a list

        properties_to_keep = list()
        actual_k = self.__k if self.__k > 0 else 0

        if len(rank.keys()) == actual_k:
            properties_to_keep = list(rank.keys())

        for prop_label in rank.keys():
            if len(properties_to_keep) < actual_k:
                properties_to_keep.append(prop_label)
            else:
                break

        return properties_to_keep


class NXTopKPageRank(NXTopKFeatureSelection):
    """
    Computes the PageRank for the new graph used by Feature Selection passed as argument. It's also possible to pass
    the parameters that one could normally pass to the NetworkX pagerank method
    """

    def __init__(self, k: int = 10, **kwargs):
        super().__init__(k, kwargs)

    def create_rank(self, graph: nx.DiGraph) -> dict:
        return nx.pagerank(graph.to_undirected(), **self.additional_arguments)


class NXTopKEigenVectorCentrality(NXTopKFeatureSelection):
    """
    Computes the EigenVector Centrality for the new graph used by Feature Selection passed as argument. It's also
    possible to pass the parameters that one could normally pass to the NetworkX eigenvector_centrality method
    """

    def __init__(self, k: int = 10, **kwargs):
        super().__init__(k, kwargs)

    def create_rank(self, graph: nx.DiGraph) -> dict:
        return nx.eigenvector_centrality(graph.to_undirected(), **self.additional_arguments)


class NXTopKDegreeCentrality(NXTopKFeatureSelection):
    """
    Computes the Degree Centrality for the new graph used by Feature Selection passed as argument.
    """

    def __init__(self, k: int = 10):
        super().__init__(k)

    def create_rank(self, graph: nx.DiGraph) -> dict:
        return nx.degree_centrality(graph.to_undirected())

