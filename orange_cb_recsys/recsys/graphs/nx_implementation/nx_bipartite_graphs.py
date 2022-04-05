from typing import List, Set, Iterable, Union

from orange_cb_recsys.content_analyzer import Ratings
from orange_cb_recsys.recsys.graphs.graph import BipartiteDiGraph, Node
import pandas as pd
import networkx as nx

from orange_cb_recsys.recsys.graphs.graph import UserNode, ItemNode
from orange_cb_recsys.utils.const import logger, get_progbar


class NXBipartiteGraph(BipartiteDiGraph):
    """
    Class that implements a Bipartite graph through networkx library.
    It supports 'user' node and 'item' node only.
    It creates a graph from an initial rating frame
    EXAMPLE::
            _| from_id | to_id | score|
            _|   u1    | Tenet | 0.6  |

        Becomes:
                u1 -----> Tenet
        where 'u1' becomes a 'user' node and 'Tenet' becomes a 'item' node,
        with the edge weighted and labelled based on the score column and on the
        'default_score_label' parameter

    Args:
        source_frame (pd.DataFrame): the initial rating frame needed to create the graph
        default_score_label (str): the default label of the link between two nodes.
            Default is 'score'
        default_weight (float): the default value with which a link will be weighted
            Default is 0.5

    """

    def __init__(self, source_frame: Ratings = None, link_label: str = None):

        self._graph = nx.DiGraph()

        if source_frame is not None:
            not_none_dict = {}
            if link_label is not None:
                not_none_dict['label'] = link_label

            with get_progbar(source_frame) as progbar:
                progbar.set_description("Creating User->Item links list")

                if len(source_frame.timestamp_column) != 0:
                    edges_with_attributes = [(UserNode(interaction.user_id), ItemNode(interaction.item_id),

                                              # {**x, **y} merges the dicts x and y
                                              {**not_none_dict, **{'weight': interaction.score,
                                                                   'timestamp': interaction.timestamp}}
                                              )
                                             for interaction in progbar]
                else:
                    edges_with_attributes = [(UserNode(interaction.user_id), ItemNode(interaction.item_id),

                                              # {**x, **y} merges the dicts x and y
                                              {**not_none_dict, **{'weight': interaction.score}})
                                             for interaction in progbar]

            logger.info("Adding User->Item links list to NetworkX graph...")
            self._graph.add_edges_from(edges_with_attributes)

    @property
    def user_nodes(self) -> Set[UserNode]:
        """
        Returns a set of all 'user' nodes in the graph
        """
        return set([node for node in self._graph.nodes if isinstance(node, UserNode)])

    @property
    def item_nodes(self) -> Set[ItemNode]:
        """
        Returns a set of all 'item' nodes in the graph
        """
        return set([node for node in self._graph.nodes if isinstance(node, ItemNode)])

    def add_node(self, node: Union[Node, List[Node]]):
        """
        Adds a 'user' node to the graph.
        If a list is passed, then every element of the list will be added as a 'user' node

        Args:
            node: node(s) that needs to be added to the graph as 'user' node(s)
        """
        if not isinstance(node, list):
            node = [node]

        if any(not isinstance(n, UserNode) and not isinstance(n, ItemNode) for n in node):
            raise ValueError("You can only add UserNodes or ItemNodes to a bipartite graph!")

        self._graph.add_nodes_from(node)

    def add_link(self, start_node: Union[Node, List[Node]], final_node: Union[Node, List[Node]],
                 weight: float = None, label: str = None, timestamp: str = None):
        """
        Creates a weighted link connecting the 'start_node' to the 'final_node'
        Both nodes must be present in the graph before calling this method

        'weight' and 'label' are optional parameters, if not specified default values
        will be used.

        Args:
            start_node (object): starting node of the link
            final_node (object): ending node of the link
            weight (float): weight of the link, default is 0.5
            label (str): label of the link, default is 'score_label'
        """
        if not isinstance(start_node, list):
            start_node = [start_node]

        if not isinstance(final_node, list):
            final_node = [final_node]

        self.add_node(start_node)
        self.add_node(final_node)

        not_none_dict = {}
        if label is not None:
            not_none_dict['label'] = label
        if weight is not None:
            not_none_dict['weight'] = weight
        if timestamp is not None:
            not_none_dict['timestamp'] = timestamp

        self._graph.add_edges_from(zip(start_node, final_node),
                                   **not_none_dict)

    def remove_link(self, start_node: Node, final_node: Node):
        """
        Removes the link connecting the 'start_node' to the 'final_node'.
        If there's no link between the two nodes, than a warning is printed

        Args:
            start_node (object): starting node of the link to remove
            final_node (object): ending node of the link to remove
        """
        try:
            self._graph.remove_edge(start_node, final_node)
        except nx.NetworkXError:
            logger.warning("No link exists between the start node and the final node!\n"
                           "No link will be removed")

    def get_link_data(self, start_node: Node, final_node: Node):
        """
        Get link data such as weight, label, between the 'start_node' and the 'final_node'.
        Returns None if said link doesn't exists

        Remember that this is a directed graph so the result differs if 'start_node' and 'final_node'
        are switched.

        Args:
            start_node (object): node where the link starts
            final_node (object): node where the link ends
        """
        return self._graph.get_edge_data(start_node, final_node)

    def get_predecessors(self, node: Node) -> List[Node]:
        """
        Returns a list containing the successors of the node passed.
        Returns None if the node doesn't exists in the graph.

        Taken from networkx library:
        "A predecessor of n is a node m such that there exists a directed
        edge from m to n"

        EXAMPLE:
                I1 <-- U1
                ↑
                U2

            get_successors(I1) ---> [U1, U2]

        Args:
            node(object): node of which we want to calculate predecessors
        """
        try:
            return list(self._graph.predecessors(node))
        except nx.NetworkXError:
            logger.warning("The node specified is not in the graph! Return None")

    def get_successors(self, node: Node) -> List[Node]:
        """
        Returns a list containing the successors of the node passed.
        Returns None if the node doesn't exists in the graph.

        Taken from networkx library:
        "A successor of n is a node m such that there exists a directed
        edge from n to m"

        EXAMPLE:
                U1 --> I2
                ↓
                I1

            get_successors(u1) ---> [I1, I2]


        Args:
            node(object): node of which we want to calculate successors
        """
        try:
            return list(self._graph.successors(node))
        except nx.NetworkXError:
            logger.warning("The node specified is not in the graph! Return None")

    def node_exists(self, node: Node) -> bool:
        """
        Returns True if the node passed exists in the graph, False otherwise

        Args:
            node(object): node to check whether it's present in the graph
        """
        r = self._graph.nodes.get(node)
        return r is not None

    def to_networkx(self):
        return self._graph

    def degree_centrality(self):
        """
        Calculate the degreee centrality for every node in the graph
        """
        return nx.degree_centrality(self._graph)

    def closeness_centrality(self):
        """
        Calculate the closeness centrality for every node in the graph
        """
        return nx.closeness_centrality(self._graph)

    def dispersion(self):
        """
        Calculate the dispersion for every node in the graph
        """
        return nx.dispersion(self._graph)

    def remove_node(self, node_to_remove: Union[Node, List[Node]]):
        """
        PRIVATE USAGE ONLY

        Used in the Feature Selection process to remove certain nodes from the graph

        Args:
            nodes_to_remove (Iterable): iterable object containing the nodes to remove from the graph
        """
        if not isinstance(node_to_remove, list):
            node_to_remove = [node_to_remove]

        self._graph.remove_nodes_from(node_to_remove)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return nx.algorithms.isomorphism.is_isomorphic(self._graph, other._graph)
        else:
            return False