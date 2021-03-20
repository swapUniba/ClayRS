from typing import List, Set
from orange_cb_recsys.recsys.graphs import TripartiteGraph
import pandas as pd
import networkx as nx

from orange_cb_recsys.recsys.graphs.graph import UserNode, ItemNode, PropertyNode
from orange_cb_recsys.utils.const import logger


class NXTripartiteGraph(TripartiteGraph):
    """
    Class that implements a Tripartite graph through networkx library.
    It supports 'from' node, 'to' and 'property' node, but the latter ones are available only for
    'to' nodes.

    It creates a graph from an initial rating frame and if the 'item_contents_dir' is specified,
    tries to add properties for every 'to' node.
    EXAMPLE:
            _| from_id | to_id | score|
            _|   u1    | Tenet | 0.6  |

        Becomes:
                u1 -----> Tenet
        with the edge weighted and labelled based on the score column and on the
        'default_score_label' parameter.
        Then tries to load 'Tenet' from the 'item_contents_dir' if it is specified and if succeeds,
        adds in the graph its loaded properties as specified with 'item_exo_representation' and
        'item_exo_properties'.

    Args:
        source_frame (pd.DataFrame): the initial rating frame needed to create the graph
        item_contents_dir (str): the path containing items serialized
        item_exo_representation (str): the exogenous representation we want to extract properties from
        item_exo_properties (list): the properties we want to extract from the exogenous representation
        default_score_label (str): the label of the link between 'from' and 'to' nodes.
            Default is 'score_label'
        default_not_rated_value (float): the default value with which the link will be weighted
            Default is 0.5

    """

    def __init__(self, source_frame: pd.DataFrame, item_contents_dir: str = None,
                 item_exo_representation: str = None, item_exo_properties: List[str] = None,
                 default_score_label: str = 'score_label', default_not_rated_value: float = 0.5):
        self.__graph: nx.DiGraph = None

        super().__init__(source_frame, item_contents_dir,
                         item_exo_representation, item_exo_properties,
                         default_score_label, default_not_rated_value)

    def create_graph(self):
        """
        Instantiates the networkx DiGraph
        """
        self.__graph = nx.DiGraph()

    @property
    def user_nodes(self) -> Set[object]:
        """
        Returns a set of all 'user' nodes in the graph
        """
        return set(node for node in self.__graph.nodes if isinstance(node, UserNode))

    @property
    def item_nodes(self) -> Set[object]:
        """
        Returns a set of all 'item' nodes in the graph
        """
        return set(node for node in self.__graph.nodes if isinstance(node, ItemNode))

    @property
    def property_nodes(self) -> Set[object]:
        """
        Returns a set of all 'property' nodes in the graph
        """
        return set(node for node in self.__graph.nodes if isinstance(node, PropertyNode))

    def add_user_node(self, node: object):
        """
        Adds a 'user' node to the graph.
        If the node is not-existent then it is created and then added to the graph.

        Args:
            node (object): node that needs to be added to the graph as a from node
        """
        self.__graph.add_node(UserNode(node))

    def add_item_node(self, node: object):
        """
        Creates a 'item' node and adds it to the graph
        If the node is not-existent then it is created and then added to the graph.

        Args:
            node (object): node that needs to be added to the graph as a 'to' node
        """
        self.__graph.add_node(ItemNode(node))

    def add_property_node(self, node: object):
        """
        Creates a 'property' node and adds it to the graph
        If the node is not-existent then it is created and then added to the graph.

        Args:
            node (object): node that needs to be added to the graph as a 'property' node
        """
        self.__graph.add_node(PropertyNode(node))

    def add_link(self, start_node: object, final_node: object,
                 weight: float = None, label: str = None):
        """
        Creates a weighted link connecting the 'start_node' to the 'final_node'
        Both nodes must be present in the graph before calling this method
        'property' nodes can be linked only to 'item' nodes, otherwise a warning is returned.

        'weight' and 'label' are optional parameters, if not specified default values
        will be used.

        Args:
            start_node (object): starting node of the link
            final_node (object): ending node of the link
            weight (float): weight of the link, default is 0.5
            label (str): label of the link, default is 'score_label'
        """
        if weight is None:
            weight = self.get_default_weight()
        if label is None:
            label = self.get_default_score_label()

        if self.__is_not_valid_link(start_node, final_node):
            logger.warning("Property nodes can only be linked to item nodes! Use a Full Graph instead")
            return

        if self.node_exists(start_node) and self.node_exists(final_node):
            self.__graph.add_edge(start_node, final_node, weight=weight, label=label)
        else:
            logger.warning("One of the nodes or both don't exist in the graph! Add them before "
                           "calling this method.")

    def __is_not_valid_link(self, start_node: object, final_node: object):
        return (self.is_property_node(final_node) and not self.is_item_node(start_node)) or \
                (self.is_property_node(start_node) and not self.is_item_node(final_node))

    def get_link_data(self, start_node: object, final_node: object):
        """
        Get link data such as weight, label, between the 'start_node' and the 'final_node'.
        Returns None if said link doesn't exists

        Remember that this is a directed graph so the result differs if 'start_node' and 'final_node'
        are switched.

        Args:
            start_node (object): node where the link starts
            final_node (object): node where the link ends
        """
        return self.__graph.get_edge_data(start_node, final_node)

    def get_predecessors(self, node: object) -> List[object]:
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
        if not self.node_exists(node):
            logger.warning("The node specified is not in the graph! Return None")
            return None
        else:
            return list(self.__graph.predecessors(node))

    def get_successors(self, node: object) -> List[object]:
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
        if not self.node_exists(node):
            logger.warning("The node specified is not in the graph! Return None")
            return None
        else:
            return list(self.__graph.successors(node))

    def node_exists(self, node: object) -> bool:
        """
        Returns True if the node passed exists in the graph, False otherwise

        Args:
            node(object): node to check whether it's present in the graph
        """
        return self.__graph.nodes.get(node) is not None

    def is_user_node(self, node: object) -> bool:
        """
        Returns True if the node passed is a 'user' node, False otherwise

        Args:
            node(object): node to check whether it's a 'user' node or not
        """
        return node in self.user_nodes

    def is_item_node(self, node: object) -> bool:
        """
        Returns True if the node passed is a 'item' node, False otherwise

        Args:
            node(object): node to check whether it's a 'item' node or not
        """
        return node in self.item_nodes

    def is_property_node(self, node: object) -> bool:
        """
        Returns True if the node passed is a 'item' node, False otherwise

        Args:
            node(object): node to check whether it's a 'item' node or not
        """
        return node in self.property_nodes

    @property
    def _graph(self):
        """
        PRIVATE USAGE ONLY!

        In case some metrics needs to be performed on the newtowrkx graph
        """
        return self.__graph
