from typing import List, Set
from orange_cb_recsys.recsys.graphs import BipartiteGraph
import pandas as pd
import networkx as nx

from orange_cb_recsys.recsys.graphs.graph import Category
from orange_cb_recsys.utils.const import logger


class NXBipartiteGraph(BipartiteGraph):
    """
    Class that implements a Bipartite graph through networkx library.
    It supports 'from' node and 'to' node only.
    It creates a graph from an initial rating frame
    EXAMPLE:
            _| from_id | to_id | score|
            _|   u1    | Tenet | 0.6  |

        Becomes:
                u1 -----> Tenet
        with the edge weighted and labeled based on the score column

    Args:
        source_frame (pd.DataFrame): the initial rating frame needed to create the graph

    """
    def __init__(self, source_frame: pd.DataFrame):
        self.__graph: nx.DiGraph = None
        super().__init__(source_frame)

    def create_graph(self):
        """
        Instantiates the networkx DiGraph
        """
        self.__graph = nx.DiGraph()

    @property
    def from_nodes(self) -> Set[object]:
        """
        Returns a set of all 'from' nodes in the graph
        """
        # node is a tuple like ('001', {'category': {'from'}})
        # we need node[0] to access the node name,
        # we need node[1] to access the dict containing the category
        return set(node[0] for node in self.__graph.nodes(data=True) if self.is_from_node(node[0]))

    @property
    def to_nodes(self) -> Set[object]:
        """
        Returns a set of all 'to' nodes in the graph
        """
        # node is a tuple like ('001', {'category': {'from'}})
        # we need node[0] to access the node name,
        # we need node[1] to access the dict containing the category
        # return set(node[0] for node in self.__graph.nodes(data=True) if 'to' in node[1]['category'])
        return set(node[0] for node in self.__graph.nodes(data=True) if self.is_to_node(node[0]))

    def add_from_node(self, node: object):
        """
        Adds a 'from' node to the graph.

        If the node is not-existent then it is created and then added to the graph.
        Otherwise if it is already existent we update the categories of the node adding a
        'From' category.

        Args:
            node (object): node that needs to be added to the graph as a from node
        """
        try:
            # If the node is already existent, add to its categories the 'From' category
            categories: set = self.__graph.nodes[node]['category']
            categories.add(Category.From)
        except KeyError:
            # If the node is not-existent, create it and add it to the graph with the
            # 'From' category
            categories: set = set()
            categories.add(Category.From)
            self.__graph.add_node(node, category=categories)

    def add_to_node(self, node: object):
        """
        Creates a 'to' node and adds it to the graph

        If the node is not-existent then it is created and then added to the graph.
        Otherwise if it is already existent we update the categories of the node adding a
        'To' category.

        Args:
            node (object): node that needs to be added to the graph as a 'to' node
        """
        try:
            # If the node is already existent, add to its categories the 'From' category
            categories: set = self.__graph.nodes[node]['category']
            categories.add(Category.To)
        except KeyError:
            # If the node is not-existent, create it and add it to the graph with the
            # 'To' category
            categories: set = set()
            categories.add(Category.To)
            self.__graph.add_node(node, category=categories)

    def link_from_to(self, from_node: object, to_node: object, weight: float, label: str = 'weight'):
        """
        Creates a weighted link connecting the 'from_node' to the 'to_node'

        If nodes are not-existent, they will be created

        Args:
            from_node (object): starting node of the link
            to_node (object): ending node of the link
            weight (float): weight of the link
            label (str): label of the link, default is 'weight'

        """
        self.add_from_node(from_node)
        self.add_to_node(to_node)
        self.__graph.add_edge(from_node, to_node, weight=weight, label=label)

    def get_link_data(self, start_node: object, final_node: object):
        """
        Get link data such as weight, label, between the 'start node' and the 'final node'.
        Returns None if said link doesn't exists

        Remember that this is a directed graph so the result differs if start_node and final_node
        are switched.

        Args:
            start_node (object): node where the link starts
            final_node (object): node where the link ends
        """
        try:
            return self.__graph.get_edge_data(start_node, final_node)
        except ValueError:
            return None

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
        if self.__graph.nodes.get(node) is None:
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
        if self.__graph.nodes.get(node) is None:
            logger.warning("The node specified is not in the graph! Return None")
            return None
        else:
            return list(self.__graph.successors(node))

    def is_from_node(self, node: object) -> bool:
        """
        Returns True if the node passed is a 'From' node, False otherwise

        In case the node has multiple categories, we just need that the 'From' category
        is one of them

        Args:
            node(object): node to check whether it's a 'from' node or not
        """
        n = self.__graph.nodes.get(node)
        if n is not None and Category.From in n['category']:
            return True
        else:
            return False

    def is_to_node(self, node: object) -> bool:
        """
        Returns True if the node passed is a 'to' node, False otherwise

        In case the node has multiple categories, we just need that the 'To' category
        is one of them

        Args:
            node(object): node to check whether it's a 'to' node or not
        """
        n = self.__graph.nodes.get(node)
        if n is not None and Category.To in n['category']:
            return True
        else:
            return False

    @property
    def _graph(self):
        """
        PRIVATE USAGE ONLY!

        In case some metrics needs to be performed on the newtowrkx graph
        """
        return self.__graph
