from __future__ import annotations
from typing import List, Set, Union, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from clayrs.content_analyzer import Ratings

import networkx as nx

from clayrs.recsys.graphs.graph import BipartiteDiGraph, Node
from clayrs.recsys.graphs.graph import UserNode, ItemNode
from clayrs.utils.const import logger
from clayrs.utils.context_managers import get_progbar


class NXBipartiteGraph(BipartiteDiGraph):
    """
    Class that implements a Bipartite graph through networkx library.

    !!! info

        A *Bipartite Graph* is a graph which supports only *User* nodes and *Item* nodes. If you need to model also
        other node categories, consider using a Tripartite Graph or a Full Graph

    It creates a graph from an initial Rating object.

    Consider the following matrix representation of the Rating object
    ```
        +------+-----------+-------+
        | User |   Item    | Score |
        +------+-----------+-------+
        | u1   | Tenet     |     4 |
        | u2   | Inception |     5 |
        | ...  | ...       |   ... |
        +------+-----------+-------+
    ```

    The graph will be created with the following interactions:

    ```
                 4
            u1 -----> Tenet
                 5
            u2 -----> Inception
    ```

    where `u1` and `u2` become *User nodes* and `Tenet` and `Inception` become *Item nodes*,
    with the edge weighted depending on the score given

    If the `link_label` parameter is specified, then each link between users and items will be labeled with the label
    specified (e.g. `link_label='score'`):

    ```
            (4, 'score')
        u1 -------------> Tenet
            (5, 'score')
        u2 -------------> Inception
    ```


    Args:
        source_frame: the initial Ratings object needed to create the graph
        link_label: If specified, each link will be labeled with the given label. Default is None
    """

    def __init__(self, source_frame: Ratings = None, link_label: str = None):

        self._graph = nx.DiGraph()

        if source_frame is not None:
            not_none_dict = {}
            if link_label is not None:
                not_none_dict['label'] = link_label

            user_column = source_frame.user_id_column
            item_column = source_frame.item_id_column
            score_column = source_frame.score_column
            timestamp_column = source_frame.timestamp_column

            if len(timestamp_column) != 0:
                frame_iterator = zip(user_column, item_column, score_column, timestamp_column)
            else:
                frame_iterator = zip(user_column, item_column, score_column)

            with get_progbar(frame_iterator, total=len(source_frame)) as progbar:
                progbar.set_description("Creating User->Item links")

                if len(timestamp_column) != 0:
                    edges_with_attributes_gen = ((UserNode(interaction[0]), ItemNode(interaction[1]),

                                                  # {**x, **y} merges the dicts x and y
                                                  {**not_none_dict, **{'weight': interaction[2],
                                                                       'timestamp': interaction[3]}}
                                                  )
                                                 for interaction in progbar)
                else:
                    edges_with_attributes_gen = ((UserNode(interaction[0]), ItemNode(interaction[1]),

                                                  # {**x, **y} merges the dicts x and y
                                                  {**not_none_dict, **{'weight': interaction[2]}})
                                                 for interaction in progbar)

                self._graph.add_edges_from(edges_with_attributes_gen)

    @property
    def user_nodes(self) -> Set[UserNode]:
        """
        Returns a set of all *User nodes* in the graph
        """
        return set([node for node in self._graph.nodes if isinstance(node, UserNode)])

    @property
    def item_nodes(self) -> Set[ItemNode]:
        """
        Returns a set of all *Item nodes* in the graph
        """
        return set([node for node in self._graph.nodes if isinstance(node, ItemNode)])

    def add_node(self, node: Union[Node, List[Node]]):
        """
        Adds one or multiple Node objects to the graph.
        Since this is a Bipartite Graph, only `User Node` and `Item Node` can be added!

        No duplicates are allowed, but different category nodes with same id are (e.g. `ItemNode('1')` and
        `UserNode('1')`)

        Args:
            node: Node(s) object(s) that needs to be added to the graph

        Raises:
            ValueError: Exception raised when one of the node to add to the graph is not a User or Item node
        """
        if not isinstance(node, list):
            node = [node]

        if any(not isinstance(n, (UserNode, ItemNode)) for n in node):
            raise ValueError("You can only add UserNodes or ItemNodes to a bipartite graph!")

        self._graph.add_nodes_from(node)

    def add_link(self, start_node: Union[Node, List[Node]], final_node: Union[Node, List[Node]],
                 weight: float = None, label: str = None, timestamp: str = None):
        """
        Creates a link connecting the `start_node` to the `final_node`. If two lists are passed, then the node in
        position $i$ in the `start_node` list will be linked to the node in position $i$ in the `final_node` list.

        If nodes to link do not exist, they will be added automatically to the graph. Please remember that since this is
        a Bipartite Graph, only *User nodes* and *Item nodes* can be added!

        A link can be weighted with the `weight` parameter and labeled with the `label` parameter.
        A timestamp can also be specified via `timestamp` parameter.
        All three are optional parameters, so they are not required

        Args:
            start_node: Single Node object or a list of Node objects. They will be the 'head' of the link, since it's a
                directed graph
            final_node (object): Single Node object or a list Node objects. They will be the 'tail' of the link,
                since it's a directed graph
            weight: weight of the link, default is None (no weight)
            label: label of the link, default is None (no label)
            timestamp: timestamp of the link, default is None (no timestamp)
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
        Removes the link connecting the `start_node` to the `final_node`.
        If there's no link between the two nodes, then a warning is printed

        Args:
            start_node: *head* node of the link to remove
            final_node: *tail* node of the link to remove
        """
        try:
            self._graph.remove_edge(start_node, final_node)
        except nx.NetworkXError:
            logger.warning("No link exists between the start node and the final node!\n"
                           "No link will be removed")

    def get_link_data(self, start_node: Node, final_node: Node):
        """
        Get link data such as weight, label, timestamp. between the `start_node` and the `final_node`.
        Returns None if said link doesn't exists

        Remember that this is a directed graph so the result differs if 'start_node' and 'final_node'
        are switched.

        Args:
            start_node: Node object from where the link starts
            final_node: Node object to where the link ends
        """
        return self._graph.get_edge_data(start_node, final_node)

    def get_predecessors(self, node: Node) -> List[Node]:
        """
        Returns a list containing the *predecessors* of the node passed.
        Raises TypeError exception if the node doesn't exists in the graph.

        Taken from networkx library:

        > A predecessor of n is a node m such that there exists a directed
        edge from m to n

        For example:
        ```
        # GRAPH:

        I1 <-- U1
        ↑
        U2
        ```

        ```python
        >>> graph.get_predecessors(ItemNode('I1'))
        [User U1, User U2]
        ```

        Args:
            node: Node for which we want to know the predecessors

        Raises:
            TypeError: Exception raised when the node it's not in the graph
        """
        try:
            return list(self._graph.predecessors(node))
        except nx.NetworkXError:
            raise TypeError("The node specified is not in the graph!")

    def get_successors(self, node: Node) -> List[Node]:
        """
        Returns a list containing the successors of the node passed.
        Returns None if the node doesn't exists in the graph.

        Taken from networkx library:
        > A successor of n is a node m such that there exists a directed
        edge from n to m

        For example:
        ```
        U1 --> I2
        ↓
        I1
        ```

        ```python

        >>> graph.get_successors(UserNode('U1'))
        [Item I1, Item I2]
        ```

        Args:
            node: Node for which we want to know the successors

        Raises:
            TypeError: Exception raised when the node it's not in the graph
        """
        try:
            return list(self._graph.successors(node))
        except nx.NetworkXError:
            raise TypeError("The node specified is not in the graph!")

    def node_exists(self, node: Node) -> bool:
        """
        Returns True if the node passed exists in the graph, False otherwise

        Args:
            node: Node to check whether it's present in the graph or not
        """
        r = self._graph.nodes.get(node)
        return r is not None

    def to_networkx(self) -> nx.DiGraph:
        """
        Returns underlying networkx implementation of the graph
        """
        return self._graph

    def degree_centrality(self) -> Dict:
        """
        Calculate the degree centrality for every node in the graph

        Returns:
            Dictionary containing the degree centrality for each node in the graph
        """
        return nx.degree_centrality(self._graph)

    def closeness_centrality(self) -> Dict:
        """
        Calculate the closeness centrality for every node in the graph

        Returns:
            Dictionary containing the closeness centrality for each node in the graph
        """
        return nx.closeness_centrality(self._graph)

    def dispersion(self) -> Dict:
        """
        Calculate the dispersion for every node in the graph

        Returns:
            Dictionary containing the dispersion computed for each node in the graph
        """
        return nx.dispersion(self._graph)

    def remove_node(self, node_to_remove: Union[Node, List[Node]]):
        """
        Removes one or multiple nodes from the graph.
        If one of the nodes to remove is not present in the graph, it is silently ignored

        Args:
            node_to_remove: Single Node object or a list of Node objects to remove from the graph
        """
        if not isinstance(node_to_remove, list):
            node_to_remove = [node_to_remove]

        self._graph.remove_nodes_from(node_to_remove)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return nx.algorithms.isomorphism.is_isomorphic(self._graph, other._graph)
        else:
            return False

    def __str__(self):
        return "NXBipartiteGraph"

    def __repr__(self):
        return str(self)
