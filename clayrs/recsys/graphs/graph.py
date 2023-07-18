import os
import pickle
import lzma
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Set, Union, Iterable, Dict

from clayrs.recsys.graphs.graph_metrics import GraphMetrics
from clayrs.content_analyzer.ratings_manager.ratings import Ratings


class Node(ABC):
    """
    Abstract class that generalizes the concept of a Node

    The Node stores the actual value of the node in the 'value' attribute

    If another type of Node must be added to the graph (EX. Context), create a subclass for
    this class and create appropriate methods for the new Node in the Graph class
    """

    def __init__(self, value: str):
        self.__value = value

    @property
    def value(self):
        return self.__value

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        # If we are comparing self to an instance of the 'Node' class
        # then compare the value stored of self with the value stored
        # of the other instance and check if they are of the same class (UserNode == UserNode),
        # else compare the stored value of self directly to the other object
        return_val = False
        if isinstance(other, Node):
            return self.value == other.value and type(self) is type(other)

        return return_val

    def __lt__(self, other):
        if isinstance(other, Node):
            return self.value < other.value

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class UserNode(Node):
    """
    Class that represents 'user' nodes

    Args:
        value (object): the value to store in the node
    """

    def __init__(self, value: str):
        super().__init__(value)

    def __str__(self):
        return "User " + str(self.value)

    def __repr__(self):
        return f"UserNode({self.value})"


class ItemNode(Node):
    """
    Class that represents 'item' nodes

    Args:
        value (object): the value to store in the node
    """

    def __init__(self, value: str):
        super().__init__(value)

    def __str__(self):
        return "Item " + str(self.value)

    def __repr__(self):
        return f"ItemNode({self.value})"


class PropertyNode(Node):
    """
    Class that represents 'property' nodes

    Args:
        value (object): the value to store in the node
    """

    def __init__(self, value: str):
        super().__init__(value)

    def __str__(self):
        return "Property " + str(self.value)

    def __repr__(self):
        return f"PropertyNode({self.value})"


class Graph(ABC):
    """
    Abstract class that generalizes the concept of a Graph

    Every Graph "is born" from a dataframe which contains
    """

    @property
    @abstractmethod
    def user_nodes(self) -> Set[UserNode]:
        """
        Returns a set of 'user' nodes
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def item_nodes(self) -> Set[ItemNode]:
        """
        Returns a set of 'item' nodes'
        """
        raise NotImplementedError

    @abstractmethod
    def add_node(self, node: Union[Node, List[Node]]):
        """
        Add a 'item' node to the graph
        """
        raise NotImplementedError

    @abstractmethod
    def add_link(self, start_node: Union[Node, List[Node]], final_node: Union[Node, List[Node]],
                 weight: float = None, label: str = 'score', timestamp: str = None):
        """
        Adds an edge between the 'start_node' and the 'final_node',
        Both nodes must be present in the graph otherwise no link is created
        """
        raise NotImplementedError

    @abstractmethod
    def remove_link(self, start_node: Node, final_node: Node):
        """
        Remove the edge between the 'start_node' and the 'final_node',
        If there's no edge between the nodes, a warning is printed
        """
        raise NotImplementedError

    @abstractmethod
    def get_link_data(self, start_node: Node, final_node: Node):
        """
        Get data of the link between two nodes
        It can be None if does not exist
        """
        raise NotImplementedError

    @abstractmethod
    def node_exists(self, node: Node) -> bool:
        """
        Returns True if node exists in the graph, false otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def remove_node(self, node: Union[Node, Iterable[Node]]):
        """
        PRIVATE USAGE ONLY

        Used in the Feature Selection process to remove certain nodes from the graph

        Args:
            nodes_to_remove (Iterable): iterable object containing the nodes to remove from the graph
        """
        raise NotImplementedError

    def copy(self):
        """
        Make a deep copy the graph
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo = {id(self): result}
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def serialize(self, output_directory: str = ".", file_name: str = 'graph.xz'):
        """
        Serialize the graph with the pickle.dump() method

        Args:
            output_directory (str): location where the graph will be serialized
            file_name (str): name of the file which will contain the graph serialized
        """
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        if not file_name.endswith('.xz'):
            file_name += '.xz'

        path = os.path.join(output_directory, file_name)
        with lzma.open(path, 'wb') as f:
            pickle.dump(self, f)


class BipartiteDiGraph(Graph, GraphMetrics):
    """
    Abstract class that generalizes the concept of a BipartiteGraph

    A BipartiteGraph is a Graph containing only 'user' and 'item' nodes.

    Attributes:
        source_frame (pandas.DataFrame): must contains at least 'from_id', 'to_id', 'score' columns. The graph will be
            generated from this DataFrame
        default_score_label (str): the default label of the link between two nodes.
            Default is 'score'
        default_weight (float): the default value with which a link will be weighted
            Default is 0.5
    """

    @abstractmethod
    def get_predecessors(self, node: Node) -> List[Node]:
        """
        Get all predecessors of a node
        """
        raise NotImplementedError

    @abstractmethod
    def get_successors(self, node: Node) -> List[Node]:
        """
        Get all successors of a node
        """
        raise NotImplementedError

    # will only contain users and items
    def to_ratings(self, user_map=None, item_map=None):

        node_list = list(self.user_nodes)

        interaction_list = [(node.value, succ.value, float(self.get_link_data(node, succ).get('weight')))
                            for node in node_list
                            for succ in self.get_successors(node)
                            if isinstance(succ, ItemNode) and self.get_link_data(node, succ).get('weight')]

        return Ratings.from_list(interaction_list, user_map=user_map, item_map=item_map)


class TripartiteDiGraph(BipartiteDiGraph):
    """
    Abstract class that generalize the concept of a TripartiteGraph

    A TripartiteGraph is a Graph containing 'user', 'item' and 'property' nodes, but the latter ones
    are only allowed to be linked to 'item' nodes.

    Attributes:
        source_frame (pandas.DataFrame): must contains at least 'from_id', 'to_id', 'score' columns. The graph will be
            generated from this DataFrame
        item_contents_dir (str): the path containing items serialized
        item_exo_representation (str): the exogenous representation we want to extract properties from
        item_exo_properties (list): the properties we want to extract from the exogenous representation
        default_score_label (str): the default label of the link between two nodes.
            Default is 'score'
        default_weight (float): the default value with which a link will be weighted
            Default is 0.5
    """

    @property
    @abstractmethod
    def property_nodes(self) -> Set[Node]:
        """
        Returns a set of 'property' nodes
        """
        raise NotImplementedError

    @abstractmethod
    def add_node(self, node: Union[Node, List[Node]]):
        raise NotImplementedError

    @abstractmethod
    def add_node_with_prop(self, node: Union[ItemNode, List[ItemNode]], item_exo_properties: Union[Dict, set],
                           item_contents_dir: str,
                           item_id: Union[str, List[str]] = None):
        raise NotImplementedError


class FullDiGraph(TripartiteDiGraph):
    """
    Abstract class that generalize the concept of a FullGraph

    A FullGraph is a Graph containing 'user', 'item', 'property' nodes, and every other type of node that may be
    implemented, with no restrictions in linking.

    Attributes:
        source_frame (pandas.DataFrame): must contains at least 'from_id', 'to_id', 'score' columns. The graph will be
            generated from this DataFrame
        user_contents_dir (str): the path containing users serialized
        item_contents_dir (str): the path containing items serialized
        user_exo_representation (str): the exogenous representation we want to extract properties from for the users
        user_exo_properties (list): the properties we want to extract from the exogenous representation for the users
        item_exo_representation (str): the exogenous representation we want to extract properties from for the items
        item_exo_properties (list): the properties we want to extract from the exogenous representation for the items
        default_score_label (str): the default label of the link between two nodes.
            Default is 'score'
        default_weight (float): the default value with which a link will be weighted
            Default is 0.5
    """

    @abstractmethod
    def add_node_with_prop(self, node: Union[Node, List[Node]], exo_properties: Union[Dict, set],
                           contents_dir: str,
                           content_filename: Union[str, List[str]] = None):
        raise NotImplementedError
