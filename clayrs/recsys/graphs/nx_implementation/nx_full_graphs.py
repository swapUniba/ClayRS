from __future__ import annotations
from typing import List, Union, Dict, TYPE_CHECKING

from clayrs.utils.const import logger

if TYPE_CHECKING:
    from clayrs.content_analyzer import Ratings, Content
    from clayrs.recsys.graphs.graph import Node

from clayrs.recsys.content_based_algorithm.contents_loader import LoadedContentsDict
from clayrs.recsys.graphs.nx_implementation import NXTripartiteGraph
from clayrs.recsys.graphs.graph import FullDiGraph, UserNode, PropertyNode
from clayrs.utils.context_managers import get_progbar


# Multiple Inheritance so that we will use NXTripartite as an interface (we only use its methods)
# and FullGraph as its proper father class (we'll call its __init__)
class NXFullGraph(NXTripartiteGraph, FullDiGraph):
    """
    Class that implements a Full graph through networkx library.

    !!! info

        A *Full Graph* is a graph which doesn't impose any particular restriction

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

    Then the framework tries to load 'Tenet' and 'Inception' from the `item_contents_dir` and 'u1' and 'u2' from
    `user_contents_dir` if they are specified and if it succeeds, adds in the graph their loaded properties as
    specified in the `item_exo_properties` parameter and `user_exo_properties`.

    !!! info "Load exogenous properties"

        In order to load properties in the graph, we must specify where users (and/or) items are serialized and ***which
        properties to add*** (the following is the same for *item_exo_properties*):

        *   If *user_exo_properties* is specified as a **set**, then the graph will try to load **all properties**
        from **said exogenous representation**

        ```python
        {'my_exo_id'}
        ```

        *   If *user_exo_properties* is specified as a **dict**, then the graph will try to load **said properties**
        from **said exogenous representation**

        ```python
        {'my_exo_id': ['my_prop1', 'my_prop2']]}
        ```

    Args:
        source_frame: The initial Ratings object needed to create the graph
        item_exo_properties: Set or Dict which contains representations to load from items. Use a `Set` if you want
            to load all properties from specific representations, or use a `Dict` if you want to choose which properties
            to load from specific representations
        item_contents_dir: The path containing items serialized with the Content Analyzer
        user_exo_properties: Set or Dict which contains representations to load from items. Use a `Set` if you want
            to load all properties from specific representations, or use a `Dict` if you want to choose which properties
            to load from specific representations
        user_contents_dir: The path containing users serialized with the Content Analyzer
        link_label: If specified, each link will be labeled with the given label. Default is None

    """

    def __init__(self, source_frame: Ratings = None,
                 item_exo_properties: Union[Dict, set] = None,
                 item_contents_dir: str = None,
                 user_exo_properties: Union[Dict, set] = None,
                 user_contents_dir: str = None,
                 link_label: str = None):

        NXTripartiteGraph.__init__(self, source_frame, item_exo_properties, item_contents_dir, link_label)

        if user_exo_properties and not user_contents_dir:
            logger.warning("`user_exo_properties` parameter set but `user_contents_dir` is None! "
                           "No property will be loaded")
        elif not user_exo_properties and user_contents_dir:
            logger.warning("`user_contents_dir` parameter set but `user_exo_properties` is None! "
                           "No property will be loaded")

        if source_frame is not None and user_contents_dir is not None and user_exo_properties is not None:
            self.add_node_with_prop([UserNode(user_id) for user_id in source_frame.unique_user_id_column],
                                    user_exo_properties,
                                    user_contents_dir)

    def add_node(self, node: Union[Node, List[Node]]):
        """
        Adds one or multiple Node objects to the graph.
        Since this is a Full Graph, any category of node is allowed

        No duplicates are allowed, but different category nodes with same id are (e.g. `ItemNode('1')` and
        `UserNode('1')`)

        Args:
            node: Node(s) object(s) that needs to be added to the graph
        """
        if not isinstance(node, list):
            node = [node]

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

    def add_node_with_prop(self, node: Union[Node, List[Node]], exo_properties: Union[Dict, set],
                           contents_dir: str,
                           content_filename: Union[str, List[str]] = None):
        """
        Adds one or multiple Node objects and its/their properties to the graph
        Since this is a Full Graph, no restriction are imposed and you can add any category of node together with its
        properties.

        In order to load properties in the graph, we must specify where contents are serialized and ***which
        properties to add*** (the following is the same for *item_exo_properties*):

        *   If *exo_properties* is specified as a **set**, then the graph will try to load **all properties**
        from **said exogenous representation**

        ```python
        {'my_exo_id'}
        ```

        *   If *exo_properties* is specified as a **dict**, then the graph will try to load **said properties**
        from **said exogenous representation**

        ```python
        {'my_exo_id': ['my_prop1', 'my_prop2']]}
        ```

        In case you want your node to have a different id from serialized contents, via the `content_filename` parameter
        you can specify what is the filename of the node that you are adding, e.g.

        ```
        item_to_add = ItemNode('different_id')

        # content_filename is 'item_serialized_1.xz'

        graph.add_node_with_prop(item_to_add, ..., content_filename='item_serialized_1')
        ```

        In case you are adding a list of nodes, you can specify the filename for each node in the list.

        Args:
            node: Node(s) object(s) that needs to be added to the graph along with their properties
            exo_properties: Set or Dict which contains representations to load from items. Use a `Set` if you want
                to load all properties from specific representations, or use a `Dict` if you want to choose which
                properties to load from specific representations
            contents_dir: The path containing items serialized with the Content Analyzer
            content_filename: Filename(s) of the node(s) to add

        Raises:
            ValueError: Exception raised when one of the node to add to the graph with their properties is not
                an ItemNode
        """
        def node_prop_link_generator():
            for n, id in zip(progbar, content_filename):
                item: Content = loaded_items.get(id)

                if item is not None:
                    exo_props = self._get_exo_props(exo_properties, item)

                    single_item_prop_edges = [(n,
                                               PropertyNode(prop_dict[prop]),
                                               {'label': prop})
                                              for prop_dict in exo_props for prop in prop_dict]
                else:
                    single_item_prop_edges = []

                yield from single_item_prop_edges

        if not isinstance(node, list):
            node = [node]

        if isinstance(exo_properties, set):
            exo_properties = dict.fromkeys(exo_properties, None)

        if content_filename is None:
            content_filename = [n.value for n in node]

        if not isinstance(content_filename, list):
            content_filename = [content_filename]

        loaded_items = LoadedContentsDict(contents_dir, contents_to_load=set(content_filename))
        with get_progbar(node) as progbar:
            progbar.set_description("Creating Node->Properties links")

            self._graph.add_edges_from((tuple_to_add for tuple_to_add in node_prop_link_generator()))

    def __str__(self):
        return "NXFullGraph"

    def __repr__(self):
        return str(self)
