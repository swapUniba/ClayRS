from __future__ import annotations
from collections.abc import Iterable
from typing import List, Set, Union, Dict, TYPE_CHECKING

from clayrs.utils.const import logger

if TYPE_CHECKING:
    from clayrs.content_analyzer import Ratings, Content
    from clayrs.recsys.graphs.graph import Node

from clayrs.recsys.content_based_algorithm.contents_loader import LoadedContentsDict
from clayrs.recsys.graphs.nx_implementation.nx_bipartite_graphs import NXBipartiteGraph
from clayrs.recsys.graphs.graph import TripartiteDiGraph, ItemNode, UserNode
from clayrs.recsys.graphs.graph import PropertyNode
from clayrs.utils.context_managers import get_progbar


# Multiple Inheritance so that we will use NXBipartite as its proper father class (we'll its __init__ method)
# and TripartiteDiGraph as its interface
class NXTripartiteGraph(NXBipartiteGraph, TripartiteDiGraph):
    """
    Class that implements a Tripartite graph through networkx library.

    !!! info

        A *Tripartite Graph* is a graph which supports *User* nodes, *Item* nodes and *Property* nodes, but the latter
        can only be linked to *Item* nodes.
        If you need maximum flexibility, consider using a Full Graph

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

    Then the framework tries to load 'Tenet' and 'Inception' from the `item_contents_dir` if it is specified and if
    succeeds, adds in the graph their loaded properties as specified in the `item_exo_properties` parameter.

    !!! info "Load exogenous properties"

        In order to load properties in the graph, we must specify where items are serialized and ***which
        properties to add***:

        *   If *item_exo_properties* is specified as a **set**, then the graph will try to load **all properties**
        from **said exogenous representation**

        ```python
        {'my_exo_id'}
        ```

        *   If *item_exo_properties* is specified as a **dict**, then the graph will try to load **said properties**
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
        link_label: If specified, each link will be labeled with the given label. Default is None

    """

    def __init__(self, source_frame: Ratings = None,
                 item_exo_properties: Union[Dict, set] = None,
                 item_contents_dir: str = None,
                 link_label: str = None):

        NXBipartiteGraph.__init__(self, source_frame, link_label)

        if item_exo_properties and not item_contents_dir:
            logger.warning("`item_exo_properties` parameter set but `item_contents_dir` is None! "
                           "No property will be loaded")
        elif not item_exo_properties and item_contents_dir:
            logger.warning("`item_contents_dir` parameter set but `item_exo_properties` is None! "
                           "No property will be loaded")

        if source_frame is not None and item_contents_dir is not None and item_exo_properties is not None:
            self.add_node_with_prop([ItemNode(item_id) for item_id in source_frame.unique_item_id_column],
                                    item_exo_properties,
                                    item_contents_dir)

    @property
    def property_nodes(self) -> Set[PropertyNode]:
        """
        Returns a set of all *Property nodes* in the graph
        """
        return set(node for node in self._graph.nodes if isinstance(node, PropertyNode))

    def add_node(self, node: Union[Node, List[Node]]):
        """
        Adds one or multiple Node objects to the graph.
        Since this is a Tripartite Graph, only `User Node`, `Item Node` and `Property Node` can be added!

        No duplicates are allowed, but different category nodes with same id are (e.g. `ItemNode('1')` and
        `UserNode('1')`)

        Args:
            node: Node(s) object(s) that needs to be added to the graph

        Raises:
            ValueError: Exception raised when one of the node to add to the graph is not a User, Item or Property node
        """
        if not isinstance(node, list):
            node = [node]

        if any(not isinstance(n, (UserNode, ItemNode, PropertyNode)) for n in node):
            raise ValueError("You can only add UserNodes or ItemNodes to a bipartite graph!")

        self._graph.add_nodes_from(node)

    def add_node_with_prop(self, node: Union[ItemNode, List[ItemNode]], item_exo_properties: Union[Dict, set],
                           item_contents_dir: str,
                           item_filename: Union[str, List[str]] = None):
        """
        Adds one or multiple Node objects and its/their properties to the graph.
        Since this is a Tripartite Graph, only `Item Node` are allowed to have properties!

        In order to load properties in the graph, we must specify where items are serialized and ***which
        properties to add***:

        *   If *item_exo_properties* is specified as a **set**, then the graph will try to load **all properties**
        from **said exogenous representation**

        ```python
        {'my_exo_id'}
        ```

        *   If *item_exo_properties* is specified as a **dict**, then the graph will try to load **said properties**
        from **said exogenous representation**

        ```python
        {'my_exo_id': ['my_prop1', 'my_prop2']]}
        ```

        In case you want your node to have a different id from serialized contents, via the `item_filename` parameter
        you can specify what is the filename of the node that you are adding, e.g.

        ```
        item_to_add = ItemNode('different_id')

        # item_filename is 'item_serialized_1.xz'

        graph.add_node_with_prop(item_to_add, ..., item_filename='item_serialized_1')
        ```

        In case you are adding a list of nodes, you can specify the filename for each node in the list.

        Args:
            node: Node(s) object(s) that needs to be added to the graph along with their properties
            item_exo_properties: Set or Dict which contains representations to load from items. Use a `Set` if you want
                to load all properties from specific representations, or use a `Dict` if you want to choose which
                properties to load from specific representations
            item_contents_dir: The path containing items serialized with the Content Analyzer
            item_filename: Filename(s) of the node(s) to add

        Raises:
            ValueError: Exception raised when one of the node to add to the graph with their properties is not
                an ItemNode
        """
        def node_prop_link_generator():
            for n, id in zip(progbar, item_filename):
                item: Content = loaded_items.get(id)

                if item is not None:
                    exo_props = self._get_exo_props(item_exo_properties, item)

                    single_item_prop_edges = [(n,
                                               PropertyNode(prop_dict[prop]),
                                               {'label': prop})
                                              for prop_dict in exo_props for prop in prop_dict]

                else:
                    single_item_prop_edges = []

                yield from single_item_prop_edges

        if not isinstance(node, list):
            node = [node]

        if any(not isinstance(n, ItemNode) for n in node):
            raise ValueError("Only item nodes can be linked to property nodes in a Tripartite Graph!")

        if isinstance(item_exo_properties, set):
            item_exo_properties = dict.fromkeys(item_exo_properties, None)

        if item_filename is None:
            item_filename = [n.value for n in node]

        if not isinstance(item_filename, list):
            item_filename = [item_filename]

        loaded_items = LoadedContentsDict(item_contents_dir, contents_to_load=set(item_filename))
        with get_progbar(node) as progbar:

            progbar.set_description("Creating Item->Properties links")
            self._graph.add_edges_from((tuple_to_add for tuple_to_add in node_prop_link_generator()))

    def _get_exo_props(self, desired_exo_dict: Dict, item: Content):
        extracted_prop = []
        for exo_id, prop_list_desired in desired_exo_dict.items():
            all_exo_prop: dict = item.get_exogenous_representation(exo_id).value
            # here we filter properties if user specified only certain properties to get
            if prop_list_desired is not None:
                filtered_keys = all_exo_prop.keys() & set(prop_list_desired)
                all_exo_prop = {prop: all_exo_prop[prop] for prop in filtered_keys}

            # we must split eventual properties that are lists
            # eg. {film_director: ['tarantino uri', 'nolan uri']} ->
            # [{film_director: 'tarantino uri'}, {film_director: 'nolan uri'}]
            prop_list_val = {prop_label: all_exo_prop[prop_label] for prop_label in all_exo_prop
                             if isinstance(all_exo_prop[prop_label], Iterable) and
                             not isinstance(all_exo_prop[prop_label], str)}
            if len(prop_list_val) != 0:
                splitted_dict = [{prop_label: prop_val} for prop_label in prop_list_val
                                 for prop_val in prop_list_val[prop_label]]

                prop_single_val = {prop_label: all_exo_prop[prop_label]
                                   for prop_label in all_exo_prop.keys() - prop_list_val.keys()}

                extracted_prop.extend(splitted_dict)
                if len(prop_single_val) != 0:
                    extracted_prop.append(prop_single_val)
            else:
                extracted_prop.append(all_exo_prop)

        return extracted_prop

    def add_link(self, start_node: Union[Node, List[Node]], final_node: Union[Node, List[Node]],
                 weight: float = None, label: str = None, timestamp: str = None):
        """
        Creates a link connecting the `start_node` to the `final_node`. If two lists are passed, then the node in
        position $i$ in the `start_node` list will be linked to the node in position $i$ in the `final_node` list.

        If nodes to link do not exist, they will be added automatically to the graph. Please remember that since this is
        a Tripartite Graph, only *User nodes*, *Item nodes* and *Property nodes* can be added! And *Property nodes* can
        only be linked to *Item nodes*!

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

        Raises:
            ValueError: Exception raised when Property nodes are tried to be linked with non-Item nodes
        """

        def is_not_valid_link(start_n: Node, final_n: Node):
            return (isinstance(final_n, PropertyNode) and not isinstance(start_n, ItemNode)) or \
                   (isinstance(start_n, PropertyNode) and not isinstance(final_n, ItemNode))

        if not isinstance(start_node, list):
            start_node = [start_node]

        if not isinstance(final_node, list):
            final_node = [final_node]

        if any(is_not_valid_link(start_n, final_n) for start_n, final_n in zip(start_node, final_node)):
            raise ValueError("Only item nodes can be linked to property nodes in a Tripartite Graph!")

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

    def __str__(self):
        return "NXTripartiteGraph"

    def __repr__(self):
        return str(self)
