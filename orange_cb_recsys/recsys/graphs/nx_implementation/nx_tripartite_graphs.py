from collections.abc import Iterable
from typing import List, Set, Union, Dict

from orange_cb_recsys.content_analyzer import Ratings, Content
from orange_cb_recsys.recsys.content_based_algorithm.contents_loader import LoadedContentsDict
from orange_cb_recsys.recsys.graphs.nx_implementation.nx_bipartite_graphs import NXBipartiteGraph

from orange_cb_recsys.recsys.graphs.graph import TripartiteDiGraph, ItemNode, Node
import pandas as pd

from orange_cb_recsys.recsys.graphs.graph import PropertyNode
from orange_cb_recsys.utils.const import logger, get_progbar


# Multiple Inheritance so that we will use NXBipartite as its proper father class (we'll its __init__ method)
# and TripartiteDiGraph as its interface
class NXTripartiteGraph(NXBipartiteGraph, TripartiteDiGraph):
    """
    Class that implements a Tripartite graph through networkx library.
    It supports 'user' node, 'item' and 'property' node, but the latter ones are available only for
    are only allowed to be linked to 'item' nodes.

    It creates a graph from an initial rating frame and if the 'item_contents_dir' is specified,
    tries to add properties for every 'item' node.
    EXAMPLE::
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
            Default is 'score'
        default_not_rated_value (float): the default value with which the link will be weighted
            Default is 0.5

    """

    def __init__(self, source_frame: Ratings = None,
                 item_exo_properties: Union[Dict, set] = None,
                 item_contents_dir: str = None,
                 link_label: str = None):

        NXBipartiteGraph.__init__(self, source_frame, link_label)

        if source_frame is not None and item_contents_dir is not None:
            self.add_node_with_prop([ItemNode(item_id) for item_id in set(source_frame.item_id_column)],
                                    item_exo_properties,
                                    item_contents_dir)

    @property
    def property_nodes(self) -> Set[PropertyNode]:
        """
        Returns a set of all 'property' nodes in the graph
        """
        return set(node for node in self._graph.nodes if isinstance(node, PropertyNode))

    def add_node(self, node: Union[Node, List[Node]]):
        """
        Adds a 'user' node to the graph.
        If a list is passed, then every element of the list will be added as a 'user' node

        Args:
            node: node(s) that needs to be added to the graph as 'user' node(s)
        """
        if not isinstance(node, list):
            node = [node]

        self._graph.add_nodes_from(node)

    def add_node_with_prop(self, node: Union[ItemNode, List[ItemNode]], item_exo_properties: Union[Dict, set],
                           item_contents_dir: str,
                           item_filename: Union[str, List[str]] = None):

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
            progbar.set_description("Creating Item->Properties links list")

            all_item_prop_edges = []
            for n, id in zip(progbar, item_filename):
                item: Content = loaded_items.get(id)

                exo_props = self._get_exo_props(item_exo_properties, item)

                single_item_prop_edges = [(n,
                                           PropertyNode(prop_dict[prop]),
                                           {'label': prop})
                                          for prop_dict in exo_props for prop in prop_dict]

                all_item_prop_edges.extend(single_item_prop_edges)

            logger.info("Adding Item->Properties links list to NetworkX graph...")
            self._graph.add_edges_from(all_item_prop_edges)

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
