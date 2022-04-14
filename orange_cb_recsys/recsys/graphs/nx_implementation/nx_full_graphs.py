from typing import List, Union, Dict

from orange_cb_recsys.content_analyzer import Ratings, Content
from orange_cb_recsys.recsys.content_based_algorithm.contents_loader import LoadedContentsDict
from orange_cb_recsys.recsys.graphs.nx_implementation import NXTripartiteGraph
from orange_cb_recsys.recsys.graphs.graph import FullDiGraph, UserNode, PropertyNode, Node
import pandas as pd

from orange_cb_recsys.utils.const import logger, get_progbar


# Multiple Inheritance so that we will use NXTripartite as an interface (we only use its methods)
# and FullGraph as its proper father class (we'll call its __init__)
class NXFullGraph(NXTripartiteGraph, FullDiGraph):
    """
    Class that implements a Full graph through networkx library.
    It supports every node implemented in the framework with no restriction in linking

    It creates a graph from an initial rating frame and if the 'item_contents_dir' or 'user_contents_dir'
    are specified, tries to add properties for every 'to' or 'from' node respectively.
    EXAMPLE::
            _| from_id | to_id | score|
            _|   u1    | Tenet | 0.6  |

        Becomes:
                u1 -----> Tenet
        with the edge weighted and labelled based on the score column and on the
        'default_score_label' parameter
        Then tries to load 'Tenet' from the 'item_contents_dir' if it is specified and if succeeds,
        adds in the graph its loaded properties as specified with 'item_exo_representation' and
        'item_exo_properties'.
        Then tries to load 'u1' from the 'item_contents_dir' if it is specified and if succeeds,
        adds in the graph its loaded properties as specified with 'user_exo_representation' and
        'user_exo_properties'.

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
                 user_exo_properties: Union[Dict, set] = None,
                 user_contents_dir: str = None,
                 link_label: str = None):

        NXTripartiteGraph.__init__(self, source_frame, item_exo_properties, item_contents_dir, link_label)

        if source_frame is not None and user_contents_dir is not None:
            self.add_node_with_prop([UserNode(user_id) for user_id in set(source_frame.user_id_column)],
                                    user_exo_properties,
                                    user_contents_dir)

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

        def node_prop_link_generator():
            for n, id in zip(progbar, content_filename):
                item: Content = loaded_items.get(id)

                exo_props = self._get_exo_props(exo_properties, item)

                single_item_prop_edges = [(n,
                                           PropertyNode(prop_dict[prop]),
                                           {'label': prop})
                                          for prop_dict in exo_props for prop in prop_dict]

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
