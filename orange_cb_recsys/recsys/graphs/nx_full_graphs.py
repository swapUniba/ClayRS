from typing import List, Union

from orange_cb_recsys.recsys.graphs import NXTripartiteGraph
from orange_cb_recsys.recsys.graphs.graph import FullGraph
import networkx as nx
import pandas as pd

from orange_cb_recsys.utils.const import logger


# Multiple Inheritance so that we will use NXTripartite as an interface (we only use its methods)
# and FullGraph as its proper father class (we'll call its __init__)
class NXFullGraph(NXTripartiteGraph, FullGraph):
    """
    Class that implements a Full graph through networkx library.
    It supports 'from' node, 'to' and 'property' node

    It creates a graph from an initial rating frame and if the 'item_contents_dir' or 'user_contents_dir'
    are specified, tries to add properties for every 'to' or 'from' node respectively.
    EXAMPLE:
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
            Default is 'score_label'
        default_not_rated_value (float): the default value with which the link will be weighted
            Default is 0.5

    """

    def __init__(self, source_frame: pd.DataFrame, user_contents_dir: str = None, item_contents_dir: str = None,
                 user_exo_properties: List[str] = None, user_exo_representation: Union[str, int] = None,
                 item_exo_properties: List[str] = None, item_exo_representation: Union[str, int] = None,
                 default_score_label: str = 'score', default_not_rated_value: float = 0.5):

        FullGraph.__init__(self, source_frame=source_frame,
                           user_contents_dir=user_contents_dir, item_contents_dir=item_contents_dir,
                           user_exo_properties=user_exo_properties, user_exo_representation=user_exo_representation,
                           item_exo_properties=item_exo_properties, item_exo_representation=item_exo_representation,
                           default_score_label=default_score_label, default_weight=default_not_rated_value)

    def add_link(self, start_node: object, final_node: object,
                 weight: float = None, label: str = None):
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
        if weight is None:
            weight = self.get_default_weight()
        if label is None:
            label = self.get_default_score_label()

        if self.node_exists(start_node) and self.node_exists(final_node):
            # We must to this so that if the 'final' node passed is 'i1' and in the graph it's a 'ItemNode'
            # we get its instance and link the start node to the instance, otherwise networkx
            # links 'start' node to the string 'i1' and not the ItemNode!!
            nodes_list = list(self._graph.nodes)
            index_first = nodes_list.index(start_node)
            index_second = nodes_list.index(final_node)

            self._graph.add_edge(nodes_list[index_first], nodes_list[index_second], weight=weight, label=label)
        else:
            logger.warning("One of the nodes or both don't exist in the graph! Add them before "
                           "calling this method.")

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return \
                self.get_user_contents_dir() == other.get_user_contents_dir() and \
                self.get_item_contents_dir() == other.get_item_contents_dir() and \
                self.get_user_exogenous_properties() == other.get_user_exogenous_properties() and \
                self.get_user_exogenous_representation() == other.get_user_exogenous_representation() and \
                self.get_item_exogenous_properties() == other.get_item_exogenous_properties() and \
                self.get_item_exogenous_representation() == other.get_item_exogenous_representation() and \
                self.get_default_score_label() == other.get_default_score_label() and \
                self.get_default_weight() == other.get_default_weight() and \
                nx.algorithms.isomorphism.is_isomorphic(self._graph, other._graph)
        else:
            return False
