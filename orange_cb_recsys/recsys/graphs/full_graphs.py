from typing import List, Tuple

from orange_cb_recsys.recsys.graphs.graph import FullGraph
import networkx as nx
import pandas as pd


class NXFullGraph(FullGraph):
    def __init__(self, source_frame: pd.DataFrame, user_contents_dir: str = None, item_contents_dir: str = None,
                 user_exogenous_properties: List[str] = None,
                 item_exogenous_properties: List[str] = None,
                 **options):
        super().__init__(source_frame=source_frame, user_contents_dir=user_contents_dir, item_contents_dir=item_contents_dir,
                         user_exogenous_properties=user_exogenous_properties,
                         item_exogenous_properties=item_exogenous_properties,
                         **options)

    def create_graph(self):
        self.__graph = nx.DiGraph()

    @property
    def graph(self):
        return self.__graph

    def add_node(self, node: object):
        self.__graph.add_node(node)

    def add_edge(self, from_node: object, to_node: object, weight: float, label: str = 'weight'):
        self.__graph.add_edge(from_node, to_node, weight=weight, label=label)

    def get_edge_data(self, from_node: object, to_node: object):
        try:
            return self.__graph.get_edge_data(from_node, to_node)
        except ValueError:
            return None

    def get_adj(self, node: object) -> List[Tuple[object, object, float]]:
        return self.__graph.neighbors(node)

    def get_predecessors(self, node: object) -> List[Tuple[object, object, float]]:
        return self.__graph.predecessors(node)

    def get_successors(self, node: object) -> List[Tuple[object, object, float]]:
        return self.__graph.successors(node)
