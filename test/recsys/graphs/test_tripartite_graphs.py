from unittest import TestCase
import pandas as pd
from orange_cb_recsys.recsys.graphs.tripartite_graphs import NXTripartiteGraph


class TestNXTripartiteGraph(TestCase):
    def test_all(self):
        df = pd.DataFrame.from_dict({'from_id': ["001", "001", "002", "002", "002", "003", "004", "004"],
                                     'to_id': ["tt0112281", "tt0112302", "tt0112281", "ddd",
                                               "tt0112346", "tt0112346", "ddd", "tt0112346"],
                                     'score': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7]})
        g = NXTripartiteGraph(df)
        g.add_node('000')
        g.add_edge('000', 'aaa', 0.5)
        g.get_adj('aaa')
        g.get_edge_data('002', 'ccc')
        g.get_predecessors('aaa')
        g.get_successors('000')

        g = NXTripartiteGraph(df, 'contents/movielens_test_exo_prop1593012430.4516225')
        g.add_node('000')
        g.add_edge('000', 'aaa', 0.5)
        g.get_adj('aaa')
        g.get_edge_data('002', 'ccc')
        g.get_predecessors('aaa')
        g.get_successors('000')
        g.add_tree('tt0113845')


