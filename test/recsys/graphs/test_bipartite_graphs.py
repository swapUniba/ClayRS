from unittest import TestCase
import pandas as pd
from orange_cb_recsys.recsys.graphs import NXBipartiteGraph


class TestNXBipartiteGraph(TestCase):
    def test_all(self):
        df = pd.DataFrame.from_dict({'from_id': ["001", "001", "002", "002", "002", "003", "004", "004"],
                                     'to_id': ["aaa", "bbb", "aaa", "ddd", "ccc", "ccc", "ddd", "ccc"],
                                     'score': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7]})
        g = NXBipartiteGraph(df)
        g.add_node('000')
        g.add_edge('000', 'aaa', 0.5)
        g.get_adj('aaa')
        g.get_edge_data('002', 'ccc')
        g.get_predecessors('aaa')
        g.get_successors('000')
