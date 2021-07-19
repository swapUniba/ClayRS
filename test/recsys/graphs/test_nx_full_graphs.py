from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.raw_information_source import CSVFile
from orange_cb_recsys.recsys.graphs.nx_full_graphs import NXFullGraph
import os
import pandas as pd
import numpy as np

from orange_cb_recsys.recsys.graphs.graph import Node
from test.recsys.graphs.test_nx_tripartite_graphs import TestNXTripartiteGraph

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

contents_path = os.path.join(THIS_DIR, '../../../contents')

ratings_filename = os.path.join(contents_path, 'exo_prop/new_ratings_small.csv')
movies_dir = os.path.join(contents_path, 'movies_codified/')
user_dir = os.path.join(contents_path, 'users_codified/')


class TestNXFullGraph(TestNXTripartiteGraph):

    def setUp(self) -> None:
        self.df = pd.DataFrame.from_dict({'from_id': ["1", "1", "2", "2", "2", "3", "4", "4"],
                                          'to_id': ["tt0112281", "tt0112302", "tt0112281", "tt0112346",
                                                    "tt0112453", "tt0112453", "tt0112346", "tt0112453"],
                                          'score': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7]})

        self.g: NXFullGraph = NXFullGraph(self.df,
                                          user_contents_dir=user_dir,
                                          item_contents_dir=movies_dir,
                                          item_exo_representation="dbpedia",
                                          user_exo_representation='local',
                                          item_exo_properties=['film director'],
                                          user_exo_properties=['1']  # It's the column in the users .DAT which
                                          # identifies the gender
                                          )

    def test_populate_from_dataframe_w_labels(self):
        df_label = pd.DataFrame.from_dict({'from_id': ["1", "1", "2", "2", "2", "3", "4", "4"],
                                           'to_id': ["tt0112281", "tt0112302", "tt0112281", "tt0112346",
                                                     "tt0112453", "tt0112453", "tt0112346", "tt0112453"],
                                           'score': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7],
                                           'label': ['score_df', 'score_df', 'score_df', 'score_df',
                                                     'score_df', 'score_df', 'score_df', 'score_df']
                                           })

        g: NXFullGraph = NXFullGraph(df_label,
                                     user_contents_dir=user_dir,
                                     item_contents_dir=movies_dir,
                                     item_exo_representation="dbpedia",
                                     user_exo_representation="local",
                                     item_exo_properties=['starring'],
                                     user_exo_properties=['1']  # It's the column in the users .DAT which
                                     # identifies the gender
                                     )

        for user, item, score in zip(df_label['from_id'], df_label['to_id'], df_label['score']):
            expected = {'label': 'score_df', 'weight': score}
            result = g.get_link_data(user, item)

            self.assertEqual(expected, result)

    def test_graph_created(self):
        # Simple assert just to make sure the graph is created
        self.assertGreater(len(self.g.user_nodes), 0)
        self.assertGreater(len(self.g.item_nodes), 0)
        self.assertGreater(len(self.g.property_nodes), 0)

    def test_add_link_user_property(self):
        # Link existent 'user' node to existent 'property' node
        self.g.add_user_node('u1')
        self.g.add_property_node('Nolan')
        self.g.add_link('u1', 'Nolan', weight=0.5, label='Friend')
        result = self.g.get_properties('u1')
        expected = [{'Friend': 'Nolan'}]
        self.assertEqual(expected, result)

        # Link existent 'user' node to a list of existent 'property'
        self.g.add_user_node('u_list')
        properties_list = ['prop1', 'prop2', 'prop3']
        self.g.add_property_node(properties_list)
        self.g.add_link('u_list', properties_list, weight=0.5, label='starring')
        result = self.g.get_properties('u_list')
        expected = [{'starring': 'prop1'}, {'starring': 'prop2'}, {'starring': 'prop3'}]
        self.assertEqual(expected, result)

        # Link existent 'property' node to existent 'user' node
        self.g.add_property_node('Nolan')
        self.g.add_user_node('u2')
        self.g.add_link('Nolan', 'u2', weight=0.5, label='Friend')
        result = self.g.get_link_data('Nolan', 'u2')
        expected = {'label': 'Friend', 'weight': 0.5}
        self.assertEqual(expected, result)

        # Link existent 'property' node to a list of existent 'user' node
        self.g.add_property_node('prop_list')
        users_list = ['u1_list', 'u2_list', 'u3_list']
        self.g.add_item_node(users_list)
        self.g.add_link('prop_list', users_list, weight=0.5, label='Director of')
        for item in users_list:
            result = self.g.get_link_data('prop_list', item)
            expected = {'label': 'Director of', 'weight': 0.5}
            self.assertEqual(expected, result)

        # Try to link non-existent 'user' node and non-existent 'property' node,
        # so no link is created
        self.assertFalse(self.g.node_exists('u_new'))
        self.assertFalse(self.g.node_exists('prop_new'))
        self.g.add_link('u_new', 'prop_new', weight=0.5, label="PropertyNew")
        self.assertFalse(self.g.is_user_node('u_new'))
        self.assertFalse(self.g.is_property_node('prop_new'))
        self.assertIsNone(self.g.get_link_data('u_new', 'prop_new'))

    def test_add_user_tree(self):
        # Add from_tree, so the 'from' node and its properties
        self.assertFalse(self.g.is_user_node('11'))
        self.g.add_user_tree('11')
        self.assertTrue(self.g.is_user_node('11'))
        self.assertTrue(self.g.is_property_node('M'))
        self.assertTrue(self.g.is_property_node('F'))

        # Try to add 'item' tree
        self.g.add_item_node('tt0112641')
        self.assertTrue(self.g.is_item_node('tt0112641'))
        self.g.add_user_tree('tt0112641')
        expected = []
        result = self.g.get_properties('tt0112641')
        self.assertEqual(expected, result)
        self.assertTrue(self.g.is_user_node('tt0112641'))

    def test_graph_creation(self):
        # Test multiple graph creation possibilities

        # Import ratings as DataFrame
        ratings_import = RatingsImporter(
            source=CSVFile(ratings_filename),
            from_id_column='user_id',
            to_id_column='item_id',
            score_column='points',
            timestamp_column='timestamp',
            score_processor=NumberNormalizer()
        )
        ratings_frame = ratings_import.import_ratings()

        # Create graph without setting the representation
        # EX. Create graph with properties 'producer' and 'starring' from
        # all exo representation, since there can be multiple exo representation
        # containing the same properties
        g = NXFullGraph(
            source_frame=ratings_frame,
            item_contents_dir=movies_dir,
            user_contents_dir=user_dir,
            item_exo_properties=['producer', 'starring'],
            user_exo_properties=['1']  # It's the column in the users DAT which identifies the gender
        )

        # Simple assert just to make sure the graph is created
        self.assertGreater(len(g.user_nodes), 0)
        self.assertGreater(len(g.item_nodes), 0)
        self.assertGreater(len(g.property_nodes), 0)

        # Create graph without setting properties,
        # so ALL exo properties of the representation 0 will be retrieved
        g = NXFullGraph(
            source_frame=ratings_frame,
            item_contents_dir=movies_dir,
            user_contents_dir=user_dir,
            item_exo_representation="dbpedia",
            user_exo_representation="local"
        )

        # Simple assert just to make sure the graph is created
        self.assertGreater(len(g.user_nodes), 0)
        self.assertGreater(len(g.item_nodes), 0)
        self.assertGreater(len(g.property_nodes), 0)

        # Create graph specifying without properties
        g = NXFullGraph(ratings_frame)

        # Simple assert just to make sure the graph is created
        self.assertGreater(len(g.user_nodes), 0)
        self.assertGreater(len(g.item_nodes), 0)
        self.assertEqual(len(g.property_nodes), 0)

    def test_graph_creation_exo_missing(self):
        # Test multiple graph creation possibilities with not existent exo_representations/exo_properties

        # Import ratings as DataFrame
        ratings_import = RatingsImporter(
            source=CSVFile(ratings_filename),
            from_id_column='user_id',
            to_id_column='item_id',
            score_column='points',
            timestamp_column='timestamp',
            score_processor=NumberNormalizer()
        )
        ratings_frame = ratings_import.import_ratings()

        # Create graph with non-existent exo_properties
        g = NXFullGraph(
            source_frame=ratings_frame,
            item_contents_dir=movies_dir,
            user_contents_dir=user_dir,
            item_exo_properties=['asdds', 'dsdds'],
            user_exo_properties=['vvvv']  # It's the column in the users DAT which identifies the gender
        )

        # Simple assert just to make sure the graph is created
        self.assertGreater(len(g.user_nodes), 0)
        self.assertGreater(len(g.item_nodes), 0)
        self.assertEqual(len(g.property_nodes), 0)

        # Create graph with non-existent exo_representations
        g = NXFullGraph(
            source_frame=ratings_frame,
            item_contents_dir=movies_dir,
            user_contents_dir=user_dir,
            item_exo_representation="asdsa",
            user_exo_representation="dsdssd"
        )

        # Simple assert just to make sure the graph is created
        self.assertGreater(len(g.user_nodes), 0)
        self.assertGreater(len(g.item_nodes), 0)
        self.assertEqual(len(g.property_nodes), 0)

        # Create graph with non-existent exo_representations and non-existent exo_properties
        g = NXFullGraph(
            source_frame=ratings_frame,
            item_contents_dir=movies_dir,
            user_contents_dir=user_dir,
            user_exo_representation='not_exist',
            item_exo_representation='not_Exist2',
            item_exo_properties=["asdsa"],
            user_exo_properties=["dsdssd"]
        )

        # Simple assert just to make sure the graph is created
        self.assertGreater(len(g.user_nodes), 0)
        self.assertGreater(len(g.item_nodes), 0)
        self.assertEqual(len(g.property_nodes), 0)

    def test_convert_to_dataframe(self):
        converted_df = self.g.convert_to_dataframe()
        self.assertNotIn('label', converted_df.columns)
        for user, item in zip(converted_df['from_id'], converted_df['to_id']):
            self.assertIsInstance(user, Node)
            self.assertIsInstance(item, Node)

        converted_df = converted_df.query('to_id not in @self.g.property_nodes')
        result = np.sort(converted_df, axis=0)
        expected = np.sort(self.df, axis=0)
        self.assertTrue(np.array_equal(expected, result))

        converted_df = self.g.convert_to_dataframe(only_values=True, with_label=True)
        self.assertIn('label', converted_df.columns)
        for user, item in zip(converted_df['from_id'], converted_df['to_id']):
            self.assertNotIsInstance(user, Node)
            self.assertNotIsInstance(item, Node)

        converted_df = converted_df.query('to_id not in @self.g.property_nodes')[['from_id', 'to_id', 'score']]
        result = np.sort(converted_df, axis=0)
        expected = np.sort(self.df, axis=0)
        self.assertTrue(np.array_equal(expected, result))
