from unittest import TestCase

from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import CSVFile
from orange_cb_recsys.recsys.graphs.tripartite_graphs import NXTripartiteGraph
import networkx as nx


class TestNXTripartiteGraph(TestCase):
    def test_all(self):
        ratings_filename = "contents/exo_prop/new_ratings_small.csv"
        movies_dir = 'contents/exo_prop/movielens_exo_1612956350.7812138/'
        try:
            open(ratings_filename)
        except FileNotFoundError:
            ratings_filename = "../../../contents/exo_prop/new_ratings_small.csv"
            movies_dir = '../../../contents/exo_prop/movielens_exo_1612956350.7812138/'

        ratings_import = RatingsImporter(
            source=CSVFile(ratings_filename),
            rating_configs=[RatingsFieldConfig(
                field_name='points',
                processor=NumberNormalizer(min_=1, max_=5))],
            from_field_name='user_id',
            to_field_name='item_id',
            timestamp_field_name='timestamp',
        )
        ratings_frame = ratings_import.import_ratings()

        # Create graph using the property 'starring' from representation '0'
        g = NXTripartiteGraph(ratings_frame, movies_dir,
                              item_exo_representation="0", item_exo_properties=['starring'])

        # Add 'from' node
        self.assertFalse(g.is_from_node('u1'))
        g.add_from_node('u1')
        self.assertTrue(g.is_from_node('u1'))

        # Add 'to' node
        self.assertFalse(g.is_to_node('Tenet'))
        g.add_to_node('Tenet')
        self.assertTrue(g.is_to_node('Tenet'))

        # Add 'property' node
        self.assertFalse(g.is_property_node('Nolan'))
        g.add_prop_node('Nolan')
        self.assertTrue(g.is_property_node('Nolan'))

        # Link existent 'from' node and existent 'to' node
        self.assertIsNone(g.get_link_data('u1', 'Tenet'))
        g.link_from_to('u1', 'Tenet', 0.5)
        self.assertIsNotNone(g.get_link_data('u1', 'Tenet'))

        # Link non-existent 'from' node and non-existent 'to' node,
        # so both nodes will be created
        self.assertFalse(g.is_from_node('u2'))
        self.assertFalse(g.is_to_node('Birdman'))
        self.assertIsNone(g.get_link_data('u2', 'Birdman'))
        g.link_from_to('u2', 'Birdman', 0.5)
        self.assertTrue(g.is_from_node('u2'))
        self.assertTrue(g.is_to_node('Birdman'))
        self.assertIsNotNone(g.get_link_data('u2', 'Birdman'))

        # Make a 'from' node also a 'to' node and a 'to' node also a 'from' node
        g.add_from_node('000')
        g.add_to_node('i1')
        g.link_from_to('i1', '000', 0.5)
        self.assertTrue(g.is_from_node('i1'))
        self.assertTrue(g.is_to_node('000'))

        # Make a 'from' node also a 'prop' node
        self.assertFalse(g.is_property_node('000'))
        g.add_prop_node('000')
        self.assertTrue(g.is_property_node('000'))

        # Link existent 'to' node to existent 'property'
        g.link_prop_node('Tenet', 'Nolan', weight=0.5, label='Director')
        result = g.get_properties('Tenet')
        expected = [{'Director': 'Nolan'}]
        self.assertEqual(expected, result)

        g.link_prop_node('Inception', 'Nolan', weight=0.5, label='Director')
        result = g.get_properties('Inception')
        expected = [{'Director': 'Nolan'}]
        self.assertEqual(expected, result)

        # Link non existent 'to' node to non existent 'property', so both
        # are created
        self.assertFalse(g.is_to_node('Titanic'))
        self.assertFalse(g.is_property_node('DiCaprio'))
        g.link_prop_node('Titanic', 'DiCaprio', weight=0.5, label='Starring')
        self.assertTrue(g.is_to_node('Titanic'))
        self.assertTrue(g.is_property_node('DiCaprio'))

        # Get all predecessors of a node
        result = g.get_predecessors('Nolan')
        expected = ['Tenet', 'Inception']
        self.assertEqual(expected, result)

        # Get all successors of a node
        result = g.get_successors('u1')
        expected = ['Tenet']
        self.assertEqual(expected, result)

        # Add 'to' tree, so add 'to' node and its properties to the graph
        self.assertFalse(g.is_to_node('tt0114709'))
        self.assertFalse(g.is_property_node('http://dbpedia.org/resource/Tom_Hanks'))
        g.add_to_tree('tt0114709')
        self.assertTrue(g.is_property_node('http://dbpedia.org/resource/Tom_Hanks'))
        self.assertTrue(g.is_to_node('tt0114709'))

        # Create graph specifying only the exo representation
        g = NXTripartiteGraph(ratings_frame, movies_dir,
                              item_exo_representation="0")

        # Simple assert just to make sure the graph is created
        self.assertGreater(len(g.from_nodes), 0)
        self.assertGreater(len(g.to_nodes), 0)
        self.assertGreater(len(g.property_nodes), 0)

        # Create graph specifying only the exo representation
        g = NXTripartiteGraph(ratings_frame, movies_dir,
                              item_exo_properties=['starring'])

        # Simple assert just to make sure the graph is created
        self.assertGreater(len(g.from_nodes), 0)
        self.assertGreater(len(g.to_nodes), 0)
        self.assertGreater(len(g.property_nodes), 0)

        # simple assert just to test the _graph method
        self.assertIsInstance(g._graph, nx.DiGraph)
