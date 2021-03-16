from unittest import TestCase

from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import CSVFile
from orange_cb_recsys.evaluation.graph_metrics import nx_degree_centrality, nx_closeness_centrality, nx_dispersion
from orange_cb_recsys.recsys.graphs.full_graphs import NXFullGraph


class TestNXFullGraph(TestCase):
    def test_all(self):

        ratings_filename = "contents/exo_prop/new_ratings_small.csv"
        movies_dir = 'contents/exo_prop/movielens_exo_1612956350.7812138'
        user_dir = 'contents/exo_prop/user_exo_1612956381.4652517'
        try:
            open(ratings_filename)
        except FileNotFoundError:
            ratings_filename = "../../../contents/exo_prop/new_ratings_small.csv"
            movies_dir = '../../../contents/exo_prop/movielens_exo_1612956350.7812138'
            user_dir = '../../../contents/exo_prop/user_exo_1612956381.4652517'

        # movies_ca_config = ContentAnalyzerConfig(
        #     content_type='Item',
        #     source=JSONFile(movies_filename),
        #     id_field_name_list=['imdbID'],
        #     output_directory=output_dir
        # )
        #
        # movies_ca_config.append_exogenous_properties_retrieval(
        #     DBPediaMappingTechnique(
        #         entity_type='Film',
        #         lang='EN',
        #         label_field='Title'
        #     )
        # )
        #
        # ContentAnalyzer(movies_ca_config).fit()
        #
        # users_ca_config = ContentAnalyzerConfig(
        #     content_type='User',
        #     source=DATFile(user_filename),
        #     id_field_name_list=['0'],
        #     output_directory=output_dir
        # )
        #
        # users_ca_config.append_exogenous_properties_retrieval(
        #     PropertiesFromDataset()
        # )
        #
        # ContentAnalyzer(config=users_ca_config).fit()

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

        # Create graph with those properties from that representation
        # EX. create graph with properties 'producer' and 'starring'
        # from representation 0
        g = NXFullGraph(
            source_frame=ratings_frame,
            item_contents_dir=movies_dir,
            user_contents_dir=user_dir,
            item_exo_representation="0",
            user_exo_representation="0",
            item_exo_properties=['producer', 'starring'],
            user_exo_properties=['1']  # It's the column in the users .DAT which identifies the gender
        )

        # Add 'from' node
        self.assertFalse(g.is_from_node('u1'))
        g.add_from_node('u1')
        self.assertTrue(g.is_from_node('u1'))

        # Add 'to' node
        self.assertFalse(g.is_to_node('Tenet'))
        g.add_to_node('Tenet')
        self.assertTrue(g.is_to_node('Tenet'))

        # Add 'property' node
        self.assertFalse(g.is_to_node('Inarritu'))
        g.add_prop_node('Inarritu')
        self.assertTrue(g.is_property_node('Inarritu'))

        # link 'from' and 'to' node already presents in the graph
        self.assertIsNone(g.get_link_data('u1', 'Tenet'))
        g.link_from_to('u1', 'Tenet', 0.5)
        self.assertIsNotNone(g.get_link_data('u1', 'Tenet'))

        # link 'from' and 'to' node which are not presents in the graph
        self.assertFalse(g.is_from_node('u2'))
        self.assertFalse(g.is_to_node('Inception'))
        g.link_from_to('u2', 'Inception', 0.5)
        self.assertTrue(g.is_from_node('u2'))
        self.assertTrue(g.is_to_node('Inception'))

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

        # Create property 'Director' for item 'Tenet', then get all properties from item 'Tenet'
        g.link_prop_node('Tenet', 'Nolan', 0.5, 'Director')
        result = g.get_properties('Tenet')
        expected = [{'Director': 'Nolan'}]
        self.assertEqual(expected, result)

        # Create property for a non existent node, so no property is not created
        # since the node must be existent
        g.link_prop_node('not existent', 'prop', 0.5, 'Director')
        self.assertFalse(g.is_property_node('prop'))
        self.assertIsNone(g.get_link_data('not existent', 'prop'))

        # Get all predecessors of a node
        result = g.get_predecessors('Tenet')
        expected = ['u1']
        self.assertEqual(expected, result)

        # Get all successors of a node
        result = g.get_successors('u1')
        expected = ['Tenet']
        self.assertEqual(expected, result)

        # Add to_tree, so the 'to' node and its properties
        self.assertFalse(g.is_to_node('tt0114709'))
        self.assertFalse(g.is_property_node('http://dbpedia.org/resource/Tom_Hanks'))
        g.add_to_tree('tt0114709')
        self.assertTrue(g.is_to_node('tt0114709'))
        self.assertTrue(g.is_property_node('http://dbpedia.org/resource/Tom_Hanks'))

        # Add from_tree, so the 'from' node and its properties
        self.assertFalse(g.is_from_node('11'))
        g.add_from_tree('11')
        self.assertTrue(g.is_from_node('11'))
        self.assertTrue(g.is_property_node('M'))
        self.assertTrue(g.is_property_node('F'))

        # add a link from u1 to inception for testing multiple voted_contents
        g.link_from_to('u1', 'Inception', 0.5)
        # Get all voted elements by the user u1
        result = g.get_voted_contents('u1')
        expected = ['Tenet', 'Inception']
        self.assertEqual(expected, result)

        # Create graph without setting particular representation,
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

        # simple assert just to make sure the graph is created
        self.assertGreater(len(g.from_nodes), 0)
        self.assertGreater(len(g.to_nodes), 0)
        self.assertGreater(len(g.property_nodes), 0)

        # Create graph without setting particular properties,
        # so ALL exo properties of the representation 0 will be retrieved
        g = NXFullGraph(
            source_frame=ratings_frame,
            item_contents_dir=movies_dir,
            user_contents_dir=user_dir,
            item_exo_representation="0",
            user_exo_representation="0"
        )

        # simple assert just to make sure the graph is created
        self.assertGreater(len(g.from_nodes), 0)
        self.assertGreater(len(g.to_nodes), 0)
        self.assertGreater(len(g.property_nodes), 0)

        # we calculate some metrics, simple assert to make sure they are
        # calculated
        self.assertGreater(len(nx_degree_centrality(g)), 0)
        self.assertGreater(len(nx_closeness_centrality(g)), 0)
        self.assertGreater(len(nx_dispersion(g)), 0)
