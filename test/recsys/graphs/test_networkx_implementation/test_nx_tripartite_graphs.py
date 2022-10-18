from clayrs.recsys.graphs.graph import PropertyNode, UserNode, ItemNode
from clayrs.recsys.graphs.nx_implementation.nx_tripartite_graphs import NXTripartiteGraph
import os

from clayrs.utils import load_content_instance
from test import dir_test_files
from test.recsys.graphs.test_networkx_implementation.test_nx_bipartite_graphs import TestNXBipartiteGraph, rat, \
    rat_timestamp

ratings_filename = os.path.join(dir_test_files, 'new_ratings_small.csv')
movies_dir = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')
users_dir = os.path.join(dir_test_files, 'complex_contents', 'users_codified/')


class TestNXTripartiteGraph(TestNXBipartiteGraph):

    def setUp(self) -> None:
        # graphs that will be used for testing
        self.g: NXTripartiteGraph = NXTripartiteGraph(rat,
                                                      item_contents_dir=movies_dir,
                                                      item_exo_properties={'dbpedia': ['film director',
                                                                                       'runtime (m)']})

        self.graph_custom_label: NXTripartiteGraph = NXTripartiteGraph(rat, link_label='my_label',
                                                                       item_contents_dir=movies_dir,
                                                                       item_exo_properties={'dbpedia': ['film director',
                                                                                                        'runtime (m)']})

        self.graph_timestamp: NXTripartiteGraph = NXTripartiteGraph(rat_timestamp,
                                                                    item_contents_dir=movies_dir,
                                                                    item_exo_properties={'dbpedia': ['film director',
                                                                                                     'runtime (m)']})

        # this will be empty even if other attributes are specified since ratings are missing
        self.empty_graph: NXTripartiteGraph = NXTripartiteGraph(item_contents_dir=movies_dir,
                                                                item_exo_properties={'dbpedia'})

        # item_exo_properties set but no item_contents_dir specified
        self.graph_missing_item_dir_parameter = NXTripartiteGraph(rat, item_exo_properties={'dbpedia': ['film director',
                                                                                                        'runtime (m)']})

        # item_contents_dir set but no item_exo_properties_specified specified
        self.graph_missing_item_prop_parameter = NXTripartiteGraph(rat, item_contents_dir=movies_dir)

    def test_graph_creation(self):
        # the super class test will check if every user and item have a link
        # as they are present in the ratings frame
        super().test_graph_creation()

        # here we test if item nodes are linked to their exogenous property as specified in the constructor
        for item_node in self.g.item_nodes:
            loaded_item = load_content_instance(movies_dir, item_node.value)
            exogenous_representation: dict = loaded_item.get_exogenous_representation("dbpedia").value
            director_prop_expected = exogenous_representation.get("film director", [])
            runtime_prop_expected = exogenous_representation.get("runtime (m)", [])

            if not isinstance(director_prop_expected, list):
                director_prop_expected = [director_prop_expected]

            if not isinstance(runtime_prop_expected, list):
                runtime_prop_expected = [runtime_prop_expected]

            for director in director_prop_expected:
                self.assertTrue(PropertyNode(director) in self.g.property_nodes)
                result_link_data = self.g.get_link_data(item_node, PropertyNode(director))
                expected_link_data = {'label': 'film director'}

                self.assertEqual(expected_link_data, result_link_data)

            for runtime in runtime_prop_expected:
                self.assertTrue(PropertyNode(runtime) in self.g.property_nodes)
                result_link_data = self.g.get_link_data(item_node, PropertyNode(runtime))
                expected_link_data = {'label': 'runtime (m)'}

                self.assertEqual(expected_link_data, result_link_data)

    def test_graph_creation_missing_parameter(self):

        self.assertTrue(len(self.graph_missing_item_dir_parameter.user_nodes) != 0)
        self.assertTrue(len(self.graph_missing_item_dir_parameter.item_nodes) != 0)
        # warning is printed and no prop will be loaded
        self.assertTrue(len(self.graph_missing_item_dir_parameter.property_nodes) == 0)

        self.assertTrue(len(self.graph_missing_item_dir_parameter.user_nodes) != 0)
        self.assertTrue(len(self.graph_missing_item_dir_parameter.item_nodes) != 0)
        # warning is printed and no prop will be loaded
        self.assertTrue(len(self.graph_missing_item_dir_parameter.property_nodes) == 0)

    def test_graph_creation_empty(self):
        super().test_graph_creation_empty()

        self.assertTrue(len(self.empty_graph.property_nodes) == 0)

    def test_add_property(self):
        # Add 'property' node
        self.assertFalse(PropertyNode('Nolan') in self.g.property_nodes)
        self.g.add_node(PropertyNode('Nolan'))
        self.assertTrue(self.g.node_exists(PropertyNode('Nolan')))
        self.assertTrue(PropertyNode('Nolan') in self.g.property_nodes)

        # Add a list of 'property' nodes
        list_nodes = [PropertyNode('prop1'), PropertyNode('prop2'), PropertyNode('prop3')]
        self.g.add_node(list_nodes)
        for n in list_nodes:
            self.assertTrue(self.g.node_exists(n))
            self.assertTrue(n in self.g.property_nodes)

        # Add 'property' node but it already exists as
        # a 'user' node
        self.g.add_node(UserNode('0'))
        self.assertTrue(self.g.node_exists(UserNode('0')))
        self.assertTrue(UserNode('0') in self.g.user_nodes)
        self.g.add_node(PropertyNode('0'))
        self.assertTrue(self.g.node_exists(PropertyNode('0')))
        self.assertTrue(self.g.node_exists(UserNode('0')))
        self.assertTrue(UserNode('0') in self.g.user_nodes)
        self.assertTrue(PropertyNode('0') in self.g.property_nodes)

    def test_add_link_item_prop_existent(self):
        # Link existent 'item' node to an existent 'prop' node
        item_node = ItemNode('Tenet')
        prop_node = PropertyNode('Nolan')
        self.g.add_node(item_node)
        self.g.add_node(prop_node)
        self.assertIsNone(self.g.get_link_data(item_node, prop_node))
        self.g.add_link(item_node, prop_node, timestamp='now', label='directed_by')
        expected = {'timestamp': 'now', 'label': 'directed_by'}
        result = self.g.get_link_data(item_node, prop_node)
        self.assertEqual(expected, result)

        # Link list of existent 'item' nodes to a list of existent 'property' nodes
        items_list = [ItemNode('i1_list'), ItemNode('i2_list'), ItemNode('i3_list')]
        self.g.add_node(items_list)
        props_list = [PropertyNode('p1_list'), PropertyNode('p2_list'), PropertyNode('p3_list')]
        self.g.add_node(props_list)
        self.g.add_link(items_list, props_list, 0.5)
        for user, item in zip(items_list, props_list):
            result = self.g.get_link_data(user, item)
            expected = {'weight': 0.5}

            self.assertEqual(expected, result)

        # Link existent 'property' node to an existent 'item' node
        item_node = ItemNode('Tenet')
        prop_node = PropertyNode('Nolan')
        self.g.add_node(item_node)
        self.g.add_node(prop_node)
        self.assertIsNone(self.g.get_link_data(prop_node, item_node))
        self.g.add_link(prop_node, item_node)
        self.assertIsNotNone(self.g.get_link_data(item_node, prop_node))

        # Link list of existent 'prop' nodes to a list of existent 'item' nodes
        items_list = [ItemNode('i1_list'), ItemNode('i2_list'), ItemNode('i3_list')]
        self.g.add_node(items_list)
        users_list = [UserNode('u1_list'), UserNode('u2_list'), UserNode('u3_list')]
        self.g.add_node(users_list)
        self.g.add_link(items_list, users_list, 0.5)
        for item, user in zip(items_list, users_list):
            result = self.g.get_link_data(item, user)
            expected = {'weight': 0.5}

            self.assertEqual(expected, result)

    def test_add_link_item_prop_non_existent(self):
        # Link non-existent 'user' node and non-existent 'item' node,
        # so both nodes are created and then linked
        item_new = ItemNode('i_new')
        prop_new = PropertyNode('p_new')
        self.assertFalse(self.g.node_exists(item_new))
        self.assertFalse(self.g.node_exists(prop_new))
        self.g.add_link(item_new, prop_new, 0.5)
        self.assertTrue(self.g.node_exists(item_new))
        self.assertTrue(self.g.node_exists(prop_new))
        self.assertIsNotNone(self.g.get_link_data(item_new, prop_new))

        # Link non-existent 'user' node list and non-existent 'item' node list,
        # so all nodes of the two lists are created and then linked
        item_new_list = [ItemNode('i_new_new1'), ItemNode('i_new_new2')]
        prop_new_list = [PropertyNode('u_new_new1'), PropertyNode('u_new_new2')]
        for item in item_new_list:
            self.assertFalse(self.g.node_exists(item))
        for prop in prop_new_list:
            self.assertFalse(self.g.node_exists(prop))

        self.g.add_link(item_new_list, prop_new_list, 0.5)

        for user, item in zip(item_new_list, prop_new_list):
            result = self.g.get_link_data(user, item)
            expected = {'weight': 0.5}

            self.assertEqual(expected, result)

    def test_add_link_raise_error(self):
        # test link existent 'user' node to existent 'property' node
        self.g.add_node(UserNode('u1'))
        self.g.add_node(PropertyNode('Nolan'))
        with self.assertRaises(ValueError):
            self.g.add_link(UserNode('u1'), PropertyNode('Nolan'), weight=0.5, label='Friend')

        # test link 'property' node to 'user' node
        self.g.add_node(UserNode('u1'))
        self.g.add_node(PropertyNode('Nolan'))
        with self.assertRaises(ValueError):
            self.g.add_link(PropertyNode('Nolan'), UserNode('u1'), weight=0.5, label='Friend')

        # test link list of non-existent user node to list of non-existent property node
        user_list = [UserNode('u_new'), UserNode('u_new1')]
        prop_list = [PropertyNode('p_new'), PropertyNode('p_new1')]
        for n in user_list:
            self.assertFalse(self.g.node_exists(n))
        for n in prop_list:
            self.assertFalse(self.g.node_exists(n))

        with self.assertRaises(ValueError):
            self.g.add_link(user_list, prop_list, weight=0.5, label="PropertyNew")

    def test_add_node_with_prop(self):
        # add 'item' node and only properties 'film_director' to the graph
        self.assertFalse(ItemNode('tt0114709') in self.g.item_nodes)
        self.assertFalse(PropertyNode('http://dbpedia.org/resource/John_Lasseter') in self.g.property_nodes)
        self.g.add_node_with_prop(ItemNode('tt0114709'), {'dbpedia': ['film director']}, movies_dir)
        self.assertTrue(ItemNode('tt0114709') in self.g.item_nodes)
        self.assertTrue(PropertyNode('http://dbpedia.org/resource/John_Lasseter') in self.g.property_nodes)
        self.assertIsNotNone(self.g.get_link_data(ItemNode('tt0114709'),
                                                  PropertyNode('http://dbpedia.org/resource/John_Lasseter')))

        self.g.remove_node(ItemNode('tt0114709'))
        self.g.remove_node(PropertyNode('http://dbpedia.org/resource/John_Lasseter'))

        # add 'item' node and only properties 'film_director' to the graph with different id from the file name
        self.assertFalse(ItemNode('different') in self.g.item_nodes)
        self.assertFalse(PropertyNode('http://dbpedia.org/resource/John_Lasseter') in self.g.property_nodes)
        self.g.add_node_with_prop(ItemNode('different'), {'dbpedia': ['film director']}, movies_dir,
                                  'tt0114709')
        self.assertTrue(ItemNode('different') in self.g.item_nodes)
        self.assertTrue(PropertyNode('http://dbpedia.org/resource/John_Lasseter') in self.g.property_nodes)
        self.assertIsNotNone(self.g.get_link_data(ItemNode('different'),
                                                  PropertyNode('http://dbpedia.org/resource/John_Lasseter')))

        self.g.remove_node(ItemNode('different'))
        self.g.remove_node(PropertyNode('http://dbpedia.org/resource/John_Lasseter'))

        # add 'item' node and all its properties
        loaded_item = load_content_instance(movies_dir, 'tt0114709')
        exogenous_representation: dict = loaded_item.get_exogenous_representation("dbpedia").value

        self.assertFalse(ItemNode('tt0114709') in self.g.item_nodes)
        self.assertFalse(PropertyNode('http://dbpedia.org/resource/John_Lasseter') in self.g.property_nodes)
        self.g.add_node_with_prop(ItemNode('tt0114709'), {'dbpedia'}, movies_dir)
        self.assertTrue(ItemNode('tt0114709') in self.g.item_nodes)

        for prop_label in exogenous_representation.keys():
            prop_val_expected = exogenous_representation.get(prop_label, [])

            if not isinstance(prop_val_expected, list):
                prop_val_expected = [prop_val_expected]

            for prop_val in prop_val_expected:
                self.assertTrue(PropertyNode(prop_val) in self.g.property_nodes)
                result_link_data_list = self.g.get_link_data(ItemNode('tt0114709'), PropertyNode(prop_val))

                # we make sure that this is a property of the exogenous representation
                self.assertIn(result_link_data_list['label'], exogenous_representation)

    def test_add_raise_error(self):
        # Try to add 'user' node with its prop
        with self.assertRaises(ValueError):
            self.g.add_node_with_prop(UserNode('user'), {'representation_id'}, users_dir)

    # def test_convert_to_dataframe(self):
    #     converted_df = self.g.convert_to_dataframe()
    #     self.assertNotIn('label', converted_df.columns)
    #     for user, item in zip(converted_df['from_id'], converted_df['to_id']):
    #         self.assertIsInstance(user, Node)
    #         self.assertIsInstance(item, Node)
    #
    #     converted_df = converted_df.query('to_id not in @self.g.property_nodes')
    #     result = np.sort(converted_df, axis=0)
    #     expected = np.sort(self.df, axis=0)
    #     self.assertTrue(np.array_equal(expected, result))
    #
    #     converted_df = self.g.convert_to_dataframe(only_values=True, with_label=True)
    #     self.assertIn('label', converted_df.columns)
    #     for user, item in zip(converted_df['from_id'], converted_df['to_id']):
    #         self.assertNotIsInstance(user, Node)
    #         self.assertNotIsInstance(item, Node)
    #
    #     converted_df = converted_df.query('to_id not in @self.g.property_nodes')[['from_id', 'to_id', 'score']]
    #     result = np.sort(converted_df, axis=0)
    #     expected = np.sort(self.df, axis=0)
    #     self.assertTrue(np.array_equal(expected, result))
