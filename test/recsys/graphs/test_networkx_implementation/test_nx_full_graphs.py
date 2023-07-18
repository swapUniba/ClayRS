from clayrs.recsys.graphs.nx_implementation import NXFullGraph
import os

from clayrs.recsys.graphs.graph import PropertyNode, UserNode
from clayrs.utils import load_content_instance
from test.recsys.graphs.test_networkx_implementation.test_nx_tripartite_graphs import TestNXTripartiteGraph
from test.recsys.graphs.test_networkx_implementation.test_nx_bipartite_graphs import rat, rat_timestamp
from test import dir_test_files

ratings_filename = os.path.join(dir_test_files, 'new_ratings_small.csv')
movies_dir = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')
users_dir = os.path.join(dir_test_files, 'complex_contents', 'users_codified/')


class TestNXFullGraph(TestNXTripartiteGraph):

    def setUp(self) -> None:
        # graphs that will be used for testing
        self.g: NXFullGraph = NXFullGraph(rat,
                                          item_contents_dir=movies_dir,
                                          item_exo_properties={'dbpedia': ['film director',
                                                                           'runtime (m)']},

                                          # It's the column in the users .DAT which identifies the gender
                                          user_exo_properties={'local': '1'},
                                          user_contents_dir=users_dir)

        self.graph_custom_label: NXFullGraph = NXFullGraph(rat, link_label='my_label',
                                                           item_contents_dir=movies_dir,
                                                           item_exo_properties={'dbpedia': ['film director',
                                                                                            'runtime (m)']},

                                                           # It's the column in the users .DAT which identifies the
                                                           # gender
                                                           user_exo_properties={'local': '1'},
                                                           user_contents_dir=users_dir)

        self.graph_timestamp: NXFullGraph = NXFullGraph(rat_timestamp,
                                                        item_contents_dir=movies_dir,
                                                        item_exo_properties={'dbpedia': ['film director',
                                                                                         'runtime (m)']},

                                                        # It's the column in the users .DAT which identifies the gender
                                                        user_exo_properties={'local': '1'},
                                                        user_contents_dir=users_dir)

        # this will be empty even if other attributes are specified since ratings are missing
        self.empty_graph: NXFullGraph = NXFullGraph(item_contents_dir=movies_dir,
                                                    item_exo_properties={'dbpedia': ['film director',
                                                                                     'runtime (m)']},

                                                    # It's the column in the users .DAT which identifies the gender
                                                    user_exo_properties={'local': '1'},
                                                    user_contents_dir=users_dir)

        # item_exo_properties set but no item_contents_dir specified
        self.graph_missing_item_dir_parameter = NXFullGraph(rat, item_exo_properties={'dbpedia': ['film director',
                                                                                                  'runtime (m)']})

        # item_contents_dir set but no item_exo_properties_specified specified
        self.graph_missing_item_prop_parameter = NXFullGraph(rat, item_contents_dir=movies_dir)

        # user_exo_properties set but no user_contents_dir specified
        self.graph_missing_user_dir_parameter = NXFullGraph(rat, user_exo_properties={'local'})

        # user_contents_dir set but no user_exo_properties_specified specified
        self.graph_missing_user_prop_parameter = NXFullGraph(rat, user_contents_dir=users_dir)

    def test_graph_creation(self):
        # the super class test will check if every user and item have a link
        # as they are present in the ratings frame and that each item is linked to its property
        super().test_graph_creation()

        # here we test if item nodes are linked to their exogenous property as specified in the constructor
        for user_node in self.g.user_nodes:
            loaded_item = load_content_instance(users_dir, user_node.value)
            exogenous_representation: dict = loaded_item.get_exogenous_representation("local").value
            gender_prop_expected = exogenous_representation.get("1", [])

            if not isinstance(gender_prop_expected, list):
                gender_prop_expected = [gender_prop_expected]

            for gender in gender_prop_expected:
                self.assertTrue(PropertyNode(gender) in self.g.property_nodes)
                result_link_data = self.g.get_link_data(user_node, PropertyNode(gender))
                expected_link_data = {'label': '1'}

                self.assertEqual(expected_link_data, result_link_data)

    def test_graph_creation_missing_parameter(self):

        # super class method will test missing parameters related to items
        super().test_graph_creation_missing_parameter()

        self.assertTrue(len(self.graph_missing_user_dir_parameter.user_nodes) != 0)
        self.assertTrue(len(self.graph_missing_user_dir_parameter.item_nodes) != 0)
        # warning is printed and no prop will be loaded
        self.assertTrue(len(self.graph_missing_user_dir_parameter.property_nodes) == 0)

        self.assertTrue(len(self.graph_missing_user_prop_parameter.user_nodes) != 0)
        self.assertTrue(len(self.graph_missing_user_prop_parameter.item_nodes) != 0)
        # warning is printed and no prop will be loaded
        self.assertTrue(len(self.graph_missing_user_prop_parameter.property_nodes) == 0)

    def test_add_link_user_prop_existent(self):
        # Link existent 'user' node to an existent 'prop' node
        user_node = UserNode('u_prop')
        prop_node = PropertyNode('Nolan')
        self.g.add_node(user_node)
        self.g.add_node(prop_node)
        self.assertIsNone(self.g.get_link_data(user_node, prop_node))
        self.g.add_link(user_node, prop_node, timestamp='now', label='directed_by')
        expected = {'timestamp': 'now', 'label': 'directed_by'}
        result = self.g.get_link_data(user_node, prop_node)
        self.assertEqual(expected, result)

        # Link list of existent 'user' nodes to a list of existent 'property' nodes
        items_list = [UserNode('i1_list'), UserNode('i2_list'), UserNode('i3_list')]
        self.g.add_node(items_list)
        props_list = [PropertyNode('p1_list'), PropertyNode('p2_list'), PropertyNode('p3_list')]
        self.g.add_node(props_list)
        self.g.add_link(items_list, props_list, 0.5)
        for user, item in zip(items_list, props_list):
            result = self.g.get_link_data(user, item)
            expected = {'weight': 0.5}

            self.assertEqual(expected, result)

        # Link existent 'property' node to an existent 'item' node
        user_node = UserNode('u_prop')
        prop_node = PropertyNode('Nolan')
        self.g.add_node(user_node)
        self.g.add_node(prop_node)
        self.assertIsNone(self.g.get_link_data(prop_node, user_node))
        self.g.add_link(prop_node, user_node)
        self.assertIsNotNone(self.g.get_link_data(user_node, prop_node))

        # Link list of existent 'prop' nodes to a list of existent 'item' nodes
        items_list = [UserNode('i1_list'), UserNode('i2_list'), UserNode('i3_list')]
        self.g.add_node(items_list)
        users_list = [UserNode('u1_list'), UserNode('u2_list'), UserNode('u3_list')]
        self.g.add_node(users_list)
        self.g.add_link(items_list, users_list, 0.5)
        for item, user in zip(items_list, users_list):
            result = self.g.get_link_data(item, user)
            expected = {'weight': 0.5}

            self.assertEqual(expected, result)

    def test_add_link_user_prop_non_existent(self):
        # Link non-existent 'user' node and non-existent 'item' node,
        # so both nodes are created and then linked
        user_new = UserNode('u_new')
        prop_new = PropertyNode('p_new')
        self.assertFalse(self.g.node_exists(user_new))
        self.assertFalse(self.g.node_exists(prop_new))
        self.g.add_link(user_new, prop_new, 0.5)
        self.assertTrue(self.g.node_exists(user_new))
        self.assertTrue(self.g.node_exists(prop_new))
        self.assertIsNotNone(self.g.get_link_data(user_new, prop_new))

        # Link non-existent 'user' node list and non-existent 'item' node list,
        # so all nodes of the two lists are created and then linked
        user_new_list = [UserNode('i_new_new1'), UserNode('i_new_new2')]
        prop_new_list = [PropertyNode('u_new_new1'), PropertyNode('u_new_new2')]
        for user in user_new_list:
            self.assertFalse(self.g.node_exists(user))
        for prop in prop_new_list:
            self.assertFalse(self.g.node_exists(prop))

        self.g.add_link(user_new_list, prop_new_list, 0.5)

        for user, item in zip(user_new_list, prop_new_list):
            result = self.g.get_link_data(user, item)
            expected = {'weight': 0.5}

            self.assertEqual(expected, result)

    # this needs to override superclass test, since for tripartite graph we have no possible error
    # when adding new links
    def test_add_link_raise_error(self):
        pass

    def test_add_node_with_prop(self):
        super().test_add_node_with_prop()

        # add 'user' node and only properties '2' to the graph
        self.assertFalse(UserNode('6') in self.g.user_nodes)
        self.assertFalse(PropertyNode('50') in self.g.property_nodes)
        self.g.add_node_with_prop(UserNode('6'), {'local': ['2']}, users_dir)
        self.assertTrue(UserNode('6') in self.g.user_nodes)
        self.assertTrue(PropertyNode('50') in self.g.property_nodes)
        self.assertIsNotNone(self.g.get_link_data(UserNode('6'),
                                                  PropertyNode('50')))

        self.g.remove_node(UserNode('6'))
        self.g.remove_node(PropertyNode('50'))

        # add 'user' node and only property'2' to the graph with different id from the file name
        self.assertFalse(UserNode('different') in self.g.user_nodes)
        self.assertFalse(PropertyNode('50') in self.g.property_nodes)
        self.g.add_node_with_prop(UserNode('different'), {'local': ['2']}, users_dir,
                                  '6')
        self.assertTrue(UserNode('different') in self.g.user_nodes)
        self.assertTrue(PropertyNode('50') in self.g.property_nodes)
        self.assertIsNotNone(self.g.get_link_data(UserNode('different'),
                                                  PropertyNode('50')))

        self.g.remove_node(UserNode('different'))
        self.g.remove_node(PropertyNode('50'))

        # add 'item' node and all its properties
        loaded_item = load_content_instance(users_dir, '6')
        exogenous_representation: dict = loaded_item.get_exogenous_representation("local").value

        self.assertFalse(UserNode('6') in self.g.item_nodes)
        self.assertFalse(PropertyNode('50') in self.g.property_nodes)
        self.g.add_node_with_prop(UserNode('6'), {'local'}, users_dir)
        self.assertTrue(UserNode('6') in self.g.user_nodes)

        for prop_label in exogenous_representation.keys():
            prop_val_expected = exogenous_representation.get(prop_label, [])

            if not isinstance(prop_val_expected, list):
                prop_val_expected = [prop_val_expected]

            for prop_val in prop_val_expected:
                self.assertTrue(PropertyNode(prop_val) in self.g.property_nodes)
                result_link_data_list = self.g.get_link_data(UserNode('6'), PropertyNode(prop_val))

                # we make sure that this is a property of the exogenous representation
                self.assertIn(result_link_data_list['label'], exogenous_representation)

    # this needs to override superclass test, since for tripartite graph we have no possible error
    # when adding new nodes
    def test_add_raise_error(self):
        pass
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
