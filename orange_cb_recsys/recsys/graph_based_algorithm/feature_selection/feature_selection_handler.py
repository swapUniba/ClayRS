from typing import List, Iterable

from orange_cb_recsys.recsys.graph_based_algorithm.feature_selection.exceptions import FeatureSelectionException
from orange_cb_recsys.recsys.graph_based_algorithm.feature_selection.feature_selection import FeatureSelectionAlgorithm
from orange_cb_recsys.recsys.graphs.graph import FullGraph, UserNode, ItemNode
from orange_cb_recsys.utils.const import recsys_logger


class FeatureSelectionHandler:
    """
    Class that handles the Feature Selection process. It allows to define a Feature Selection algorithm that will be
    used both for feature selection on user properties and on item properties. The Handler will perform feature
    selection on user properties first and on item properties after that. After obtaining the results from the
    Feature Selection algorithms (which will return the list of properties to keep), the Handler creates a copy of
    the original graph and removes any Property node which edges' labels are not in the list of properties to keep
    For example:

            Item some_item_node --label: a_property--> Property some_property_node

    If a_property is not in the list of properties to keep, the Property node is removed (note that other labels
    referring to that Property node should always be 'a_property')

    The goal is to remove the less useful Property Nodes so that
    the ranking and prediction graph based algorithms have to explore less nodes during computation

    Args:
        feature_selection_algorithm (FeatureSelectionAlgorithm): feature selection algorithm to use on user and item
            properties
    """

    def __init__(self, feature_selection_algorithm: FeatureSelectionAlgorithm):
        # POSSIBLE IMPROVEMENT: consider two feature selection algorithms (one for users and one for items) instead of
        # a single one
        self.__feature_selection_algorithm = feature_selection_algorithm

    def process_feature_selection_on_fullgraph(self, graph: FullGraph,
                                               user_target_nodes: List[object],
                                               item_target_nodes: List[object]) -> FullGraph:
        """
        Given a FullGraph, this method performs feature selection on said graph. It also allows to define a custom list
        of user and item nodes which properties will be considered during the feature selection process (instead of
        using the whole set of user and item nodes).

        Args:
            graph (FullGraph): original graph on which feature selection will be performed
            user_target_nodes (list): list of user nodes (or values of said nodes) to consider in the feature selection
                process
            item_target_nodes (list): list of item nodes (or values of said nodes) to consider in the feature selection
                process

        Returns:
            Copy of the original graph from which the less important Property nodes (the ones having edges with less
            important property labels) will be removed
        """

        if any(not graph.is_user_node(node) for node in user_target_nodes):
            raise FeatureSelectionException('All nodes in user_target_nodes list must be user nodes')

        if any(not graph.is_item_node(node) for node in item_target_nodes):
            raise FeatureSelectionException('All nodes in item_target_nodes list must be item nodes')

        if any(not isinstance(node, UserNode) for node in user_target_nodes):
            user_target_nodes = [UserNode(node) if not isinstance(node, UserNode) else node
                                 for node in user_target_nodes]

        if any(not isinstance(node, ItemNode) for node in item_target_nodes):
            item_target_nodes = [ItemNode(node) if not isinstance(node, ItemNode) else node
                                 for node in item_target_nodes]

        properties_to_keep = list()
        user_fs_failed = False
        item_fs_failed = False

        recsys_logger.info("Performing Feature Selection on users")
        try:
            properties_to_keep.extend(self.__feature_selection_algorithm.perform(graph, user_target_nodes))
        except FeatureSelectionException as e:
            recsys_logger.warning(str(e) + "! Users original properties will be kept")
            user_fs_failed = True

        recsys_logger.info("Performing Feature Selection on items")
        try:
            properties_to_keep.extend(self.__feature_selection_algorithm.perform(graph, item_target_nodes))
        except FeatureSelectionException as e:
            recsys_logger.warning(str(e) + "! Items original properties will be kept")
            item_fs_failed = True

        # in case user feature selection or item feature selection failed
        # if both failed the original graph is returned
        # if only one of them failed, the original properties (either for items or users) are retrieved
        if user_fs_failed and item_fs_failed:
            recsys_logger.warning("Since items and users original properties will be kept, "
                                  "the original graph will be returned")
            return graph
        elif user_fs_failed and not item_fs_failed:
            properties_to_keep.extend(self._get_property_labels_info(graph, graph.user_nodes))
        elif not user_fs_failed and item_fs_failed:
            properties_to_keep.extend(self._get_property_labels_info(graph, graph.item_nodes))

        return self.__delete_property_nodes(graph, properties_to_keep)

    @staticmethod
    def _get_property_labels_info(graph: FullGraph, nodes_to_get_properties: Iterable[object]) -> list:
        """
        This method retrieves the properties (which in the graph are property labels) and returns them in a list. It's
        possible to define a custom Iterable of nodes from the FullGraph from which properties will be extracted.

        Note that in case of multiple representations, this function will return the properties in their basic form.
        So, for example:

            [starring#0#dbpedia, producer#0#dbpedia, ...] -> [starring, producer, ...]

        Args:
            graph (FullGraph): the original graph from which the properties will be extracted
            nodes_to_get_properties (Iterable): iterable containing the nodes in the graph from which the properties
                will be extracted

        Returns:
            properties (list): list containing the properties from the original graph for the Iterable of nodes
                passed as argument
        """
        properties = list()

        # retrieves the property nodes (successors) from the original graph. For each node in the target list
        # retrieves the data regarding the link between the node in the target list and each property.
        for node in nodes_to_get_properties:
            for successor_node in graph.get_successors(node):
                if graph.is_property_node(successor_node):
                    property_label = graph.get_link_data(node, successor_node)['label']
                    # in case of multiple representations
                    property_label = property_label.split('#')[0] if '#' in property_label else property_label
                    if property_label not in properties:
                        properties.append(property_label)
        return properties

    @staticmethod
    def __delete_property_nodes(original_graph: FullGraph, properties_to_keep: List[object]) -> FullGraph:
        """
        Creates a copy of the original graph from which the Property nodes having links not defined in the
        properties_to_keep parameter will be deleted (these property nodes will be the ones for which the feature
        selection algorithm found the lowest 'importance' score for the property label of their links)

        Args:
            original_graph (FullGraph): the original graph used for Feature Selection
            properties_to_keep (list): list of properties that should be kept in the original graph.
            Note that properties are the labels in the edges connected to Property nodes (so not the Property nodes
            themselves)

        Returns:
            Graph on which the less important property nodes will be removed
        """
        nodes_to_remove = set()

        # nodes that have even one ingoing edge that is not in the properties_to_keep list are removed
        # note that these nodes should only have one type of label for each edge (that's why only the first predecessor
        # is being considered)
        # in cases where, for example, http://dbpedia.org/resource/Phil_Collins can be both a 'film director'
        # and a 'producer' (both property edge labels of the original graph) there should be two different nodes for it
        for property_node in original_graph.property_nodes:
            predecessor = original_graph.get_predecessors(property_node)[0]
            link = original_graph.get_link_data(predecessor, property_node)
            label = link['label']
            # in case of multiple representations
            label = label.split('#')[0] if '#' in label else label
            if label not in properties_to_keep:
                nodes_to_remove.add(property_node)

        graph = original_graph.copy()
        graph._remove_nodes_from_graph(nodes_to_remove)
        return graph
