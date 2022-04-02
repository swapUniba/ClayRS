from typing import List, Iterable

from orange_cb_recsys.recsys.graphs.feature_selection.exceptions import FeatureSelectionException
from orange_cb_recsys.recsys.graphs.feature_selection.feature_selection_alg import FeatureSelectionAlgorithm
from orange_cb_recsys.recsys.graphs.graph import FullDiGraph, UserNode, ItemNode
from orange_cb_recsys.utils.const import logger


def feature_selection(graph: FullDiGraph,
                      fs_algorithm_user: FeatureSelectionAlgorithm = None,
                      fs_algorithm_item: FeatureSelectionAlgorithm = None,
                      user_target_nodes: Iterable[UserNode] = None,
                      item_target_nodes: Iterable[ItemNode] = None,
                      inplace: bool = False) -> FullDiGraph:
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
    if fs_algorithm_user is not None and user_target_nodes is None:
        user_target_nodes = graph.user_nodes

    if fs_algorithm_item is not None and item_target_nodes is None:
        item_target_nodes = graph.item_nodes

    property_labels_to_remove = list()
    user_fs_failed = False
    item_fs_failed = False

    if fs_algorithm_user is not None:
        logger.info("Performing Feature Selection on users")
        try:
            user_props_to_remove = fs_algorithm_user.perform(graph, list(user_target_nodes), mode='to_remove')
            property_labels_to_remove.extend(user_props_to_remove)
        except FeatureSelectionException as e:
            logger.warning(str(e) + "!\nUsers original properties will be kept")
            user_fs_failed = True

    if fs_algorithm_item is not None:
        logger.info("Performing Feature Selection on items")
        try:
            item_props_to_remove = fs_algorithm_item.perform(graph, list(item_target_nodes), mode='to_remove')
            property_labels_to_remove.extend(item_props_to_remove)
        except FeatureSelectionException as e:
            logger.warning(str(e) + "!\nItems original properties will be kept")
            item_fs_failed = True

    # in case user feature selection or item feature selection failed
    # if both failed the original graph is returned
    # if only one of them failed, the original properties (either for items or users) are retrieved
    if user_fs_failed and item_fs_failed:
        logger.warning("Since both feature selection on items and feature selection on users failed or no fs algorithm"
                       "has been defined,\nthe original graph will be returned")

    if inplace is True:
        graph_fs = _delete_property_nodes(graph, property_labels_to_remove)
    else:
        graph_copy = graph.copy()
        graph_fs = _delete_property_nodes(graph_copy, property_labels_to_remove)

    return graph_fs


def _delete_property_nodes(graph: FullDiGraph, property_labels_to_remove: List[str]) -> FullDiGraph:
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
    # MAYBE delete only edges and then if a node is "alone" delete it? In this way we preserve the cases in which
    # the PropertyNode DiCaprio can be linked with an edge "director" and an edge "actor" and we need to remove
    # "director" properties

    prop_nodes_to_remove = []
    for property_node in graph.property_nodes:
        predecessors = graph.get_predecessors(property_node)
        for pred in predecessors:
            link_label = graph.get_link_data(pred, property_node).get('label')
            if link_label in set(property_labels_to_remove):
                prop_nodes_to_remove.append(property_node)

                # we don't check for other predecessors,
                # if node must be removed we check the others property nodes
                break

    graph.remove_node(prop_nodes_to_remove)
    return graph
