from __future__ import annotations
from typing import List, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from clayrs.recsys.graphs.feature_selection.feature_selection_alg import FeatureSelectionAlgorithm
    from clayrs.recsys.graphs.graph import FullDiGraph, UserNode, ItemNode

from clayrs.recsys.graphs.feature_selection.exceptions import FeatureSelectionException
from clayrs.utils.const import logger


def feature_selector(graph: FullDiGraph,
                     fs_algorithm_user: FeatureSelectionAlgorithm = None,
                     fs_algorithm_item: FeatureSelectionAlgorithm = None,
                     user_target_nodes: Iterable[UserNode] = None,
                     item_target_nodes: Iterable[ItemNode] = None,
                     inplace: bool = False) -> FullDiGraph:
    """
    Given a FullGraph, this method performs feature selection on it and returns the "reduced" graph.

    You can choose to reduce only *user properties* (*evaluate the `fs_algorithm_user` parameter*),
    to reduce only *item properties* (*evaluate the `fs_algorithm_item` parameter*) or both (*evaluate
    the `fs_algorithm_user` parameter* and the `fs_algorithm_item` parameter*).
    You can also choose different *feature selection algorithms* for users and items.

    You can also define a custom list of user and item nodes:

    * In this case only properties of those nodes will be considered during the feature selection process (instead of
    using properties of all users and items)

    This function changes a *copy* of the original graph by default, but you can change this behaviour with the
    `inplace` parameter.

    Examples:

        ```python
        # create a full graph
        full_graph = rs.NXFullGraph(ratings,
                                     user_contents_dir='users_codified/', # (1)
                                     item_contents_dir='movies_codified/', # (2)
                                     user_exo_properties={0}, # (3)
                                     item_exo_properties={'dbpedia'}, # (4)
                                     link_label='score')

         # perform feature selection by keeping only top 5 property labels
         # according to page rank algorithm
         fs_graph = rs.feature_selector(full_graph,
                                        fs_algorithm_item=rs.TopKPageRank(k=5))
        ```

    Args:
        graph: Original graph on which feature selection will be performed
        fs_algorithm_user: FeatureSelectionAlgorithm that will be performed on user properties. Can be different from
            `fs_algorithm_item`
        fs_algorithm_item: FeatureSelectionAlgorithm that will be performed on item properties. Can be different from
            `fs_algorithm_user`
        user_target_nodes (list): List of user nodes to consider in the feature selection process: only properties
            of user nodes in this list will be "reduced"
        item_target_nodes (list): List of item nodes to consider in the feature selection process: only properties
            of item nodes in this list will be "reduced"
        inplace: Boolean parameter that let you choose if changes must be performed on the original graph
            (`inplace=True`) or on its copy (`inplace=False`). Default is False

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
    Creates a copy of the original graph from which the Property nodes with labels defined in the
    `properties_to_remove` parameter will be deleted (these property nodes will be the ones for which the feature
    selection algorithm found the lowest 'importance' score for the property label of their links)

    Args:
        graph: The original graph used for Feature Selection
        property_labels_to_remove: List of property labels that should be removed from the original graph

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
                # if node is linked to only one node with the label to remove it will be removed
                break

    graph.remove_node(prop_nodes_to_remove)
    return graph
