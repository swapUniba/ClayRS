# Feature Selection

Via the `feature_selecter` function you are able to perform feature selection on a given graph, by keeping properties
that are the most important according to a given ***feature selection algorithm***. Check the documentation of the
method for more and for a *usage example*

::: clayrs.recsys.graphs.feature_selection.feature_selection_fn
    handler: python

---

## Feature Selection algorithms

The following are the feature selection algorithms you can use in the `fs_algorithms_user` 
and/or in the `fs_algorithm_item`

::: clayrs.recsys.graphs.feature_selection.feature_selection_alg.TopKPageRank
    handler: python
    options:
        heading_level: 3
        show_root_toc_entry: true
        show_root_heading: true
		members: none

::: clayrs.recsys.graphs.feature_selection.feature_selection_alg.TopKEigenVectorCentrality
    handler: python
    options:
        heading_level: 3
        show_root_toc_entry: true
        show_root_heading: true
		members: none

::: clayrs.recsys.graphs.feature_selection.feature_selection_alg.TopKDegreeCentrality
    handler: python
    options:
        heading_level: 3
        show_root_toc_entry: true
        show_root_heading: true
		members: none
