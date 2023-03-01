from __future__ import annotations
import abc
from typing import Union, Dict, List, Optional, TYPE_CHECKING, Set

from abc import ABC

import numpy as np

if TYPE_CHECKING:
    from clayrs.content_analyzer import Ratings
    from clayrs.recsys.graphs.graph import FullDiGraph, UserNode
    from clayrs.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm
    from clayrs.recsys.graph_based_algorithm.graph_based_algorithm import GraphBasedAlgorithm
    from clayrs.recsys.methodology import Methodology

from clayrs.content_analyzer.ratings_manager.ratings import Rank, Prediction
from clayrs.recsys.methodology import TestRatingsMethodology, AllItemsMethodology
from clayrs.recsys.content_based_algorithm.exceptions import NotFittedAlg
from clayrs.utils.const import logger


class RecSys(ABC):
    """
    Abstract class for a Recommender System

    There exists various type of recommender systems, content-based, graph-based, etc. so extend this class
    if another type must be implemented into the framework.

    Every recommender system has an algorithm which is used to compute rank/score predictions

    Args:
        algorithm: The algorithm used to compute rank/score prediction
    """

    def __init__(self, algorithm: Union[ContentBasedAlgorithm, GraphBasedAlgorithm]):
        self.__alg = algorithm

        self._yaml_report: Optional[Dict] = None

    @property
    def algorithm(self):
        return self.__alg

    @abc.abstractmethod
    def rank(self, test_set: Ratings, n_recs: int = 10) -> Rank:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, test_set: Ratings) -> Prediction:
        raise NotImplementedError


class ContentBasedRS(RecSys):
    """
    Class for recommender systems which use the items' content in order to make predictions,
    some algorithms may also use users' content, so it's an optional parameter.

    Every *CBRS* differ from each other based the algorithm used.

    Examples:

        In case you perform a splitting of the dataset which returns a single train and test set (e.g. HoldOut
        technique):

        ```python title="Single split train"
        from clayrs import recsys as rs
        from clayrs import content_analyzer as ca

        original_rat = ca.Ratings(ca.CSVFile(ratings_path))

        [train], [test] = rs.HoldOutPartitioning().split_all(original_rat)

        alg = rs.CentroidVector()  # any cb algorithm

        cbrs = rs.ContentBasedRS(alg, train, items_path)

        rank = cbrs.fit_rank(test, n_recs=10)
        ```

        In case you perform a splitting of the dataset which returns a multiple train and test sets (KFold technique):

        ```python title="Multiple split train"
        from clayrs import recsys as rs
        from clayrs import content_analyzer as ca

        original_rat = ca.Ratings(ca.CSVFile(ratings_path))

        train_list, test_list = rs.KFoldPartitioning(n_splits=5).split_all(original_rat)

        alg = rs.CentroidVector()  # any cb algorithm

        for train_set, test_set in zip(train_list, test_list):

            cbrs = rs.ContentBasedRS(alg, train_set, items_path)
            rank_to_append = cbrs.fit_rank(test_set)

            result_list.append(rank_to_append)
        ```

        `result_list` will contain recommendation lists for each split

    Args:
        algorithm: the content based algorithm that will be used in order to
            rank or make score prediction
        train_set: a Ratings object containing interactions between users and items
        items_directory: the path of the items serialized by the Content Analyzer
        users_directory: the path of the users serialized by the Content Analyzer
    """

    def __init__(self,
                 algorithm: ContentBasedAlgorithm,
                 train_set: Ratings,
                 items_directory: str,
                 users_directory: str = None):

        super().__init__(algorithm)
        self.__train_set = train_set
        self.__items_directory = items_directory
        self.__users_directory = users_directory
        self.fit_alg = None

    @property
    def algorithm(self) -> ContentBasedAlgorithm:
        """
        The content based algorithm chosen
        """
        alg: ContentBasedAlgorithm = super().algorithm
        return alg

    @property
    def train_set(self) -> Ratings:
        """
        The train set of the Content Based RecSys
        """
        return self.__train_set

    @property
    def items_directory(self) -> str:
        """
        Path of the serialized items by the Content Analyzer
        """
        return self.__items_directory

    @property
    def users_directory(self) -> str:
        """
        Path of the serialized users by the Content Analyzer
        """
        return self.__users_directory

    def fit(self, num_cpus: int = 1):
        """
        Method which will fit the algorithm chosen for each user in the train set passed in the constructor

        If the algorithm can't be fit for some users, a warning message is printed showing the number of users
        for which the alg couldn't be fit

        Args:
            num_cpus: number of processors that must be reserved for the method. If set to `0`, all cpus available will
                be used. Be careful though: multiprocessing in python has a substantial memory overhead!

        """
        self.fit_alg = self.algorithm.fit(train_set=self.train_set,
                                          items_directory=self.items_directory,
                                          num_cpus=num_cpus)

        return self

    def rank(self, test_set: Ratings, n_recs: Optional[int] = 10, user_list: Union[List[str], List[int]] = None,
             methodology: Optional[Methodology] = TestRatingsMethodology(),
             num_cpus: int = 1) -> Rank:

        """
        Method used to calculate ranking for all users in test set or all users in `user_list` parameter.
        You must first call the `fit()` method ***before*** you can compute the ranking.
        The `user_list` parameter could contain users with their string id or with their mapped integer

        If the `n_recs` is specified, then the rank will contain the top-n items for the users.
        Otherwise, the rank will contain all unrated items of the particular users.
        By default the ***top-10*** ranking is computed for each user

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so, for each user, items in its test set only will be ranked

        If the algorithm was not fit for some users, they will be skipped and a warning message is printed showing the
        number of users for which the alg couldn't produce a ranking

        Args:
            test_set: Ratings object which represents the ground truth of the split considered
            n_recs: Number of the top items that will be present in the ranking of each user.
                If `None` all candidate items will be returned for the user. Default is 10 (top-10 for each user
                will be computed)
            user_list: List of users for which you want to compute score prediction. If None, the ranking
                will be computed for all users of the `test_set`. The list should contain user id as strings or user ids
                mapped to their integers
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`. If None, AllItemsMethodology() will be used
            num_cpus: number of processors that must be reserved for the method. If set to `0`, all cpus available will
                be used. Be careful though: multiprocessing in python has a substantial memory overhead!

        Raises:
            NotFittedAlg: Exception raised when this method is called without first calling the `fit` method

        Returns:
            Rank object containing recommendation lists for all users of the test set or for all users in `user_list`
        """

        if self.fit_alg is None:
            raise NotFittedAlg("Algorithm not fit! You must call the fit() method first, or fit_rank().")

        logger.info("Don't worry if it looks stuck at first")
        logger.info("First iterations will stabilize the estimated remaining time")

        all_users = test_set.unique_user_idx_column
        if user_list is not None:
            all_users = np.array(user_list)
            if np.issubdtype(all_users.dtype, str):
                all_users = self.train_set.user_map.convert_seq_str2int(all_users)

        all_users = set(all_users)

        if methodology is None:
            methodology = AllItemsMethodology()

        methodology.setup(self.train_set, test_set)

        rank = self.algorithm.rank(self.fit_alg, self.train_set, test_set,
                                   user_idx_list=all_users,
                                   items_directory=self.items_directory, n_recs=n_recs,
                                   methodology=methodology, num_cpus=num_cpus)

        # we should remove empty uir matrices otherwise vstack won't work due to dimensions mismatch
        rank = [uir_rank for uir_rank in rank if len(uir_rank) != 0]

        # can't vstack when rank is empty
        if len(rank) == 0:
            rank = Rank.from_uir(np.array([]), user_map=test_set.user_map, item_map=test_set.item_map)
            return rank

        rank = Rank.from_uir(np.vstack(rank), user_map=test_set.user_map, item_map=test_set.item_map)

        self._yaml_report = {'mode': 'rank', 'n_recs': repr(n_recs), 'methodology': repr(methodology)}

        return rank

    def predict(self, test_set: Ratings, user_list: List = None,
                methodology: Union[Methodology, None] = TestRatingsMethodology(),
                num_cpus: int = 1) -> Prediction:
        """
        Method used to calculate score predictions for all users in test set or all users in `user_list` parameter.
        You must first call the `fit()` method ***before*** you can compute score predictions.
        The `user_list` parameter could contain users with their string id or with their mapped integer

        **BE CAREFUL**: not all algorithms are able to perform *score prediction*

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so, for each user, items in its test set only will be considered for score
        prediction

        If the algorithm was not fit for some users, they will be skipped and a warning message is printed showing the
        number of users for which the alg couldn't produce a ranking

        Args:
            test_set: Ratings object which represents the ground truth of the split considered
            user_list: List of users for which you want to compute score prediction. If None, the ranking
                will be computed for all users of the `test_set`. The list should contain user id as strings or user ids
                mapped to their integers
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`. If None, AllItemsMethodology() will be used
            num_cpus: number of processors that must be reserved for the method. If set to `0`, all cpus available will
                be used. Be careful though: multiprocessing in python has a substantial memory overhead!

        Raises:
            NotFittedAlg: Exception raised when this method is called without first calling the `fit` method

        Returns:
            Prediction object containing score prediction lists for all users of the test set or for all users in
                `user_list`
        """

        if self.fit_alg is None:
            raise NotFittedAlg("Algorithm not fit! You must call the fit() method first, or fit_rank().")

        logger.info("Don't worry if it looks stuck at first")
        logger.info("First iterations will stabilize the estimated remaining time")

        all_users = test_set.unique_user_idx_column
        if user_list is not None:
            all_users = np.array(user_list)
            if np.issubdtype(all_users.dtype, str):
                all_users = self.train_set.user_map.convert_seq_str2int(all_users)

        all_users = set(all_users)

        if methodology is None:
            methodology = AllItemsMethodology()

        methodology.setup(self.train_set, test_set)

        pred = self.algorithm.predict(self.fit_alg, self.train_set, test_set,
                                      user_idx_list=all_users,
                                      items_directory=self.items_directory,
                                      methodology=methodology, num_cpus=num_cpus)

        # we should remove empty uir matrices otherwise vstack won't work due to dimensions mismatch
        pred = [uir_pred for uir_pred in pred if len(uir_pred) != 0]

        # can't vstack when pred is empty
        if len(pred) == 0:
            pred = Prediction.from_uir(np.array([]), user_map=test_set.user_map, item_map=test_set.item_map)
            return pred

        pred = Prediction.from_uir(np.vstack(pred), user_map=test_set.user_map, item_map=test_set.item_map)

        self._yaml_report = {'mode': 'score_prediction', 'methodology': repr(methodology)}

        return pred

    def fit_rank(self, test_set: Ratings, n_recs: int = 10, user_list: List[str] = None,
                 methodology: Union[Methodology, None] = TestRatingsMethodology(),
                 save_fit: bool = False, num_cpus: int = 1) -> Rank:
        """
        Method used to both fit and calculate ranking for all users in test set or all users in `user_list`
        parameter.
        The Recommender System will first be fit for each user in the `test_set` parameter or for each
        user inside the `user_list` parameter: the `user_list` parameter could contain users with their string id or
        with their mapped integer

        If the `n_recs` is specified, then the rank will contain the top-n items for the users.
        Otherwise, the rank will contain all unrated items of the particular users.
        By default the ***top-10*** ranking is computed for each user

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so, for each user, items in its test set only will be ranked

        If the algorithm couldn't be fit for some users, they will be skipped and a warning message is printed showing
        the number of users for which the alg couldn't produce a ranking

        With the `save_fit` parameter you can decide if you want that you recommender system remains *fit* even after
        the complete execution of this method, in case you want to compute ranking with other methodologies, or
        with a different `n_recs` parameter. Be mindful since it can be memory-expensive, thus by default this behaviour
        is disabled

        Args:
            test_set: Ratings object which represents the ground truth of the split considered
            n_recs: Number of the top items that will be present in the ranking of each user.
                If `None` all candidate items will be returned for the user. Default is 10 (top-10 for each user
                will be computed)
            user_list: List of users for which you want to compute score prediction. If None, the ranking
                will be computed for all users of the `test_set`. The list should contain user id as strings or user ids
                mapped to their integers
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`. If None, AllItemsMethodology() will be used
            save_fit: Boolean value which let you choose if the Recommender System should remain fit even after the
                complete execution of this method. Default is False
            num_cpus: number of processors that must be reserved for the method. If set to `0`, all cpus available will
                be used. Be careful though: multiprocessing in python has a substantial memory overhead!

        Raises:
            NotFittedAlg: Exception raised when this method is called without first calling the `fit` method

        Returns:
            Rank object containing recommendation lists for all users of the test set or for all users in `user_list`
        """

        logger.info("Don't worry if it looks stuck at first")
        logger.info("First iterations will stabilize the estimated remaining time")

        all_users = test_set.unique_user_idx_column
        if user_list is not None:
            all_users = np.array(user_list)
            if np.issubdtype(all_users.dtype, str):
                all_users = self.train_set.user_map.convert_seq_str2int(all_users)

        all_users = set(all_users)

        if methodology is None:
            methodology = AllItemsMethodology()

        methodology.setup(self.train_set, test_set)

        fit_alg, rank = self.algorithm.fit_rank(self.train_set, test_set, user_idx_list=all_users,
                                                items_directory=self.items_directory, n_recs=n_recs,
                                                methodology=methodology, num_cpus=num_cpus, save_fit=save_fit)

        self.fit_alg = fit_alg

        # we should remove empty uir matrices otherwise vstack won't work due to dimensions mismatch
        rank = [uir_rank for uir_rank in rank if len(uir_rank) != 0]

        # can't vstack when rank is empty
        if len(rank) == 0:
            rank = Rank.from_uir(np.array([]), user_map=test_set.user_map, item_map=test_set.item_map)
            return rank

        rank = Rank.from_uir(np.vstack(rank), user_map=test_set.user_map, item_map=test_set.item_map)

        self._yaml_report = {'mode': 'rank', 'n_recs': repr(n_recs), 'methodology': repr(methodology)}

        return rank

    def fit_predict(self, test_set: Ratings, user_list: List = None,
                    methodology: Union[Methodology, None] = TestRatingsMethodology(),
                    save_fit: bool = False, num_cpus: int = 1) -> Prediction:
        """
        Method used to both fit and calculate score prediction for all users in test set or all users in `user_list`
        parameter.
        The Recommender System will first be fit for each user in the `test_set` parameter or for each
        user inside the `user_list` parameter: the `user_list` parameter could contain users with their string id or
        with their mapped integer

        **BE CAREFUL**: not all algorithms are able to perform *score prediction*

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so, for each user, items in its test set only will be considered for score
        prediction

        If the algorithm couldn't be fit for some users, they will be skipped and a warning message is printed showing
        the number of users for which the alg couldn't produce a ranking

        With the `save_fit` parameter you can decide if you want that you recommender system remains *fit* even after
        the complete execution of this method, in case you want to compute ranking/score prediction with other
        methodologies, or with a different `n_recs` parameter. Be mindful since it can be memory-expensive,
        thus by default this behaviour is disabled

        Args:
            test_set: Ratings object which represents the ground truth of the split considered
            user_list: List of users for which you want to compute score prediction. If None, the ranking
                will be computed for all users of the `test_set`. The list should contain user id as strings or user ids
                mapped to their integers
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`. If None, AllItemsMethodology() will be used
            save_fit: Boolean value which let you choose if the Recommender System should remain fit even after the
                complete execution of this method. Default is False
            num_cpus: number of processors that must be reserved for the method. If set to `0`, all cpus available will
                be used. Be careful though: multiprocessing in python has a substantial memory overhead!

        Returns:
            Prediction object containing score prediction lists for all users of the test set or for all users in
                `user_list`
        """

        logger.info("Don't worry if it looks stuck at first")
        logger.info("First iterations will stabilize the estimated remaining time")

        all_users = test_set.unique_user_idx_column
        if user_list is not None:
            all_users = np.array(user_list)
            if np.issubdtype(all_users.dtype, str):
                all_users = self.train_set.user_map.convert_seq_str2int(all_users)

        all_users = set(all_users)

        if methodology is None:
            methodology = AllItemsMethodology()

        methodology.setup(self.train_set, test_set)

        fit_alg, pred = self.algorithm.fit_predict(self.train_set, test_set, user_idx_list=all_users,
                                                   items_directory=self.items_directory,
                                                   methodology=methodology, num_cpus=num_cpus, save_fit=save_fit)

        self.fit_alg = fit_alg

        # we should remove empty uir matrices otherwise vstack won't work due to dimensions mismatch
        pred = [uir_pred for uir_pred in pred if len(uir_pred) != 0]

        # can't vstack when pred is empty
        if len(pred) == 0:
            pred = Prediction.from_uir(np.array([]), user_map=test_set.user_map, item_map=test_set.item_map)
            return pred

        pred = Prediction.from_uir(np.vstack(pred), user_map=test_set.user_map, item_map=test_set.item_map)

        self._yaml_report = {'mode': 'score_prediction', 'methodology': repr(methodology)}

        return pred

    def __repr__(self):
        return f"ContentBasedRS(algorithm={self.algorithm}, train_set={self.train_set}, " \
               f"items_directory={self.items_directory}, users_directory={self.users_directory})"


class GraphBasedRS(RecSys):
    """
    Class for recommender systems which use a graph in order to make predictions

    Every GBRS differ from each other based the algorithm used.

    Examples:

        In case you perform a splitting of the dataset which returns a single train and test set (e.g. HoldOut
        technique):

        ```python title="Single split train"
        from clayrs import recsys as rs
        from clayrs import content_analyzer as ca

        original_rat = ca.Ratings(ca.CSVFile(ratings_path))

        [train], [test] = rs.HoldOutPartitioning().split_all(original_rat)

        alg = rs.NXPageRank()  # any gb algorithm

        graph = rs.NXBipartiteGraph(original_rat)

        # remove from the graph interaction of the test set
        for user, item in zip(test.user_id_column, test.item_id_column):
            user_node = rs.UserNode(user)
            item_node = rs.ItemNode(item)

            graph.remove_link(user_node, item_node)

        gbrs = rs.GraphBasedRS(alg, graph)

        rank = gbrs.rank(test, n_recs=10)
        ```

        In case you perform a splitting of the dataset which returns a multiple train and test sets (KFold technique):

        ```python title="Multiple split train"
        from clayrs import recsys as rs
        from clayrs import content_analyzer as ca

        original_rat = ca.Ratings(ca.CSVFile(ratings_path))

        train_list, test_list = rs.KFoldPartitioning(n_splits=5).split_all(original_rat)

        alg = rs.NXPageRank()  # any gb algorithm

        for train_set, test_set in zip(train_list, test_list):

            graph = rs.NXBipartiteGraph(original_rat)

            # remove from the graph interaction of the test set
            for user, item in zip(test_set.user_id_column, test_set.item_id_column):
                user_node = rs.UserNode(user)
                item_node = rs.ItemNode(item)

                graph.remove_link(user_node, item_node)

            gbrs = rs.GraphBasedRS(alg, graph)
            rank_to_append = gbrs.rank(test_set)

            result_list.append(rank_to_append)
        ```

        `result_list` will contain recommendation lists for each split

    Args:
        algorithm: the graph based algorithm that will be used in order to
            rank or make score prediction
        graph: A graph which models interactions of users and items
    """

    def __init__(self,
                 algorithm: GraphBasedAlgorithm,
                 graph: FullDiGraph):

        self.__graph = graph
        super().__init__(algorithm)

    @property
    def users(self) -> Set[UserNode]:
        """
        Set of UserNode objects for each user of the graph
        """
        return self.__graph.user_nodes

    @property
    def graph(self) -> FullDiGraph:
        """
        The graph containing interactions
        """
        return self.__graph

    @property
    def algorithm(self) -> GraphBasedAlgorithm:
        """
        The graph based algorithm chosen
        """
        alg: GraphBasedAlgorithm = super().algorithm
        return alg

    def rank(self, test_set: Ratings, n_recs: int = 10, user_list: List[str] = None,
             methodology: Union[Methodology, None] = TestRatingsMethodology(),
             num_cpus: int = 1) -> Rank:
        """
        Method used to calculate ranking for all users in test set or all users in `user_list` parameter.
        The `user_list` parameter could contain users with their string id or with their mapped integer

        If the `n_recs` is specified, then the rank will contain the top-n items for the users.
        Otherwise, the rank will contain all unrated items of the particular users.
        By default the ***top-10*** ranking is computed for each user

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so, for each user, items in its test set only will be ranked

        If the algorithm couldn't produce a ranking for some users, they will be skipped and a warning message is
        printed showing the number of users for which the alg couldn't produce a ranking

        Args:
            test_set: Ratings object which represents the ground truth of the split considered
            n_recs: Number of the top items that will be present in the ranking of each user.
                If `None` all candidate items will be returned for the user. Default is 10 (top-10 for each user
                will be computed)
            user_list: List of users for which you want to compute score prediction. If None, the ranking
                will be computed for all users of the `test_set`. The list should contain user id as strings or user ids
                mapped to their integers
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`. If None, AllItemsMethodology() will be used
            num_cpus: number of processors that must be reserved for the method. If set to `0`, all cpus available will
                be used. Be careful though: multiprocessing in python has a substantial memory overhead!

        Returns:
            Rank object containing recommendation lists for all users of the test set or for all users in `user_list`
        """

        train_set = self.graph.to_ratings(user_map=test_set.user_map, item_map=test_set.item_map)

        logger.info("Don't worry if it looks stuck at first")
        logger.info("First iterations will stabilize the estimated remaining time")
        
        # in the graph recsys, each graph algorithm works with strings,
        # so in case we should convert int to strings
        all_users = test_set.unique_user_id_column
        if user_list is not None:
            all_users = np.array(user_list)
            if np.issubdtype(all_users.dtype, int):
                all_users = train_set.user_map.convert_seq_int2str(all_users)

        all_users = set(all_users)

        if methodology is None:
            methodology = AllItemsMethodology()

        methodology.setup(train_set, test_set)

        rank = self.algorithm.rank(self.graph, train_set, test_set, all_users, n_recs, methodology, num_cpus)
        # we should remove empty uir matrices otherwise vstack won't work due to dimensions mismatch
        rank = [uir_rank for uir_rank in rank if len(uir_rank) != 0]

        # can't vstack when rank is empty
        if len(rank) == 0:
            rank = Rank.from_uir(np.array([]), user_map=test_set.user_map, item_map=test_set.item_map)
            return rank

        rank = np.vstack(rank)

        # convert back strings and Nodes object to ints
        rank_users_idx = train_set.user_map.convert_seq_str2int(rank[:, 0])
        rank_items_idx = train_set.item_map.convert_seq_str2int([item_node.value for item_node in rank[:, 1]])
        rank[:, 0] = rank_users_idx
        rank[:, 1] = rank_items_idx
        rank = rank.astype(np.float64)

        rank = Rank.from_uir(rank, user_map=test_set.user_map, item_map=test_set.item_map)

        if len(rank) == 0:
            logger.warning("No items could be ranked for any users! Remember that items to rank must be present "
                           "in the graph.\n"
                           "Try changing methodology!")

        elif len(rank.unique_user_id_column) != len(all_users):
            logger.warning(f"No items could be ranked for users {all_users - set(rank.user_id_column)}\n"
                           f"No nodes to rank for them found in the graph. Try changing methodology! ")

        self._yaml_report = {'graph': repr(self.graph), 'mode': 'rank', 'n_recs': repr(n_recs),
                             'methodology': repr(methodology)}

        return rank

    def predict(self, test_set: Ratings, user_list: List[str] = None,
                methodology: Union[Methodology, None] = TestRatingsMethodology(),
                num_cpus: int = 1) -> Prediction:
        """
        Method used to calculate score predictions for all users in test set or all users in `user_list` parameter.
        The `user_list` parameter could contain users with their string id or with their mapped integer

        **BE CAREFUL**: not all algorithms are able to perform *score prediction*

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so for each user items in its test set only will be considered for score
        prediction

        If the algorithm couldn't perform score prediction for some users, they will be skipped and a warning message is
        printed showing the number of users for which the alg couldn't produce a score prediction

        Args:
            test_set: Ratings object which represents the ground truth of the split considered
            user_list: List of users for which you want to compute score prediction. If None, the ranking
                will be computed for all users of the `test_set`. The list should contain user id as strings or user ids
                mapped to their integers
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`. If None, AllItemsMethodology() will be used
            num_cpus: number of processors that must be reserved for the method. If set to `0`, all cpus available will
                be used. Be careful though: multiprocessing in python has a substantial memory overhead!

        Returns:
            Prediction object containing score prediction lists for all users of the test set or for all users in
                `user_list`
        """

        train_set = self.graph.to_ratings(user_map=test_set.user_map, item_map=test_set.item_map)

        logger.info("Don't worry if it looks stuck at first")
        logger.info("First iterations will stabilize the estimated remaining time")

        # in the graph recsys, each graph algorithm works with strings,
        # so in case we should convert int to strings
        all_users = test_set.unique_user_id_column
        if user_list is not None:
            all_users = np.array(user_list)
            if np.issubdtype(all_users.dtype, int):
                all_users = train_set.user_map.convert_seq_int2str(all_users)

        all_users = set(all_users)

        if methodology is None:
            methodology = AllItemsMethodology()

        methodology.setup(train_set, test_set)

        pred = self.algorithm.predict(self.graph, train_set, test_set, all_users, methodology, num_cpus)
        # we should remove empty uir matrices otherwise vstack won't work due to dimensions mismatch
        pred = [uir_pred for uir_pred in pred if len(uir_pred) != 0]

        # can't vstack when pred is empty
        if len(pred) == 0:
            pred = Prediction.from_uir(np.array([]), user_map=test_set.user_map, item_map=test_set.item_map)
            return pred

        pred = np.vstack(pred)
        pred_users_idx = train_set.user_map.convert_seq_str2int(pred[:, 0])
        pred_items_idx = train_set.item_map.convert_seq_str2int([item_node.value for item_node in pred[:, 1]])
        pred[:, 0] = pred_users_idx
        pred[:, 1] = pred_items_idx
        pred = pred.astype(np.float64)
        pred = Prediction.from_uir(pred, user_map=test_set.user_map, item_map=test_set.item_map)

        self._yaml_report = {'graph': repr(self.graph), 'mode': 'score_prediction', 'methodology': repr(methodology)}

        return pred

    def __repr__(self):
        return f"GraphBasedRS(algorithm={self.algorithm}, graph={self.graph})"
