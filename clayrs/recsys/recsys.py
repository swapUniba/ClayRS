import abc
import gc
import itertools
from copy import deepcopy
from typing import Union, Dict, List, Optional

from abc import ABC

from clayrs.content_analyzer import Ratings
from clayrs.content_analyzer.ratings_manager.ratings import Rank, Prediction
from clayrs.recsys.methodology import TestRatingsMethodology
from clayrs.recsys.graphs.graph import FullDiGraph

from clayrs.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm
from clayrs.recsys.content_based_algorithm.exceptions import UserSkipAlgFit, NotFittedAlg
from clayrs.recsys.graph_based_algorithm.graph_based_algorithm import GraphBasedAlgorithm
from clayrs.recsys.methodology import Methodology
from clayrs.utils.const import logger
from clayrs.utils.context_managers import get_iterator_parallel


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

    Every CBRS differ from each other based the algorithm used.

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
        self._user_fit_dic = {}

    @property
    def algorithm(self):
        """
        The content based algorithm chosen
        """
        alg: ContentBasedAlgorithm = super().algorithm
        return alg

    @property
    def train_set(self):
        """
        The train set of the Content Based RecSys
        """
        return self.__train_set

    @property
    def items_directory(self):
        """
        Path of the serialized items by the Content Analyzer
        """
        return self.__items_directory

    @property
    def users_directory(self):
        """
        Path of the serialized users by the Content Analyzer
        """
        return self.__users_directory

    def fit(self, num_cpus: int = 0):
        """
        Method which will fit the algorithm chosen for each user in the train set passed in the constructor

        If the algorithm can't be fit for some users, a warning message is printed
        """
        def compute_single_fit(user_id):
            user_train = self.train_set.get_user_interactions(user_id)
            user_alg = deepcopy(self.algorithm)

            try:
                user_alg.process_rated(user_train, loaded_items_interface)
                user_alg.fit()
            except UserSkipAlgFit as e:
                warning_message = str(e) + f"\nNo algorithm will be fitted for the user {user_id}"
                logger.warning(warning_message)
                user_alg = None

            return user_id, user_alg

        items_to_load = set(self.train_set.item_id_column)
        all_users = set(self.train_set.user_id_column)
        loaded_items_interface = self.algorithm._load_available_contents(self.items_directory, items_to_load)

        with get_iterator_parallel(num_cpus,
                                   compute_single_fit, all_users,
                                   progress_bar=True, total=len(all_users)) as pbar:

            pbar.set_description("Fitting algorithm")

            for user_id, fitted_user_alg in pbar:
                self._user_fit_dic[user_id] = fitted_user_alg

        # we force the garbage collector after freeing loaded items
        del loaded_items_interface
        gc.collect()

        return self

    def rank(self, test_set: Ratings, n_recs: int = 10, user_id_list: List = None,
             methodology: Union[Methodology, None] = TestRatingsMethodology(),
             num_cpus: int = 0) -> Rank:

        """
        Method used to calculate ranking for all users in test set or all users in `user_id_list` parameter.
        You must first call the `fit()` method before you can compute the ranking.

        If the `n_recs` is specified, then the rank will contain the top-n items for the users.
        Otherwise, the rank will contain all unrated items of the particular users

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so, for each user, items in its test set only will be ranked

        If the algorithm was not fit for some users, they will be skipped and a warning is printed

        Args:
            test_set: Ratings object which represents the ground truth of the split considered
            n_recs: Number of the top items that will be present in the ranking of each user.
                If `None` all candidate items will be returned for the user
            user_id_list: List of users for which you want to compute the ranking. If None, the ranking will be computed
                for all users of the `test_set`
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method. Default is 0, meaning that
                the number of cpus will be automatically detected.

        Raises:
            NotFittedAlg: Exception raised when this method is called without first calling the `fit` method

        Returns:
            Rank object containing recommendation lists for all users of the test set or for all users in `user_id_list`

        """
        def compute_single_rank(user_id):
            user_id = str(user_id)
            user_train = self.train_set.get_user_interactions(user_id)

            filter_list = None
            if methodology is not None:
                filter_list = set(methodology.filter_single(user_id, self.train_set, test_set))

            user_fitted_alg = self._user_fit_dic.get(user_id)
            if user_fitted_alg is not None:
                user_rank = user_fitted_alg.rank(user_train, loaded_items_interface,
                                                 n_recs, filter_list=filter_list)
            else:
                user_rank = []
                logger.warning(f"No algorithm fitted for user {user_id}! It will be skipped")

            return user_id, user_rank

        if len(self._user_fit_dic) == 0:
            raise NotFittedAlg("Algorithm not fit! You must call the fit() method first, or fit_rank().")

        all_users = set(test_set.user_id_column)
        if user_id_list is not None:
            all_users = set(user_id_list)

        loaded_items_interface = self.algorithm._load_available_contents(self.items_directory, set())

        rank = []

        logger.info("Don't worry if it looks stuck at first")
        logger.info("First iterations will stabilize the estimated remaining time")

        with get_iterator_parallel(num_cpus,
                                   compute_single_rank, all_users,
                                   progress_bar=True, total=len(all_users)) as pbar:

            pbar.set_description(f"Loading first items from memory...")
            for user_id, user_rank in pbar:
                pbar.set_description(f"Computing rank for user {user_id}")
                rank.append(user_rank)

        rank = itertools.chain.from_iterable(rank)
        rank = Rank.from_list(rank)

        # we force the garbage collector after freeing loaded items
        del loaded_items_interface
        gc.collect()

        self._yaml_report = {'mode': 'rank', 'n_recs': repr(n_recs), 'methodology': repr(methodology)}

        return rank

    def predict(self, test_set: Ratings, user_id_list: List = None,
                methodology: Union[Methodology, None] = TestRatingsMethodology(),
                num_cpus: int = 0) -> Prediction:
        """
        Method used to calculate score predictions for all users in test set or all users in `user_id_list` parameter.
        You must first call the `fit()` method before you can compute score predictions.

        **BE CAREFUL**: not all algorithms are able to perform *score prediction*

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so, for each user, items in its test set only will be considered for score
        prediction

        If the algorithm was not fit for some users, they will be skipped and a warning is printed

        Args:
            test_set: Ratings object which represents the ground truth of the split considered
            user_id_list: List of users for which you want to compute score prediction. If None, the ranking
                will be computed for all users of the `test_set`
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method. Default is 0, meaning that
                the number of cpus will be automatically detected.

        Raises:
            NotFittedAlg: Exception raised when this method is called without first calling the `fit` method

        Returns:
            Prediction object containing score prediction lists for all users of the test set or for all users in
                `user_id_list`

        """
        def compute_single_predict(user_id):
            user_id = str(user_id)
            user_train = self.train_set.get_user_interactions(user_id)

            filter_list = None
            if methodology is not None:
                filter_list = set(methodology.filter_single(user_id, self.train_set, test_set))

            user_fitted_alg = self._user_fit_dic.get(user_id)
            if user_fitted_alg is not None:
                user_pred = user_fitted_alg.predict(user_train, loaded_items_interface, filter_list=filter_list)
            else:
                user_pred = []
                logger.warning(f"No algorithm fitted for user {user_id}! It will be skipped")

            return user_id, user_pred

        if len(self._user_fit_dic) == 0:
            raise NotFittedAlg("Algorithm not fit! You must call the fit() method first, or fit_rank().")

        all_users = set(test_set.user_id_column)
        if user_id_list is not None:
            all_users = set(user_id_list)

        loaded_items_interface = self.algorithm._load_available_contents(self.items_directory, set())

        pred = []

        logger.info("Don't worry if it looks stuck at first")
        logger.info("First iterations will stabilize the estimated remaining time")

        with get_iterator_parallel(num_cpus,
                                   compute_single_predict, all_users,
                                   progress_bar=True, total=len(all_users)) as pbar:

            pbar.set_description(f"Loading first items from memory...")
            for user_id, user_pred in pbar:
                pbar.set_description(f"Computing score prediction for user {user_id}")
                pred.append(user_pred)

        pred = itertools.chain.from_iterable(pred)
        pred = Prediction.from_list(pred)

        # we force the garbage collector after freeing loaded items
        del loaded_items_interface
        gc.collect()

        self._yaml_report = {'mode': 'score_prediction', 'methodology': repr(methodology)}

        return pred

    def fit_predict(self, test_set: Ratings, user_id_list: List = None,
                    methodology: Union[Methodology, None] = TestRatingsMethodology(),
                    save_fit: bool = False, num_cpus: int = 0) -> Prediction:
        """
        Method used to both fit and calculate score prediction for all users in test set or all users in `user_id_list`
        parameter.
        The Recommender System will first be fit for each user in the train set passed in the constructor.
        If the algorithm can't be fit for some users, a warning message is printed

        **BE CAREFUL**: not all algorithms are able to perform *score prediction*

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so, for each user, items in its test set only will be considered for score
        prediction

        If the algorithm was not fit for some users, they will be skipped and a warning is printed

        With the `save_fit` parameter you can decide if you want that you recommender system remains *fit* even after
        the complete execution of this method, in case you want to compute ranking/score prediction with other
        methodologies, or with a different `n_recs` parameter. Be mindful since it can be memory-expensive,
        thus by default this behaviour is disabled

        Args:
            test_set: Ratings object which represents the ground truth of the split considered
            user_id_list: List of users for which you want to compute score prediction. If None, the ranking
                will be computed for all users of the `test_set`
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            save_fit: Boolean value which let you choose if the Recommender System should remain fit even after the
                complete execution of this method. Default is False
            num_cpus: number of processors that must be reserved for the method. Default is 0, meaning that
                the number of cpus will be automatically detected.

        Returns:
            Rank object containing recommendation lists for all users of the test set or for all users in `user_id_list`
        """
        def compute_single_fit_predict(user_id):
            user_train = self.train_set.get_user_interactions(user_id)

            alg = self.algorithm
            if save_fit:
                alg = deepcopy(alg)
                self._user_fit_dic[user_id] = alg

            try:
                alg.process_rated(user_train, loaded_items_interface)
                alg.fit()

            except UserSkipAlgFit as e:
                warning_message = str(e) + f"\nThe algorithm can't be fitted for the user {user_id}"
                logger.warning(warning_message)
                if save_fit:
                    self._user_fit_dic[user_id] = None
                return

            filter_list = None
            if methodology is not None:
                filter_list = set(methodology.filter_single(user_id, self.train_set, test_set))

            user_pred = alg.predict(user_train, loaded_items_interface, filter_list=filter_list)

            return user_id, user_pred

        all_users = set(test_set.user_id_column)
        if user_id_list is not None:
            all_users = set(user_id_list)

        loaded_items_interface = self.algorithm._load_available_contents(self.items_directory, set())

        pred = []

        logger.info("Don't worry if it looks stuck at first")
        logger.info("First iterations will stabilize the estimated remaining time")

        with get_iterator_parallel(num_cpus,
                                   compute_single_fit_predict, all_users,
                                   progress_bar=True, total=len(all_users)) as pbar:

            pbar.set_description(f"Loading first items from memory...")
            for user_id, user_pred in pbar:
                pbar.set_description(f"Computing fit_predict for user {user_id}")
                pred.append(user_pred)

        pred = itertools.chain.from_iterable(pred)
        pred = Prediction.from_list(pred)

        # we force the garbage collector after freeing loaded items
        del loaded_items_interface
        gc.collect()

        self._yaml_report = {'mode': 'score_prediction', 'methodology': repr(methodology)}

        return pred

    def fit_rank(self, test_set: Ratings, n_recs: int = 10, user_id_list: List = None,
                 methodology: Union[Methodology, None] = TestRatingsMethodology(),
                 save_fit: bool = False, num_cpus: int = 0) -> Rank:
        """
        Method used to both fit and calculate ranking for all users in test set or all users in `user_id_list`
        parameter.
        The Recommender System will first be fit for each user in the train set passed in the constructor.
        If the algorithm can't be fit for some users, a warning message is printed

        If the `n_recs` is specified, then the rank will contain the top-n items for the users.
        Otherwise, the rank will contain all unrated items of the particular users

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so, for each user, items in its test set only will be ranked

        If the algorithm was not fit for some users, they will be skipped and a warning is printed

        With the `save_fit` parameter you can decide if you want that you recommender system remains *fit* even after
        the complete execution of this method, in case you want to compute ranking with other methodologies, or
        with a different `n_recs` parameter. Be mindful since it can be memory-expensive, thus by default this behaviour
        is disabled

        Args:
            test_set: Ratings object which represents the ground truth of the split considered
            n_recs: Number of the top items that will be present in the ranking of each user.
                If `None` all candidate items will be returned for the user
            user_id_list: List of users for which you want to compute the ranking. If None, the ranking will be computed
                for all users of the `test_set`
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            save_fit: Boolean value which let you choose if the Recommender System should remain fit even after the
                complete execution of this method. Default is False
            num_cpus: number of processors that must be reserved for the method. Default is 0, meaning that
                the number of cpus will be automatically detected.

        Raises:
            NotFittedAlg: Exception raised when this method is called without first calling the `fit` method

        Returns:
            Rank object containing recommendation lists for all users of the test set or for all users in `user_id_list`
        """
        def compute_single_fit_rank(user_id):
            user_train = self.train_set.get_user_interactions(user_id)

            alg = self.algorithm
            if save_fit:
                alg = deepcopy(alg)
                self._user_fit_dic[user_id] = alg

            try:
                alg.process_rated(user_train, loaded_items_interface)
                alg.fit()

            except UserSkipAlgFit as e:
                warning_message = str(e) + f"\nThe algorithm can't be fitted for the user {user_id}"
                logger.warning(warning_message)
                if save_fit:
                    self._user_fit_dic[user_id] = None
                return

            filter_list = None
            if methodology is not None:
                filter_list = set(methodology.filter_single(user_id, self.train_set, test_set))

            user_rank = alg.rank(user_train, loaded_items_interface,
                                 n_recs, filter_list=filter_list)

            return user_id, user_rank

        all_users = set(test_set.user_id_column)
        if user_id_list is not None:
            all_users = set(user_id_list)

        loaded_items_interface = self.algorithm._load_available_contents(self.items_directory, set())

        rank = []

        logger.info("Don't worry if it looks stuck at first")
        logger.info("First iterations will stabilize the estimated remaining time")

        with get_iterator_parallel(num_cpus,
                                   compute_single_fit_rank, all_users,
                                   progress_bar=True, total=len(all_users)) as pbar:

            pbar.set_description(f"Loading first items from memory...")
            for user_id, user_rank in pbar:
                pbar.set_description(f"Computing fit_rank for user {user_id}")
                rank.append(user_rank)

        rank = itertools.chain.from_iterable(rank)
        rank = Rank.from_list(rank)

        # we force the garbage collector after freeing loaded items
        del loaded_items_interface
        gc.collect()

        self._yaml_report = {'mode': 'rank', 'n_recs': repr(n_recs), 'methodology': repr(methodology)}

        return rank

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

        graph = rs.NXBipartiteGraph(train)

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

            graph = rs.NXBipartiteGraph(train_set)
            gbrs = rs.GraphBasedRS(alg, graph)
            rank_to_append = gbrs.rank(test_set)

            result_list.append(rank_to_append)
        ```

        `result_list` will contain recommendation lists for each split

    Args:
        algorithm: the graph based algorithm that will be used in order to
            rank or make score prediction
        graph: a Graph object containing interactions
    """

    def __init__(self,
                 algorithm: GraphBasedAlgorithm,
                 graph: FullDiGraph):

        self.__graph = graph
        super().__init__(algorithm)

    @property
    def users(self):
        return self.__graph.user_nodes

    @property
    def graph(self):
        """
        The graph containing interactions
        """
        return self.__graph

    @property
    def algorithm(self):
        """
        The graph based algorithm chosen
        """
        alg: GraphBasedAlgorithm = super().algorithm
        return alg

    def predict(self, test_set: Ratings, user_id_list: List = None,
                methodology: Union[Methodology, None] = TestRatingsMethodology(),
                num_cpus: int = 0) -> Prediction:
        """
        Method used to calculate score predictions for all users in test set or all users in `user_id_list` parameter.

        **BE CAREFUL**: not all algorithms are able to perform *score prediction*

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so for each user items in its test set only will be considered for score
        prediction

        If the algorithm was not fit for some users, they will be skipped and a warning is printed

        Args:
            test_set: Ratings object which represents the ground truth of the split considered
            user_id_list: List of users for which you want to compute score prediction. If None, the ranking
                will be computed for all users of the `test_set`
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method. Default is 0, meaning that
                the number of cpus will be automatically detected.

        Returns:
            Prediction object containing score prediction lists for all users of the test set or for all users in
                `user_id_list`

        """
        all_users = set(test_set.user_id_column)
        if user_id_list is not None:
            all_users = set(user_id_list)

        total_predict_list = self.algorithm.predict(all_users, self.graph, test_set, methodology, num_cpus)

        total_predict = Prediction.from_list(total_predict_list)

        self._yaml_report = {'graph': repr(self.graph), 'mode': 'score_prediction', 'methodology': repr(methodology)}

        return total_predict

    def rank(self, test_set: Ratings, n_recs: int = 10, user_id_list: List = None,
             methodology: Union[Methodology, None] = TestRatingsMethodology(),
             num_cpus: int = 0) -> Rank:
        """
        Method used to calculate ranking for all users in test set or all users in `user_id_list` parameter.
        You must first call the `fit()` method before you can compute the ranking.

        If the `n_recs` is specified, then the rank will contain the top-n items for the users.
        Otherwise, the rank will contain all unrated items of the particular users

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so, for each user, items in its test set only will be ranked

        If the algorithm was not fit for some users, they will be skipped and a warning is printed

        Args:
            test_set: Ratings object which represents the ground truth of the split considered
            n_recs: Number of the top items that will be present in the ranking of each user.
                If `None` all candidate items will be returned for the user
            user_id_list: List of users for which you want to compute the ranking. If None, the ranking will be computed
                for all users of the `test_set`
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method. Default is 0, meaning that
                the number of cpus will be automatically detected.

        Raises:
            NotFittedAlg: Exception raised when this method is called without first calling the `fit` method

        Returns:
            Rank object containing recommendation lists for all users of the test set or for all users in `user_id_list`

        """
        all_users = set(test_set.user_id_column)
        if user_id_list is not None:
            all_users = set(user_id_list)

        total_rank_list = self.algorithm.rank(all_users, self.graph, test_set, n_recs, methodology, num_cpus)

        total_rank = Rank.from_list(total_rank_list)

        if len(total_rank) == 0:
            logger.warning("No items could be ranked for any users! Remember that items to rank must be present "
                           "in the graph.\n"
                           "Try changing methodology!")

        elif len(set(total_rank.user_id_column)) != len(all_users):
            logger.warning(f"No items could be ranked for users {all_users - set(total_rank.user_id_column)}\n"
                           f"No nodes to rank for them found in the graph. Try changing methodology! ")

        self._yaml_report = {'graph': repr(self.graph), 'mode': 'rank', 'n_recs': repr(n_recs),
                             'methodology': repr(methodology)}

        return total_rank

    def __repr__(self):
        return f"GraphBasedRS(algorithm={self.algorithm}, graph={self.graph})"
