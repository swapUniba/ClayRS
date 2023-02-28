from __future__ import annotations
import abc
import os
import shutil
import sys
from abc import ABC
from typing import List, Union, Dict, Callable, TYPE_CHECKING, Optional

import pyaml

# fix circular import, for the future: move Ratings class to the RecSys module
if TYPE_CHECKING:
    from clayrs.content_analyzer import Ratings
    from clayrs.evaluation.metrics.metrics import Metric
    from clayrs.recsys.algorithm import Algorithm
    from clayrs.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm
    from clayrs.recsys.graph_based_algorithm.graph_based_algorithm import GraphBasedAlgorithm
    from clayrs.recsys.partitioning import Partitioning
    from clayrs.recsys.methodology import Methodology

from clayrs.evaluation import EvalModel
from clayrs.recsys import ContentBasedRS, GraphBasedRS, NXFullGraph, UserNode, ItemNode
from clayrs.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from clayrs.utils import Report
from clayrs.utils.const import logger
from clayrs.utils.save_content import get_valid_dirname
from clayrs.recsys.methodology import TestRatingsMethodology


class Experiment(ABC):
    """
    Abstract `Experiment` class

    It provides an easy interface to perform a complete experiment by comparing different algorithms, starting from
    splitting the dataset to evaluating predictions computed.
    It is also capable of producing a `yml` report for both the recsys phase and the evaluation phase.

    Both the evaluation phase and the report are optional and are produced only if specified.

    All the results (split, ranking, evaluation results, etc.) will be saved in the folder specified with the
    `output_folder` parameter. For each algorithm a different sub-folder will be created and named after the algorithm:

    * If multiple instances of the same algorithm are present in the `algorithm_list`, sub-folders will be disambiguated
        depending on the order of execution (`algName_1`, `algName_2`, `algName_3`, etc.)

    !!! info

        Please remember that by default if a folder with same name of the `output_folder` parameter is present,
        the experiment won't run and an exception will be raised. To overcome this, simply set the `overwrite_if_exists`
        parameter to True or change the `output_folder`.

    Args:
        original_ratings: Ratings object containing original interactions between users and items

        partitioning_technique: Partitioning object which specifies how the original ratings should be split

        algorithm_list: List of algorithm for which the experiment will be conducted

        items_directory: Path to the folder containing serialized complexly represented items

        users_directory: Path to the folder containing serialized complexly represented items

        metric_list: List of metric with which predictions computed by the RecSys will be evaluated

        report: If `True`, a `yml` report will be produced for the recsys phase. It will be also produced for the
            evaluation phase, but only if the `metric_list` parameter is set

        output_folder: Path of the folder where all the results of the experiment will be saved

        overwrite_if_exists: If `True` and the path set in `output_folder` points to an already existent directory,
            it will be deleted and replaced by the folder containing the experiment results
    """

    def __init__(self, original_ratings: Ratings,
                 partitioning_technique: Partitioning,
                 algorithm_list: List[Algorithm],
                 items_directory: str,
                 users_directory: str = None,
                 metric_list: List[Metric] = None,
                 report: bool = False,
                 output_folder: str = "experiment_result",
                 overwrite_if_exists: bool = False):

        self.original_ratings = original_ratings
        self.algorithm_list = algorithm_list
        self.pt = partitioning_technique
        self.items_directory = items_directory
        self.users_directory = users_directory
        self.metrics = metric_list
        self.report = report
        self.output_folder = output_folder
        self.overwrite_if_exists = overwrite_if_exists

    def main_experiment(self, alg_function: Callable, skip_alg_error: bool = True) -> None:
        """
        Main function which will perform the whole experiment

        It is responsible for splitting the dataset, computing ranking/score predictions, evaluating them, etc.
        The key parameter is `alg_function`:
        each subclass of Experiment should call this method by passing the appropriate callable (function) which should
        expect:

        * The current split number,
        * The current algorithm,
        * The current train set,
        * The current test set,
        * The current dirname (the directory where all the results of the current algorithm should be saved)

        The `alg_function` should return a Ratings object (or one of its subclasses)

        With this abstraction, all the subclasses should just define the `alg_function` function and just call this
        method

        Args:
            alg_function: Callable which perform the ranking/score prediction. Should return a Ratings object (or one
                of its subclasses)
            skip_alg_error: Boolean which defines the behaviour of the experiment when an algorithm raises an error.
                Useful for example when a pure ranking algorithm is asked to perform score prediction, if is set to
                `True` it will just skip to the next algorithm, otherwise the exception raised will be re-raised
        """

        try:
            os.makedirs(self.output_folder)
        except FileExistsError:
            if self.overwrite_if_exists is True:
                shutil.rmtree(self.output_folder)
                os.makedirs(self.output_folder)
            else:
                raise FileExistsError(f"Folder {self.output_folder} already present!\n"
                                      "Delete it and run the experiment again or set `overwrite_if_exists` parameter "
                                      "to True!") from None

        # save user_map
        with open(os.path.join(self.output_folder, "user_map.yml"), 'w') as yaml_file:
            pyaml.dump(self.original_ratings.user_map.to_dict(), yaml_file, sort_dicts=False, safe=True)

        # save item_map
        with open(os.path.join(self.output_folder, "item_map.yml"), 'w') as yaml_file:
            pyaml.dump(self.original_ratings.user_map.to_dict(), yaml_file, sort_dicts=False, safe=True)

        train_set_list, test_set_list = self.pt.split_all(self.original_ratings)

        for i, (train_set, test_set) in enumerate(zip(train_set_list, test_set_list)):
            train_set.to_csv(self.output_folder, file_name=f"{str(self.pt)}_train_split{i}", ids_as_str=True)
            test_set.to_csv(self.output_folder, file_name=f"{str(self.pt)}_test_split{i}", ids_as_str=True)

        for alg in self.algorithm_list:

            # Control variable that lets you skip directly to the next algorithm
            # when the NotPredictionAlg exception is raised but should be skipped
            skip_to_next_alg = False

            print(file=sys.stderr)  # just a simple space
            logger.info(f"******* Processing alg {alg} *******")

            all_alg_res = []

            dirname = get_valid_dirname(self.output_folder,
                                        str(alg),
                                        overwrite=False,
                                        start_from_1=True,
                                        style="underscore")

            os.makedirs(os.path.join(self.output_folder, dirname))

            for i, (train_set, test_set) in enumerate(zip(train_set_list, test_set_list)):

                try:

                    result = alg_function(i, alg, train_set, test_set, dirname)

                    all_alg_res.append(result)

                except NotPredictionAlg as e:
                    os.rmdir(f"{self.output_folder}/{dirname}")

                    if skip_alg_error:
                        logger.warning(str(e) + "\nThe algorithm will be skipped")
                        skip_to_next_alg = True
                        break  # we skip directly to the next algorithm to perform
                    else:
                        raise e

            if not skip_to_next_alg:
                if self.metrics is not None:
                    em = EvalModel(pred_list=all_alg_res,
                                   truth_list=test_set_list,
                                   metric_list=self.metrics)

                    sys_df, users_df = em.fit()

                    sys_df.to_csv(os.path.join(self.output_folder, dirname, 'eva_sys_results.csv'))
                    users_df.to_csv(os.path.join(self.output_folder, dirname, 'eva_users_results.csv'))

                    if self.report:
                        Report(output_dir=f"{self.output_folder}/{dirname}").yaml(eval_model=em)

                print(file=sys.stderr)  # just a simple space
                logger.info(f"Results saved in '{self.output_folder}/{dirname}'")

    @abc.abstractmethod
    def predict(self, methodology: Union[Methodology, None] = TestRatingsMethodology(),
                num_cpus: int = 0,
                skip_alg_error: bool = True):
        """
        Abstract method which should perform an experiment where score predictions are involved
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rank(self, n_recs: int = 10,
             methodology: Union[Methodology, None] = TestRatingsMethodology(),
             num_cpus: int = 0):
        """
        Abstract method which should perform an experiment where rankings are involved
        """
        raise NotImplementedError


class ContentBasedExperiment(Experiment):
    """
    The `Experiment` class for ***content based algorithms***

    It provides an easy interface to perform a complete experiment by comparing different cb-algorithms, starting from
    *splitting* the dataset to *evaluating* predictions computed.
    It is also capable of producing a `yml` report for both the recsys phase and the evaluation phase.

    Both the evaluation phase and the report are *optional* and are produced only if specified.

    All the results (*split*, *ranking*, *evaluation results*, etc.) will be saved in the folder specified with the
    `output_folder` parameter. For each algorithm a different sub-folder will be created and named after it:

    * If multiple instances of the same algorithm are present in the `algorithm_list`, sub-folders will be disambiguated
        depending on the order of execution (`algName_1`, `algName_2`, `algName_3`, etc.)

    !!! info

        Please remember that by default if a folder with same name of the `output_folder` parameter is present,
        the experiment won't run and an exception will be raised. To overcome this, simply set the `overwrite_if_exists`
        parameter to `True` or change the `output_folder`.

    Examples:

        Suppose you want to compare:

        * A `CentroidVector` algorithm
        * The `SVC` classifier
        * The `KNN` classifier

        For the three different configuration, an `HoldOut` partitioning technique should be used and results should
        be evaluated on $Precision$ and $Recall$

        ```python
        from clayrs.utils import ContentBasedExperiment
        from clayrs import content_analyzer as ca
        from clayrs import content_analyzer as rs

        original_rat = ca.Ratings(ca.CSVFile(ratings_path))

        alg1 = rs.CentroidVector({'Plot': 'tfidf'},
                                 similarity=rs.CosineSimilarity()) # (1)

        alg2 = rs.ClassifierRecommender({'Plot': 'tfidf'},
                                        classifier=rs.SkSVC()) # (2)

        alg3 = rs.ClassifierRecommender({'Plot': 'tfidf'},
                                         classifier=rs.SkKNN()) # (3)

        a = ContentBasedExperiment(
            original_ratings=rat,

            partitioning_technique=rs.HoldOutPartitioning(),

            algorithm_list=[alg1, alg2, alg3],

            items_directory=movies_dir,

            metric_list=[eva.Precision(), eva.Recall()]

            output_folder="my_experiment"
        )

        a.rank()
        ```

        1. Results will be saved in *my_experiment/CentroidVector_1*
        2. Results will be saved in *my_experiment/ClassifierRecommender_1*
        3. Results will be saved in *my_experiment/ClassifierRecommender_2*

    Args:
        original_ratings: Ratings object containing original interactions between users and items

        partitioning_technique: Partitioning object which specifies how the original ratings should be split

        algorithm_list: List of **Content Based algorithms** for which the whole experiment will be conducted

        items_directory: Path to the folder containing serialized complexly represented items

        users_directory: Path to the folder containing serialized complexly represented items. Needed only if
            one or more algorithms in `algorithm_list` needs it

        metric_list: List of metric with which predictions computed by the CBRS will be evaluated

        report: If `True`, a `yml` report will be produced for the recsys phase. It will be also produced for the
            evaluation phase, but only if the `metric_list` parameter is set

        output_folder: Path of the folder where all the results of the experiment will be saved

        overwrite_if_exists: If `True` and the path set in `output_folder` points to an already existent directory,
            it will be deleted and replaced by the folder containing the experiment results
    """
    def __init__(self, original_ratings: Ratings,
                 partitioning_technique: Partitioning,
                 algorithm_list: List[ContentBasedAlgorithm],
                 items_directory: str,
                 users_directory: str = None,
                 metric_list: List[Metric] = None,
                 report: bool = False,
                 output_folder: str = "experiment_result",
                 overwrite_if_exists: bool = False):

        super().__init__(original_ratings, partitioning_technique, algorithm_list, items_directory, users_directory,
                         metric_list, report, output_folder, overwrite_if_exists)

    def predict(self,
                user_id_list: Optional[Union[List[str], List[int]]] = None,
                methodology: Optional[Methodology] = TestRatingsMethodology(),
                num_cpus: int = 1,
                skip_alg_error: bool = True) -> None:
        """
        Method used to perform an experiment which involves ***score predictions***.

        The method will first split the original ratings passed in the constructor in train and test set, then
        the Recommender System will be fit for each user in the train set.

        If the algorithm can't be fit for some users, a warning message is printed and no score predictions will be
        computed for said user

        !!! info

            **BE CAREFUL**: not all algorithms are able to perform *score prediction*. In case a pure ranking algorithm
            is asked to perform score prediction, the `NotPredictionAlg` will be raised. if the `skip_alg_error` is set
            to `True`, then said exception will be caught, a warning will be printed, and the experiment will go on with
            the next algorithm

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so, for each user, items in its test set only will be considered for score
        prediction

        Args:
            user_id_list: List of users for which you want to compute the ranking. If None, the ranking will be computed
                for all users of the `test_set`
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method. Default is 0, meaning that
                the number of cpus will be automatically detected.
            skip_alg_error: If set to `True`, a pure ranking algorithm will be skipped and the experiment will continue
                with the following algorithm. Otherwise, the `NotPredictionAlg` exception raised will be re-raised

        Raises:
            NotPredictionAlg: When a pure ranking algorithm is asked to perform score prediction and
                `skip_alg_error` == False
        """
        def cb_fit_predict(split_num, alg, train_set, test_set, dirname):
            cbrs = ContentBasedRS(alg, train_set, self.items_directory)

            predict_alg = cbrs.fit_predict(test_set, methodology=methodology, num_cpus=num_cpus,
                                           user_list=user_id_list)

            predict_alg.to_csv(f"{self.output_folder}/{dirname}", file_name=f"rs_predict_split{split_num}",
                               ids_as_str=True)

            if self.report:
                Report(output_dir=f"{self.output_folder}/{dirname}").yaml(original_ratings=self.original_ratings,
                                                                          partitioning_technique=self.pt,
                                                                          recsys=cbrs)

            return predict_alg

        self.main_experiment(cb_fit_predict, skip_alg_error=skip_alg_error)

    def rank(self,
             n_recs: Optional[int] = 10,
             user_id_list: Optional[Union[List[str], List[int]]] = None,
             methodology: Optional[Methodology] = TestRatingsMethodology(),
             num_cpus: int = 1):
        """
        Method used to perform an experiment which involves ***rankings***.

        The method will first split the original ratings passed in the constructor in train and test set, then
        the Recommender System will be fit for each user in the train.
        If the algorithm can't be fit for some users, a warning message is printed and no ranking will be
        computed for said user

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so, for each user, items in its test set only will be considered as eligible
        for ranking

        Args:
            n_recs: Number of the top items that will be present in the ranking of each user.
                If `None` all candidate items will be returned for the user
            user_id_list: List of users for which you want to compute the ranking. If None, the ranking will be computed
                for all users of the `test_set`
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method. Default is 0, meaning that
                the number of cpus will be automatically detected

        """
        def cb_fit_rank(split_num, alg, train_set, test_set, dirname):
            cbrs = ContentBasedRS(alg, train_set, self.items_directory)

            predict_alg = cbrs.fit_rank(test_set, n_recs=n_recs, methodology=methodology,
                                        user_list=user_id_list, num_cpus=num_cpus)

            predict_alg.to_csv(f"{self.output_folder}/{dirname}", file_name=f"rs_rank_split{split_num}",
                               ids_as_str=True)

            if self.report:
                Report(output_dir=f"{self.output_folder}/{dirname}").yaml(original_ratings=self.original_ratings,
                                                                          partitioning_technique=self.pt,
                                                                          recsys=cbrs)

            return predict_alg

        self.main_experiment(cb_fit_rank)


class GraphBasedExperiment(Experiment):
    """
    The `Experiment` class for ***graph based algorithms***

    It provides an easy interface to perform a **complete experiment** by comparing different gb-algorithms, starting
    from *splitting* the dataset to *evaluating* predictions computed.

    Every graph based algorithm expects a graph: that's why right before computing ranking/score predictions, a graph
    will be created depending on the current train and test split (if multiple are available):

    * All the nodes from the original graph will be present, The interactions in the test set will be
        missing (It won't be represented as a link between a user node and an item node)

    The class is also capable of producing a `yml` report for both the recsys phase and the evaluation phase.

    Both the evaluation phase and the report are optional and are produced only if specified.

    All the results (*split*, *ranking*, *evaluation results*, etc.) will be saved in the folder specified with the
    `output_folder` parameter. For each algorithm a different sub-folder will be created and named after it:

    * If multiple instances of the same algorithm are present in the `algorithm_list`, sub-folders will be disambiguated
        depending on the order of execution (`algName_1`, `algName_2`, `algName_3`, etc.)

    !!! info

        Please remember that by default if a folder with same name of the `output_folder` parameter is present,
        the experiment won't run and an exception will be raised. To overcome this, simply set the `overwrite_if_exists`
        parameter to `True` or change the `output_folder`.

    Examples:

        Suppose you want to compare:

        * The `PageRank` algorithm with `alpha=0.8`
        * The `PageRank` algorithm with `alpha=0.9`
        * The `Personalized PageRank` algorithm

        For the three different configuration, a `KFold` partitioning technique with three splits should be used and
        results should be evaluated on $Precision$, $Recall$, $NDCG$

        ```python
        from clayrs.utils import GraphBasedExperiment
        from clayrs import content_analyzer as ca
        from clayrs import content_analyzer as rs

        original_rat = ca.Ratings(ca.CSVFile(ratings_path))

        alg1 = rs.NXPageRank(alpha=0.8) # (1)

        alg2 = rs.NXPageRank(alpha=0.9) # (2)

        alg3 = rs.NXPageRank(personalized=True) # (3)

        a = GraphBasedExperiment(
            original_ratings=rat,

            partitioning_technique=rs.KFoldPartitioning(n_splits=3),

            algorithm_list=[alg1, alg2, alg3],

            items_directory=movies_dir,

            metric_list=[eva.Precision(), eva.Recall()]

            output_folder="my_experiment"
        )

        a.rank()
        ```

        1. Results will be saved in *my_experiment/NXPageRank_1*
        2. Results will be saved in *my_experiment/NXPageRank_2*
        3. Results will be saved in *my_experiment/NXPageRank_3*

    Args:
        original_ratings: Ratings object containing original interactions between users and items

        partitioning_technique: Partitioning object which specifies how the original ratings should be split

        algorithm_list: List of **Graph Based algorithms** for which the whole experiment will be conducted

        items_directory: Path to the folder containing serialized complexly represented items with one or more
            exogenous property to load

        item_exo_properties: Set or Dict which contains representations to load from items. Use a `Set` if you want
            to load all properties from specific representations, or use a `Dict` if you want to choose which properties
            to load from specific representations

        users_directory: Path to the folder containing serialized complexly represented users with one or more
            exogenous property to load

        user_exo_properties: Set or Dict which contains representations to load from items. Use a `Set` if you want
            to load all properties from specific representations, or use a `Dict` if you want to choose which properties
            to load from specific representations

        link_label: If specified, each link between user and item nodes will be labeled with the given label.
            Default is None

        metric_list: List of metric with which predictions computed by the GBRS will be evaluated

        report: If `True`, a `yml` report will be produced for the recsys phase. It will be also produced for the
            evaluation phase, but only if the `metric_list` parameter is set

        output_folder: Path of the folder where all the results of the experiment will be saved

        overwrite_if_exists: If `True` and the path set in `output_folder` points to an already existent directory,
            it will be deleted and replaced by the folder containing the experiment results
    """
    def __init__(self, original_ratings: Ratings,
                 partitioning_technique: Partitioning,
                 algorithm_list: List[GraphBasedAlgorithm],
                 items_directory: str = None,
                 item_exo_properties: Union[Dict, set] = None,
                 users_directory: str = None,
                 user_exo_properties: Union[Dict, set] = None,
                 link_label: str = None,
                 metric_list: List[Metric] = None,
                 report: bool = False,
                 output_folder: str = "experiment_result",
                 overwrite_if_exists: bool = False):

        super().__init__(original_ratings, partitioning_technique, algorithm_list, items_directory, users_directory,
                         metric_list, report, output_folder, overwrite_if_exists)

        self.item_exo_properties = item_exo_properties
        self.user_exo_properties = user_exo_properties
        self.link_label = link_label

    def predict(self,
                user_id_list: Optional[Union[List[str], List[int]]] = None,
                methodology: Optional[Methodology] = TestRatingsMethodology(),
                num_cpus: int = 1,
                skip_alg_error: bool = True):
        """
        Method used to perform an experiment which involves ***score predictions***.

        The method will first split the original ratings passed in the constructor in train and test set, then
        a graph will be built depending on them:

        * All nodes of the original ratings will be present, but the *links* (***interactions***) that are present in
        the test set will be missing, so to make the training phase *fair*

        !!! info

            **BE CAREFUL**: not all algorithms are able to perform *score prediction*. In case a pure ranking algorithm
            is asked to perform score prediction, the `NotPredictionAlg` will be raised. if the `skip_alg_error` is set
            to `True`, then said exception will be caught, a warning will be printed, and the experiment will go on with
            the next algorithm

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so, for each user, items in its test set only will be considered for score
        prediction

        Args:
            user_id_list: List of users for which you want to compute the ranking. If None, the ranking will be computed
                for all users of the `test_set`
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method. Default is 0, meaning that
                the number of cpus will be automatically detected.
            skip_alg_error: If set to `True`, a pure ranking algorithm will be skipped and the experiment will continue
                with the following algorithm. Otherwise, the `NotPredictionAlg` exception raised will be re-raised

        Raises:
            NotPredictionAlg: When a pure ranking algorithm is asked to perform score prediction
                and `skip_alg_error` == False
        """
        def gb_fit_predict(split_num, alg, train_set, test_set, dirname):

            graph = NXFullGraph(self.original_ratings,
                                item_contents_dir=self.items_directory,
                                item_exo_properties=self.item_exo_properties,
                                user_contents_dir=self.users_directory,
                                user_exo_properties=self.user_exo_properties,
                                link_label=self.link_label)

            for user, item in zip(test_set.user_id_column, test_set.item_id_column):
                graph.remove_link(UserNode(user), ItemNode(item))

            gbrs = GraphBasedRS(alg, graph)

            predict_alg = gbrs.predict(test_set, methodology=methodology,
                                       num_cpus=num_cpus, user_list=user_id_list)

            predict_alg.to_csv(f"{self.output_folder}/{dirname}", file_name=f"rs_predict_split{split_num}")

            if self.report:
                Report(output_dir=f"{self.output_folder}/{dirname}").yaml(original_ratings=self.original_ratings,
                                                                          partitioning_technique=self.pt,
                                                                          recsys=gbrs)

            return predict_alg

        self.main_experiment(gb_fit_predict, skip_alg_error=skip_alg_error)

    def rank(self,
             n_recs: Optional[int] = 10,
             user_id_list: Optional[Union[List[str], List[int]]] = None,
             methodology: Optional[Methodology] = TestRatingsMethodology(),
             num_cpus: int = 1):
        """
        Method used to perform an experiment which involves ***rankings***.

        The method will first split the original ratings passed in the constructor in train and test set, then
        a graph will be built depending on them:

        * All nodes of the original ratings will be present, but the *links* (***interactions***) that are present in
        the test set will be missing, so to make the training phase *fair*

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used, so for each user, items in its test set only will be eligible for ranking

        Args:
            n_recs: Number of the top items that will be present in the ranking of each user.
                If `None` all candidate items will be returned for the user
            user_id_list: List of users for which you want to compute the ranking. If None, the ranking will be computed
                for all users of the `test_set`
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method. Default is 0, meaning that
                the number of cpus will be automatically detected.

        """
        def gb_fit_rank(split_num, alg, train_set, test_set, dirname):

            graph = NXFullGraph(self.original_ratings,
                                item_contents_dir=self.items_directory,
                                item_exo_properties=self.item_exo_properties,
                                user_contents_dir=self.users_directory,
                                user_exo_properties=self.user_exo_properties,
                                link_label=self.link_label)

            for user, item in zip(test_set.user_id_column, test_set.item_id_column):
                graph.remove_link(UserNode(user), ItemNode(item))

            gbrs = GraphBasedRS(alg, graph)

            predict_alg = gbrs.rank(test_set, n_recs=n_recs, methodology=methodology,
                                    num_cpus=num_cpus, user_list=user_id_list)

            predict_alg.to_csv(f"{self.output_folder}/{dirname}", file_name=f"rs_rank_split{split_num}")

            if self.report:
                Report(output_dir=f"{self.output_folder}/{dirname}").yaml(original_ratings=self.original_ratings,
                                                                          partitioning_technique=self.pt,
                                                                          recsys=gbrs)

            return predict_alg

        self.main_experiment(gb_fit_rank)
