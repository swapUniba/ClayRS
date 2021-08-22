import os
import pathlib as pl
import pandas as pd
from unittest import TestCase

from orange_cb_recsys.content_analyzer.embeddings.embedding_learner.embedding_learner import EmbeddingLearner
from orange_cb_recsys.recsys.recsys import ContentBasedRS
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.evaluation import MetricCalculator
from orange_cb_recsys.script.exceptions import ScriptConfigurationError, ParametersError, NoOutputDirectoryDefined, \
    InvalidFilePath
from orange_cb_recsys.script.script_handling import handle_script_contents, RecSysRun, EvalRun, MethodologyRun, MetricCalculatorRun, \
    PartitioningRun, Run, NeedsSerializationRun, script_run_standard, script_run_with_classes_file
from orange_cb_recsys.content_analyzer.information_processor import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile
from orange_cb_recsys.runnable_instances import serialize_classes
from orange_cb_recsys.utils.const import root_path
from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
movies_info_reduced = os.path.join(root_path, "datasets/movies_info_reduced.json")
user_info = os.path.join(root_path, "datasets/users_info.json")
raw_ratings = os.path.join(root_path, "datasets/examples/new_ratings.csv")
ranking_path = os.path.join(THIS_DIR, "ranking")
prediction_path = os.path.join(THIS_DIR, "prediction")
eval_path = os.path.join(THIS_DIR, "eval")
multiple_params_dir = os.path.join(THIS_DIR, "multiple_params_test")
run_dir = os.path.join(THIS_DIR, "run_test")
ratings_example = os.path.join(run_dir, "ratings_test_script.csv")
users_example = os.path.join(run_dir, "user_test_script")
items_example = os.path.join(run_dir, "movielens_test_script")
embedding_example = os.path.join(run_dir, "word2vec_test_script.bin")


class TestRun(TestCase):

    @staticmethod
    def resetRuns():
        RecSysRun.recsys_number = 0
        EvalRun.eval_number = 0
        MethodologyRun.methodology_number = 0
        MetricCalculatorRun.metric_calculator_number = 0
        PartitioningRun.partitioning_number = 0

    def setUp(self) -> None:

        item_config_dict = {
            "module": "ContentAnalyzer",
            "config": {"class": "itemanalyzerconfig",
                       "source": {"class": "JSONFile", "file_path": movies_info_reduced},
                       "id": "imdbID",
                       "output_directory": items_example,
                       "field_dict": {"Plot": [{"class": "FieldConfig", "content_technique": {"class": "SKLearnTfIdf"},
                                                "preprocessing": {"class": "NLTK"}, "id": "test"},
                                               {"class": "fieldconfig", "content_technique": {"class": "sklearntfidf"},
                                                "preprocessing": {"class": "nltk"}}]},
                       },
            "fit": {}
        }

        user_config_dict = {
            "module": "ContentAnalyzer",
            "config": {"class": "useranalyzerconfig",
                       "source": {"class": "JSONFile", "file_path": user_info},
                       "id": "user_id",
                       "output_directory": users_example,
                       "field_dict": {"name": [{"class": "FieldConfig", "id": "test"}]},
                       "exogenous_representation_list": [
                           {"class": "ExogenousConfig",
                            "exogenous_technique": {"class": "PropertiesFromDataset", "field_name_list": ["gender", "occupation"]}}
                       ]
                       },
            "fit": {}
        }

        rating_config_dict = {
            "module": "RatingsImporter",
            "source": {"class": "csvfile", "file_path": raw_ratings},
            "from_id_column": 0,
            "to_id_column": 1,
            "score_column": 2,
            "timestamp_column": 3,
            "score_processor": {"class": "NumberNormalizer"},
            "import_ratings": {},
            "imported_ratings_to_csv": {
                "output_directory": run_dir,
                "file_name": "ratings_test_script",
                "overwrite": True
            }
        }

        embedding_learner_dict = {
            "module": "gensimword2vec",
            "reference": embedding_example,
            "workers": 4,
            "fit": {"source": {"class": "JSONFile", "file_path": movies_info_reduced},
                    "field_list": ["Plot"],
                    "preprocessor_list": [{"class": "nltk"}]}
        }

        recsys_config_dict = {
            "module": "contentbasedrs",
            "users_directory": users_example,
            "items_directory": items_example,
            "rating_frame": ratings_example,
            "algorithm": {
                "class": "LinearPredictor", "item_field": {"Plot": [0]}, "regressor": {"class": "sklinearregression"}},
            "fit_rank": {"user_id": "8", "recs_number": 10,
                         "filter_list": ["tt0114885", "tt0113987", "tt0114709", "tt0112641", "tt0114388"]},
            "fit_predict": {"user_id": "8", "filter_list": ["tt0112281", "tt0112896"]},
            "multiple_fit_rank": {"user_id_list": ["8", "9"]},
            "multiple_fit_predict": {"user_id_list": ["8", "9"]},
            "output_directory": run_dir
        }

        eval_dict = {
            "module": "evalmodel",
            "recsys": {"class": "contentbasedrs", "users_directory": users_example, "items_directory": items_example,
                       "rating_frame": ratings_example,
                       "algorithm": {"class": "LinearPredictor", "item_field": {"Plot": [0]},
                                     "regressor": {"class": "sklinearregression"}}},
            "partitioning": {"class": "KFoldPartitioning"},
            "metric_list": [{"class": "Precision"},
                            {"class": "PredictionCoverage", "catalog": os.path.join(root_path, 'datasets/test_script/catalog.json')},
                            {"class": "NDCG"}],
            "methodology": {"class": "TestRatingsMethodology"},
            "output_directory": run_dir,
            "fit": {}
        }

        metric_calculator_dict = {
            "module": "metriccalculator",
            "predictions_truths": [{"class": "Split",
                                    "first_set": os.path.join(run_dir, "rank_0_0.csv"),
                                    "second_set": ratings_example}],
            "output_directory": run_dir,
            "eval_metrics": {
                "metric_list": [{"class": "Precision"}, {"class": "GiniIndex"}, {"class": "NDCG"}]
            }
        }

        methodology_dict = {
            "module": "testratingsmethodology",
            "output_directory": run_dir,
            "only_greater_eq": 0.5,
            "get_item_to_predict": {
                "split_list": [{"class": "Split",
                                "first_set": os.path.join(run_dir, "training_0_0#0.csv"),
                                "second_set": os.path.join(run_dir, "testing_0_0#0.csv")}]
            }
        }

        partitioning_dict = {
            "module": "partitionmodule",
            "output_directory": run_dir,
            "partition_technique": {"class": "KFoldPartitioning"},
            "split_all": {
                "ratings": ratings_example,
                "user_id_list": ["8"]
            }
        }

        graph_dict = {
            "module": "Nxfullgraph",
            "source_frame": ratings_example,
            "user_contents_dir": users_example,
            "user_exo_representation": 0,
            "serialize": {"output_directory": os.path.join(run_dir, "graph_test")}
        }

        graph_based_recsys_dict = {
            "module": "GraphBasedRs",
            "algorithm": {"class": "NXPageRank", "personalized": True},
            "graph": os.path.join(run_dir, "graph_test/graph.xz"),
            "output_directory": run_dir,
            "fit_rank": {"user_id": "8", "recs_number": 10}
        }

        self.config_list = [item_config_dict, user_config_dict, rating_config_dict, recsys_config_dict,
                            eval_dict, embedding_learner_dict, partitioning_dict,
                            metric_calculator_dict, methodology_dict, graph_dict, graph_based_recsys_dict]

    def test_run(self):
        try:
            if not os.path.exists(run_dir):
                os.mkdir(run_dir)

            handle_script_contents(self.config_list)

            self.assertEqual(pl.Path(os.path.join(run_dir, "eval_sys_results_0_0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(run_dir, "eval_user_results_0_0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(run_dir, "item_to_predict_0_0#0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(run_dir, "mc_sys_results_0_0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(run_dir, "mc_user_results_0_0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(run_dir, "predict_0_0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(run_dir, "rank_0_0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(run_dir, "multiple_rank_0_0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(run_dir, "multiple_predict_0_0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(run_dir, "testing_0_0#0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(run_dir, "testing_0_0#1.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(run_dir, "training_0_0#0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(run_dir, "training_0_0#1.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(run_dir, "graph_test/graph.xz")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(run_dir, "rank_1_0.csv")).is_file(), True)
            self.assertEqual(pl.Path(ratings_example).is_file(), True)
            self.assertEqual(pl.Path(embedding_example).is_file(), True)
            self.assertEqual(pl.Path(items_example).is_dir(), True)
            self.assertEqual(pl.Path(users_example).is_dir(), True)

        finally:
            self.resetRuns()

    def test_serialization_multiple_parameters(self):
        try:

            # test for the majority of the modules that serialize results (while also giving the methods multiple params)

            if not os.path.exists(multiple_params_dir):
                os.mkdir(multiple_params_dir)

            ratings = pd.read_csv(raw_ratings)
            ratings.columns = ['from_id', 'to_id', 'score', 'timestamp']
            ratings_path = os.path.join(multiple_params_dir, "ratings.csv")
            ratings.to_csv(ratings_path, index=False)

            recsys_config_dict_multiple_params = {
                "users_directory": os.path.join(root_path, 'contents/users_codified'),
                "items_directory": os.path.join(root_path, 'contents/movies_codified'),
                "rating_frame": ratings_path,
                "algorithm": {
                    "class": "LinearPredictor", "item_field": {"Plot": [0]}, "regressor": {"class": "sklinearregression"}},
                "fit_rank": [{"user_id": "5"}, {"user_id": "6"}],
                "fit_predict": [{"user_id": "5"}, {"user_id": "6"}],
                "output_directory": multiple_params_dir
            }

            RecSysRun.run(recsys_config_dict_multiple_params, "contentbasedrs")

            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "rank_0_0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "rank_0_1.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "predict_0_0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "predict_0_1.csv")).is_file(), True)

            metric_calculator_dict_multiple_params = {
                "predictions_truths": [{"class": "Split", "first_set": os.path.join(multiple_params_dir, "rank_0_0.csv"),
                                        "second_set": ratings_path}],
                "output_directory": multiple_params_dir,
                "eval_metrics": [{"metric_list": [{"class": "Precision"}]},
                                 {"metric_list": [{"class": "NDCG"}]}]
            }

            MetricCalculatorRun.run(metric_calculator_dict_multiple_params, "metriccalculator")

            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "mc_sys_results_0_0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "mc_user_results_0_0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "mc_sys_results_0_1.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "mc_user_results_0_1.csv")).is_file(), True)

            partitioning_dict_multiple_params = {
                "output_directory": multiple_params_dir,
                "partition_technique": {"class": "KFoldPartitioning"},
                "split_all": [{"ratings": ratings_path, "user_id_list": ["5"]},
                              {"ratings": ratings_path, "user_id_list": ["6"]}]
            }

            PartitioningRun.run(partitioning_dict_multiple_params, "partitionmodule")

            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "training_0_0#0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "training_0_0#1.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "training_0_1#0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "training_0_1#1.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "testing_0_0#0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "testing_0_0#1.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "testing_0_1#0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "testing_0_1#1.csv")).is_file(), True)

            methodology_dict_multiple_params = {
                "output_directory": multiple_params_dir,
                "only_greater_eq": 2,
                "get_item_to_predict": [
                    {"split_list": [{"class": "Split",
                                     "first_set": os.path.join(multiple_params_dir, "training_0_0#0.csv"),
                                     "second_set": os.path.join(multiple_params_dir, "testing_0_0#0.csv")}]},
                    {"split_list": [{"class": "Split",
                                     "first_set": os.path.join(multiple_params_dir, "training_0_0#1.csv"),
                                     "second_set": os.path.join(multiple_params_dir, "testing_0_0#1.csv")},
                                    {"class": "Split",
                                     "first_set": os.path.join(multiple_params_dir, "training_0_1#0.csv"),
                                     "second_set": os.path.join(multiple_params_dir, "testing_0_1#0.csv")}]}]
            }

            MethodologyRun.run(methodology_dict_multiple_params, "testratingsmethodology")

            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "item_to_predict_0_0#0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "item_to_predict_0_1#0.csv")).is_file(), True)
            self.assertEqual(pl.Path(os.path.join(multiple_params_dir, "item_to_predict_0_1#1.csv")).is_file(), True)

        finally:
            self.resetRuns()

    def test_exceptions(self):
        try:
            # test for list not containing dictionaries only
            test_config_list_dict = [set(), dict()]
            with self.assertRaises(ScriptConfigurationError):
                handle_script_contents(test_config_list_dict)

            # test for dictionary in the list with no "module" parameter
            test_config_list_dict = {"parameter": "test"}
            with self.assertRaises(ScriptConfigurationError):
                handle_script_contents(test_config_list_dict)

            # test for dictionary in the list with "module" parameter but not valid value
            test_config_list_dict = [{"module": "test"}]
            with self.assertRaises(ScriptConfigurationError):
                handle_script_contents(test_config_list_dict)

            # test for not defined output directory in module that needs it
            test_dict = {"module": "contentbasedrs",
                         "rating_frame": ratings_example}
            with self.assertRaises(NoOutputDirectoryDefined):
                NeedsSerializationRun.setup_output_directory(test_dict, ContentBasedRS)

            # test for not valid ratings frame csv path
            test_dict = {"rating_frame": "not_valid_path",
                         "items_directory": "some_dir",
                         "algorithm": {
                            "class": "LinearPredictor", "item_field": {"Plot": [0]}, "regressor": {"class": "sklinearregression"}},
                         "output_directory": "some_dir"
                         }
            with self.assertRaises(InvalidFilePath):
                RecSysRun.run(test_dict, "contentbasedrs")

            # test for not valid graph path
            test_dict = {"graph": "not_valid_path",
                         "algorithm": {"class": "NXPageRank", "personalized": True},
                         "output_directory": "some_dir"}
            with self.assertRaises(InvalidFilePath):
                RecSysRun.run(test_dict, "graphbasedrs")

            # test for not valid json list path
            test_dict = {"partition_technique": {"class": "KFoldPartitioning"},
                         "output_directory": "some_dir",
                         "split_all": {"user_id_list": "not_valid_path", "ratings": ratings_example}}
            with self.assertRaises(InvalidFilePath):
                PartitioningRun.run(test_dict, "partitionmodule")
        finally:
            self.resetRuns()

    def test_extract_parameters(self):
        # test for a normal extract parameters run
        parameters = Run.extract_parameters({
            "source": {"class": "JSONFile", "file_path": movies_info_reduced},
            "field_list": ["Plot"],
            "preprocessor_list": [{"class": "nltk"}]
        }, EmbeddingLearner.fit)

        self.assertIsInstance(parameters["source"], JSONFile)
        self.assertEqual(parameters["field_list"][0], "Plot")
        self.assertIsInstance(parameters["preprocessor_list"][0], NLTK)

        # test for extract parameters with a pd.DataFrame variable
        parameters = Run.extract_parameters({
            "predictions_truths": [{"class": "Split",
                                    "first_set": raw_ratings, "second_set": raw_ratings}]
        }, MetricCalculator)

        self.assertIsInstance(parameters["predictions_truths"][0], Split)
        self.assertIsInstance(parameters["predictions_truths"][0].truth, pd.DataFrame)

        # test for parameters extraction on wrong parameter for fit method
        with self.assertRaises(ParametersError):
            Run.extract_parameters({
                "wrong_parameter": "test"
            }, EmbeddingLearner.fit)

    def test_dict_detector(self):
        # test for dictionary detector where each key of the dictionary represents a different case
        dictionary = {"1": [{"class": "NumberNormalizer"}, "NumberNormalizer"],
                      "2": {"class": "NumberNormalizer"},
                      "3": {"NumberNormalizer"},
                      "4": 0,
                      "5": None,
                      "6": {"class": "MetricCalculator",
                            "predictions_truths": [{"class": "Split",
                                                    "first_set": raw_ratings, "second_set": raw_ratings}]}}

        detected_dictionary = Run.dict_detector(dictionary)

        self.assertIsInstance(detected_dictionary["1"][0], NumberNormalizer)
        self.assertEqual(detected_dictionary["1"][1], "NumberNormalizer")
        self.assertIsInstance(detected_dictionary["2"], NumberNormalizer)
        self.assertEqual(detected_dictionary["3"], {"NumberNormalizer"})
        self.assertEqual(detected_dictionary["4"], 0)
        self.assertEqual(detected_dictionary["5"], None)
        self.assertIsInstance(detected_dictionary["6"], MetricCalculator)
        self.assertIsInstance(detected_dictionary["6"]._split_list[0].truth, pd.DataFrame)

        # test for dictionary detector when a parameter for a class constructor doesn't exist
        dictionary = {"1": {"class": "NumberNormalizer", "not_existing_parameter": "value"}}

        with self.assertRaises(ParametersError):
            Run.dict_detector(dictionary)

    def test_config_file_loading(self):
        json_path = os.path.join(root_path, 'datasets/test_script/empty_list.json')
        yml_path = os.path.join(root_path, 'datasets/test_script/empty_list.yml')
        # there are no asserts because the files loaded by the script run only contain an empty list
        # therefore nothing will be done
        # this test is used to make sure that nothing happens
        script_run_standard(json_path)
        script_run_standard(yml_path)

        serialize_classes(THIS_DIR)
        classes_file_path = os.path.join(THIS_DIR, 'classes.xz')

        # same as above
        script_run_with_classes_file(json_path, classes_file_path)
        script_run_with_classes_file(yml_path, classes_file_path)

        # tests for loading a script file witt a non supported format (csv)
        with self.assertRaises(ScriptConfigurationError):
            script_run_standard(os.path.join(root_path, 'datasets/movies_info_reduced.csv'))

        with self.assertRaises(ScriptConfigurationError):
            script_run_with_classes_file(os.path.join(root_path, 'datasets/movies_info_reduced.csv'), classes_file_path)
