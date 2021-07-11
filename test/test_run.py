import os
import unittest
from unittest import TestCase

# from orange_cb_recsys.__main__ import script_run

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
movies_info_reduced = os.path.join(THIS_DIR, "../../datasets/movies_info_reduced.json")
user_info = os.path.join(THIS_DIR, "../../datasets/users_info.json")
new_ratings = os.path.join(THIS_DIR, "../../datasets/examples/new_ratings.csv")
test_ratings_file = os.path.join(THIS_DIR, "../../datasets/test_ratings_file_script")
users_example_1 = os.path.join(THIS_DIR, "user_test_script")
items_example_1 = os.path.join(THIS_DIR, "../../contents/examples/ex_1/movies_1600355972.49884")
items_example_2 = os.path.join(THIS_DIR, "movielens_test_script")
ratings_example = os.path.join(THIS_DIR, "../../datasets/test_ratings/test_ratings_1618757236.csv")
ranking_path = os.path.join(THIS_DIR, "ranking")
prediction_path = os.path.join(THIS_DIR, "prediction")
eval_path = os.path.join(THIS_DIR, "eval")

@unittest.skip('Script not yet updated')
class TestRun(TestCase):

    def setUp(self) -> None:

        item_config_dict = {
            "module": "item_analyzer",
            "source": {"class": "json", "file_path": movies_info_reduced},
            "id": "imdbID",
            "output_directory": "movielens_test_script",
            "field_dict": {"Plot": [{"class": "field_config", "content_technique": {"class": "sk_learn_tf-idf"},
                                     "preprocessing": {"class": "nltk"}, "id": "test"},
                                    {"class": "field_config", "content_technique": {"class": "sk_learn_tf-idf"},
                                     "preprocessing": {"class": "nltk"}}]}
        }

        user_config_dict = {
            "module": "user_analyzer",
            "source": {"class": "json", "file_path": user_info},
            "id": "user_id",
            "output_directory": "user_test_script",
            "field_dict": {"name": [{"class": "field_config", "id": "test"}]}
        }

        rating_config_dict = {
            "module": "ratings",
            "source": {"class": "csv", "file_path": new_ratings},
            "from_field_name": "user_id",
            "to_field_name": "item_id",
            "timestamp_field_name": "timestamp",
            "output_directory": test_ratings_file,
            "rating_configs": {
                "class": "ratings_config", "field_name": "points",
                "processor": {"class": "number_normalizer", "max_": 5.0, "min_": 0.0}}
        }

        recsys_config_dict = {
            "module": "recsys_config",
            "users_directory": users_example_1,
            "items_directory": items_example_2,
            "rating_frame": ratings_example,
            "ranking_algorithm": {
                "class": "classifier", "item_field": {"Plot": [0]}, "classifier": {"class": "knn"}},
            "rankings": [{"user_id": "10", "recs_number": 2},
                         {"user_id": "10", "recs_number": 3}],
            "path_ranking": ranking_path
        }

        eval_config_dict = {
            "module": "ranking_alg_eval_model",
            "recsys_config_module": "recsys_config",
            "users_directory": users_example_1,
            "items_directory": items_example_2,
            "ranking_algorithm": {
                "class": "classifier", "item_field": {"Plot": [0]}, "classifier": {"class": "knn"}},
            "rating_frame": ratings_example,
            "partitioning": {"class": "k_fold", "n_splits": 2},
            "metric_list": [{"class": "fnmeasure", "n": 2}, {"class": "precision"}],
            "path_eval": eval_path
        }

        self.config_list = [item_config_dict, user_config_dict, rating_config_dict, recsys_config_dict, eval_config_dict]

    def test_run(self):
        script_run(self.config_list)

    def test_exceptions(self):
        # test for list not containing dictionaries only
        test_config_list_dict = [set(), dict()]
        with self.assertRaises(ValueError):
            script_run(test_config_list_dict)

        # test for dictionary in the list with no "module" parameter
        test_config_list_dict = {"parameter": "test"}
        with self.assertRaises(KeyError):
            script_run(test_config_list_dict)

        # test for dictionary in the list with "module" parameter but not valid value
        test_config_list_dict = [{"module": "test"}]
        with self.assertRaises(ValueError):
            script_run(test_config_list_dict)

        # test for not valid parameter name in dictionary representing object
        test_dict = {"module": "recsys",
                     "users_directory": users_example_1,
                     "items_directory": items_example_1,
                     "ranking_algorithm": {
                         "class": "classifier", "test": {"Plot": ["0"]}, "classifier": {"class": "knn"}
                     },
                     "rating_frame": ratings_example,
                     "rankings": {"user_id": "10", "recs_number": 2}}
        with self.assertRaises(TypeError):
            script_run(test_dict)

        # test for not existing config_line in ratings
        test_dict = {"module": "ratings",
                     "test": "test"}
        with self.assertRaises(ValueError):
            script_run(test_dict)

        # test for not existing config_line in item_analyzer
        test_dict = {"module": "item_analyzer",
                     "test": "test"}
        with self.assertRaises(ValueError):
            script_run(test_dict)

        # test for not existing config_line in eval model
        test_dict = {"module": "ranking_alg_eval_model",
                     "recsys_config_module": "recsys_config",
                     "users_directory": users_example_1,
                     "items_directory": items_example_1,
                     "ranking_algorithm": {
                         "class": "classifier", "item_field": {"Plot": [0]}, "classifier": {"class": "knn"}
                     },
                     "rating_frame": ratings_example,
                     "test": "test"}
        with self.assertRaises(ValueError):
            script_run(test_dict)

        # test for not existing config_line in recsys
        test_dict = {"module": "recsys",
                     "test": "test",
                     "users_directory": users_example_1,
                     "items_directory": items_example_1,
                     "ranking_algorithm": {
                         "class": "classifier", "item_field": {"Plot": [0]}, "classifier": {"class": "knn"}
                     },
                     "rating_frame": ratings_example}
        with self.assertRaises(ValueError):
            script_run(test_dict)

        # test eval model with no defined recsys_config_module
        test_dict = {"module": "ranking_alg_eval_model",
                     "users_directory": users_example_1,
                     "items_directory": items_example_1,
                     "ranking_algorithm": {
                         "class": "classifier", "item_field": {"Plot": ["0"]}, "classifier": {"class": "knn"}
                     },
                     "rating_frame": ratings_example}
        with self.assertRaises(KeyError):
            script_run(test_dict)

