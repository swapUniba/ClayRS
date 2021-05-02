import os
from unittest import TestCase

from orange_cb_recsys.__main__ import script_run

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
movies_info_reduced = os.path.join(THIS_DIR, "../../datasets/movies_info_reduced.json")
user_info = os.path.join(THIS_DIR, "../../datasets/users_info.json")
new_ratings = os.path.join(THIS_DIR, "../../datasets/examples/new_ratings.csv")
test_ratings_file = os.path.join(THIS_DIR, "../../datasets/test_ratings_file_script")
users_example_1 = os.path.join(THIS_DIR, "../../contents/examples/ex_1/users_1600355755.1935306")
items_example_1 = os.path.join(THIS_DIR, "../../contents/examples/ex_1/movies_1600355972.49884")
items_example_2 = os.path.join(THIS_DIR, "../../contents/movies_multiple_repr")
ratings_example = os.path.join(THIS_DIR, "../../datasets/test_ratings/test_ratings_1618757236.csv")


class TestRun(TestCase):

    def setUp(self) -> None:

        item_config_dict = {
            "module": "content_analyzer",
            "content_type": "ITEM",
            "source": {"class": "json", "file_path": movies_info_reduced},
            "id": "imdbID",
            "output_directory": "movielens_test_script",
            "field_dict": {"Plot": [{"class": "field_config", "content_technique": {"class": "sk_learn_tf-idf"},
                                     "preprocessing": {"class": "nltk"}, "id": "test"},
                                    {"class": "field_config", "content_technique": {"class": "sk_learn_tf-idf"},
                                     "preprocessing": {"class": "nltk"}}]}
        }

        user_config_dict = {
            "module": "content_analyzer",
            "content_type": "user",
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
            "module": "recsys",
            "users_directory": users_example_1,
            "items_directory": items_example_2,
            "rating_frame": ratings_example,
            "ranking_algorithm": {
                "class": "classifier", "item_field": {"Plot": ["0"]}, "classifier": {"class": "knn"}},
            "rankings": {"user_id": "10", "recs_number": 2}
        }

        eval_config_dict = {
            "module": "eval",
            "eval_type": "ranking_alg_eval_model",
            "users_directory": users_example_1,
            "items_directory": items_example_2,
            "ranking_algorithm": {
                "class": "classifier", "item_field": {"Plot": ["0"]}, "classifier": {"class": "knn"}},
            "rating_frame": ratings_example,
            "partitioning": {"class": "k_fold", "n_splits": 2},
            "metric_list": [{"class": "fnmeasure", "n": 2}, {"class": "precision"}]
        }

        self.config_list = [item_config_dict, user_config_dict, rating_config_dict, recsys_config_dict, eval_config_dict]

    def test_run(self):
        self.assertEqual(len(script_run(self.config_list)), 3)

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

        # test for not existing config_line in content_analyzer
        test_dict = {"module": "content_analyzer",
                     "test": "test"}
        with self.assertRaises(ValueError):
            script_run(test_dict)

        # test for not existing config_line in eval model
        test_dict = {"module": "eval",
                     "eval_type": "ranking_alg_eval_model",
                     "users_directory": users_example_1,
                     "items_directory": items_example_1,
                     "ranking_algorithm": {
                         "class": "classifier", "item_field": {"Plot": ["0"]}, "classifier": {"class": "knn"}
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
                         "class": "classifier", "item_field": {"Plot": ["0"]}, "classifier": {"class": "knn"}
                     },
                     "rating_frame": ratings_example}
        with self.assertRaises(ValueError):
            script_run(test_dict)

        # test eval model with no defined eval type
        test_dict = {"module": "eval",
                     "users_directory": users_example_1,
                     "items_directory": items_example_1,
                     "ranking_algorithm": {
                         "class": "classifier", "item_field": {"Plot": ["0"]}, "classifier": {"class": "knn"}
                     },
                     "rating_frame": ratings_example}
        with self.assertRaises(KeyError):
            script_run(test_dict)
