import json
from unittest import TestCase

from orange_cb_recsys.__main__ import content_config_run, check_for_available, rating_config_run

content_config_dict = '[{"content_type": "ITEM", ' \
                      '"output_directory": "movielens_test", ' \
                      '"raw_source_path": "datasets/movies_info_reduced.json", ' \
                      '"source_type": "json", ' \
                      '"id_field_name": ["imdbID"], ' \
                      '"start_from": "0", ' \
                      '"end_up_at": "all", ' \
                      '"fields": [{' \
                      '"field_name": "Title", ' \
                      '"memory_interface": "None", ' \
                      '"memory_interface_path": "./test-index-plot",' \
                      '"pipeline_list": [' \
                      '{"field_content_production": {"class": "babelpy", ' \
                      '"api_key": "bd7835be-12f7-4717-8c5f-429e3e968998"}, "preprocessing_list": []}]}]}]'

rating_config_dict = '[{"content_type": "ratings", ' \
                     '"source_type": "csv", ' \
                     '"from": "user_id", ' \
                     '"to": "user_id", ' \
                     '"output_directory": "datasets/test_ratings",' \
                     '"timestamp": "timestamp", ' \
                     '"raw_source_path": "datasets/test_import_ratings.json",' \
                     '"fields": [{"preference_field_name": "text", ' \
                     '"rating_processor": {"class": "number_normalizer"}}]}]'

content_config_dict_ = {
    "content_type": "ITEM",
    "output_directory": "movielens_test",
    "raw_source_path": "datasets/movies_info_reduced.json",
    "source_type": "json",
    "id_field_name": ["imdbID"],
    "get_lod_properties": [{
        "class": "dbpedia_mapping",
        "mode": 'only_retrieved_evaluated',
        "entity_type": 'Film',
        "lang": 'EN',
        "label_field": 'Title'
    }],
    "fields": [{
        "field_name": "Title",
        "memory_interface": "None",
        "memory_interface_path": "./test-index-plot",
        "pipeline_list": [
            {"field_content_production": {"class": "babelpy", "api_key": "bd7835be-12f7-4717-8c5f-429e3e968998"},
            "preprocessing_list": []}
        ]
    }]
}

rating_config_dict_ = {
    "content_type": "ratings",
    "source_type": "json",
    "from_field_name": "user_id",
    "to_field_name": "item_id",
    "timestamp_field_name": "timestamp",
    "raw_source_path": "datasets/test_import_ratings.json",
    "output_directory": "datasets/test_ratings",
    "fields": [
        {"field_name": "stars", "processor": {"class": "number_normalizer", "max_": 5.0, "min_": 0.0}},
    ],
}

rating_config_dict_2 = {
    "content_type": "ratings",
    "source_type": "json",
    "from_field_name": "user_id",
    "to_field_name": "item_id",
    "timestamp_field_name": "timestamp",
    "raw_source_path": "datasets/test_import_ratings.json",
    "output_directory": "datasets/test_ratings",
    "fields": [
        {"field_name": "stars", "processor": {"class": "number_normalizer", "max_": 5.0, "min_": 0.0}},
        {"field_name": "text", "processor": {"class": "text_blob_sentiment"}},
    ],
}


class TestRun(TestCase):
    def test_config(self):
        # test only if the key in the config.json are valid
        try:
            with open("test/content_analyzer/config.json") as file:
                config_list = json.load(file)
        except FileNotFoundError:
            with open("config.json") as file:
                config_list = json.load(file)

        msg: str = "You have to put the {} in the {}"
        self.assertEqual(type(config_list), type(list()), "the config must contain a list of dict")
        for content_config in config_list:
            self.assertEqual(type(content_config), type(dict()), "the config must contain a list of dict")
            for key in ["content_type", "raw_source_path", "source_type", "id_field_name", "fields"]:
                self.assertIn(key, content_config.keys(), msg.format(key, "content config"))
            for field in content_config['fields']:
                if field is not None:
                    for key in ["field_name", "memory_interface", "pipeline_list"]:
                        self.assertIn(key, field.keys(), msg.format(key, "field config"))
                    for pipeline in field["pipeline_list"]:
                        if pipeline is not None:
                            for key in ["field_content_production", "preprocessing_list"]:
                                self.assertIn(key, pipeline.keys(), msg.format(key, "pipeline config"))
                            self.assertIn("class", pipeline["field_content_production"].keys(),
                                          msg.format(key, "field_content_production config"))
                            for preprocessing in pipeline["preprocessing_list"]:
                                self.assertIn("class", preprocessing.keys(), msg.format(key, "preprocessing config"))

    def test_run(self):
        # self.skipTest("test in the submodules.")
        global content_config_dict_, rating_config_dict_, rating_config_dict_2
        try:
            content_config_run([content_config_dict_])
            rating_config_run(rating_config_dict_)
            rating_config_run(rating_config_dict_2)
        except FileNotFoundError:
            self.skipTest("LOCAL MACHINE")

    def test_check_for_available(self):
        in_dict = {"content_type": "item", "source_type": "text"}
        self.assertFalse(check_for_available(in_dict))
        in_dict = {"content_type": "item", "source_type": "json", "fields": [{"memory_interface": "not-index"}]}
        self.assertFalse(check_for_available(in_dict))
        in_dict = {"content_type": "item", "source_type": "json", "fields": [
            {"memory_interface": "index", "pipeline_list": [{"field_content_production": {"class": "no-class"}}]}]}
        self.assertFalse(check_for_available(in_dict))
        in_dict = {"content_type": "item", "source_type": "json", "fields": [
            {"memory_interface": "index", "pipeline_list": [{"field_content_production": {"class": "babelpy"},
                                                             "preprocessing_list": [{"class": "no-class"}]}]}]}
        self.assertFalse(check_for_available(in_dict))
        in_dict = {"content_type": "item", "source_type": "json", "fields": [
            {"memory_interface": "index", "pipeline_list": [{"field_content_production": {"class": "babelpy"},
                                                             "preprocessing_list": [{"class": "nltk"}]}]}]}
        self.assertTrue(check_for_available(in_dict))
        in_dict = {"content_type": "ratings", "source_type": "csv"}
        self.assertFalse(check_for_available(in_dict))
        in_dict = {"content_type": "ratings", "source_type": "csv",
                   "fields": [{"preference_field_name": "_", "processor": {"class": "text_blob"}}]}
        self.assertFalse(check_for_available(in_dict))
        in_dict = {"content_type": "ratings", "source_type": "csv", "from_field_name": "_", "to_field_name": "_", "output_directory": "_",
                   "timestamp_field_name": "_",
                   "fields": [{"preference_field_name": "_", "processor": {"class": "text_blob_sentiment"}}]}
        self.assertTrue(check_for_available(in_dict))
        in_dict = {"content_type": "ratings", "source_type": "csv", "from_field_name": "_", "to_field_name": "_", "output_directory": "_",
                   "timestamp_field_name": "_",
                   "fields": [{"preference_field_name": "_", "processor": {"class": "boh"}}]}
        self.assertFalse(check_for_available(in_dict))
