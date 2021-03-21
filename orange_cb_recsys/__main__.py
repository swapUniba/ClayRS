from typing import List, Dict
import orange_cb_recsys.utils.runnable_instances as r_i

import json
import sys
import yaml

from orange_cb_recsys.content_analyzer.config import ContentAnalyzerConfig, FieldConfig, \
    FieldRepresentationPipeline
from orange_cb_recsys.content_analyzer.content_analyzer_main import ContentAnalyzer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import \
    RatingsImporter, RatingsFieldConfig


DEFAULT_CONFIG_PATH = "web_GUI/app/configuration_files/config.json"  # "content_analyzer/config_prova2.json"

implemented_preprocessing = r_i.get_cat('preprocessor')

implemented_content_prod = r_i.get_cat('content_production')

implemented_rating_proc = r_i.get_cat('rating_processor')

runnable_instances = r_i.get()


def check_for_available(content_config: Dict):
    # check if need_interface is respected
    # check runnable_instances
    if content_config['source_type'] not in ['json', 'csv', 'sql', 'dat']:
        return False
    if content_config['content_type'].lower() == 'rating' or content_config['content_type'].lower() == 'ratings':
        if "from_field_name" not in content_config.keys() \
                or "to_field_name" not in content_config.keys() \
                or "timestamp_field_name" not in content_config.keys() \
                or "output_directory" not in content_config.keys():
            return False
        for field in content_config['fields']:
            if field['processor']['class'] not in implemented_rating_proc:
                return False
        return True
    for field_dict in content_config['fields']:
        if field_dict['memory_interface'] not in ['index', 'None']:
            return False
        for pipeline_dict in field_dict['pipeline_list']:
            if pipeline_dict['field_content_production'] != "None":
                if pipeline_dict['field_content_production']['class'] \
                        not in implemented_content_prod:
                    return False
            for preprocessing in pipeline_dict['preprocessing_list']:
                if preprocessing['class'] not in implemented_preprocessing:
                    return False
    return True


def dict_detector(technique_dict):
    """
    detect a a class constructor call in a sub-dict of a dict
    """
    for key in technique_dict.keys():
        value = technique_dict[key]
        if isinstance(value, dict) and 'class' in value.keys():
            parameter_class_name = value.pop('class')
            technique_dict[key] = runnable_instances[parameter_class_name](**value)

    return technique_dict


def content_config_run(config_list: List[Dict]):
    for content_config in config_list:
        # content production
        search_index = False
        if 'search_index' in content_config.keys():
            search_index = content_config['search_index']

        content_analyzer_config = ContentAnalyzerConfig(
            content_config["content_type"],
            runnable_instances[content_config['source_type']]
            (file_path=content_config["raw_source_path"]),
            content_config['id_field_name'],
            content_config['output_directory'],
            search_index)

        if 'get_lod_properties' in content_config.keys():
            for ex_retrieval in content_config['get_lod_properties']:
                class_name = ex_retrieval.pop('class')
                args = dict_detector(ex_retrieval)
                content_analyzer_config.append_exogenous_properties_retrieval(runnable_instances[class_name](**args))

        for field_dict in content_config['fields']:
            try:
                field_config = FieldConfig(field_dict['lang'])
            except KeyError:
                field_config = FieldConfig()

            # setting the content analyzer config

            for pipeline_dict in field_dict['pipeline_list']:
                preprocessing_list = list()
                for preprocessing in pipeline_dict['preprocessing_list']:
                    # each preprocessing settings
                    class_name = preprocessing.pop('class')  # extract the class acronyms
                    preprocessing = dict_detector(preprocessing)
                    preprocessing_list.append(
                        runnable_instances[class_name](**preprocessing))  # params for the class
                # content production settings
                if isinstance(pipeline_dict['field_content_production'], dict):
                    class_name = \
                        pipeline_dict['field_content_production'].pop('class')
                    # append each field representation pipeline to the field config
                    technique_dict = pipeline_dict["field_content_production"]
                    technique_dict = dict_detector(technique_dict)
                    field_config.append_pipeline(
                        FieldRepresentationPipeline(
                            runnable_instances[class_name]
                            (**technique_dict), preprocessing_list))
                else:
                    field_config.append_pipeline(
                        FieldRepresentationPipeline(None, preprocessing_list))
            # verify that the memory interface is set
            if field_dict['memory_interface'] != "None":
                field_config.memory_interface = runnable_instances[
                    field_dict['memory_interface']](field_dict['memory_interface_path'])

            content_analyzer_config.append_field_config(field_dict["field_name"], field_config)

        # fitting the data for each
        content_analyzer = \
            ContentAnalyzer(content_analyzer_config)  # need the id list (id configuration)
        content_analyzer.fit()


def rating_config_run(config_dict: Dict):
    rating_configs = []
    for field in config_dict["fields"]:
        class_name = field['processor'].pop('class')
        class_dict = dict_detector(field["processor"])
        rating_configs.append(
            RatingsFieldConfig(field_name=field["field_name"],
                               processor=runnable_instances[class_name](**class_dict))
        )
        args = {}
        if config_dict["source_type"] == 'sql':
            pass
    RatingsImporter(
        source=runnable_instances[
            config_dict["source_type"]](file_path=config_dict["raw_source_path"], **args),
        output_directory=config_dict["output_directory"],
        rating_configs=rating_configs,
        from_field_name=config_dict["from_field_name"],
        to_field_name=config_dict["to_field_name"],
        timestamp_field_name=config_dict["timestamp_field_name"]
    ).import_ratings()


if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = DEFAULT_CONFIG_PATH
    if config_path.endswith('.yml'):
        config_list_dict = yaml.load(open(config_path), Loader=yaml.FullLoader)
    elif config_path.endswith('.json'):
        config_list_dict = json.load(open(config_path))
    else:
        raise Exception("Wrong file extension")

    for config_dict in config_list_dict:
        if check_for_available(config_dict):
            if config_dict["content_type"].lower() == "rating":
                rating_config_run(config_dict)
            else:
                content_config_run([config_dict])
        else:
            raise Exception("Check for available instances failed.")
