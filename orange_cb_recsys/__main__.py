from typing import Dict, Union
import orange_cb_recsys.utils.runnable_instances as r_i

import json
import sys
import yaml
import pandas as pd
from inspect import signature, isclass

from orange_cb_recsys.content_analyzer.config import ContentAnalyzerConfig
from orange_cb_recsys.content_analyzer.content_analyzer_main import ContentAnalyzer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsImporter
from orange_cb_recsys.recsys import RecSysConfig, RecSys

DEFAULT_CONFIG_PATH = "web_GUI/app/configuration_files/config.json"

"""
All the available implementations are extracted
"""

runnable_instances = r_i.get()


def __dict_detector(technique: Union[dict, list]):
    """
    Detects a class constructor (defined by a dictionary with a class parameter that stores the alias for the class)
    and replaces it with the object instance.
    {"class": 'test_class'} will be transformed into TestClass()
    where TestClass is the class associated to the alias 'test_class' in the runnable_instances file

    This method is also useful in case there are class constructors inside another class constructor
    {"class": 'test_class', "parameter": {"class": 'test_class'}} will be transformed into
    TestClass(parameter=TestClass())

    If a list of objects is specified such as: [{"class": 'class1'}, {"class": 'class2'}]
    this method will call itself recursively so that each dictionary will be transformed into the corresponding object

    If the technique is a standard dictionary (and not one representing an object instance),
    for example: {"parameter": "value", "parameter2": "value2"}, the values will be checked in order to transform
    any possible object representation
    So for example, {"parameter": {"class": 'test_class'}} will be transformed into {"parameter": TestClass()}

    If the technique doesn't match any of these cases, no operation is done and it is returned as it is

    Args:
        technique (Union[list,dict]): dictionary or list to check in order to transform any class constructors into
            actual objects
    Returns:
        technique: processed and transformed dictionary or list of dictionaries or object instance
    """
    if isinstance(technique, list):
        techniques = []
        # each element in the list will be processed so that any dictionary in the list (or in lists inside of the
        # first list) will be processed and transformed if it represents an object instance
        for element in technique:
            techniques.append(__dict_detector(element))
        return techniques
    elif isinstance(technique, dict):
        # if a parameter class is defined it means that the dictionary represents an object instance
        if 'class' in technique.keys():
            parameter_class_name = technique.pop('class')
            try:
                # checks if any value is a dictionary representing an object
                for parameter in technique.keys():
                    technique[parameter] = __dict_detector(technique[parameter])
                return runnable_instances[parameter_class_name.lower()](**technique)
            except TypeError:
                passed_parameters = list(technique.keys())
                actual_parameters = list(signature(
                    runnable_instances[parameter_class_name.lower()].__init__).parameters.keys())
                actual_parameters.remove("self")
                raise TypeError("The following parameters: " + str(passed_parameters) + "\n" +
                                "Don't match the class constructor parameters: " + str(actual_parameters))
        # otherwise it's just a standard dictionary and every value is checked (in case one of them is a dictionary
        # representing an object)
        else:
            for parameter in technique.keys():
                technique[parameter] = __dict_detector(technique[parameter])
            return technique
    else:
        return technique


def __extract_parameters(config_dict: Dict, class_instance) -> dict:
    """
    Used to extract the parameters for the main modules of the framework or their most important methods.
    In order to instantiate a content_analyzer (for example), the content_analyzer_config has to be created first.
    To do so, this method checks every line of the dictionary containing the parameters for the content_analyzer_config
    and runs different operations according to the value for the parameter.
    If the value is a dictionary or a list it is passed to the dict_detector, in any other cases the value is simply
    kept as it is.
    If during the checking operation it finds any parameter that doesn't match the class or method signature, it
    raises a ValueError exception.

    EXAMPLE:
        {"parameter_list_value": ["value1", "value2"], "parameter_dict_value": {"class": "class_alias_value"},
        "parameter_str_value": "VALUE_str", "parameter_int_value": int_value}

        for parameter_list_value and parameter_dict_value the framework will pass their values to the
        dict_detector method. For explanation on how it works check its relative documentation.
        To sum it up, it replaces the dictionaries representing an object with the actual object instance.
        So for example, in this case, parameter_list_value wouldn't be subject to any modification, instead
        the dictionary in parameter_dict_value will be transformed into the object matching the
        alias name (this is so because there is a class parameter).

        for parameter_str_value and parameter_int_value no modifications are done.

        the final output will be:

        {"parameter_list_value": ["value1", "value2"], "parameter_dict_value": Class_from_alias(),
        "parameter_str_value": "VALUE_str", "parameter_int_value": int_value

    Args:
        config_dict (dict): dictionary representing the parameters for a method or class constructor
        class_instance: the class or method to refer to when retrieving the parameters

    Returns:
        parameters (dict): dictionary with the modified parameters
    """
    try:
        # if the method receives a class it will extract the parameters from the constructor, otherwise
        # it will just extract the parameters directly (this happens, for example, if the passed argument is a method)
        if isclass(class_instance):
            signature_parameters = list(signature(class_instance.__init__).parameters.keys())
        else:
            signature_parameters = list(signature(class_instance).parameters.keys())
        if "self" in signature_parameters:
            signature_parameters.remove("self")

        # checks if the signature of the method or class is able to take any number of parameters
        # the next operation done by the method will use this information and won't check for
        # matches between the parameters passed by the user (in the dictionary) and the signature parameters
        any_parameters = False
        if "kwargs" in signature_parameters or "args" in signature_parameters:
            any_parameters = True

        # the method checks every key of the dictionary in order to modify the value (in case it is a dictionary
        # representing an object or a string to be lowered) and makes sure that all the keys are in the
        # previously extracted parameters
        parameters = dict()
        for config_line in config_dict.keys():
            if not any_parameters and config_line not in signature_parameters:
                raise ValueError("%s is not a parameter for %s"
                                 "\nThe actual parameters are: " %
                                 (config_line, class_instance) + str(signature_parameters))
            else:
                if isinstance(config_dict[config_line], dict) or isinstance(config_dict[config_line], list):
                    parameters[config_line] = __dict_detector(config_dict[config_line])
                else:
                    parameters[config_line] = config_dict[config_line]

        return parameters
    except (KeyError, ValueError, TypeError) as e:
        raise e


def __content_config_run(content_config: Dict):
    """
    Method that extracts the parameters for the creation and fitting of a Content Analyzer

    Args:
        content_config (dict): dictionary that represents a config defined in the config file, in this case it
            represents the config of the Content Analyzer
    """
    try:
        content_analyzer_config = ContentAnalyzerConfig(**__extract_parameters(content_config, ContentAnalyzerConfig))
        content_analyzer = ContentAnalyzer(content_analyzer_config)
        content_analyzer.fit()
    except (KeyError, ValueError, TypeError, FileNotFoundError) as e:
        raise e


def __rating_config_run(config_dict: Dict):
    """
    Method that extracts the parameters for the creation of a Ratings Importer and saves the DataFrame produced by
    the method import_ratings()

    Args:
        config_dict (dict): dictionary that represents a config defined in the config file, in this case it
            represents the config of the Ratings Importer
    """
    try:
        ratings_parameters = __extract_parameters(config_dict, RatingsImporter)
        RatingsImporter(**ratings_parameters).import_ratings()
    except (KeyError, ValueError, TypeError, FileNotFoundError) as e:
        raise e


def __recsys_config_run(config_dict: Dict) -> RecSysConfig:
    """
    Method that extracts the parameters for the creation of a Recsys Config.

    Since the recsys config is used by different modules (like recsys or eval model) it was made more dynamic
    so that it can be easily re-used.
    When the config_dict is passed, it should contain all the available parameters for the main module.
    So, for example, if the recsys_config was being run for a recsys instance, the config_dict parameter should
    contain not only the parameters for the recsys_config but also the parameters for the recsys.
    The parameters in the config_dict for the recsys_config are removed and stored in a local dictionary.
    By doing so, the recsys_config can be instantiated without problems and it is then returned.

    Args:
        config_dict (dict): dictionary that represents a config defined in the config file, in this case it represents
            the config of the recommender system and it can also contain other parameters from the main module
            who needed the recsys config
    Returns:
        recsys_config (RecSysConfig): instance of the RecSysConfig instantiated from the extracted parameters
    """
    try:
        recsys_config_parameters = list(signature(RecSysConfig.__init__).parameters.keys())
        recsys_config_parameters.remove("self")

        # extracts the parameters for the Recsys config from the dictionary passed as an argument
        # and removes said parameters from said dictionary
        recsys_config_dict = dict()
        for config_line in config_dict.keys():
            if config_line in recsys_config_parameters:
                recsys_config_dict[config_line] = config_dict[config_line]
        for recsys_config_parameter in recsys_config_dict.keys():
            config_dict.pop(recsys_config_parameter)

        recsys_parameters = __extract_parameters(recsys_config_dict, RecSysConfig)
        recsys_config = RecSysConfig(**recsys_parameters)
        return recsys_config
    except (KeyError, ValueError, TypeError, FileNotFoundError) as e:
        raise e


def __recsys_run(config_dict: Dict) -> list:
    """
    Method that extracts the parameters for the creation of a Recsys. Also it allows
    to define what kind of ranking or predictions the user wants from the recommender system. In order to do so,
    two additional parameters (not in the RecSys class constructor) are considered, the first being the
    'predictions' parameter containing a list of parameters for the prediction algorithm, the second is
    'rankings' which works as the first one but for the ranking algorithm.

    Args:
        config_dict (dict): dictionary that represents a config defined in the config file, in this case it represents
            the config of the predictions or rankings the user wants to run

    Returns:
        recsys_results (list): list containing two lists where the first one contains the results for the prediction
        algorithm and the second one contains the results for the ranking algorithm
    """
    try:
        recsys_config = __recsys_config_run(config_dict)
        config_dict["config"] = recsys_config

        # prediction and ranking parameters are extracted and the 'predictions' and 'rankings' keys are removed
        # from the dictionary (since they aren't parameters for the RecSys constructor
        prediction_parameters = []
        if 'predictions' in config_dict.keys():
            if not isinstance(config_dict['predictions'], list):
                config_dict['predictions'] = [config_dict['predictions']]
            for prediction in config_dict['predictions']:
                prediction_parameters.append(prediction)
            config_dict.pop('predictions')

        ranking_parameters = []
        if 'rankings' in config_dict.keys():
            if not isinstance(config_dict['rankings'], list):
                config_dict['rankings'] = [config_dict['rankings']]
            for ranking in config_dict['rankings']:
                ranking_parameters.append(ranking)
            config_dict.pop('rankings')

        recsys = RecSys(**__extract_parameters(config_dict, RecSys))

        # predictions and rankings are computed and stored in two separate lists
        prediction_results = []
        for prediction in prediction_parameters:
            prediction_results.append(recsys.fit_predict(**__extract_parameters(prediction, recsys.fit_predict)))

        ranking_results = []
        for ranking in ranking_parameters:
            ranking_results.append(recsys.fit_ranking(**__extract_parameters(ranking, recsys.fit_ranking)))

        recsys_results = list()
        recsys_results.append(prediction_results)
        recsys_results.append(ranking_results)
        return recsys_results

    except (KeyError, ValueError, TypeError, FileNotFoundError) as e:
        raise e


def __eval_config_run(config_dict: Dict) -> pd.DataFrame():
    """
    Method that extracts the parameters for the creation of a Eval Model. In order to define what kind of eval_model
    has to be run and the parameters for said model (ranking_alg, prediction_alg, report, ...), an additional parameter
    is added to the config_dict, that being 'eval_type'. This allows the user to define what kind of eval model has to
    be used (available in the runnable_instances file)

    Args:
        config_dict (dict): dictionary that represents a config defined in the config file, in this case it represents
        the config for the Eval Model
    Returns:
        eval_results (pd.DataFrame): dataframe containing the results for the eval process for the metrics defined in
        the metric_list
    """
    try:
        recsys_config = __recsys_config_run(config_dict)
        config_dict["config"] = recsys_config

        if "eval_type" not in config_dict.keys():
            raise KeyError("Eval model class type must be defined in order to use the module")
        eval_class = config_dict.pop("eval_type").lower()
        eval_class = runnable_instances[eval_class]

        eval_parameters = __extract_parameters(config_dict, eval_class)
        eval_model = eval_class(**eval_parameters)
        return eval_model.fit()

    except (KeyError, ValueError, TypeError, FileNotFoundError) as e:
        raise e


implemented_modules = {
    "content_analyzer": __content_config_run,
    "ratings": __rating_config_run,
    "recsys": __recsys_run,
    "eval": __eval_config_run
}


def script_run(config_list_dict: Union[dict, list]):
    """
    Method that controls the entire process of script running. It checks that the contents loaded from a script match
    the required prerequisites. The data must contain dictionaries and each dictionary must have a "module" key
    that is used to understand what kind of operation it is supposed to perform (if it's meant for a Recommender System,
    a Content Analyzer, ...). If any of these prerequisites aren't matched a ValueError or KeyError exception is thrown.

    If any of the functions (made for the modules) returns anything (could be the results of the predict method in the
    Recsys), the return value is stored in a list which is then returned at the end of the script processing

    Args:
        config_list_dict: single dictionary or list of dictionaries extracted from the config file

    Returns:
        script_results (list): list containing the results returned by some of the functions (for example, it contains
        the results of the RecSys or of the EvalModel)
    """
    try:

        if not isinstance(config_list_dict, list):
            config_list_dict = [config_list_dict]

        if not all(isinstance(config_dict, dict) for config_dict in config_list_dict):
            raise ValueError("The list in the script must contain dictionaries only")

        script_results = []
        for config_dict in config_list_dict:

            if "module" in config_dict.keys():
                if config_dict["module"] in implemented_modules:
                    module = config_dict.pop("module")
                    returned_results = implemented_modules[module](config_dict)
                    if returned_results is not None:
                        if isinstance(returned_results, list):
                            for returned_result in returned_results:
                                script_results.append(returned_result)
                        else:
                            script_results.append(returned_results)
                else:
                    raise ValueError("You must specify a valid module: " + str(implemented_modules.keys()))
            else:
                raise KeyError("A 'module' parameter must be specified and the value must be one of the following: " +
                               str(implemented_modules.keys()))

        return script_results

    except (KeyError, ValueError, TypeError, FileNotFoundError) as e:
        raise e


if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = DEFAULT_CONFIG_PATH

    if config_path.endswith('.yml'):
        extracted_data = yaml.load(open(config_path), Loader=yaml.FullLoader)
    elif config_path.endswith('.json'):
        extracted_data = json.load(open(config_path))
    else:
        raise ValueError("Wrong file extension")

    results = script_run(extracted_data)
