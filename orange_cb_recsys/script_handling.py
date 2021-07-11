import os
import pandas as pd
from typing import Dict, Union, Type, Callable
from abc import ABC, abstractmethod
from inspect import signature, isclass, isfunction, getmembers, Parameter
import orange_cb_recsys.utils.runnable_instances as r_i

from orange_cb_recsys.content_analyzer.content_analyzer_main import ContentAnalyzer
from orange_cb_recsys.content_analyzer.embeddings.embedding_learner.embedding_learner import EmbeddingLearner
from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.evaluation.eval_model import EvalModel
from orange_cb_recsys.evaluation.eval_pipeline_modules.methodology import Methodology
from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import PartitionModule
from orange_cb_recsys.evaluation.eval_pipeline_modules.metric_evaluator import MetricCalculator
from orange_cb_recsys.exceptions import ScriptConfigurationError, NoOutputDirectoryDefined, ParametersError
from orange_cb_recsys.recsys.recsys import RecSys
from orange_cb_recsys.utils.class_utils import get_all_implemented_classes, get_all_implemented_subclasses

"""
All the available implementations are extracted
"""
runnable_instances = r_i.get_classes()


class Run(ABC):
    """
    Abstract base class that enables all classes in the framework that can be instantiated like a stand alone module to
    configure the behavior of the dictionary related to it in the script file

    This class is mainly used by the script_run method to recognise the run configuration related to a main module of
    the framework (for example, the Run class related to RecSys is RecSysRun)
    """

    @classmethod
    @abstractmethod
    def get_associated_class(cls) -> Type:
        """
        Method that defines which class is associated to the Run class (for example, for the class that defines what
        to do for the RecSys, the RecSys class will be returned by this method)

        Returns:
            Framework's class associated to the Run subclass
        """
        raise NotImplementedError

    @classmethod
    def run(cls, config_dict: Dict, module: str):
        """
        Main method of the class that defines what happens as a result of using the run configuration

        In particular, there is a pipeline of operations:
            - The module is instantiated by retrieving the module's class
                (using the file serialized in the runnable_instances.py file)

            - The methods that are defined in the configuration dictionary are extracted from it

                EXAMPLE:

                    {"constructor_parameter": value,
                     "method_name": {"method_parameter": value}
                     }

                method_name will be extracted with the associated value (which is a dictionary containing the parameters
                associated to the method or a list of said dictionary) and saved in a new dictionary containing all
                these methods ( {"method_name": {"method_parameter": value} )

            - The parameters for the class constructor defined in the configuration dictionary are extracted from it

            - An object of the class associated with the module is instantiated and the parameters extracted from the
                configuration dictionary as argument are passed to the constructor

            - The methods extracted are executed

        Args:
            config_dict (dict): dictionary extracted from the script file which contains the parameters for the
                constructor (and methods eventually) for the framework's class associated to the Run subclass

            the dictionary will be in the following form:

                {
                    "output_directory": "dir",
                    "object_example": {"class": "object_class", "constructor_parameter": "value"},
                    ...
                }

            module (str): name of the module to use (example: "ContentAnalyzer")
        """
        run_class = runnable_instances[module]
        methods = cls.check_for_methods(config_dict, run_class)
        class_parameters = cls.extract_parameters(config_dict, run_class)
        class_instance = run_class(**class_parameters)
        cls.execute_methods(methods, class_instance)

    @classmethod
    def dict_detector(cls, technique: Union[dict, list]):
        """
        Detects a class constructor (defined by a dictionary with a class parameter that stores the alias for the class)
        and replaces it with the object instance.
        {"class": 'test_class'} will be transformed into TestClass()
        where TestClass is the class associated to the alias 'test_class' in the runnable_instances file

        This method is also useful in case there are class constructors inside another class constructor
        {"class": 'test_class', "parameter": {"class": 'test_class'}} will be transformed into
        TestClass(parameter=TestClass())

        If a list of objects is specified such as: [{"class": 'class1'}, {"class": 'class2'}]
        this method will call itself recursively so that
        each dictionary will be transformed into the corresponding object

        If the technique is a standard dictionary (and not one representing an object instance),
        for example: {"parameter": value, "parameter2": value2}, the values will be checked in order to transform
        any possible object representation
        So for example, {"parameter": {"class": 'test_class'}} will be transformed into {"parameter": TestClass()}

        If the technique doesn't match any of these cases, no operation is done and it is returned as it is

        If the wrong parameters are defined for an object, a ParametersException is thrown

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
                techniques.append(cls.dict_detector(element))
            return techniques
        elif isinstance(technique, dict):
            # if a parameter class is defined it means that the dictionary represents an object instance
            if 'class' in technique.keys():
                parameter_class_name = technique.pop('class')
                class_signature = signature(runnable_instances[parameter_class_name.lower()]).parameters
                try:
                    # checks if any value is a dictionary representing an object
                    for parameter in technique.keys():
                        # checks if the parameter should be a DataFrame, in which case the configuration dictionary will
                        # contain a path to a csv file. The csv file is loaded into a DataFrame
                        if parameter in class_signature.keys():
                            parameter_signature = class_signature[parameter]
                            if (not parameter_signature.annotation == Parameter.empty and parameter_signature.annotation == pd.DataFrame) or \
                                    (parameter_signature.annotation == Parameter.empty and type(parameter_signature.default) == pd.DataFrame):
                                technique[parameter] = pd.read_csv(technique[parameter])

                        # recursively calls dict_detector to check if the value for the parameter is
                        # an object to instantiate
                        technique[parameter] = cls.dict_detector(technique[parameter])
                    return runnable_instances[parameter_class_name.lower()](**technique)
                except TypeError:
                    passed_parameters = list(technique.keys())
                    actual_parameters = list(signature(
                        runnable_instances[parameter_class_name.lower()].__init__).parameters.keys())
                    actual_parameters.remove("self")
                    raise ParametersError("The following parameters: " + str(passed_parameters) + "\n" +
                                          "Don't match the actual parameters: " + str(actual_parameters))
            # otherwise it's just a standard dictionary and every value is checked (in case one of them is a dictionary
            # representing an object)
            else:
                for parameter in technique.keys():
                    technique[parameter] = cls.dict_detector(technique[parameter])
                return technique
        else:
            return technique

    @classmethod
    def extract_parameters(cls, config_dict: Dict, class_or_function: Union[Type, Callable]) -> dict:
        """
        Used to extract the parameters for the main modules of the framework or their most important methods.
        In order to instantiate a content_analyzer (for example), the content_analyzer_config has to be created first.
        To do so, this method checks every line of the dictionary containing the parameters for the
        content_analyzer_config and runs different operations according to the value for the parameter.
        If the value is a dictionary or a list it is passed to the dict_detector, in any other cases the value is simply
        kept as it is.
        If during the checking operation it finds any parameter that doesn't match the class or method signature, it
        raises a ParametersError exception.

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
            "parameter_str_value": "VALUE_str", "parameter_int_value": int_value}

        Args:
            config_dict (dict): dictionary representing the parameters for a method or class constructor
            class_or_function (Type or Callable): the class or function to refer to when retrieving the parameters

        Returns:
            parameters (dict): dictionary with the modified parameters
        """
        try:
            # if the method receives a class it will extract the parameters from the constructor, otherwise
            # it will just extract the parameters directly (this happens if the passed argument is a function)
            if isclass(class_or_function):
                signature_parameters = list(signature(class_or_function.__init__).parameters.keys())
            else:
                signature_parameters = list(signature(class_or_function).parameters.keys())
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
                    raise ParametersError("%s is not a parameter for %s"
                                          "\nThe actual parameters are: " %
                                          (config_line, class_or_function.__name__) + str(signature_parameters))
                else:
                    if isinstance(config_dict[config_line], dict) or isinstance(config_dict[config_line], list):
                        parameters[config_line] = cls.dict_detector(config_dict[config_line])
                    else:
                        if config_line in signature_parameters:
                            # checks if the parameter should be a DataFrame, in which case the configuration dictionary
                            # will contain a path to a csv file. The csv file is loaded into a DataFrame
                            parameter_signature = signature(class_or_function).parameters[config_line]
                            if (not parameter_signature.annotation == Parameter.empty and parameter_signature.annotation == pd.DataFrame) or \
                                    (parameter_signature.annotation == Parameter.empty and type(parameter_signature.default) == pd.DataFrame):
                                config_dict[config_line] = pd.read_csv(config_dict[config_line])

                        parameters[config_line] = config_dict[config_line]

            return parameters
        except ParametersError as e:
            raise e

    @staticmethod
    def check_for_methods(config_dict: dict, cls: Type) -> Dict[str, Union[dict, list]]:
        """
        Method that searches for the public methods implemented in the class passed as argument. If the
        configuration dict passed as argument contains one of them, it is removed from the configuration
        dictionary and added to a dictionary that will keep all the results

        The final dictionary will be in the following form:

            {
                "fit": {dictionary containing the parameters in the config_dict for the fit method},
                method_name: {same dictionary as above but for the method_name method},
                ...
            }

        Note that it is also possible to pass lists of parameters, for example:

            method_name: [dictionary containing the parameters for method_name,
                          dictionary containing the parameters for method_name]

        In that case, the execute_methods method will use the method with every dictionary containing the parameters

        Args:
            config_dict (dict): configuration dictionary for a specific module in the script file
            cls (Type): class from which the methods will be retrieved

        Returns:
            found_methods (dict): dictionary containing the methods to execute, the keys will be the methods' names and
                the values will be the parameters for said methods
        """
        # [method_name, method_name, ...]
        methods = [name[0] for name in getmembers(cls, predicate=isfunction) if not name[0].startswith("_")]
        found_methods = {}
        # removes the method name from the list of methods and from the config_dict and adds the name as key
        # to the dictionary to return and the parameters defined in the config_dict as value
        for parameter in list(config_dict.keys()):
            if parameter in methods:
                found_methods[parameter] = config_dict.pop(parameter)

        return found_methods

    @classmethod
    def execute_methods(cls, methods_dict: Dict[str, Union[dict, list]], obj: object) -> Dict[str, list]:
        """
        Executes the methods passed in the methods_dict parameter related to the object passed as argument.
        The methods dictionary is in the following form:
            {
            "name": {"parameter1": "somevalue", "parameter2": 0, "parameter3": {"class": "someclass"},
            method_name: dictionary containing the parameters for the method,
            other_method_name: [dictionary containing the parameters for the method,
                                dictionary containing the parameters for the method]
            }

        It's also possible to pass a list of parameters instead of a single one (like other_method_name in the example),
        in which case the method will be executed as many times as many parameters are defined.

        Args:
            methods_dict (dict): dictionary containing the method name as key and the parameters as value
            obj (object): object to which the methods refer to

        Returns:
            results (dict): dictionary containing the method name as key and the results obtained from the method
                as values. The results values will be stored in a list, so for example

                    {
                    method_name: [result],
                    other_method_name: [result_1, result_2]
                    }
        """
        results = {}
        # executes all the methods defined in the methods dictionary
        for method in methods_dict.keys():
            func = getattr(obj, method)
            # if only one parameter was passed (so not a list)
            if not isinstance(methods_dict[method], list):
                methods_dict[method] = [methods_dict[method]]
            results[method] = []
            for parameters in methods_dict[method]:
                results[method].append(func(**cls.extract_parameters(parameters, func)))
        return results


class NeedsSerializationRun(Run):
    """
    Abstract class that inherits from the Run class and allows the configuration of modules in the framework that need
    serialization of the results obtained using some of their methods

    To do so, an extra parameter has to be defined in the configuration dictionary associated with the module, that
    parameter is called 'output_directory' and in said directory the results will be serialized
    """

    @classmethod
    @abstractmethod
    def get_associated_class(cls) -> Type:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def serialize_results(cls, executed_methods_results: Dict[str, list], output_directory: str):
        """
        Method that defines how the results are serialized and for which methods of the object

        Args:
            executed_methods_results(dict): dictionary containing the methods names as keys and the related results
                as values (stored in a list that contains one result for each method call)
            output_directory (str): directory where the results will be serialized
        """
        raise NotImplementedError

    @classmethod
    def run(cls, config_dict: Dict, module: str):
        """
        Works as the run method defined in the Run class but also considers the output directory
        """
        run_class = runnable_instances[module]
        output_directory = cls.setup_output_directory(config_dict, module)
        methods = cls.check_for_methods(config_dict, run_class)
        class_parameters = cls.extract_parameters(config_dict, run_class)
        class_instance = run_class(**class_parameters)
        executed_methods_results = cls.execute_methods(methods, class_instance)
        cls.serialize_results(executed_methods_results, output_directory)

    @staticmethod
    def setup_output_directory(config_dict: dict, module: str):
        """
        This method extracts the output_directory from the configuration dictionary and allows for the basic setup of
        said output directory.

        If the directory doesn't exist it is created.

        If the dictionary passed as argument doesn't contain the output_directory parameter, a NoOutputDirectoryDefined
        exception is thrown.

        Args:
            config_dict (dict): dictionary representing the parameters for a class constructor
            module (str): name of the module to use (example: "ContentAnalyzer")

        Returns:
            output_directory(str): output_directory extracted from the config_dict
        """
        try:
            output_directory = config_dict.pop('output_directory')
        except KeyError:
            raise NoOutputDirectoryDefined(
                "Output directory must be defined for %s" % runnable_instances[module].__name__)

        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        return output_directory

    @staticmethod
    def save_to_csv(dataframe: pd.DataFrame, path: str):
        """
        Saves a dataframe passed as argument in the path passed as argument (adds the ".csv" extension if it is not
        present in the path)
        """
        if not path.endswith(".csv"):
            path += ".csv"

        dataframe.to_csv(path)


class ContentAnalyzerRun(Run):
    """
    Run associated with the ContentAnalyzer
    """

    @classmethod
    def get_associated_class(cls) -> Type:
        return ContentAnalyzer


class EmbeddingLearnerRun(Run):
    """
    Run associated with the EmbeddingLearner
    """

    @classmethod
    def get_associated_class(cls):
        return EmbeddingLearner


class RatingsRun(Run):
    """
    Run associated with the RatingsImporter
    """

    @classmethod
    def get_associated_class(cls):
        return RatingsImporter


class RecSysRun(NeedsSerializationRun):
    """
    Run associated with the RecSys

    The output files will be in the following form:

        predict_0_0
        predict_0_1
        ...
        rank_0_0
        rank_0_1

        or

        multiple_predict_0_0
        multiple_predict_0_1
        ...
        multiple_rank_0_0
        multiple_rank_0_1

    where the first number identifies the RecSysRun (if multiple RecSys with the same output directory are defined in
    the script file, this allows identification of the files created by each RecSys) and the second number identifies
    the parameters used (it's possible to call fit_rank and fit_predict multiple times with the same RecSys, the
    number identifies the fit_rank and fit_predict run)
    """

    recsys_number = 0

    @classmethod
    def get_associated_class(cls):
        return RecSys

    @classmethod
    def serialize_results(cls, executed_methods_results: Dict[str, list], output_directory: str):
        try:
            if 'fit_rank' in executed_methods_results.keys():
                for i, ranking in enumerate(executed_methods_results['fit_rank']):
                    cls.save_to_csv(
                        ranking,
                        os.path.join(output_directory, "rank_{}_{}".format(str(cls.recsys_number), str(i))))

            if 'fit_predict' in executed_methods_results.keys():
                for i, prediction in enumerate(executed_methods_results['fit_predict']):
                    cls.save_to_csv(
                        prediction,
                        os.path.join(output_directory, "predict_{}_{}".format(str(cls.recsys_number), str(i))))

            if 'multiple_fit_rank' in executed_methods_results.keys():
                for i, ranking in enumerate(executed_methods_results['multiple_fit_rank']):
                    cls.save_to_csv(
                        ranking,
                        os.path.join(output_directory, "multiple_rank_{}_{}".format(str(cls.recsys_number), str(i))))

            if 'multiple_fit_predict' in executed_methods_results.keys():
                for i, prediction in enumerate(executed_methods_results['multiple_fit_predict']):
                    cls.save_to_csv(
                        prediction,
                        os.path.join(output_directory, "multiple_predict_{}_{}".format(str(cls.recsys_number), str(i))))
        finally:
            cls.recsys_number += 1


class EvalRun(NeedsSerializationRun):
    """
    Run associated with the EvalModel

    The output will be serialized in the following form:

        eval_sys_results_0_0
        eval_user_results_0_0

    where the first number is used to identify the EvalModel in the script file (in case multiple EvalModel objects
    with the same output directory are defined in the script file, this identifies each output) and the second number
    identifies the method (since it is possible to call the same method multiple times) (note that this shouldn't
    really be the case for the eval model since the fit method doesn't have any parameters,
    but it was made for consistency)
    """

    eval_number = 0

    @classmethod
    def get_associated_class(cls):
        return EvalModel

    @classmethod
    def serialize_results(cls, executed_methods_results: Dict[str, list], output_directory: str):
        try:
            if 'fit' in executed_methods_results.keys():
                for i, results in enumerate(executed_methods_results['fit']):
                    cls.save_to_csv(
                        results[0],
                        os.path.join(output_directory, "eval_sys_results_{}_{}".format(str(cls.eval_number), str(i))))
                    cls.save_to_csv(
                        results[1],
                        os.path.join(output_directory, "eval_user_results_{}_{}".format(str(cls.eval_number), str(i))))
        finally:
            cls.eval_number += 1


class MetricCalculatorRun(NeedsSerializationRun):
    """
    Run associated with the MetricCalculator

    The output will be serialized in the following form:

        mc_sys_results_0_0
        mc_user_results_0_0

    where the first number is used to identify the MetricCalculator in the script file (in case multiple
    MetricCalculator objects with the same output directory are defined in the script file, this identifies each output)
    The second number instead, identifies the method call (in case multiple parameters for the 'eval_metrics' method are
    defined)
    """

    metric_calculator_number = 0

    @classmethod
    def get_associated_class(cls):
        return MetricCalculator

    @classmethod
    def serialize_results(cls, executed_methods_results: Dict[str, list], output_directory: str):
        try:
            if 'eval_metrics' in executed_methods_results.keys():
                for i, results in enumerate(executed_methods_results['eval_metrics']):
                    cls.save_to_csv(
                        results[0],
                        os.path.join(output_directory, "mc_sys_results_{}_{}".format(str(cls.metric_calculator_number), str(i))))
                    cls.save_to_csv(
                        results[1],
                        os.path.join(output_directory, "mc_user_results_{}_{}".format(str(cls.metric_calculator_number), str(i))))
        finally:
            cls.metric_calculator_number += 1


class MethodologyRun(NeedsSerializationRun):
    """
    Run associated with the Methodology

    The output will be serialized in the following form:

        item_to_predict_0_0#0
        item_to_predict_0_0#1
        item_to_predict_0_1#0
        ...

    where the first number is used to identify the Methodology in the script file (in case multiple Methodology
    objects with the same output directory are defined in the script file, this identifies each output), the
    second one is used to identify each method call (since it's possible to call methods multiple times with different
    parameters) and the last one identifies each Dataframe in the list returned by the get_item_to_predict method
    """

    methodology_number = 0

    @classmethod
    def get_associated_class(cls):
        return Methodology

    @classmethod
    def serialize_results(cls, executed_methods_results: Dict[str, list], output_directory: str):
        try:
            if 'get_item_to_predict' in executed_methods_results.keys():
                for i, item_to_predict in enumerate(executed_methods_results['get_item_to_predict']):
                    for split_number, split in enumerate(item_to_predict):
                        cls.save_to_csv(
                            split,
                            os.path.join(output_directory, "item_to_predict_{}_{}#{}".format(str(cls.methodology_number), str(i),
                                                                                             str(split_number))))
        finally:
            cls.methodology_number += 1


class PartitioningRun(NeedsSerializationRun):
    """
    Run associated with the PartioningModule

    The output will be serialized in the following form:

        training_0_0#0
        training_0_0#1
        ...
        testing_0_0#0
        testing_0_0#1

    where the first number is used to identify the Partitioning in the script file (in case multiple PartitionModule
    objects with the same output directory are defined in the script file, this identifies each output) and the
    second one is used to identify each method call (since it's possible to call methods multiple times with
    different parameters). The last number after the # symbol is used to
    identify each partition (so training_0_0#0 matches testing_0_0#0)
    """

    partitioning_number = 0

    @classmethod
    def get_associated_class(cls):
        return PartitionModule

    @classmethod
    def serialize_results(cls, executed_methods_results: Dict[str, list], output_directory: str):
        try:
            for i, result_list in enumerate(executed_methods_results['split_all']):
                for split_number, split in enumerate(result_list):
                    training: pd.DataFrame = split.train
                    testing: pd.DataFrame = split.test

                    cls.save_to_csv(
                        training,
                        os.path.join(output_directory, "training_{}_{}#{}".format(str(cls.partitioning_number),
                                                                                  str(i), str(split_number))))
                    cls.save_to_csv(
                        testing,
                        os.path.join(output_directory, "testing_{}_{}#{}".format(str(cls.partitioning_number),
                                                                                 str(i), str(split_number))))
        finally:
            cls.partitioning_number += 1


def __setup_implemented_modules_dictionary():
    """
    This function creates the implemented modules dictionary which will contain all implemented classes and
    subclasses for the classes defined in the get_associated_class method of the Run class

    The final dictionary will be in the following form:

        {
            "ContentAnalyzer": ContentAnalyzerRun,
            class_name: Run class associated with the class name,
            ...
        }

    Returns:
        implemented_modules (dict): dictionary with the name of the implemented main modules as keys and the run
            associated to the class as values
    """
    implemented_modules: Dict[str, Type[Run]] = dict()

    for run in get_all_implemented_subclasses(Run):
        implemented_classes = get_all_implemented_classes(run.get_associated_class())
        for cls in implemented_classes:
            implemented_modules[cls.__name__.lower()] = run

    return implemented_modules


def script_run(config_list_dict: Union[dict, list]):
    """
    Method that controls the entire process of script running. It checks that the loaded contents from a
    script match the required prerequisites. The data must contain dictionaries and each dictionary must have a "module"
    key that is used to understand what kind of operation it is supposed to perform (if it's meant for a Recommender
    System, a Content Analyzer, ...). If any of these prerequisites aren't matched a ScriptConfigurationError exception
    is thrown.

    Args:
        config_list_dict: single dictionary or list of dictionaries extracted from the config file
    """
    implemented_modules = __setup_implemented_modules_dictionary()
    try:

        if not isinstance(config_list_dict, list):
            config_list_dict = [config_list_dict]

        if not all(isinstance(config_dict, dict) for config_dict in config_list_dict):
            raise ScriptConfigurationError("The list in the script must contain dictionaries only")

        for config_dict in config_list_dict:

            if "module" in config_dict.keys():
                module = config_dict.pop("module").lower()
                try:
                    run_class = implemented_modules[module]
                except KeyError:
                    raise ScriptConfigurationError(
                        "You must specify a valid module: " + str(implemented_modules.keys()))

                run_class.run(config_dict, module)
            else:
                raise ScriptConfigurationError("A 'module' parameter must be specified and the value must be one "
                                               "of the following: " + str(implemented_modules.keys()))
    except (ScriptConfigurationError, NoOutputDirectoryDefined, ParametersError, FileNotFoundError) as e:
        raise e
