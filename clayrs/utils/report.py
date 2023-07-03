from __future__ import annotations
import os.path
import re
from pathlib import Path

import numpy as np
import pyaml


from typing import TYPE_CHECKING

# to avoid circular import. Maybe a little 'hacky', better organization for the future?
# This is almost inevitable though, since Report MUST refer to other modules for type hints
if TYPE_CHECKING:
    from clayrs_can_see.content_analyzer.content_analyzer_main import ContentAnalyzer
    from clayrs_can_see.content_analyzer import Ratings
    from clayrs_can_see.evaluation import EvalModel
    from clayrs_can_see.recsys.partitioning import Partitioning
    from clayrs_can_see.recsys.recsys import RecSys


class Report:
    """
    Class which will generate a YAML report for the whole experiment (or a part of it) depending on the objects
    passed to the `yaml()` function.

    A report will be generated for each module used (`Content Analyzer`, `RecSys`, `Evaluation`).

    Args:
        output_dir: Path of the folder where reports generated will be saved
        ca_report_filename: Filename of the Content Analyzer report
        rs_report_filename: Filename of the Recsys report
        eva_report_filename: Filename of the evaluation report

    """

    def __init__(self, output_dir: str = '.',
                 ca_report_filename: str = 'ca_report',
                 rs_report_filename: str = 'rs_report',
                 eva_report_filename: str = 'eva_report'):

        self._output_dir = output_dir
        self._ca_report_filename = ca_report_filename
        self._rs_report_filename = rs_report_filename
        self._eva_report_filename = eva_report_filename

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, path):
        self._output_dir = path

    @property
    def ca_report_filename(self):
        return self._ca_report_filename

    @ca_report_filename.setter
    def ca_report_filename(self, filename: str):
        self._ca_report_filename = filename

    @property
    def rs_report_filename(self):
        return self._rs_report_filename

    @rs_report_filename.setter
    def rs_report_filename(self, filename: str):
        self._rs_report_filename = filename

    @property
    def eva_report_filename(self):
        return self._eva_report_filename

    @eva_report_filename.setter
    def eva_report_filename(self, filename: str):
        self._eva_report_filename = filename

    @staticmethod
    def _extract_arguments(repr_string):

        # this is the regex pattern for matching anything inside balanced '(' ')' parenthesis.
        # It only works if the maximum number of recursion is specified (in this case 25),
        # so basically we are saying that if an attribute of object A is an object B, which has an attribute
        # that is object C and this object has an attribute which is an object ...
        # this will break after the 25th object
        n = 25
        balanced_parenthesis_pattern = r"[^()]*?(?:\(" * n + r"[^()]*?" + r"\)[^()]*?)*?" * n

        # here we match the name of the object: we are matching any \w before the open parenthesis '('
        name = re.match(r"(\w+)(?=\()", repr_string)
        parameters_dict = dict()

        if name is not None:
            name = name.group()

            # We separate in tuples attributes and values.
            # NameObject(a1=1.0, a2=2.0) -> [(a1, 1.0), (a2, 2.0)]
            # we are also matching whole nested objects as attributes, with the max recursion discussed above
            # NameObject(a1=1.0, a2=2.0, a3=NestedObject()) -> [(a1, 1.0), (a2, 2.0), (a3, NestedObject())]
            # we are matching any \w preceded by '(' (the beginning) or by a space (it's a follow up parameter)
            # which has as follow-up character the '=' and a nested object after that or anything else.
            # The last part is the bounding, the match ends when ', \w+=' is found (another parameter= or the ')' (the
            # end)
            tuples_list_args = re.findall(r"(?<=\(|\s)(\w+)=(\w+"+balanced_parenthesis_pattern+r"|.*?)(?=,\s\w+=|\)$)",
                                          repr_string)

            dict_args_string = dict(tuples_list_args)

            for key, val in dict_args_string.items():

                # Recursive call so that each nested object (if any) will be expanded:
                # {'a3': 'NestedObject(a1=NestedObject())'} -> {'a3': 'NestedObject': {'a1': NestedObject: {}}}
                name_nested, dict_args_nested = Report._extract_arguments(val)

                # if a value can be evaluated we transform it into an object (better serializability) or else
                # we take it as a string
                if name_nested is None:
                    try:
                        parameters_dict[key] = eval(val)
                    except (NameError, SyntaxError):
                        parameters_dict[key] = val
                else:
                    parameters_dict[key] = {name_nested: dict_args_nested}

        return name, parameters_dict

    def _report_ca_module(self, content_analyzer: ContentAnalyzer):

        ca_dict = dict()

        ca_dict['source_file'] = content_analyzer._config.source.representative_name
        ca_dict['id_each_content'] = content_analyzer._config.id

        # EXO REPRESENTATIONS
        exo_list = content_analyzer._config.exogenous_representation_list

        ca_dict['exogenous_representations'] = dict() if len(exo_list) != 0 else None

        for exo_config in exo_list:
            string_exo = repr(exo_config.exogenous_technique)
            name_exo, exo_parameters_dict = self._extract_arguments(string_exo)
            ca_dict['exogenous_representations'][name_exo] = exo_parameters_dict

        # FIELD REPRESENTATIONS
        ca_dict['field_representations'] = dict()

        field_names = content_analyzer._config.get_field_name_list()
        for field_name in field_names:

            field_config_list = content_analyzer._config.get_configs_list(field_name)

            for i, field_config in enumerate(field_config_list):

                ca_dict['field_representations']['{}/{}'.format(field_name, str(i))] = dict()

                single_representation_dict = dict()

                string_technique = repr(field_config.content_technique)

                name_technique, parameter_dict_technique = self._extract_arguments(string_technique)

                single_representation_dict[name_technique] = parameter_dict_technique

                single_representation_dict['preprocessing'] = dict() if len(field_config.preprocessing) != 0 else None
                for preprocessing in field_config.preprocessing:
                    string_preprocessing = repr(preprocessing)

                    name_preprocessing, parameter_dict_preprocessing = self._extract_arguments(string_preprocessing)

                    single_representation_dict['preprocessing'][name_preprocessing] = parameter_dict_preprocessing

                ca_dict['field_representations']['{}/{}'.format(field_name, str(i))] = single_representation_dict

        return ca_dict

    def _report_rs_module(self, original_ratings: Ratings, partitioning_technique: Partitioning, recsys: RecSys):

        # To provide a yaml report for recsys object, a rank or predict method must be called first
        if recsys is not None and recsys._yaml_report is None:
            raise ValueError("You must first call with the rank() or predict() method of the recsys object "
                             "before computing the report!")

        rs_dict = dict()

        if original_ratings is not None:
            rs_dict['interactions'] = dict()

            rs_dict['interactions']['n_users'] = len(set(original_ratings.user_id_column))
            rs_dict['interactions']['n_items'] = len(set(original_ratings.item_id_column))
            rs_dict['interactions']['total_interactions'] = len(original_ratings.user_id_column)

            sparsity_numerator = rs_dict['interactions']['total_interactions']
            sparsity_denominator = rs_dict['interactions']['n_users'] * rs_dict['interactions']['n_items']

            sparsity = 1 - sparsity_numerator / sparsity_denominator

            rs_dict['interactions']['sparsity'] = sparsity
            rs_dict['interactions']['min_score'] = float(np.min(original_ratings.score_column))
            rs_dict['interactions']['max_score'] = float(np.max(original_ratings.score_column))
            rs_dict['interactions']['mean_score'] = float(np.mean(original_ratings.score_column))

        if partitioning_technique is not None:
            rs_dict['partitioning'] = dict()

            partitioning_string = repr(partitioning_technique)
            name_partitioning, parameters_partitioning_dict = self._extract_arguments(partitioning_string)

            rs_dict['partitioning'][name_partitioning] = parameters_partitioning_dict

        if recsys is not None:
            rs_dict['recsys'] = dict()

            recsys_string = repr(recsys)
            name_recsys, _ = self._extract_arguments(recsys_string)

            parameters_recsys_dict = dict()

            name_alg, parameters_alg = self._extract_arguments(repr(recsys.algorithm))

            parameters_recsys_dict['algorithm'] = {name_alg: parameters_alg}

            for key, val in recsys._yaml_report.items():

                val = val if isinstance(val, str) else repr(val)

                # each item of the dict of the recsys may be an object with nested objects as attributes,
                # we check this
                name, parameters_dict = self._extract_arguments(val)

                # in case an element of the dict is an object with nested object, name will not be None.
                # else it's a "base" attribute and we add it as it is to the yaml dict
                if name is not None:
                    parameters_recsys_dict[key] = {name: parameters_dict}
                else:
                    parameters_recsys_dict[key] = val

            rs_dict['recsys'][name_recsys] = parameters_recsys_dict

        return rs_dict

    def _report_eva_module(self, eval_model: EvalModel):

        # To provide a yaml report for eval_model object, the fit() method must be called first
        if eval_model._yaml_report_result is None:
            raise ValueError("You must first call with the fit() method of the eval_model object "
                             "before computing the report!")

        eva_dict = dict()

        eva_dict['n_split'] = len(eval_model.pred_list)

        eva_dict['metrics'] = dict()

        for metric in eval_model.metric_list:
            string_metric = repr(metric)

            name_metric, parameters_metric_dict = self._extract_arguments(string_metric)

            eva_dict['metrics'][name_metric] = parameters_metric_dict

        eva_dict['sys_results'] = eval_model._yaml_report_result

        return eva_dict

    def yaml(self, content_analyzer: ContentAnalyzer = None,
             original_ratings: Ratings = None,
             partitioning_technique: Partitioning = None,
             recsys: RecSys = None,
             eval_model: EvalModel = None):
        """
        Main module responsible of generating the `YAML` reports based on the objects passed to this function:

        * If `content_analyzer` is set, then the report for the Content Analyzer will be produced
        * If one between `original_ratings`, `partitioning_technique`, `recsys` is set, then the report for the recsys
        module will be produced.
        * If `eval_model` is set, then the report for the evaluation module will be produced

        **PLEASE NOTE**: by setting the `recsys` parameter, the last experiment conducted will be documented! If no
        experiment is conducted in the current run, then a `ValueError` exception is raised!

        * Same goes for the `eval_model`

        Examples:

            * Generate a report for the Content Analyzer module
            >>> from clayrs_can_see import content_analyzer as ca
            >>> from clayrs_can_see import utils as ut
            >>> # movies_ca_config = ...  # user defined configuration
            >>> content_a = ca.ContentAnalyzer(movies_config)
            >>> content_a.fit()  # generate and serialize contents
            >>> ut.Report().yaml(content_analyzer=content_a)  # generate yaml

            * Generate a partial report for the RecSys module
            >>> from clayrs_can_see import utils as ut
            >>> from clayrs_can_see import recsys as rs
            >>> ratings = ca.Ratings(ca.CSVFile(ratings_path))
            >>> pt = rs.HoldOutPartitioning()
            >>> [train], [test] = pt.split_all(ratings)
            >>> ut.Report().yaml(original_ratings=ratings, partitioning_technique=pt)

            * Generate a full report for the RecSys module and evaluation module
            >>> from clayrs_can_see import utils as ut
            >>> from clayrs_can_see import recsys as rs
            >>> from clayrs_can_see import evaluation as eva
            >>>
            >>> # Generate recommendations
            >>> ratings = ca.Ratings(ca.CSVFile(ratings_path))
            >>> pt = rs.HoldOutPartitioning()
            >>> [train], [test] = pt.split_all(ratings)
            >>> alg = rs.CentroidVector()
            >>> cbrs = rs.ContentBasedRS(alg, train_set=train, items_directory=items_path)
            >>> rank = cbrs.fit_rank(test, n_recs=10)
            >>>
            >>> # Evaluate recommendations and generate report
            >>> em = eva.EvalModel([rank], [test], metric_list=[eva.Precision(), eva.Recall()])
            >>> ut.Report().yaml(original_ratings=ratings,
            >>>                  partitioning_technique=pt,
            >>>                  recsys=cbrs,
            >>>                  eval_model=em)

        Args:
            content_analyzer: `ContentAnalyzer` object used to generate complex representation in the experiment
            original_ratings: `Ratings` object representing the original dataset
            partitioning_technique: `Partitioning` object used to split the original dataset
            recsys: `RecSys` object used to produce recommendations/score predictions. Please note that the latest
                experiment run will be documented. If no experiment is run, then an exception is thrown
            eval_model: `EvalModel` object used to evaluate predictions generated. Please note that the latest
                evaluation run will be documented. If no evaluation is run, then an exception is thrown
        """

        def represent_none(self, _):
            return self.represent_scalar('tag:yaml.org,2002:null', 'null')

        def dump_yaml(output_dir, data):
            with open(output_dir, 'w') as yaml_file:
                pyaml.dump(data, yaml_file, sort_dicts=False, safe=True,)

        # None values will be represented as 'null' in yaml file.
        # without this, they will simply be represented as an empty string
        pyaml.add_representer(type(None), represent_none)

        if content_analyzer is not None:
            ca_dict = self._report_ca_module(content_analyzer)

            # create folder if it doesn't exist
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

            output_dir = os.path.join(self.output_dir, f'{self._ca_report_filename}.yml')
            dump_yaml(output_dir, ca_dict)

        if original_ratings is not None or partitioning_technique is not None or recsys is not None:
            rs_dict = self._report_rs_module(original_ratings, partitioning_technique, recsys)

            # create folder if it doesn't exist
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

            output_dir = os.path.join(self.output_dir, f'{self._rs_report_filename}.yml')
            dump_yaml(output_dir, rs_dict)

        if eval_model is not None:
            eva_dict = self._report_eva_module(eval_model)

            # create folder if it doesn't exist
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

            output_dir = os.path.join(self.output_dir, f'{self._eva_report_filename}.yml')
            dump_yaml(output_dir, eva_dict)
