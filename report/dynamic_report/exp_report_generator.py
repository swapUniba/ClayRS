import os
import re
import shutil
import yaml
import jinja2
import subprocess
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod

# GLOBAL VARIABLES IN THE SCRIPT

# template non indentato
# TEMPLATE_FILE = "report_template_not_indented.html"
# template indentato
# TEMPLATE_FILE = "report_template_indented.html"
# TEMPLATE_FILE = "report_template.html"
# TEMPLATE_FILE = "templates_latex/report_template.html"
# TEMPLATE_FILE = "report_template.tex"
# il report del modulo evaluation
# DATA_FILE = "data/eva_report.yml"
# il report del modulo recsys
# DATA_FILE = "data/rs_report.yml"
# il report del modulo content analyzer
# DATA_FILE = "data/ca_report.yml"
OUTPUT_TEX = "output/report.TEX"
OUTPUT_PATH = "output/report.pdf"
"""
# test with linear predictor
LIST_YAML_FILES = ["data/data_to_test/item_ca_report_nxPageRank.yml",
                   "data/data_to_test/rs_report_linearPredictor.yml",
                   "data/data_to_test/eva_report_linearPredictor.yml"]
"""
"""
# test with yml of Index query
LIST_YAML_FILES = ["data/data_to_test/item_ca_report_nxPageRank.yml",
                   "data/data_to_test/rs_report_indexQuery.yml",
                   "data/data_to_test/eva_report_indexQuery.yml"]
"""
"""
# test with yml of classifier recommender
LIST_YAML_FILES = ["data/data_to_test/item_ca_report_nxPageRank.yml",
                   "data/data_to_test/rs_report_classifierRecommender.yml",
                   "data/data_to_test/eva_report_classifierRecommender.yml"]
"""

# test with yml of centroid vector
LIST_YAML_FILES = ["data/data_to_test/item_ca_report_nxPageRank.yml",
                   "data/data_to_test/rs_report_centroidVector.yml",
                   "data/data_to_test/eva_report_centroidVector.yml"]

"""
# test with yml of amar double source
LIST_YAML_FILES = ["data/data_to_test/item_ca_report_nxPageRank.yml",
                   "data/data_to_test/rs_report_amarDoubleSource.yml",
                   "data/data_to_test/eva_report_armarDoubleSource.yml"]
"""
# LIST_YAML_FILES = ["data/ca_report.yml", "data/rs_report.yml", "data/eva_report.yml"]
# TEMPLATE_FILE = "report_templateNew.tex"
TEMPLATE_FILE = "dynamic_fin_rep.tex"


class ReportManager(ABC):
    def __init__(self, file_destination):
        self.file_destination = file_destination
        self.ca_report_dict = {}
        self.rs_report_dict = {}
        self.eva_report_dict = {}
        self.unified_report_dict = {}
        self.template_to_render = ""

    def get_current_date_string(self):
        """
           The function will take the current day, formatting as DD-MM-YYYY and returning as a string.

           Author:
           - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
        """
        today_date = datetime.now()
        today_date_string = today_date.strftime('%d-%m-%Y')
        return today_date_string

    def get_keys_at_level(self, dictionary, parent_key):
        """
            Retrieve all keys at a specific level under a given parent key in a nested dictionary.

            Parameters:
            - dictionary (dict): The nested dictionary to explore.
            - parent_key (str): The key under which to retrieve the sub-keys.

            Returns:
            - list: List of keys at the specified level under the parent key.
                    If the parent key is not present or does not have a dictionary value, an empty list is returned.

            Example:
               # >>> my_dict = {
               # ...     "source_file": {"a": {}, "b": {}},
               # ...     "id_each_conten": {"mario": {"carry": {}, "boxes": {}}, "anna": {}, "dora": {"carry": {}, "boxes": {}}},
               # ...     "field_representations": {"bar": {"carry": {}, "boxes": {}}, "lane": {"tarry": {}, "door": {}}, "car": {}, "park": {}},
               # ... }
               # >>> get_keys_at_level(my_dict, "bar")
                ['carry', 'boxes']

            Author:
            - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
        """
        keys_at_level = []

        def explore_dictionary(current_dict, current_parent):
            if current_parent == parent_key and isinstance(current_dict, dict):
                keys_at_level.extend(current_dict.keys())
            else:
                for key, value in current_dict.items():
                    if isinstance(value, dict):
                        explore_dictionary(value, key)

        explore_dictionary(dictionary, None)
        return keys_at_level

    def get_subkeys_at_path(self, dictionary, *keys_in_order):
        """
            Retrieve all subkeys under the specified path in a nested dictionary.

            Parameters:
            - dictionary (dict): The nested dictionary to explore.
            - keys_in_order (str): Keys in the order of the path.

            Returns:
            - list: List of subkeys under the specified path in the dictionary.
                    If the path is not present or does not have a dictionary value, an empty list is returned.

            Example:
               # >>> my_dict = {
               # ...     "a": {"b": {"x": {}, "y": {}}, "c": {}},
               # ...     "t": {"b": {"z": {}}, "c": {}},
               # ...     "r": {"b": {"f": {"a": {}, "w": {}}}, "h": {}}
               # ... }
               # >>> get_subkeys_at_path(my_dict, "r", "b")
                ['f']

            Author:
            - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
        """
        subkeys_at_path = []

        def explore_dictionary(current_dict, current_keys):
            if len(current_keys) == 0 or current_dict is None:
                return

            if isinstance(current_dict, dict) and current_keys[0] in current_dict:
                next_dict = current_dict[current_keys[0]]

                if next_dict is not None:
                    if len(current_keys) == 1:
                        subkeys_at_path.extend(next_dict.keys())
                    else:
                        explore_dictionary(next_dict, current_keys[1:])

        explore_dictionary(dictionary, keys_in_order)
        return subkeys_at_path

    def merge_subkeys_for_keys(self, dictionary, keys_list, additional_args):
        merged_subkeys = []

        for key in keys_list:
            # Aggiungi la chiave corrente alla fine di additional_args
            args_with_key = additional_args + [key]

            subkeys = self.get_subkeys_at_path(dictionary, *args_with_key)
            merged_subkeys.extend(subkeys)

        return merged_subkeys

    def read_yaml_file(self, file_path):
        """
            Read a YAML file and return the corresponding dictionary.

            Parameters:
            - file_path (str): Path to the YAML file.

            Returns:
            - dict: Dictionary representing the structure of the YAML file.

            Example:
                # >>> yaml_content = read_yaml_file("example.yaml")
                # >>> print(yaml_content)
                 {'key1': 'value1', 'key2': {'key3': 'value3', 'key4': 'value4'}}

            Author:
            - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
        """
        with open(file_path, "r") as file:
            yaml_content = yaml.safe_load(file)

        return yaml_content

    def get_data(self, rendering_file):
        with open(rendering_file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                return data
            except yaml.YAMLError as exc:
                print(exc)

    # considera di spostarli direttamente
    def read_file_latex(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            contenuto = file.read()
        return contenuto

    def write_on_file_latex(self, testo):
        with open(self.file_destination, 'a', encoding='utf-8') as file:
            file.write(testo)

    def get_text_from_latex(self, contenuto_latex):
        pattern = re.compile(r'###(.*?)###', re.DOTALL)
        match = pattern.search(contenuto_latex)
        if match:
            return match.group(1)
        else:
            return None
    """
    @abstractmethod
    def build_template_file(self):
        pass
    """
    @abstractmethod
    def build_template_file_simplex(self):
        pass

class DynamicReportManager(ReportManager):
    # dictionary to find the path for mini template chunks of content analyzer
    # the key are the key found in the yaml file for content analyzer and the
    # value are the path to the corresponding templates_latex.
    CA_DICT = {
        'OriginalData': './templates_chunks/templates_ca_mini_chunks/OriginalData_techniques_ca.tex',
        'WhooshTfIdf': './templates_chunks/templates_ca_mini_chunks/WhooshTfIdf_techinques_ca.tex',
        'SkLearnTfIdf': './templates_chunks/templates_ca_mini_chunks/SkLearnTfIdf_tech_ca.tex',
        'WordEmbeddingTechnique': './templates_chunks/templates_ca_mini_chunks/WordEmbedding_tech_ca.tex',
        'SentenceEmbeddingTechnique': './templates_chunks/templates_ca_mini_chunks/SentenceEmb_tech_ca.tex',
        'DocumentEmbeddingTechnique': './templates_chunks/templates_ca_mini_chunks/DocumentEmb_tech_ca.tex',
        'Word2SentenceEmbedding': './templates_chunks/templates_ca_mini_chunks/Word2SentenceEmb_tech_ca.tex',
        'Word2DocEmbedding': './templates_chunks/templates_ca_mini_chunks/Word2DocEmb_tech_ca.tex',
        'Sentence2DocEmbedding': './templates_chunks/templates_ca_mini_chunks/Word2SentenceEmb_tech_ca.tex',
        'PyWSDSynsetDocumentFrequency': './templates_chunks/templates_ca_mini_chunks/PyWSD_tech_ca.tex',
        'FromNPY': './templates_chunks/templates_ca_mini_chunks/FromNPY_tech_ca.tex',
        'SkImageHogDescriptor': './templates_chunks/templates_ca_mini_chunks/SkImgHogDescr_tech_ca.tex',
        'MFCC': './templates_chunks/templates_ca_mini_chunks/mfcc_tech_ca.tex',
        'VGGISH': './templates_chunks/templates_ca_mini_chunks/TorchVVM_tech_ca.tex',
        'PytorchImageModels': './templates_chunks/templates_ca_mini_chunks/PytorchImgMod_tech_ca.tex',
        'TorchVisionVideoModels': './templates_chunks/templates_ca_mini_chunks/TorchVVM_tech_ca.tex',
        'Spacy': './templates_chunks/templates_ca_mini_chunks/spacy_prepro_ca.tex',
        'Ekphrasis': './templates_chunks/templates_ca_mini_chunks/ekphrasis_prepro_ca.tex',
        'NLTK': './templates_chunks/templates_ca_mini_chunks/nltk_prepro_ca.tex',
        'TorchUniformTemporalSubSampler': './templates_chunks/templates_ca_mini_chunks/TorchUniformTSS_prepro_ca.tex',
        'Resize': './templates_chunks/templates_ca_mini_chunks/Resize_prepro_ca.tex',
        'CenterCrop': './templates_chunks/templates_ca_mini_chunks/centerCrop_prepro_ca.tex',
        'Lambda': './templates_chunks/templates_ca_mini_chunks/Lambda_prepro_ca.tex',
        'Normalize': './templates_chunks/templates_ca_mini_chunks/normilize_prepro_ca.tex',
        'ClipSampler': './templates_chunks/templates_ca_mini_chunks/clipSampler_prepro_ca.tex',
        'FVGMM': './templates_chunks/templates_ca_mini_chunks/fvgmm_postpro_ca.tex',
        'VLADGMM': './templates_chunks/templates_ca_mini_chunks/vladgmm_postpro_ca.tex',
        'SkLearnPCA': './templates_chunks/templates_ca_mini_chunks/skLearnPCA_postpro_ca.tex',
        'DBPediaMappingTechnique': './templates_chunks/templates_ca_mini_chunks/DBpediaMT_ex_tech_ca.tex',
        'ConvertToMono': './templates_chunks/templates_ca_mini_chunks/Convert2Mono_prepro_ca.tex',
        'TorchResample': './templates_chunks/templates_ca_mini_chunks/TorchResample_prepro_ca.tex',
        'intro': './templates_chunks/templates_ca_mini_chunks/intro_ca.tex',
        'end': './templates_chunks/templates_ca_mini_chunks/end_of_ca.tex',
        'field_sec': './templates_chunks/templates_ca_mini_chunks/field_ca.tex',
        'preprocessing': './templates_chunks/templates_ca_mini_chunks/no_preprocessing_ca.tex',
        'postprocessing': './templates_chunks/templates_ca_mini_chunks/no_postprocessing_ca.tex',
        'exogenous_representations': './templates_chunks/templates_ca_mini_chunks/no_exogenous_tech.tex',
        'pst': './templates_chunks/templates_ca_mini_chunks/postprocessing_general.tex',
        'pre': './templates_chunks/templates_ca_mini_chunks/preprocess_field.tex',
        'post': './templates_chunks/templates_ca_mini_chunks/postprocessing_field.tex',
        'repr': './templates_chunks/templates_ca_mini_chunks/content_representation_field.tex'
    }

    # dictionary to find path for the recsys module template
    RS_DICT = {
        'recsys': './templates_chunks/templates_rs/recsys_template_complete_new.tex'
    }

    # dictionary to find path for the evaluation module template
    EVA_DICT = {
        'intro': './templates_chunks/templates_eva_mini_chunks/intro_eva_all_metrics.tex',
        'end': './templates_chunks/templates_eva_mini_chunks/end_eva.tex',
        'result': './templates_chunks/templates_eva_mini_chunks/sys_result_on_fold_eva_new.tex'
    }

    # dictionary to find path for template used to start and complete the report
    REP_DICT = {
        'intro': './templates_chunks/intro_report_start.tex',
        'end': './templates_chunks/conclusion.tex'
    }

    def __init__(self, file_destination, ca_rep_yml=None, rs_rep_yml=None, eva_rep_yml=None):
        super().__init__(file_destination)
        self.ca_rep_yml = ca_rep_yml
        self.rs_rep_yml = rs_rep_yml
        self.eva_rep_yml = eva_rep_yml
        # used to load the dictionary related to each yaml file
        self.ca_report_dict = super().read_yaml_file(self.ca_rep_yml)
        self.rs_report_dict = super().read_yaml_file(self.rs_rep_yml)
        self.eva_report_dict = super().read_yaml_file(self.eva_rep_yml)
        # Imposta la directory di lavoro corrente
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        print("Current working directory:", self.base_path)

        # Imposta il percorso del template LaTeX
        self.template_path = os.path.join(self.base_path, "dynamic_fin_rep.tex")
        print("Template path:", self.template_path)
        # setting environment based on latex needs
        self.latex_jinja_env = jinja2.Environment(
            block_start_string="\BLOCK{",
            block_end_string="}",
            variable_start_string="\VAR{",
            variable_end_string="}",
            comment_start_string="\#{",
            comment_end_string="}",
            trim_blocks=True,
            autoescape=False,
            loader=jinja2.FileSystemLoader(searchpath=self.base_path),
        )
        # load filter for jinja environment
        self.load_filters()

    def safe_text(self, text: str) -> str:
        special_chars = ['&', '%', '$', '_', '{', '}', '#']
        for char in special_chars:
            text = str(text)
            text = text.replace(char, "\\" + char)
        return text

    def truncate(self, text: str) -> str:
        number = float(text)
        number = round(number, 5)
        text = str(number)
        print(text)
        return text

    def load_filters(self):
        self.latex_jinja_env.filters["safe_text"] = self.safe_text
        self.latex_jinja_env.filters["truncate"] = self.truncate

    def merge_yaml_files(self, input_paths_list, output_folder, output_filename):
        """
        Merge multiple YAML files into a single YAML file.

        Parameters:
        - input_paths (list): List of paths to input YAML files.
        - output_folder (str): Path to the folder where the output YAML file will be created.
        - output_filename (str): Name of the output YAML file.

        Returns:
        - str: Path to the merged YAML file.

        Author:
        - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
        """
        if not input_paths_list:
            raise ValueError("The list of input files is empty. It must contain at least one path.")

        merged_data = {}

        # read from input file yaml and write data in dict merged_data
        for input_path in input_paths_list:
            data = super().get_data(input_path)
            if data is not None:
                merged_data.update(data)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # create and adjust path to yaml merged file
        output_path = os.path.join(output_folder, output_filename)
        output_path = os.path.normpath(output_path)

        # write data in file unified YAML
        with open(output_path, 'w') as output_file:
            yaml.dump(merged_data, output_file, default_flow_style=False)

        return output_path

    # è la nuova funzione che sostituisce get_latex_template() in text_generate_latex.py
    def load_template_into_enviroment(self):
        template_path = "dynamic_fin_rep.tex"
        template = self.latex_jinja_env.get_template(template_path)
        return template

    def generate_dynamic_report(self, path_data_in, output_tex_path):
        my_dict = {}
        data = self.get_data(path_data_in)

        # dictionary check
        try:
            my_dict = dict(data)
        except TypeError as e:
            print(f"Impossible to convert data to dictionary: {e}")
            my_dict = {}  # o un altro valore di default a tua scelta

        # type(my_dict)
        # print(my_dict)
        # Load template LaTeX
        template = self.load_template_into_enviroment()
        # print(template)

        # Rendering template with data
        output_text = template.render(my_dict=data)
        # print(output_text)

        try:
            # Extract the directory path from the LaTeX file path
            output_directory = os.path.dirname(output_tex_path)

            # Check if the output directory exists; otherwise, create the directory
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            # Write the content to the LaTeX file
            with open(output_tex_path, 'w') as ofile:
                ofile.write(output_text)

            # Return the path of the written file
            return output_tex_path

        except Exception as e:
            print(f"Error during writing LaTeX file: {e}")
            return None

    """
    def build_template_file(self):
        # support functions to adding the text to the finale latex file, after processing and change placeholder
        # with the object needed.
        def add_single_mini_template(path_map, access_key, path_destiny, content_of_field_container,
                                     text_extract_container):
            content = self.read_file_latex(path_map.get(access_key))
            text = self.get_text_from_latex(content)
            # check point
            if text:
                self.write_on_file_latex(text)
                print(f"Content of {path_map.get(access_key)} added to {path_destiny} successfully")
            else:
                print(f"Impossible to extract text from LaTeX document: {path_map.get(access_key)}")

            # updating the external parameter
            content_of_field_container[0] = content
            text_extract_container[0] = text

        def process_and_write_to_file(path_dict, pros, scope, content_container, text_container, file_path):
            # we are extracting the written part of the latex file which is retrieved from
            # the dictionary which contain for each key the path of the file that needs to be added
            content = self.read_file_latex(path_dict.get(pros))

            # This passage allows to specify the field in the report and prepare text
            content = content.replace('X', scope)
            text = self.get_text_from_latex(content)

            # add text to file after check point
            if text:
                self.write_on_file_latex(text)
                print(f"Content of {path_dict.get(pros)} added to {self.file_destination} successfully")
            else:
                print(f"Impossible to extract text from LaTeX document: {path_dict.get(pros)}")
            # update external parameters
            content_container[0] = content
            text_container[0] = text

        # used to add mini template at the final template latex
        content_of_field = [""]
        text_extract = [""]

        # first part of the report intro of the experiment report template
        # add_single_mini_template(REPORT_DICT, 'intro', file_destination, content_of_field, text_extract)
        process_and_write_to_file(DynamicReportManager.REP_DICT, 'intro', super().get_current_date_string(),
                                  content_of_field, text_extract, self.file_destination)

        # dealing with Content Analyzer
        add_single_mini_template(DynamicReportManager.CA_DICT, 'intro', self.file_destination,
                                 content_of_field, text_extract)

        # add all the field that have been represented using the content analyzer and specify their
        # preprocessing and postprocessing received.
        if 'source_file' in self.ca_report_dict:

            # extraction of the field being analyzed
            list_of_field = super().get_keys_at_level(self.ca_report_dict, "field_representations")

            print(list_of_field)

            # dealing with all field that have been represented with content analyzer
            for field in list_of_field:
                # add the highlight field in the report
                process_and_write_to_file(DynamicReportManager.CA_DICT, 'field_sec',
                                          field, content_of_field, text_extract,
                                          self.file_destination)

                # list of primary subkey of key field
                processing_list = super().get_keys_at_level(self.ca_report_dict, field)

                print(processing_list)

                # dealing with all process applied to a specific field --> dealing with field subkeys
                for process in processing_list:
                    # this first check allow to identify the techniques for field representation
                    if process != 'preprocessing' and process != 'postprocessing':
                        process_and_write_to_file(DynamicReportManager.CA_DICT, process,
                                                  field, content_of_field,
                                                  text_extract, self.file_destination)
                    # dealing with preprocessing part
                    elif process == 'preprocessing':
                        prepro_list = super().get_subkeys_at_path(self.ca_report_dict,
                                                                  "field_representations",
                                                                  field, process)
                        # check if list empty
                        # then preprocessing hasn't been applied
                        if not prepro_list:
                            # add no preprocessing template part
                            process_and_write_to_file(DynamicReportManager.CA_DICT, 'preprocessing',
                                                      field, content_of_field,
                                                      text_extract, self.file_destination)
                        else:
                            # add all file latex corresponding to preprocessing techniques used
                            for prep in prepro_list:
                                process_and_write_to_file(DynamicReportManager.CA_DICT, prep,
                                                          field, content_of_field,
                                                          text_extract, self.file_destination)
                    # dealing with postprocessing part
                    else:
                        postpro_list = super().get_subkeys_at_path(self.ca_report_dict,
                                                                   "field_representations",
                                                                   field, process)
                        # check if list empty then postprocessing hasn't been applied
                        if not postpro_list:
                            # add no postprocessing template part
                            process_and_write_to_file(DynamicReportManager.CA_DICT, 'postprocessing',
                                                      field, content_of_field,
                                                      text_extract, self.file_destination)
                        else:
                            # add general template post_processing
                            process_and_write_to_file(DynamicReportManager.CA_DICT, 'pst',
                                                      field, content_of_field,
                                                      text_extract, self.file_destination)

        # closing the content analyzer section
        add_single_mini_template(DynamicReportManager.CA_DICT, 'end',
                                 self.file_destination, content_of_field, text_extract)

        # dealing with recsys report template
        add_single_mini_template(DynamicReportManager.RS_DICT, 'recsys',
                                 self.file_destination, content_of_field, text_extract)

        # dealing with eva report template
        add_single_mini_template(DynamicReportManager.EVA_DICT, 'intro',
                                 self.file_destination, content_of_field, text_extract)

        # add as mini template all the possible fold used during the training of the recommender system
        if 'sys_results' in self.eva_report_dict:
            result_fold_list = super().get_keys_at_level(self.eva_report_dict, 'sys_results')

            if not result_fold_list:
                print("result list is empty, no result on the partition, check eva report yml file.")
            else:
                for res in result_fold_list:
                    process_and_write_to_file(DynamicReportManager.EVA_DICT, 'result',
                                              res, content_of_field,
                                              text_extract, self.file_destination)

        # closing eva report section
        add_single_mini_template(DynamicReportManager.EVA_DICT, 'end',
                                 self.file_destination, content_of_field, text_extract)

        # dealing with conclusion
        add_single_mini_template(DynamicReportManager.REP_DICT, 'end',
                                 self.file_destination, content_of_field, text_extract)
    """

    def build_template_file_simplex(self):
        # support functions to adding the text to the finale latex file, after processing and change placeholder
        # with the object needed.
        def add_single_mini_template(path_map, access_key, path_destiny, content_of_field_container,
                                     text_extract_container):
            content = self.read_file_latex(path_map.get(access_key))
            text = self.get_text_from_latex(content)
            # check point
            if text:
                self.write_on_file_latex(text)
                print(f"Content of {path_map.get(access_key)} added to {path_destiny} successfully")
            else:
                print(f"Impossible to extract text from LaTeX document: {path_map.get(access_key)}")

            # updating the external parameter
            content_of_field_container[0] = content
            text_extract_container[0] = text

        def process_and_write_to_file(path_dict, pros, scope, content_container, text_container, file_path):
            # we are extracting the written part of the latex file which is retrieved from
            # the dictionary which contain for each key the path of the file that needs to be added
            content = self.read_file_latex(path_dict.get(pros))

            # This passage allows to specify the field in the report and prepare text
            content = content.replace('X', scope)
            text = self.get_text_from_latex(content)

            # add text to file after check point
            if text:
                self.write_on_file_latex(text)
                print(f"Content of {path_dict.get(pros)} added to {self.file_destination} successfully")
            else:
                print(f"Impossible to extract text from LaTeX document: {path_dict.get(pros)}")
            # update external parameters
            content_container[0] = content
            text_container[0] = text

        # used to add mini template at the final template latex
        content_of_field = [""]
        text_extract = [""]

        # first part of the report intro of the experiment report template
        # add_single_mini_template(REPORT_DICT, 'intro', file_destination, content_of_field, text_extract)
        process_and_write_to_file(DynamicReportManager.REP_DICT, 'intro', super().get_current_date_string(),
                                  content_of_field, text_extract, self.file_destination)

        # dealing with Content Analyzer
        add_single_mini_template(DynamicReportManager.CA_DICT, 'intro', self.file_destination,
                                 content_of_field, text_extract)

        # add all the field that have been represented using the content analyzer and specify their
        # preprocessing and postprocessing received.
        if 'source_file' in self.ca_report_dict:

            # extraction of the field being analyzed
            list_of_field = super().get_keys_at_level(self.ca_report_dict, "field_representations")

            print(list_of_field)

            # dealing with all field that have been represented with content analyzer
            for field in list_of_field:
                # add the highlight field in the report
                process_and_write_to_file(DynamicReportManager.CA_DICT, 'repr',
                                          field, content_of_field, text_extract,
                                          self.file_destination)

                process_and_write_to_file(DynamicReportManager.CA_DICT, 'pre',
                                          field, content_of_field, text_extract,
                                          self.file_destination)

                process_and_write_to_file(DynamicReportManager.CA_DICT, 'post',
                                          field, content_of_field, text_extract,
                                          self.file_destination)

        # closing the content analyzer section
        add_single_mini_template(DynamicReportManager.CA_DICT, 'end',
                                 self.file_destination, content_of_field, text_extract)

        # dealing with recsys report template
        add_single_mini_template(DynamicReportManager.RS_DICT, 'recsys',
                                 self.file_destination, content_of_field, text_extract)

        # dealing with eva report template
        add_single_mini_template(DynamicReportManager.EVA_DICT, 'intro',
                                 self.file_destination, content_of_field, text_extract)

        # add as mini template all the possible fold used during the training of the recommender system
        if 'sys_results' in self.eva_report_dict:
            result_fold_list = super().get_keys_at_level(self.eva_report_dict, 'sys_results')

            if not result_fold_list:
                print("result list is empty, no result on the partition, check eva report yml file.")
            else:
                for res in result_fold_list:
                    process_and_write_to_file(DynamicReportManager.EVA_DICT, 'result',
                                              res, content_of_field,
                                              text_extract, self.file_destination)

        # closing eva report section
        add_single_mini_template(DynamicReportManager.EVA_DICT, 'end',
                                 self.file_destination, content_of_field, text_extract)

        # dealing with conclusion
        add_single_mini_template(DynamicReportManager.REP_DICT, 'end',
                                 self.file_destination, content_of_field, text_extract)
    def generate_pdf_report(self, latex_file_path, output_folder=None):
        try:
            # Extract the name and extension of the LaTeX file.
            latex_file_name, _ = os.path.splitext(os.path.basename(latex_file_path))

            # Build the path of the PDF file in the output folder or in the same folder as the LaTeX file
            pdf_file_path = os.path.join(output_folder,
                                         f"tex_to_pdf_report.pdf") if output_folder else f"{latex_file_name}_to_pdf_report.pdf"

            # Copy the LaTeX file to the output folder if specified
            if output_folder:
                output_latex_path = os.path.join(output_folder, f"{latex_file_name}_copied.tex")
                shutil.copy2(latex_file_path, output_latex_path)
            else:
                output_latex_path = latex_file_path

            # Compile LaTeX file
            subprocess.run(['pdflatex', '-interaction=nonstopmode', output_latex_path])

            # Move the PDF file to the output folder with the specified name
            os.rename(f"{output_latex_path[:-4]}.pdf", pdf_file_path)

            # Return the path of the PDF file
            return pdf_file_path

        except Exception as e:
            print(f"Error during PDF generation: {e}")
            return None


# la parte nuova per la gestione del post processing
"""
# dealing with postprocessing part
else:
    postpro_list_super = super().get_subkeys_at_path(self.ca_report_dict,
                                                     "field_representations",
                                                     field, process)
    print(postpro_list_super)
    key_postpros_path_list = ["field_representations", str(field), str(process)]
    postpro_list = []
    if postpro_list_super:
        postpro_list = super().merge_subkeys_for_keys(self.ca_report_dict, postpro_list_super,
                                                      key_postpros_path_list)
        print(postpro_list)

    print(postpro_list)
    # check if list empty then postprocessing hasn't been applied
    if not postpro_list:
        # add no postprocessing template part
        process_and_write_to_file(DynamicReportManager.CA_DICT, 'postprocessing',
                                  field, content_of_field,
                                  text_extract, self.file_destination)
    else:
        # add all file latex corresponding to postprocessing techniques used
        for postp in postpro_list:
            # print(postp)
            process_and_write_to_file(DynamicReportManager.CA_DICT, postp,
                                      field, content_of_field,
                                      text_extract, self.file_destination)
"""
