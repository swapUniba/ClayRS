import os
import re
import shutil
import yaml
from datetime import datetime

# NEEDED TO TEST THE SCRIPT
"""
# test with yml of amar single source
CA_YML = "../data/data_to_test/item_ca_report_nxPageRank.yml"
EVA_YML = "../data/data_to_test/eva_report_amarSingleSource.yml"
RS_YML = "../data/data_to_test/rs_report_amarSingleSource.yml"
"""
"""
# test with yml of amar double source
CA_YML = "../data/data_to_test/item_ca_report_nxPageRank.yml"
EVA_YML = "../data/data_to_test/eva_report_armarDoubleSource.yml"
RS_YML = "../data/data_to_test/rs_report_amarDoubleSource.yml"
"""

# test with yml of centroid vector
CA_YML = "../data/data_to_test/item_ca_report_nxPageRank.yml"
EVA_YML = "../data/data_to_test/eva_report_centroidVector.yml"
RS_YML = "../data/data_to_test/rs_report_centroidVector.yml"

"""
# test with yml of classifier recommender
CA_YML = "../data/data_to_test/item_ca_report_nxPageRank.yml"
EVA_YML = "../data/data_to_test/eva_report_classifierRecommender.yml"
RS_YML = "../data/data_to_test/rs_report_classifierRecommender.yml"
"""
"""
# test with yml of Index query
CA_YML = "../data/data_to_test/item_ca_report_nxPageRank.yml"
EVA_YML = "../data/data_to_test/eva_report_indexQuery.yml"
RS_YML = "../data/data_to_test/rs_report_indexQuery.yml"
"""
"""
# test with yml of linear predictor
CA_YML = "../data/data_to_test/item_ca_report_nxPageRank.yml"
EVA_YML = "../data/data_to_test/eva_report_linearPredictor.yml"
RS_YML = "../data/data_to_test/rs_report_linearPredictor.yml"
"""
"""
# PATH USED TO TEXT THE SCRIPT
CA_YML = "../data/ca_report.yml"
EVA_YML = "../data/eva_report.yml"
RS_YML = "../data/rs_report.yml"
"""

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
    'exogenous_representations': './templates_chunks/templates_ca_mini_chunks/no_exogenous_tech.tex'
}

RS_DICT = {
    'recsys': './templates_chunks/templates_rs/recsys_template_complete_new.tex'
}

EVA_DICT = {
    'intro': './templates_chunks/templates_eva_mini_chunks/intro_eva_all_metrics.tex',
    'end': './templates_chunks/templates_eva_mini_chunks/end_eva.tex',
    'result': './templates_chunks/templates_eva_mini_chunks/sys_result_on_fold_eva_new.tex'
}

REPORT_DICT = {
    'intro': './templates_chunks/intro_report_start.tex',
    'end': './templates_chunks/conclusion.tex'
}

# the list will contain all the fields that
# have been represented by the content analyzer.
list_of_fields = []

# the list will contain teh keys of the eva yaml that represent
# the fold n-th of the evaluation performed.
list_fold_eva = []


def get_current_date_string():
    """
       The function will take the current day, formatting as DD-MM-YYYY and returning as a string.

       Author:
       - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
       """
    today_date = datetime.now()
    today_date_string = today_date.strftime('%d-%m-%Y')
    return today_date_string


def get_keys_at_level(dictionary, parent_key):
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


def get_subkeys_at_path(dictionary: object, *keys_in_order: object) -> object:
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


"""
    def explore_dictionary(current_dict, current_keys):
        if len(current_keys) == 0:
            return

        if isinstance(current_dict, dict) and current_keys[0] in current_dict:
            if len(current_keys) == 1:
                subkeys_at_path.extend(current_dict[current_keys[0]].keys())
            else:
                explore_dictionary(current_dict[current_keys[0]], current_keys[1:])

    explore_dictionary(dictionary, keys_in_order)
    return subkeys_at_path
    """


def read_yaml_file(file_path):
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


def read_file_latex(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        contenuto = file.read()
    return contenuto


def write_on_file_latex(testo, destinazione):
    with open(destinazione, 'a', encoding='utf-8') as file:
        file.write(testo)


def get_text_from_latex(contenuto_latex):
    pattern = re.compile(r'###(.*?)###', re.DOTALL)
    match = pattern.search(contenuto_latex)
    if match:
        return match.group(1)
    else:
        return None


def build_final_latex_file(file_destination):
    # support functions to adding the text to the finale latex file, after processing and change placeholder
    # with the object needed.
    def add_single_mini_template(path_map, access_key, path_destiny, content_of_field_container,
                                 text_extract_container):
        content = read_file_latex(path_map.get(access_key))
        text = get_text_from_latex(content)
        # check point
        if text:
            write_on_file_latex(text, path_destiny)
            print(f"Content of {path_map.get(access_key)} added to {path_destiny} successfully")
        else:
            print(f"Impossible to extract text from LaTeX document: {path_map.get(access_key)}")

        # updating the external parameter
        content_of_field_container[0] = content
        text_extract_container[0] = text

    def process_and_write_to_file(path_dict, pros, scope, content_container, text_container, file_path):
        # we are extracting the written part of the latex file which is retrieved from
        # the dictionary which contain for each key the path of the file that needs to be added
        content = read_file_latex(path_dict.get(pros))

        # This passage allows to specify the field in the report and prepare text
        content = content.replace('X', scope)
        text = get_text_from_latex(content)

        # add text to file after check point
        if text:
            write_on_file_latex(text, file_path)
            print(f"Content of {path_dict.get(pros)} added to {file_path} successfully")
        else:
            print(f"Impossible to extract text from LaTeX document: {path_dict.get(pros)}")
        # update external parameters
        content_container[0] = content
        text_container[0] = text

    # used to load the dictonary related to each yaml file
    ca_dict = read_yaml_file(CA_YML)
    rs_dict = read_yaml_file(RS_YML)
    eva_dict = read_yaml_file(EVA_YML)

    # used to add mini template at the final template latex
    content_of_field = [""]
    text_extract = [""]

    # first part of the report intro of the experiment report template
    # add_single_mini_template(REPORT_DICT, 'intro', file_destination, content_of_field, text_extract)
    process_and_write_to_file(REPORT_DICT, 'intro', get_current_date_string(),
                              content_of_field, text_extract, file_destination)

    # dealing with Content Analyzer
    add_single_mini_template(CA_DICT, 'intro', file_destination, content_of_field, text_extract)

    # add all the field that have been represented using the content analyzer and specify their
    # preprocessing and postprocessing received.
    if 'source_file' in ca_dict:

        # extraction of the field being analyzed
        list_of_field = get_keys_at_level(ca_dict, "field_representations")

        print(list_of_field)

        # dealing with all field that have been represented with content analyzer
        for field in list_of_field:
            # add the highlight field in the report
            process_and_write_to_file(CA_DICT, 'field_sec', field, content_of_field, text_extract, file_destination)

            # list of primary subkey of key field
            processing_list = get_keys_at_level(ca_dict, field)

            print(processing_list)

            # dealing with all process applied to a specific field --> dealing with field subkeys
            for process in processing_list:
                # this first check allow to identify the techniques for field representation
                if process != 'preprocessing' and process != 'postprocessing':
                    process_and_write_to_file(CA_DICT, process, field, content_of_field, text_extract, file_destination)
                # dealing with preprocessing part
                elif process == 'preprocessing':
                    prepro_list = get_subkeys_at_path(ca_dict, "field_representations", field, process)

                    # check if list empty then preprocessing hasn't been applied
                    if not prepro_list:
                        # add no preprocessing template part
                        process_and_write_to_file(CA_DICT, 'preprocessing', field, content_of_field,
                                                  text_extract, file_destination)
                    else:
                        # add all file latex corresponding to preprocessing techniques used
                        for prep in prepro_list:
                            process_and_write_to_file(CA_DICT, prep, field, content_of_field,
                                                      text_extract, file_destination)
                # dealing with postprocessing part
                else:
                    postpro_list = get_subkeys_at_path(ca_dict, "field_representations", field, process)

                    # check if list empty then postprocessing hasn't been applied
                    if not postpro_list:
                        # add no postprocessing template part
                        process_and_write_to_file(CA_DICT, 'postprocessing', field, content_of_field,
                                                  text_extract, file_destination)
                    else:
                        # add all file latex corresponding to postprocessing techniques used
                        for postp in postpro_list:
                            process_and_write_to_file(CA_DICT, postp, field, content_of_field,
                                                      text_extract, file_destination)

    # closing the content analyzer section
    add_single_mini_template(CA_DICT, 'end', file_destination, content_of_field, text_extract)

    # dealing with recsys report template
    add_single_mini_template(RS_DICT, 'recsys', file_destination, content_of_field, text_extract)

    # dealing with eva report template
    add_single_mini_template(EVA_DICT, 'intro', file_destination, content_of_field, text_extract)

    # add as mini template all the possible fold used during the training of the recommender system
    if 'sys_results' in eva_dict:
        result_fold_list = get_keys_at_level(eva_dict, 'sys_results')

        if not result_fold_list:
            print("result list is empty, no result on the partition, check eva report yml file.")
        else:
            for res in result_fold_list:
                process_and_write_to_file(EVA_DICT, 'result', res, content_of_field, text_extract, file_destination)

    # closing eva report section
    add_single_mini_template(EVA_DICT, 'end', file_destination, content_of_field, text_extract)

    # dealing with conclusion
    add_single_mini_template(REPORT_DICT, 'end', file_destination, content_of_field, text_extract)


# RUN script
if __name__ == "__main__":
    build_final_latex_file("./dynamic_fin_rep.tex")
    print()
