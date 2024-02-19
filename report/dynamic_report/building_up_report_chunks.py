# This script wil contain a bunch of function able to deal with renderization of chunks of the report
# in order to make flessibile the process of creating a report in latex from yml file, basically the function
# will make some adjustment of basic template for each section rendering this section with jinja2 lib and once
# rendered the output will be used as a string with function of the support_reporting_lib in order to add this
# string well-formed on the file.latex that will be the report of the experiment conducted
import os
import re
import yaml
import jinja2
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import table_comparison as tbl_comp

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
    'pre_min': './templates_chunks/templates_ca_mini_chunks/preprocess_field_minimised.tex',
    'post': './templates_chunks/templates_ca_mini_chunks/postprocessing_field.tex',
    'post_min': './templates_chunks/templates_ca_mini_chunks/postprocessing_field_minimised.tex',
    'repr': './templates_chunks/templates_ca_mini_chunks/content_representation_field.tex',
    'repr_min': './templates_chunks/templates_ca_mini_chunks/content_representation_field_minimised.tex',
    'exo': './templates_chunks/templates_ca_mini_chunks/exogenous_tech_report_ca.tex',
    'exo_min': './templates_chunks/templates_ca_mini_chunks/exogenous_tech_report_ca_min.tex',
    'dataset': './templates_chunks/templates_ca_mini_chunks/Daset_name_source.tex',
    'stats': './templates_chunks/templates_ca_mini_chunks/Dataset_Stat.tex',
    'field_flat': './templates_chunks/templates_ca_mini_chunks/field_ca_report_flat.tex',
    'field_rep_flat': './templates_chunks/templates_ca_mini_chunks/field_represantation_ca_report_flat.tex',
    'field_prep_flat': './templates_chunks/templates_ca_mini_chunks/field_preprocess_ca_report_falt.tex',
    'field_post_flat': './templates_chunks/templates_ca_mini_chunks/field_postprocessing_ca_report_flat.tex'
}

# dictionary to find path for the recsys module template
RS_DICT = {
    'recsys': './templates_chunks/templates_rs/recsys_template_complete_new.tex',
    'general_rec': './templates_chunks/templates_rs/recsys_general.tex',
    'split': './templates_chunks/templates_rs/split_technique_on_data.tex',
    'starting_sec': './templates_chunks/templates_rs/starting_sec_recsys.tex',
    'starting_sec_flat': './templates_chunks/templates_rs/starting_sec_recsys_flat.tex',
    'algo': './templates_chunks/templates_rs/algorithm_used_recsys.tex',
    'algo_flat': './templates_chunks/templates_rs/algorithm_used_recsys_flat.tex'
}

# dictionary to find path for the evaluation module template
EVA_DICT = {
    'intro': './templates_chunks/templates_eva_mini_chunks/intro_eva_all_metrics.tex',
    'intro_min': './templates_chunks/templates_eva_mini_chunks/intro_eva_all_metrics_minimised.tex',
    'intro_flat': './templates_chunks/templates_eva_mini_chunks/intro_eva_all_metrics_flat.tex',
    'end': './templates_chunks/templates_eva_mini_chunks/end_eva.tex',
    'result': './templates_chunks/templates_eva_mini_chunks/sys_result_on_fold_eva_new.tex',
    'sys - mean': './templates_chunks/templates_eva_mini_chunks/sys_mean_result.tex',
    'no_res': './templates_chunks/templates_eva_mini_chunks/no_results_on_fold.tex',
    'comparison_intro': './templates_chunks/templates_eva_mini_chunks/comparison_algo_section.tex',
    'stats_rel_intro': './templates_chunks/templates_eva_mini_chunks/stats_relevance_subsection.tex'
}

# dictionary to find path for template used to start and complete the report
REP_DICT = {
    'intro': './templates_chunks/intro_report_start.tex',
    'end': './templates_chunks/conclusion.tex'
}


def merge_yaml_files(input_paths_list, output_folder, output_filename):
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
        data = get_data(input_path)
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


def get_data(rendering_file):
    with open(rendering_file, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)


def read_yaml_file(file_path):
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


def retrieve_subkeys_at_3(dictionary, key1, key2, key3):
    subkeys = set()

    if key1 in dictionary and isinstance(dictionary[key1], dict):
        if key2 in dictionary[key1] and isinstance(dictionary[key1][key2], dict):
            if key3 in dictionary[key1][key2] and isinstance(dictionary[key1][key2][key3], dict):
                subkeys.update(dictionary[key1][key2][key3].keys())

    return list(subkeys)


def retrieve_subkeys_at_4(dictionary, key1, key2, key3, key4):
    subkeys = set()

    if key1 in dictionary and isinstance(dictionary[key1], dict):
        if key2 in dictionary[key1] and isinstance(dictionary[key1][key2], dict):
            if key3 in dictionary[key1][key2] and isinstance(dictionary[key1][key2][key3], dict):
                if key4 in dictionary[key1][key2][key3] and isinstance(dictionary[key1][key2][key3][key4], dict):
                    subkeys.update(dictionary[key1][key2][key3][key4].keys())

    return list(subkeys)


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


def build_relative_path(folder, file_name):
    # Unisci la stringa della cartella con il nome del file
    relative_path = os.path.join(folder, file_name)

    # Ritorna il percorso relativo completo
    return relative_path


def ca_processing_report_minimal(render_dict, working_path, file_path):
    # relative path complete
    file_destination = build_relative_path(working_path, file_path)

    # used to add mini template at the final template latex
    content_of_field = [""]
    text_extract = [""]

    # dealing with Content Analyzer
    add_single_mini_template(CA_DICT, 'intro', file_destination,
                             content_of_field, text_extract)

    # add all the field that have been represented using the content analyzer and specify their
    # preprocessing and postprocessing received.
    if render_dict is not None:
        if 'source_file' in render_dict:
            # extraction of the field being analyzed
            list_of_field = get_keys_at_level(render_dict, "field_representations")

            print(list_of_field)

            # dealing with all field that have been represented with content analyzer
            for field in list_of_field:
                # add the highlight field in the report
                process_and_write_to_file(CA_DICT, 'repr_min',
                                          field, content_of_field, text_extract,
                                          file_destination)

                process_and_write_to_file(CA_DICT, 'pre_min',
                                          field, content_of_field, text_extract,
                                          file_destination)

                process_and_write_to_file(CA_DICT, 'post_min',
                                          field, content_of_field, text_extract,
                                          file_destination)

    # adding reporting on exogenous techniques
    add_single_mini_template(CA_DICT, 'exo_min', file_destination,
                             content_of_field, text_extract)

    # closing the content analyzer section
    add_single_mini_template(CA_DICT, 'end',
                             file_destination, content_of_field, text_extract)


def ca_processing_report_flat(render_dict, working_path, file_path):
    # relative path complete
    file_destination = build_relative_path(working_path, file_path)

    def stringify(strings):
        # Verifica se la lista è vuota
        if not strings:
            return ""

        # Verifica se tutti gli elementi della lista sono stringhe o possono essere convertiti in stringhe
        for index, item in enumerate(strings):
            if not isinstance(item, str):
                try:
                    # Tenta di convertire l'elemento in stringa
                    strings[index] = str(item)
                except:
                    # Se la conversione fallisce, restituisci una stringa vuota
                    return ""

        # Unisci gli elementi della lista (ora convertiti in stringhe, se necessario) in una stringa
        return ', '.join(strings)

    def delete_copy(reduntant_list):
        # Utilizziamo un insieme per tenere traccia degli elementi unici
        unique_set = set()
        # Utilizziamo una lista per mantenere l'ordine originale
        result = []

        for elemento in reduntant_list:
            # Aggiungiamo l'elemento all'insieme solo se non è già presente
            if elemento not in unique_set:
                unique_set.add(elemento)
                # Aggiungiamo l'elemento alla lista risultato
                result.append(elemento)

        return result

    def process_list_of_strings(list_of_strings):
        processed_strings = []
        for string in list_of_strings:
            processed_string = tbl_comp.sanitize_latex_string(string)
            processed_strings.append(processed_string)
        return processed_strings

    field_list = []
    represantation_tech_list = []
    preprocessing_list = []
    postprocessing_list = []

    if render_dict is not None:
        if 'source_file' in render_dict:
            # extraction of the field being analyzed
            field_list = get_keys_at_level(render_dict, "field_representations")

            print(field_list)

            # dealing with all field that have been represented with content analyzer
            for field in field_list:
                elaborating_list = get_keys_at_level(render_dict, field)
                print(f" Questa è elaborating list per ogni campo {elaborating_list}")
                for process in elaborating_list:
                    if process != "preprocessing" and process != "postprocessing":
                        represantation_tech_list.append(process)
                        print(f"added to representation_tech_list la rapresentazione {process}")
                    elif process == "preprocessing":
                        prep_list = retrieve_subkeys_at_3(render_dict, "field_representations", key2=field,
                                                          key3=process)
                        print(f"Questa è la lista delle chiavi sotto preprocessing: {prep_list}")
                        if prep_list:
                            for p in prep_list:
                                preprocessing_list.append(p)
                    else:
                        post_list = retrieve_subkeys_at_3(render_dict, "field_representations", key2=field,
                                                          key3=process)
                        print(f"Questa è la lista delle chiavi sotto postprocessing: {post_list}")
                        if post_list:
                            for p in post_list:
                                if isinstance(p, int):
                                    post_processeing_technique = retrieve_subkeys_at_4(render_dict,
                                                                                       "field_representations",
                                                                                       key2=field,
                                                                                       key3=process,
                                                                                       key4=p)
                                    for t in post_processeing_technique:
                                         postprocessing_list.append(t)
                                else:
                                    if isinstance(p, str):
                                        postprocessing_list.append(p)

    str_fields = stringify(process_list_of_strings(delete_copy(field_list)))
    str_rapresentation = stringify(process_list_of_strings(delete_copy(represantation_tech_list)))
    str_prep = stringify(process_list_of_strings(delete_copy(preprocessing_list)))
    str_post = stringify(process_list_of_strings(delete_copy(postprocessing_list)))

    # used to add mini template at the final template latex
    content_of_field = [""]
    text_extract = [""]

    # dealing with Content Analyzer
    add_single_mini_template(CA_DICT, 'intro', file_destination,
                             content_of_field, text_extract)

    process_and_write_to_file(CA_DICT, 'field_flat',
                              str_fields, content_of_field, text_extract,
                              file_destination)

    process_and_write_to_file(CA_DICT, 'field_rep_flat',
                              str_rapresentation, content_of_field, text_extract,
                              file_destination)

    process_and_write_to_file(CA_DICT, 'field_prep_flat',
                              str_prep, content_of_field, text_extract,
                              file_destination)

    process_and_write_to_file(CA_DICT, 'field_post_flat',
                              str_post, content_of_field, text_extract,
                              file_destination)

    # closing the content analyzer section
    add_single_mini_template(CA_DICT, 'end',
                             file_destination, content_of_field, text_extract)


def ca_processing_report_verb(render_dict, working_path, file_path):
    # relative path complete
    file_destination = build_relative_path(working_path, file_path)

    # used to add mini template at the final template latex
    content_of_field = [""]
    text_extract = [""]

    # dealing with Content Analyzer
    add_single_mini_template(CA_DICT, 'intro', file_destination,
                             content_of_field, text_extract)

    # add all the field that have been represented using the content analyzer and specify their
    # preprocessing and postprocessing received.
    if render_dict is not None:
        if 'source_file' in render_dict:
            # extraction of the field being analyzed
            list_of_field = get_keys_at_level(render_dict, "field_representations")

            print(list_of_field)

            # dealing with all field that have been represented with content analyzer
            for field in list_of_field:
                # add the highlight field in the report
                process_and_write_to_file(CA_DICT, 'repr',
                                          field, content_of_field, text_extract,
                                          file_destination)

                process_and_write_to_file(CA_DICT, 'pre',
                                          field, content_of_field, text_extract,
                                          file_destination)

                process_and_write_to_file(CA_DICT, 'post',
                                          field, content_of_field, text_extract,
                                          file_destination)

    # adding reporting on exogenous techniques
    add_single_mini_template(CA_DICT, 'exo', file_destination,
                             content_of_field, text_extract)

    # closing the content analyzer section
    add_single_mini_template(CA_DICT, 'end',
                             file_destination, content_of_field, text_extract)


def data_statistic_report(render_dict, name_of_dataset, working_path, file_path):
    # relative path complete
    file_destination = build_relative_path(working_path, file_path)

    # used to add mini template at the final template latex
    content_of_field = [""]
    text_extract = [""]

    # dealing with subsection of dataset and its statistics
    process_and_write_to_file(CA_DICT, 'dataset', name_of_dataset,
                              content_of_field, text_extract,
                              file_destination)

    # add stats table
    add_single_mini_template(CA_DICT, 'stats',
                             file_destination, content_of_field,
                             text_extract)


def splitting_technique_report(render_dict, working_path, file_path):
    # relative path complete
    file_destination = build_relative_path(working_path, file_path)

    # used to add mini template at the final template latex
    content_of_field = [""]
    text_extract = [""]

    add_single_mini_template(RS_DICT, 'split',
                             file_destination, content_of_field,
                             text_extract)


def make_content_analyzer_sec(render_dict, name_of_dataset="no name", mode="minimise", working_path="working_dir"):
    if mode not in ["flat", "minimise", "verbose"]:
        raise ValueError("Parameter Error: 'mode' can be only 'falt', 'minimise' or 'verbose'.")

    # Crea il nome del file che farà da template per la renderizzazione di questa
    # parte di report che stiamo andando a produrre
    file_name = "ca_report_latex.tex"
    file_path = os.path.join(working_path, file_name)
    print(file_path)

    if mode == "flat":
        ca_processing_report_flat(render_dict, working_path, file_name)
    elif mode == "minimise":
        ca_processing_report_minimal(render_dict, working_path, file_name)
    else:
        # scelta della funzione in base alla verbosità del report che si vuole ottenere
        # a riguardo delle elaborazioni fatte dal content analyzer sui dati
        ca_processing_report_verb(render_dict, working_path, file_name)
        # print()

    # procediamo con l'inserimento della tabella statistica sui dati
    data_statistic_report(render_dict, name_of_dataset, working_path, file_name)

    # aggiungiamo la sezione di spit del dataset
    splitting_technique_report(render_dict, working_path, file_name)

    # Ritorna il percorso di lavoro e il nome del file creato
    return working_path, file_name


def make_recsys_sec(dict_render, insert_intro=True, mode="flat", working_path="working_dir"):
    def create_file_name(base_name, descriptor):
        return f"{base_name}_{descriptor}.tex"

    if mode not in ["flat", "verbose"]:
        raise ValueError("Il parametro 'mode' può essere solo 'flat' o 'verbose'.")

    # estrazione del nome dell'algoritmo che usiamo
    algo_name = get_keys_at_level(dict_render, 'algorithm')
    # print(f"questo è il nome dell'algoritmo utilizzato: {algo_name}")

    # Crea il nome del file che farà da template per la renderizzazione di questa
    # parte di report che stiamo andando a produrre
    file_name = create_file_name("recsys_report", algo_name[0])
    file_path = os.path.join(working_path, file_name)
    # print(file_path)

    # used to add mini template at the final template latex
    content_of_field = [""]
    text_extract = [""]

    if insert_intro:
        # adding intro of recsys section
        add_single_mini_template(RS_DICT, 'starting_sec_flat',
                                 file_path, content_of_field,
                                 text_extract)

    process_and_write_to_file(RS_DICT, 'algo_flat', algo_name[0],
                              content_of_field, text_extract,
                              file_path)

    if mode != "flat":
        if insert_intro:
            # adding intro of recsys section
            add_single_mini_template(RS_DICT, 'starting_sec',
                                     file_path, content_of_field,
                                     text_extract)

        process_and_write_to_file(RS_DICT, 'algo', algo_name[0],
                                  content_of_field, text_extract,
                                  file_path)

    # Ritorna il percorso di lavoro e il nome del file creato
    return working_path, file_name


def make_eval_metric_sec(dict_render, mode="minimised", working_path="working_dir"):
    if mode not in ["flat", "minimised", "verbose"]:
        raise ValueError("Il parametro 'mode' può essere solo 'flat' o 'verbose'.")

    # Crea il nome del file che farà da template per la renderizzazione di questa
    # parte di report che stiamo andando a produrre
    file_name = "eval_metric_report_latex.tex"
    file_path = os.path.join(working_path, file_name)
    # print(file_path)

    # used to add mini template at the final template latex
    content_of_field = [""]
    text_extract = [""]

    if mode == "flat":
        # solo elenco delle metriche usate
        add_single_mini_template(EVA_DICT, 'intro_flat',
                                 file_path, content_of_field,
                                 text_extract)

    if mode == "minimised":
        # report metric minimizzato
        add_single_mini_template(EVA_DICT, 'intro_min',
                                 file_path, content_of_field,
                                 text_extract)

    if mode == "verbose":
        add_single_mini_template(EVA_DICT, 'intro',
                                 file_path, content_of_field,
                                 text_extract)

    add_single_mini_template(EVA_DICT, 'end',
                             file_path, content_of_field,
                             text_extract)

    # Ritorna il percorso di lavoro e il nome del file creato
    return working_path, file_name


def make_eval_result_sec(dict_render, working_path="working_dir"):
    def create_file_name(base_name, descriptor):
        return f"{base_name}_{descriptor}.tex"

    # estrazione del nome dell'algoritmo che usiamo
    algo_name = get_keys_at_level(dict_render, 'algorithm')
    # print(f"questo è il nome dell'algoritmo utilizzato: {algo_name}")

    # Crea il nome del file che farà da template per la renderizzazione di questa
    # parte di report che stiamo andando a produrre
    file_name = create_file_name("mean_res", algo_name[0])
    file_path = os.path.join(working_path, file_name)
    # print(file_path)

    # used to add mini template at the final template latex
    content_of_field = [""]
    text_extract = [""]

    process_and_write_to_file(EVA_DICT, 'sys - mean', algo_name[0],
                              content_of_field, text_extract,
                              file_path)

    # Ritorna il percorso di lavoro e il nome del file creato
    return working_path, file_name


def load_and_add_comparison_table_with_relevance(eva_yaml_paths, recsys_yaml_paths, data_frame_ref, pv_ref):
    # set list of dictionary from eva yaml report and make key change
    dictionary_res_sys = tbl_comp.from_yaml_list_to_dict_list(eva_yaml_paths)
    # print(dictionary_res_sys)
    dict_sys_mean = []
    for e in dictionary_res_sys:
        tmp = tbl_comp.extract_subdictionary(e, "sys - mean")
        dict_sys_mean.append(tmp)

    # clean key unused from dictionary list
    my_dictio = tbl_comp.remove_key_from_nested_dicts(dict_sys_mean, "CatalogCoverage (PredictionCov)")
    """
    for i in dict_sys_mean:
        print(i)
    print("\n")
    """

    # get dictionary from the recsys ymal and extract a list of key with name of algorithm used
    dictionarylist = tbl_comp.from_yaml_list_to_dict_list(recsys_yaml_paths)
    """
    for r in dictionarylist:
        print(i)
    """

    keys = tbl_comp.get_algorithm_keys(dictionarylist)  # lista dei nomi degli algoritmi usati
    print(f"\nLa LISTA CONTENENTE I NOMI DEGLI ALGORITMI ESTRATTI DAI YML FILE\n {keys} \n\n")

    # show the dictonary extracted after processing them
    result_dictionary = tbl_comp.nest_dictionaries(keys, my_dictio)
    print("I DIZIONARI USATI PER CREARE LA LISTA DA DARE IN INPUT PER LA CREAZIONE DELLA TABELLA:")
    for r in result_dictionary:
        print(r)

    # andiamo a caricare il dataframe che sarà usato per il confronto dei p-value
    # partiamo con la creazione del dizionario di mapping per i nomi dei sistemi
    system_map = tbl_comp.list_to_dict_system_map(keys)
    print(f"\n\nIL DIZIONARIO USATO PER IL MAPPING DEI SISTEMI \n SYSTEM MAP: {system_map} \n\n")

    # effettuiamo le modifiche degli indici di accesso per riga al dataframe caricato p_value_ref_df
    p_value_ref_df = tbl_comp.stt.change_system_name(data_frame_ref, system_map)
    print(f"\n\nDATAFRAME CON INDICI DI RIGA MODIFICATI\n {p_value_ref_df}")

    # generate_latex_table_pvalued(algorithms, stats_rel, comparison="", treshold_pvalue=0.5,
    #                                  decimal_place=3, column_width=3.0,
    #                                  max_columns_per_part=5, caption_for_table="Comparison between algorithms")
    # ora in base alla funzione che stiamo testando di cui abbiamo riportato sopra la signature andiammo a visualizzare
    # e comprenderre chi sono i parametri che gli andremo a passare prima di effettuare la chiamata
    print("ELENCO E VISULIZZAZIONE DEI PARAMETRI CHE PASSEREMO A generate_latex_table_pvalued:")
    print(f"\n1. algorithms ovvero la lista di dizionari creata a partire da i file yml sarà:\n"
          f"result_dictionary ovvero:\n {result_dictionary} \n")
    print(f"\n2. stats_rel sarà il dataframe caricato e modificato opportunamente:\n"
          f"p_value_ref_df:\n {p_value_ref_df} \n")
    reference_alg = 'CentroidVector'
    print(f"\n3. comparison sarà uno tra i 5 algoritmi presenti ed utilizzati in particolare:\n"
          f"reference_alg è {reference_alg} \n")

    print(f"\n4. treshold_pvalue=0.5 di default in questo caso settato con:\n"
          f"pv_ref = {pv_ref} \n")
    print(f"\n5. i restanti paremetri saranno usati di default o cambiati all'interno della chiamata.\n\n")

    # with the dictnory processed create the latex table
    latex_table = tbl_comp.generate_latex_table_pvalued(result_dictionary, p_value_ref_df, reference_alg,
                                                        pv_ref, max_columns_per_part=3)
    return latex_table


def prepare_data_frame_for_stats_relevance(recsys_yaml_paths, data_frame_ref, reference_mode):
    if reference_mode not in ["complete", "statistic", "pvalue"]:
        raise ValueError("Parameter Error: reference_mode can be only complete or statistic or pvalue")

    # get dictionary from the recsys yml and extract a list of key with name of algorithm used
    dictionarylist = tbl_comp.from_yaml_list_to_dict_list(recsys_yaml_paths)
    """
    for r in dictionarylist:
        print(i)
    """
    keys = tbl_comp.get_algorithm_keys(dictionarylist)  # lista dei nomi degli algoritmi usati
    # print(f"\nLa LISTA CONTENENTE I NOMI DEGLI ALGORITMI ESTRATTI DAI YML FILE\n {keys} \n\n")

    # andiamo a caricare il dataframe che sarà usato per il confronto dei p-value
    # partiamo con la creazione del dizionario di mapping per i nomi dei sistemi
    system_map = tbl_comp.list_to_dict_system_map(keys)
    # print(f"\n\nIL DIZIONARIO USATO PER IL MAPPING DEI SISTEMI \n SYSTEM MAP: {system_map} \n\n")

    # effettuiamo le modifiche degli indici di accesso per riga al dataframe caricato p_value_ref_df
    stats_relevant_df = tbl_comp.stt.change_system_name(data_frame_ref, system_map)
    # print(f"\n\nDATAFRAME CON INDICI DI RIGA MODIFICATI\n {p_value_ref_df}")

    if reference_mode == "pvalue":
        # apportiamo le modifiche sul dataframe per rimuovere le colonne che contengono le statistiche
        stats_relevant_df = tbl_comp.stt.remove_stats_from_df(stats_relevant_df, 'statistic')
    elif reference_mode == "statistic":
        stats_relevant_df = tbl_comp.stt.remove_stats_from_df(stats_relevant_df, 'pvalue')
    else:
        pass

    return stats_relevant_df


def load_and_add_statistic_relevance_table(recsys_yaml_paths, data_frame_ref, reference_mode, n_col, tab_title):
    p_value_ref_df = prepare_data_frame_for_stats_relevance(recsys_yaml_paths, data_frame_ref, reference_mode)

    # Adesso chiamiamo la funzione di stampa per la tabella latex
    stats_reverence_table = tbl_comp.stt.from_dataframe_to_latex_table_second(p_value_ref_df,
                                                                              n_col,
                                                                              tab_title)

    return stats_reverence_table


def make_statistical_relevance_subsection(render_dict, table_to_add, only_table=False, working_path="working_dir"):
    # Crea il nome del file che farà da template per la renderizzazione di questa
    # parte di report che stiamo andando a produrre
    file_name = "statistical_relevance.tex"
    file_path = os.path.join(working_path, file_name)
    # print(file_path)

    # used to add mini template at the final template latex
    content_of_field = [""]
    text_extract = [""]

    if not only_table:
        # introduction of the stats relevance subsection
        add_single_mini_template(EVA_DICT, 'stats_rel_intro', file_path,
                                 content_of_field, text_extract)

    write_on_file_latex(table_to_add, file_path)

    # Ritorna il percorso di lavoro e il nome del file creato
    return working_path, file_name


def load_and_add_statistic_relevance_tab_single_comparison(recsys_yaml_paths, data_frame_ref,
                                                           reference_mode, idx,
                                                           tab_title, sci_not, approximation):
    p_value_ref_df = prepare_data_frame_for_stats_relevance(recsys_yaml_paths, data_frame_ref, reference_mode)

    pair_comparison_strats_relevance_tab = tbl_comp.stt.stats_relevance_tab(p_value_ref_df, idx[0][0],
                                                                            tab_title,
                                                                            sci_not, approximation)

    return pair_comparison_strats_relevance_tab


def make_comparison_algo_sec(dict_render, eva_yaml_paths, recsys_yaml_paths,
                             data_frame_ref, pv_treshold, only_table=False, working_path="working_dir"):
    # Crea il nome del file che farà da template per la renderizzazione di questa
    # parte di report che stiamo andando a produrre
    file_name = "comparison_algo_table.tex"
    file_path = os.path.join(working_path, file_name)
    # print(file_path)

    # used to add mini template at the final template latex
    content_of_field = [""]
    text_extract = [""]

    if not only_table:
        # introduction of the comparison section
        add_single_mini_template(EVA_DICT, 'comparison_intro', file_path,
                                 content_of_field, text_extract)

    # dealing with tab
    table_comparison_latex = load_and_add_comparison_table_with_relevance(eva_yaml_paths,
                                                                          recsys_yaml_paths,
                                                                          data_frame_ref, pv_treshold)
    write_on_file_latex(table_comparison_latex, file_path)

    # Ritorna il percorso di lavoro e il nome del file creato
    return working_path, file_name


def render_latex_template(template_name, search_path, my_dict):
    # setting environment based on latex needs
    latex_jinja_env = jinja2.Environment(
        block_start_string="\BLOCK{",
        block_end_string="}",
        variable_start_string="\VAR{",
        variable_end_string="}",
        comment_start_string="\#{",
        comment_end_string="}",
        trim_blocks=True,
        autoescape=False,
        loader=jinja2.FileSystemLoader(searchpath=search_path),
    )

    def safe_text(text: str) -> str:
        special_chars = ['&', '%', '$', '_', '{', '}', '#']
        for char in special_chars:
            text = str(text)
            text = text.replace(char, "\\" + char)
        return text

    def truncate(text: str) -> str:
        # Verifica se text non è nullo o None
        if text is None or text == '':
            return None  # o qualsiasi altro valore di default che desideri restituire

        # Verifica se text è una stringa che può essere convertita in float
        try:
            number = float(text)
        except (ValueError, TypeError):
            # Se non può essere convertito in float, restituisci il valore originale
            return str(text)

        # Esegui il codice di troncamento se text è un numero
        number = round(number, 5)
        text = str(number)
        return text

    # adding filter to the environment
    latex_jinja_env.filters["safe_text"] = safe_text
    latex_jinja_env.filters["truncate"] = truncate

    # Load the template
    template = latex_jinja_env.get_template(template_name)

    # Render the template using the context
    latex_output = template.render(my_dict=my_dict)

    print(f"inside my_dict {my_dict}")
    return latex_output


"""
# func1 la funzione sottostante ha un caso d'uso
def render_latex_template(template_path, context):
    # Carica l'ambiente Jinja con il caricatore di file system
    env = Environment(loader=FileSystemLoader('.'))

    # Carica il template LaTeX
    template = env.get_template(template_path)

    # Renderizza il template utilizzando il contesto fornito
    latex_output = template.render(context)

    return latex_output
"""

# Esempio di utilizzo
if __name__ == "__main__":
    # Caso di utilizzo e test della pipeline per la renderizzazione
    # test with yml of centroid vector
    CA_YML = "../data/data_to_test/item_ca_report_nxPageRank.yml"
    EVA_YML = "../data/data_to_test/eva_report_centroidVector.yml"
    EVA_YML2 = "../data/data_to_test/eva_report_linearPredictor.yml"
    EVA_YML3 = "../data/data_to_test/eva_report_indexQuery.yml"
    EVA_YML4 = "../data/data_to_test/eva_report_classifierRecommender.yml"
    EVA_YML5 = "../data/data_to_test/eva_report_armarDoubleSource.yml"
    RS_YML = "../data/data_to_test/rs_report_centroidVector.yml"
    RS_YML2 = "../data/data_to_test/rs_report_linearPredictor.yml"
    RS_YML3 = "../data/data_to_test/rs_report_indexQuery.yml"
    RS_YML4 = "../data/data_to_test/rs_report_classifierRecommender.yml"
    RS_YML5 = "../data/data_to_test/rs_report_amarDoubleSource.yml"

    # used to load the dictonary related to each yaml file
    ca_dict = read_yaml_file(CA_YML)
    rs_dict = read_yaml_file(RS_YML)
    eva_dict = read_yaml_file(EVA_YML)

    # GESTIONE DEL REPORT SEZIONE CONTENT ANALYZER
    # vado a creare un nuovo yml che userò per la renderizzazione della
    # sezione del content analyzer
    path_rendering_dict = merge_yaml_files([CA_YML, RS_YML],
                                           "working_dir",
                                           "ca_rcs_yml_union.yml")

    dict_for_render = read_yaml_file(path_rendering_dict)
    print(dict_for_render)
    
    # chiamata alle funzioni che andranno ad aggiungere il report sul content analyzer
    route_path, file_to_render = make_content_analyzer_sec(dict_for_render, name_of_dataset="1000K data video movie",
                                                           mode="flat")
    # print(route_path)
    # print(file_to_render)
    part_of_report = render_latex_template(file_to_render, route_path, dict_for_render)
    print(part_of_report)


    # preparing list of dict with the information on the recsys used
    render_list = [rs_dict]
    rs_2_dict = read_yaml_file(RS_YML2)
    rs_3_dict = read_yaml_file(RS_YML3)
    rs_4_dict = read_yaml_file(RS_YML4)
    rs_5_dict = read_yaml_file(RS_YML5)
    render_list.append(rs_2_dict)
    render_list.append(rs_3_dict)
    render_list.append(rs_4_dict)
    render_list.append(rs_5_dict)

    # GESTIONE DEL REPORT SEZIONE RECSYS E ALGORITMI USATI
    """
    first_iteration = True  # Flag per tracciare se è la prima iterazione

    for render in render_list:
        # print(render)
        if first_iteration:
            route_path, file_to_render = make_recsys_sec(render, insert_intro=True, working_path="working_dir")
            first_iteration = False  # Imposta il flag a False dopo la prima iterazione
        else:
            route_path, file_to_render = make_recsys_sec(render, insert_intro=False, working_path="working_dir")

        part_of_report = render_latex_template(file_to_render, route_path,  render)
        print(part_of_report)
    """

    # GESTIONE DEL REPORT SEZIONE METRICE DEL EVAL
    """
    # vado a creare un nuovo yml che userò per la renderizzazione della
    # sezione del content analyzer
    path_rendering_dict = merge_yaml_files([EVA_YML, RS_YML],
                                           "working_dir",
                                           "eva_rcs_yml_union.yml")

    dict_for_render_metric = read_yaml_file(path_rendering_dict)

    route_path, file_to_render = make_eval_metric_sec(dict_for_render_metric,
                                                      mode="flat", working_path="working_dir")
    # print(route_path)
    # print(file_to_render)
    part_of_report = render_latex_template(file_to_render, route_path, dict_for_render_metric)
    print(part_of_report)
    """

    # GESTIONE DEL REPORT SEZIONE RISULTATI OTTENUTI PER OGNI ALGORITMO
    """
    centroidV_eva_res_render = read_yaml_file(merge_yaml_files([EVA_YML, RS_YML],
                                                               "working_dir",
                                                               "eva_rcs_CentroidVector.yml"))

    linearP_eva_rec_render = read_yaml_file(merge_yaml_files([EVA_YML2, RS_YML2],
                                                             "working_dir",
                                                             "eva_rcs_LinearPredictor.yml"))

    indexQ_eva_rec_render = read_yaml_file(merge_yaml_files([EVA_YML3, RS_YML3],
                                                            "working_dir",
                                                            "eva_rcs_QueryIndex.yml"))

    clsR_eva_rec_render = read_yaml_file(merge_yaml_files([EVA_YML4, RS_YML4],
                                                          "working_dir",
                                                          "eva_rcs_ClassifierRecommender.yml"))

    amarDS_eva_rec_render = read_yaml_file(merge_yaml_files([EVA_YML5, RS_YML5],
                                                            "working_dir",
                                                            "eva_rcs_amarDB.yml"))

    list_render_dict_for_sys_mean = []
    list_render_dict_for_sys_mean.append(centroidV_eva_res_render)
    list_render_dict_for_sys_mean.append(linearP_eva_rec_render)
    list_render_dict_for_sys_mean.append(indexQ_eva_rec_render)
    list_render_dict_for_sys_mean.append(clsR_eva_rec_render)
    list_render_dict_for_sys_mean.append(amarDS_eva_rec_render)

    for sys_render_dict in list_render_dict_for_sys_mean:
        route_path, file_to_render = make_eval_result_sec(sys_render_dict, working_path="working_dir")
        part_of_report = render_latex_template(file_to_render, route_path, sys_render_dict)
        print(part_of_report)
    """

    # GESTIONE DELLA TABELLA CONFRONTO TRA ALGORITMI CON RILEVANZA STATISTICA
    """
    eva_yaml_paths_list = ["../data/data_for_test_two/eva_report_centroidVector.yml",
                           "../data/data_for_test_two/eva_report_linearPredictor.yml",
                           "../data/data_for_test_two/eva_report_indexQuery.yml",
                           "../data/data_for_test_two/eva_report_classifierRecommender.yml",
                           "../data/data_to_test/eva_report_armarDoubleSource.yml"]

    recsys_yaml_paths_list = ["../data/data_for_test_two/rs_report_centroidVector.yml",
                              "../data/data_for_test_two/rs_report_linearPredictor.yml",
                              "../data/data_for_test_two/rs_report_indexQuery.yml",
                              "../data/data_for_test_two/rs_report_classifierRecommender.yml",
                              "../data/data_to_test/rs_report_amarDoubleSource.yml"]

    # procediamo con il recupero del dataframe ottenuto dai test statistici
    file_ref_df = 'ttest_expand.xlsx'
    ref_df = tbl_comp.pd.read_excel(file_ref_df, header=[0, 1], index_col=0)
    # controlliamo che il dataframe caricato non presenti problemi
    # print(f"IL DATAFRAME CARICATO CHE UTILIZZEREMO PER LE REFERENZE DOPO LE OPPORTUNE MODIFICHE\n {ref_df}")

    route_path, file_to_render = make_comparison_algo_sec({}, eva_yaml_paths_list, recsys_yaml_paths_list,
                                                          ref_df, 1.0,only_table=True,
                                                          working_path="working_dir")
    comparison_sec = render_latex_template(file_to_render, route_path, {})
    # print(comparison_sec)

    stats_rev = load_and_add_statistic_relevance_table(recsys_yaml_paths_list, ref_df, "complete",
                                                       2, "relevance table")
    # print(stats_rev)
    """

    # GESTIONE DELLA SOTtOSEZIONE PER LA RILEVANZA STATISTICA DEI RISULTATI
    """
    # qui andiamo a creare un indice di accesso al dataframe che contiene i confronti dei test statistici
    # e l'indice ci servirà per recuperare il confronto tra i due sistemi che vogliamo mettere in evidenza
    # stampando la tabella della rilevanza statica
    idx_access = tbl_comp.stt.set_access_index('CentroidVector', 'IndexQuery',
                                               'Precision - macro', type_val='pvalue')
    pair_stas_rel = load_and_add_statistic_relevance_tab_single_comparison(recsys_yaml_paths_list, ref_df,
                                                                            "complete", idx_access,
                                                                           tab_title="CentroidVector and IndexQuery",
                                                                           sci_not=True, approximation=4)
    # print(pair_stas_rel)

    route_path, file_to_render = make_statistical_relevance_subsection({}, stats_rev ,
                                                                       only_table=False, working_path="working_dir")

    stats_report_sec = render_latex_template(file_to_render, route_path, {})
    """

    # Caso di utilizzo della funzione di renderizzazione render_latex_template(template_name, search_path, my_dict)
    # l'idea è che questa funzione sarà usata per renderizzare pezzi costruiti appositamente da aggiungere al template
    # per farlo necessiterà di un file contenente il template ovvero template_name e di un path verso
    # tale file search_path inoltre cercheromo di riutilizzare i template_chunks già presenti in modo da unirli
    # effettuare alcuni cambiamenti e in seguito dare in pasto alla funzione di renderizzazione il file template
    # creato per mezzo dell'unione di più mini template
    """
    current_path = os.getcwd()
    print("Percorso corrente:", current_path)
    src_path = "templates_chunks/templates_ca/"
    file_per_tmplt = "intro_ca2_test.tex"
    # "../../../templates_chunks/templates_ca/intro_ca.tex"
    # "report/dynamic_report/templates_chunks/templates_ca/intro_ca.tex"

    my_dict = {
        'source_file': "inr",
        'title': 'Il mio documento LaTeX',
        'author': 'Io stesso'
    }

    print(my_dict)
    # Renderizza il template
    rendered_latex = render_latex_template(file_per_tmplt, src_path, my_dict)

    print(type(rendered_latex))
    print(rendered_latex)
    """

    # caso di uso della prima funzione base di renderizzazione func1
    """
    # Definisci il percorso del template LaTeX
    template_path = 'template.tex'

    # Definisci il dizionario con i dati da passare al template
    context = {
        'title': 'Il mio documento LaTeX',
        'author': 'Io stesso',
        'content': 'Questo è il contenuto del documento. Potrebbe contenere più righe.'
    }

    # Renderizza il template
    rendered_latex = render_latex_template(template_path, context)

    # Salva il risultato in un file LaTeX
    with open('output.tex', 'w') as f:
        f.write(rendered_latex)

    print("Template LaTeX renderizzato con successo!")
    """
