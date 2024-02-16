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
    'stats': './templates_chunks/templates_ca_mini_chunks/Dataset_Stat.tex'
}

# dictionary to find path for the recsys module template
RS_DICT = {
    'recsys': './templates_chunks/templates_rs/recsys_template_complete_new.tex',
    'general_rec': './templates_chunks/templates_rs/recsys_general.tex'
}

# dictionary to find path for the evaluation module template
EVA_DICT = {
    'intro': './templates_chunks/templates_eva_mini_chunks/intro_eva_all_metrics.tex',
    'end': './templates_chunks/templates_eva_mini_chunks/end_eva.tex',
    'result': './templates_chunks/templates_eva_mini_chunks/sys_result_on_fold_eva_new.tex',
    'no_res': './templates_chunks/templates_eva_mini_chunks/no_results_on_fold.tex'
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
    print()


def make_content_analyzer_sec(render_dict, name_of_dataset="no name", mode="minimise", working_path="working_dir"):
    if mode not in ["minimise", "verbose"]:
        raise ValueError("Il parametro 'mode' può essere solo 'minimise' o 'verbose'.")

    # Crea il nome del file che farà da template per la renderizzazione di questa
    # parte di report che stiamo andando a produrre
    file_name = "ca_report_latex.tex"
    file_path = os.path.join(working_path, file_name)
    print(file_path)

    ca_processing_report_minimal(render_dict, working_path, file_name)
    if mode != "minimise":
        # scelta della funzione in base alla verbosità del report che si vuole ottenere
        # a riguardo delle elaborazioni fatte dal content analyzer sui dati
        ca_processing_report_verb(render_dict, working_path, file_name)
        # print()

    # procediamo con l'inserimento della tabella statistica sui dati
    data_statistic_report(render_dict, name_of_dataset,working_path, file_name)

    # aggiungiamo la sezione di spit del dataset
    # splitting_technique_report(render_dict, working_path, file_name)

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
        number = float(text)
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
    RS_YML = "../data/data_to_test/rs_report_centroidVector.yml"

    # used to load the dictonary related to each yaml file
    ca_dict = read_yaml_file(CA_YML)
    rs_dict = read_yaml_file(RS_YML)
    eva_dict = read_yaml_file(EVA_YML)

    path_rendering_dict = merge_yaml_files([CA_YML, RS_YML],
                                       "working_dir",
                                       "ca_rcs_yml_union.yml")

    dict_for_render = read_yaml_file(path_rendering_dict)
    print(dict_for_render)

    route_path, file_to_render = make_content_analyzer_sec(dict_for_render, name_of_dataset="1000K data video movie")
    print(route_path)
    print(file_to_render)
    part_of_report = render_latex_template(file_to_render, route_path, dict_for_render)
    print(part_of_report)

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
