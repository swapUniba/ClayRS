import os
import re
import shutil
import yaml

# PATH USED TO TEXT THE SCRIPT
CA_YML = "../data/ca_report.yml"
EVA_YML = "../data/eva_report.yml"
RS_YML = "../data/rs_report.yml"


# dictionary to find the path for mini template chunks of content analyzer
# the key are the key found in the yaml file for content analyzer and the
# value are the path to the corresponding templates.
CA_DICT = {
    'OriginalData': 'OriginalData_techniques_ca.tex',
    'WhooshTfIdf': 'WhooshTfIdf_techinques_ca.tex',
    'SkLearnTfIdf': 'SkLearnTfIdf_tech_ca.tex',
    'WordEmbeddingTechnique': 'WordEmbedding_tech_ca.tex',
    'SentenceEmbeddingTechnique': 'SentenceEmb_tech_ca.tex',
    'DocumentEmbeddingTechnique': 'DocumentEmb_tech_ca.tex',
    'Word2SentenceEmbedding': 'Word2SentenceEmb_tech_ca.tex',
    'Word2DocEmbedding': 'Word2DocEmb_tech_ca.tex',
    'Sentence2DocEmbedding': 'Word2SentenceEmb_tech_ca.tex',
    'PyWSDSynsetDocumentFrequency': 'PyWSD_tech_ca.tex',
    'FromNPY': 'FromNPY_tech_ca.tex',
    'SkImageHogDescriptor': 'SkImgHogDescr_tech_ca.tex',
    'MFCC': 'mfcc_tech_ca.tex',
    'VGGISH:': 'TorchVVM_tech_ca.tex',
    'PytorchImageModels': 'PytorchImgMod_tech_ca.tex',
    'TorchVisionVideoModels': '',
    'Spacy': 'spacy_prepro_ca.tex',
    'Ekphrasis': 'ekphrasis_prepro_ca.tex',
    'NLTK': 'nltk_prepro_ca.tex',
    'TorchUniformTemporalSubSampler': '',
    'Resize': 'Resize_prepro_ca.tex',
    'CenterCrop': 'centerCrop_prepro_ca.tex',
    'Lambda': 'Lambda_prepro_ca.tex',
    'Normalize': 'normilize_prepro_ca.tex',
    'ClipSampler': 'clipSampler_prepro_ca.tex',
    'FVGMM': 'fvgmm_postpro_ca.tex',
    'VLADGMM': 'vladgmm_postpro_ca.tex',
    'SkLearnPCA': 'skLearnPCA_postpro_ca.tex',
    'DBPediaMappingTechnique': 'DBpediaMT_ex_tech_ca.tex',
    'ConvertToMono': 'Convert2Mono_prepro_ca.tex',
    'TorchResample': 'TorchResample_prepro_ca.tex'
}

# the list will contain all the fields that
# have been represented by the content analyzer.
list_of_fields = []

# the list will contain teh keys of the eva yaml that represent
# the fold n-th of the evaluation performed.
list_fold_eva = []


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

# -------------- to test this code -----------
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

def build_final_latex_file(file_list, customizations, file_destinazione):
    for file_path in file_list:
        contenuto_file = read_file_latex(file_path)

        if file_path in customizations:
            personalizzazione = customizations[file_path]
            contenuto_file = contenuto_file.replace('XXX', personalizzazione)

        testo_estratto = get_text_from_latex(contenuto_file)

        if testo_estratto:
            write_on_file_latex(testo_estratto, file_destinazione)
            print(f"Contenuto di {file_path} aggiunto con successo a {file_destinazione}")
        else:
            print(f"Impossibile estrarre il testo dal documento LaTeX: {file_path}")

    # Condizione 1: Aggiungi un numero predefinito di file LaTeX
    numero_file_predefiniti = 3
    file_list_predefiniti = file_list[:numero_file_predefiniti]

    for file_path in file_list_predefiniti:
        contenuto_file = read_file_latex(file_path)

        if file_path in customizations:
            personalizzazione = customizations[file_path]
            contenuto_file = contenuto_file.replace('XXX', personalizzazione)

        testo_estratto = get_text_from_latex(contenuto_file)

        if testo_estratto:
            write_on_file_latex(testo_estratto, file_destinazione)
            print(f"Contenuto di {file_path} aggiunto con successo a {file_destinazione}")
        else:
            print(f"Impossibile estrarre il testo dal documento LaTeX: {file_path}")

    # Condizione 2: Aggiungi un numero variabile di file LaTeX
    file_list_variabili = get_variable_files()
    for file_path in file_list_variabili:
        contenuto_file = read_file_latex(file_path)

        if file_path in customizations:
            personalizzazione = customizations[file_path]
            contenuto_file = contenuto_file.replace('XXX', personalizzazione)

        testo_estratto = get_text_from_latex(contenuto_file)

        if testo_estratto:
            write_on_file_latex(testo_estratto, file_destinazione)
            print(f"Contenuto di {file_path} aggiunto con successo a {file_destinazione}")
        else:
            print(f"Impossibile estrarre il testo dal documento LaTeX: {file_path}")

    # Condizione 3: Aggiungi ulteriori file o esegui ulteriori logiche secondo necessità
    # ...

# Funzione di esempio per ottenere una lista di file LaTeX variabili
def get_variable_files():
    # Implementa la logica per recuperare dinamicamente i file LaTeX variabili
    # In questo esempio, restituisce una lista vuota, ma dovrebbe essere adattata alle tue esigenze.
    return []


# ------ up to this part we have to test ------------
# Chiamata principale
if __name__ == "__main__":
    file_list = ['templates_chunks/intro.tex',
                 'templates_chunks/content_analyzer_section.tex',
                 'templates_chunks/recsys_section.tex',
                 'templates_chunks/evaluation_section.tex']

    customizations = {'templates_chunks/intro.tex': 'DIEGO',
                      'templates_chunks/content_analyzer_section.tex': 'MARCO',
                      'templates_chunks/recsys_section.tex': 'GIULIO',
                      'templates_chunks/evaluation_section.tex': 'FAY'
                      }

    file_destinazione = 'final_report.tex'

    build_final_latex_file(file_list, customizations, file_destinazione)
    my_dict = read_yaml_file(CA_YML)

    desired_parent_key = "plot_0"
    keys_at_desired_level = get_keys_at_level(my_dict, desired_parent_key)

    print(keys_at_desired_level)

    # list_of_fields = get_keys_at_level(my_dict, "field_representations")
    # list_fold_eva = get_keys_at_level(my_dict, "sys_results")
    # print(list_of_fields )
    # print(list_fold_eva)
