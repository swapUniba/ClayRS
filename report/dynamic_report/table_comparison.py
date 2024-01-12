import yaml


def generate_latex_table(algorithms, decimal_places=3, column_width=3.0, max_columns_per_part=5):
    # Estrai le chiavi (nomi di colonne) dal primo dizionario
    first_algorithm = algorithms[0]
    column_names = list(next(iter(first_algorithm.values())).keys())
    num_columns = len(column_names)

    # Calcola il numero di parti necessarie
    num_parts = -(-num_columns // max_columns_per_part)  # Divisione arrotondata per eccesso

    # Inizializza il codice LaTeX
    latex_code = ""

    for part_index in range(num_parts):
        # Calcola gli indici delle colonne per questa parte
        start_col_index = part_index * max_columns_per_part
        end_col_index = (part_index + 1) * max_columns_per_part
        current_column_names = column_names[start_col_index:end_col_index]

        # Calcola la larghezza totale della tabella
        total_width = len(current_column_names) * column_width + 1
        latex_code += "\\begin{table}[ht]\n"
        latex_code += "\\centering\n"
        latex_code += "\\resizebox{\\textwidth}{!}{%\n"
        latex_code += "\\begin{tabular}{@{}c" + " *{" + str(len(current_column_names)) + "}{" + "p{" + str(
            column_width) + "cm}}@{}}\n"
        latex_code += "\\toprule\n"
        latex_code += "\\multirow{2}{*}{Algorithms} & \\multicolumn{" + str(
            len(current_column_names)) + "}{c}{Colonne} \\\\\n"
        latex_code += "\\cmidrule{2-" + str(len(current_column_names) + 1) + "}\n"

        # Aggiungi i nomi delle colonne
        for col_index, column_name in enumerate(current_column_names):
            latex_code += "& \\multirow{2}{*}{\\makecell{" + column_name.replace("_", "\\_") + "}} "

        latex_code += "\\\\\n"
        latex_code += "\\cmidrule{2-" + str(len(current_column_names) + 1) + "}\n"

        # Aggiungi i dati delle righe
        for algorithm in algorithms:
            algorithm_name = list(algorithm.keys())[0]
            values = list(algorithm.values())[0]
            latex_code += algorithm_name
            for column_name in current_column_names:
                # Verifica se la colonna è presente nel dizionario prima di accedere
                column_value = values.get(column_name, '')
                # Converte il valore in un numero (float) prima di arrotondarlo
                try:
                    column_value = float(column_value)
                    rounded_value = round(column_value, decimal_places)
                except (ValueError, TypeError):
                    # Se la conversione non è possibile, mantieni il valore come stringa
                    rounded_value = column_value

                latex_code += " & " + str(rounded_value)
            latex_code += " \\\\\n"
            latex_code += "\\addlinespace[5pt]\n"
            latex_code += "\\midrule\n"

        # Aggiungi la parte finale del codice LaTeX
        latex_code += "\\bottomrule\n"
        latex_code += "\\end{tabular}}\n"
        latex_code += "\\caption{Tabella generata automaticamente (Parte " + str(part_index + 1) + ")}\n"
        latex_code += "\\end{table}\n"

        # Aggiungi alcune righe vuote tra le parti
        if part_index < num_parts - 1:
            latex_code += "\\vspace{10pt}\n"

    return latex_code


# prima versione che spezza la tabellla su più pagine [LASCIA TROPPO SPAZIO]
"""
def generate_latex_table(algorithms, decimal_places=3, column_width=3.0, max_columns_per_part=5):
    # Estrai le chiavi (nomi di colonne) dal primo dizionario
    first_algorithm = algorithms[0]
    column_names = list(next(iter(first_algorithm.values())).keys())
    num_columns = len(column_names)

    # Calcola il numero di parti necessarie
    num_parts = -(-num_columns // max_columns_per_part)  # Divisione arrotondata per eccesso

    # Costruisci l'intestazione della tabella LaTeX
    latex_code = "\\begin{table}[ht]\n"
    latex_code += "\\centering\n"

    # Aggiungi la dimensione del testo
    latex_code += "\\small\n"

    for part_index in range(num_parts):
        # Calcola gli indici delle colonne per questa parte
        start_col_index = part_index * max_columns_per_part
        end_col_index = (part_index + 1) * max_columns_per_part
        current_column_names = column_names[start_col_index:end_col_index]

        # Calcola la larghezza totale della tabella
        total_width = len(current_column_names) * column_width + 1
        latex_code += "\\resizebox{\\textwidth}{!}{%\n"
        latex_code += "\\begin{tabular}{@{}c" + " *{" + str(len(current_column_names)) + "}{" + "p{" + str(column_width) + "cm}}@{}}\n"
        latex_code += "\\toprule\n"
        latex_code += "\\multirow{2}{*}{Algorithms} & \\multicolumn{" + str(len(current_column_names)) + "}{c}{Colonne} \\\\\n"
        latex_code += "\\cmidrule{2-" + str(len(current_column_names) + 1) + "}\n"

        # Aggiungi i nomi delle colonne
        for col_index, column_name in enumerate(current_column_names):
            latex_code += "& \\multirow{2}{*}{\\makecell{" + column_name.replace("_", "\\_") + "}} "

        latex_code += "\\\\\n"
        latex_code += "\\cmidrule{2-" + str(len(current_column_names) + 1) + "}\n"

        # Aggiungi i dati delle righe
        for algorithm in algorithms:
            algorithm_name = list(algorithm.keys())[0]
            values = list(algorithm.values())[0]
            latex_code += algorithm_name
            for column_name in current_column_names:
                # Verifica se la colonna è presente nel dizionario prima di accedere
                column_value = values.get(column_name, '')
                # Converte il valore in un numero (float) prima di arrotondarlo
                try:
                    column_value = float(column_value)
                    rounded_value = round(column_value, decimal_places)
                except (ValueError, TypeError):
                    # Se la conversione non è possibile, mantieni il valore come stringa
                    rounded_value = column_value

                latex_code += " & " + str(rounded_value)
            latex_code += " \\\\\n"
            latex_code += "\\addlinespace[5pt]\n"
            latex_code += "\\midrule\n"

        # Aggiungi la parte finale del codice LaTeX
        latex_code += "\\bottomrule\n"
        latex_code += "\\end{tabular}}\n"
        latex_code += "\\caption{Tabella generata automaticamente (Parte " + str(part_index + 1) + ")}\n"
        latex_code += "\\end{table}\n"
        latex_code += "\\clearpage"  # Nuova pagina tra le parti

    return latex_code
"""

# VERSIONE FUNZIONANTE PER LE COPLONNE MA CARATTERI TROPPO PICCOLI E TABELLA TROPPO PICCOLA
"""
def generate_latex_table(algorithms, decimal_places=3, column_width=3.0):
    # Estrai le chiavi (nomi di colonne) dal primo dizionario
    first_algorithm = algorithms[0]
    column_names = list(next(iter(first_algorithm.values())).keys())
    num_columns = len(column_names)

    # Costruisci l'intestazione della tabella LaTeX
    latex_code = "\\begin{table}[ht]\n"
    latex_code += "\\centering\n"

    # Aggiungi la dimensione del testo
    latex_code += "\\small\n"

    # Calcola la larghezza totale della tabella
    total_width = num_columns * column_width + 1
    latex_code += "\\resizebox{\\textwidth}{!}{%\n"
    latex_code += "\\begin{tabular}{@{}c" + " *{" + str(num_columns) + "}{" + "p{" + str(column_width) + "cm}}@{}}\n"
    latex_code += "\\toprule\n"
    latex_code += "\\multirow{2}{*}{Algorithms} & \\multicolumn{" + str(num_columns) + "}{c}{Colonne} \\\\\n"
    latex_code += "\\cmidrule{2-" + str(num_columns + 1) + "}\n"

    # Aggiungi i nomi delle colonne
    for col_index, column_name in enumerate(column_names):
        latex_code += "& \\multirow{2}{*}{\\makecell{" + column_name.replace("_", "\\_") + "}} "

    latex_code += "\\\\\n"
    latex_code += "\\cmidrule{2-" + str(num_columns + 1) + "}\n"

    # Aggiungi i dati delle righe
    for algorithm in algorithms:
        algorithm_name = list(algorithm.keys())[0]
        values = list(algorithm.values())[0]
        latex_code += algorithm_name
        for column_name in column_names:
            # Verifica se la colonna è presente nel dizionario prima di accedere
            column_value = values.get(column_name, '')
            # Converte il valore in un numero (float) prima di arrotondarlo
            try:
                column_value = float(column_value)
                rounded_value = round(column_value, decimal_places)
            except (ValueError, TypeError):
                # Se la conversione non è possibile, mantieni il valore come stringa
                rounded_value = column_value

            latex_code += " & " + str(rounded_value)
        latex_code += " \\\\\n"
        latex_code += "\\addlinespace[5pt]\n"
        latex_code += "\\midrule\n"

    # Aggiungi la parte finale del codice LaTeX
    latex_code += "\\bottomrule\n"
    latex_code += "\\end{tabular}}\n"
    latex_code += "\\caption{Tabella generata automaticamente}\n"
    latex_code += "\\end{table}\n"

    return latex_code
"""

# implementazione diversa della funzione generate_latex_table()
"""
Questa è già buona ma serve risolvere il problema della sovrapposizione 
# dei nomi di colonna
def generate_latex_table(algorithms, decimal_places=3):
    # Estrai le chiavi (nomi di colonne) dal primo dizionario
    first_algorithm = algorithms[0]
    column_names = list(next(iter(first_algorithm.values())).keys())
    num_columns = len(column_names)

    # Costruisci l'intestazione della tabella LaTeX
    latex_code = "\\begin{table}[ht]\n"
    latex_code += "\\centering\n"

    # Aggiungi la dimensione del testo
    latex_code += "\\small\n"

    # Calcola la larghezza totale della tabella
    total_width = num_columns * 1.5 + 1
    latex_code += "\\resizebox{\\textwidth}{!}{%\n"
    latex_code += "\\begin{tabular}{@{}c" + " *{" + str(num_columns) + "}{>{\\centering\\arraybackslash}p{1.5cm}}@{}}\n"
    latex_code += "\\toprule\n"
    latex_code += "& \\multicolumn{" + str(num_columns) + "}{c}{Colonne} \\\\\n"
    latex_code += "\\cmidrule{2-" + str(num_columns + 1) + "}\n"

    # Aggiungi i nomi delle colonne
    latex_code += "Algorithms & " + " & ".join(column_names) + " \\\\\n"
    latex_code += "\\midrule\n"

    # Aggiungi i dati delle righe
    for algorithm in algorithms:
        algorithm_name = list(algorithm.keys())[0]
        values = list(algorithm.values())[0]
        latex_code += algorithm_name
        for column_name in column_names:
            # Verifica se la colonna è presente nel dizionario prima di accedere
            column_value = values.get(column_name, '')
            # Converte il valore in un numero (float) prima di arrotondarlo
            try:
                column_value = float(column_value)
                rounded_value = round(column_value, decimal_places)
            except (ValueError, TypeError):
                # Se la conversione non è possibile, mantieni il valore come stringa
                rounded_value = column_value

            latex_code += " & " + str(rounded_value)
        latex_code += " \\\\\n"
        latex_code += "\\addlinespace[5pt]\n"
        latex_code += "\\midrule\n"

    # Aggiungi la parte finale del codice LaTeX
    latex_code += "\\bottomrule\n"
    latex_code += "\\end{tabular}}\n"
    latex_code += "\\caption{Tabella generata automaticamente}\n"
    latex_code += "\\end{table}\n"

    return latex_code
"""

# algorithms è una lista di dizionari
"""
def generate_latex_table(algorithms):
    # Estrai le chiavi (nomi di colonne) dal primo dizionario
    first_algorithm = algorithms[0]
    column_names = list(next(iter(first_algorithm.values())).keys())
    num_columns = len(column_names)

    # Costruisci l'intestazione della tabella LaTeX
    latex_code = "\\begin{table}[ht]\n"
    latex_code += "\\centering\n"
    latex_code += "\\begin{tabular}{@{}c" + " *{" + str(num_columns) + "}{>{\\centering\\arraybackslash}p{1.5cm}}@{}}\n"
    latex_code += "\\toprule\n"
    latex_code += "& \\multicolumn{" + str(num_columns) + "}{c}{Colonne} \\\\\n"
    latex_code += "\\cmidrule{2-" + str(num_columns + 1) + "}\n"

    # Aggiungi i nomi delle colonne
    latex_code += "Algorithms & " + " & ".join(column_names) + " \\\\\n"
    latex_code += "\\midrule\n"

    # Aggiungi i dati delle righe
    for algorithm in algorithms:
        algorithm_name = list(algorithm.keys())[0]
        values = list(algorithm.values())[0]
        latex_code += algorithm_name
        for column_name in column_names:
            latex_code += " & " + str(values[column_name])
        latex_code += " \\\\\n"
        latex_code += "\\addlinespace[5pt]\n"
        latex_code += "\\midrule\n"

    # Aggiungi la parte finale del codice LaTeX
    latex_code += "\\bottomrule\n"
    latex_code += "\\end{tabular}\n"
    latex_code += "\\caption{Tabella generata automaticamente}\n"
    latex_code += "\\end{table}\n"

    return latex_code
"""


def remove_key_from_nested_dicts(dictionary_list, key_to_remove):
    """
    Remove a specified key and its associated value from nested dictionaries in a list.

    Parameters:
    - dictionary_list (list): List of dictionaries.
    - key_to_remove (str): Key to be removed from each dictionary.

    Returns:
    - list: List of dictionaries with the specified key removed.

    Example:
        # >>> input_list = [{'a': 1, 'b': {'c': 2, 'd': 3}}, {'x': 4, 'y': {'z': 5, 'w': 6}}]
        # >>> key_to_remove = 'y'
        # >>> result = remove_key_from_nested_dicts(input_list, key_to_remove)
        # >>> print(result)
        [{'a': 1, 'b': {'c': 2, 'd': 3}}, {'x': 4}]
    """

    def remove_key_recursive(d):
        if key_to_remove in d:
            del d[key_to_remove]
        for value in d.values():
            if isinstance(value, dict):
                remove_key_recursive(value)

    modified_list = [dict(d) for d in dictionary_list]

    for d in modified_list:
        remove_key_recursive(d)

    return modified_list


def extract_subdictionary(data, key):
    if key in data:
        return {key: data[key]}
    for value in data.values():
        if isinstance(value, dict):
            result = extract_subdictionary(value, key)
            if result:
                return result
    return None


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


def from_yaml_list_to_dict_list(yaml_file_paths):
    """
    Convert a list of YAML file paths to a list of dictionaries.

    Parameters:
    - yaml_file_paths (list): List of strings representing YAML file paths.

    Returns:
    - list: List of dictionaries representing the structures of the YAML files.

    Example:
        # >>> yaml_paths = ["file1.yaml", "file2.yaml"]
        # >>> result = from_yaml_list_to_dict_list(yaml_paths)
        # >>> print(result)
        [{'key1': 'value1', 'key2': {'key3': 'value3', 'key4': 'value4'}}, {'key5': 'value5'}]
    """
    dict_list = []

    for path in yaml_file_paths:
        yaml_dict = read_yaml_file(path)
        dict_list.append(yaml_dict)

    return dict_list


def get_keys_from_parent_key(dictionary, main_key, current_parent=None):
    """
    Retrieve all keys at a specific level under a given parent key in a nested dictionary.

    Parameters:
    - dictionary (dict): The nested dictionary to explore.
    - main_key (str): The key under which to retrieve the sub-keys.
    - current_parent (str): The current parent key (used for recursion).

    Returns:
    - list: List of keys at the specified level under the parent key.
            If the parent key is not present or does not have a dictionary value, an empty list is returned.

    Example:
        # >>> my_dict = {...}  # Dizionario come mostrato nei tuoi esempi
        # >>> result = get_keys_at_level(my_dict, 'algorithm')
        # >>> print(result)  # Output: ['AmarDoubleSource', 'CentroidVector', 'ClassifierRecommender', 'IndexQuery', 'LinearPredictor']
    """
    keys_at_level = []

    def explore_dictionary(current_dict, current_parent):
        if current_parent == main_key and isinstance(current_dict, dict):
            keys_at_level.extend(current_dict.keys())
        else:
            for key, value in current_dict.items():
                if isinstance(value, dict):
                    explore_dictionary(value, key)

    explore_dictionary(dictionary, current_parent)
    return keys_at_level


def get_algorithm_keys(dict_list, main_key='algorithm'):
    """
    Get the first subkey associated with the specified main key from a list of dictionaries.

    Parameters:
    - dict_list (list): List of dictionaries.
    - main_key (str): Main key to search for in each dictionary.

    Returns:
    - list: List of first subkeys associated with the main key.

    Example:
        # >>> recsys_list = [ {...}, {...}, {...} ]  # Lista dei dizionari come mostrato nei tuoi esempi
        # >>> result = get_algorithm_keys(recsys_list)
        # >>> print(result)  # Output: ['AmarDoubleSource', 'CentroidVector', 'ClassifierRecommender', 'IndexQuery', 'LinearPredictor']
    """
    algorithm_keys = []

    for recsys_dict in dict_list:
        for recsys_key, recsys_value in recsys_dict.items():
            subkeys = get_keys_from_parent_key(recsys_value, main_key)
            if subkeys:
                first_subkey = subkeys[0]  # Prendi il primo elemento dalla lista di sottochiavi
                algorithm_keys.append(first_subkey)

    return algorithm_keys


def nest_dictionaries(keys, dictionaries, key_to_replace='sys - mean'):
    """
    Create a new list of dictionaries by nesting each dictionary with its corresponding key from the given lists.

    Parameters:
    - keys (list): List of keys.
    - dictionaries (list): List of dictionaries.
    - key_to_replace (str): The key to be replaced in each dictionary.

    Returns:
    - list: List of new dictionaries obtained by nesting the dictionaries with their corresponding keys.

    Example:
        # >>> keys = ['AmarDoubleSource', 'CentroidVector', 'ClassifierRecommender', 'IndexQuery', 'LinearPredictor']
        # >>> dictionaries = [ {...}, {...}, {...}, {...}, {...} ]  # Lista dei dizionari come mostrato nei tuoi esempi
        # >>> result = nest_dictionaries(keys, dictionaries, key_to_replace='sys - mean')
        # >>> print(result)
    """
    if len(keys) != len(dictionaries):
        raise ValueError("Number of keys and dictionaries must be the same.")

    nested_dicts = []

    for key, dictionary in zip(keys, dictionaries):
        # Ottieni il dizionario corrispondente a key_to_replace
        sub_dict = dictionary.pop(key_to_replace, None)

        # Assicurati che sub_dict sia un dizionario
        if sub_dict is not None and isinstance(sub_dict, dict):
            nested_dicts.append({key: sub_dict})

    return nested_dicts


# Esegui lo script
if __name__ == "__main__":

    eva_yaml_paths = ["./../data/data_to_test/eva_report_amarSingleSource.yml",
                      "./../data/data_to_test/eva_report_centroidVector.yml",
                      "./../data/data_to_test/eva_report_classifierRecommender.yml",
                      "./../data/data_to_test/eva_report_indexQuery.yml",
                      "./../data/data_to_test/eva_report_linearPredictor.yml"]

    recsys_yaml_paths = ["./../data/data_to_test/rs_report_amarSingleSource.yml",
                         "./../data/data_to_test/rs_report_centroidVector.yml",
                         "./../data/data_to_test/rs_report_classifierRecommender.yml",
                         "./../data/data_to_test/rs_report_indexQuery.yml",
                         "./../data/data_to_test/rs_report_linearPredictor.yml"]

    # set list of dictionary from eva yaml report and make key change
    dictionary_res_sys = from_yaml_list_to_dict_list(eva_yaml_paths)
    dict_sys_mean = []
    for e in dictionary_res_sys:
        tmp = extract_subdictionary(e, "sys - mean")
        dict_sys_mean.append(tmp)

    # clean key unused from dictionary list
    my_dictio = remove_key_from_nested_dicts(dict_sys_mean, "CatalogCoverage (PredictionCov)")

    # get dictionary from the recsys ymal and extract a list of key with name of algorithm used
    dictyonary_list = from_yaml_list_to_dict_list(recsys_yaml_paths)
    keys = get_algorithm_keys(dictyonary_list)

    # show the dictonary extracted after processing them
    result = nest_dictionaries(keys, my_dictio)
    for r in result:
        print(r)

    # with the dictnory processed create the latex table
    latex_table = generate_latex_table(result, max_columns_per_part=7)
    print(latex_table)

    """
    # prova per get_algorithm_keys ovvero per recuperare una lista di chiavi da dei dizionari
    recsys_yaml_paths = ["./../data/data_to_test/rs_report_amarSingleSource.yml",
                          "./../data/data_to_test/rs_report_centroidVector.yml",
                          "./../data/data_to_test/rs_report_classifierRecommender.yml",
                          "./../data/data_to_test/rs_report_indexQuery.yml",
                          "./../data/data_to_test/rs_report_linearPredictor.yml"]
    dictyonary_list = from_yaml_list_to_dict_list(recsys_yaml_paths)
    for d in dictyonary_list:
        print(d)
    result = get_algorithm_keys(dictyonary_list)

    for k in result:
        print(k)
    """

    """
    # prova per il recupero di una lista di dizionari a partire da una lista di file yaml
    eva_yaml_paths = ["./../data/data_to_test/eva_report_amarSingleSource.yml",
                      "./../data/data_to_test/eva_report_centroidVector.yml",
                      "./../data/data_to_test/eva_report_classifierRecommender.yml",
                      "./../data/data_to_test/eva_report_indexQuery.yml",
                      "./../data/data_to_test/eva_report_linearPredictor.yml"]
    result = from_yaml_list_to_dict_list(eva_yaml_paths)
    for d in result:
        print(d)
    """

    """
    # prova per la formatazione e creazione della tabella di comparazione tra algortitmi
    algorithms_data = [
        {'CentroidVector': {'F1@1 - macro': 0.85, 'MRR': 0.92, 'MAE': 0.12, 'Recall': 0.78, 'Precision': 0.91}},
        {'AmarDoubleSource': {'F1@1 - macro': 0.92, 'MRR': 0.88, 'MAE': 0.14, 'Recall': 0.85, 'Precision': 0.89}},
        {'LinearPredictor': {'F1@1 - macro': 0.88, 'MRR': 0.91, 'MAE': 0.11, 'Recall': 0.82, 'Precision': 0.90}},
    ]

    latex_table = generate_latex_table(algorithms_data)
    print(latex_table)
    """

    """
    # prova per la funzione che recupera un sottodizionario innestato da un dizionario fatto
    # di dizionari innestati
    # Esempio di utilizzo
    nested_dict = {'A': {
        'CentroidVector': {
            'F1@1 - macro': 0.85,
            'MRR': 0.92,
            'MAE': 0.12,
            'Recall': 0.78,
            'Precision': 0.91
        },
        'AmarDoubleSource': {
            'F1@1 - macro': 0.92,
            'MRR': 0.88,
            'MAE': 0.14,
            'Recall': 0.85,
            'Precision': 0.89
        },
        'LinearPredictor': {
            'F1@1 - macro': 0.88,
            'MRR': 0.91,
            'MAE': 0.11,
            'Recall': 0.82,
            'Precision': 0.90
        }
    }}

    key_to_extract = 'LinearPredictor'
    result = extract_subdictionary(nested_dict, key_to_extract)

    if result:
        print(result)
    else:
        print(f"Chiave '{key_to_extract}' non trovata nel dizionario.")
    """
