from collections import defaultdict

import pandas as pd

import statistic_test_table as stt
import yaml


# ------------------------------ generate_table_for_comparsion() + support functions ------------------------------
# Funzione per generare la tabella che mette a confronto le diverse istanziazioni dello stesso algoritmo per il
# recsys utilizzato
def generate_table_for_comparison(list_of_dicts, n_col_desired, width=3.0,
                                  alg_type="killer", round_to=3, set_title="Results of metrics"):
    """
        The function  use a list of dictionary to extract their information and createa table that will compare
        the performance of the same algorithm trained on different field representations. In the table the best
        result for each metric showed will be highlighted in bold the best and underlined the second_best.

        Parameters:
        - list_of_dicts (list): The dictionary contained will be used to create the table.
        - n_col_desired (int): number of columns to show for each table created.
        - width=3.0 (int) : number used for the formatting the table in particular to deal with its resizing.
        - alg_type="killer" (str) : the name of the algorithm used .
        - round_to=3 (int) : indicate the number of approximation to be applied on the values.
        - set_title="Results of metrics" (str): string that will be the title for the table.

        Returns:
        - latex_table (str): The table with the value in string form.

        Author:
        - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
    """
    # Extract fixed columns name (Alg., Repr., Content, Emb.)
    fixed_columns = ['Alg.', 'Repr.', 'Content', 'Emb.']

    # Lista di metriche il cui punteggio migliore è quello minimo
    metrics_minimum_score = ['RMSE', 'MSE', 'MAE', 'Gini']

    # extraction of the dictionary containing the metrics and their values
    metrics_dicts = [get_metrics(d) for d in list_of_dicts]
    # print(metrics_dicts)

    # get columns from best_higher_dict
    # higher_dict is the dictionary containing for each metrics the best value and the second-best value
    higher_dict = best_higher_dict(metrics_dicts)
    # print(higher_dict)
    dynamic_columns = list(higher_dict.keys())

    # dictionary contaiting the best and second-best value form the metrics evaluated as lower is better
    lower_dict = best_lower_dict(metrics_dicts)
    # print(lower_dict)

    # Get total number of columns for the table
    total_columns = len(fixed_columns) + len(dynamic_columns)

    # check on max number of columns per table
    if n_col_desired > 8:
        n_col_desired = 8

    table_string = ""

    # Calculating the number of table that needs to be generated
    num_tables, remainder = divmod(len(dynamic_columns), n_col_desired - len(fixed_columns))

    # add 1 table if previous division has got rest
    if remainder > 0:
        num_tables += 1

    for i in range(num_tables):
        start_idx = i * (n_col_desired - len(fixed_columns))
        end_idx = (i + 1) * (n_col_desired - len(fixed_columns))
        current_dynamic_columns = dynamic_columns[start_idx:end_idx]

        # header of the table
        table_string += "\\begin{table}\n"
        table_string += "\\begin{adjustwidth}{-1 in}{-1 in}\n"
        table_string += "  \\centering\n"
        table_string += f"   \\caption{{{set_title} - Tabella {i + 1}}}\n"
        table_string += f"  \\begin{{tabular}}{{{'l' + 'c' * (len(fixed_columns) + len(current_dynamic_columns))}}}\n"
        table_string += "    \\toprule\n"
        table_string += "    " + " & ".join(fixed_columns + current_dynamic_columns) + " \\\\\n"
        table_string += "    \\midrule\n"
        table_string += "    \\midrule\n"

        # fill row with values
        for i, (dictionary, metrics_dict) in enumerate(zip(list_of_dicts, metrics_dicts)):
            # make the header of the multirow only for fist row of each table
            if i % len(list_of_dicts) == 0:
                table_string += f"    \\multirow{{{len(list_of_dicts)}}}{{*}}{{{alg_type}}}"

            # filling of the columns fixed
            for col in fixed_columns[1:]:  # Parti da 'Repr.' per evitare di ripetere 'Alg.'
                if col == 'Repr.':
                    table_string += f" & {sanitize_latex_string(get_representation(dictionary))}"
                elif col == 'Content':
                    table_string += f" & {sanitize_latex_string(get_content(dictionary))}"
                elif col == 'Emb.':
                    table_string += f" & {sanitize_latex_string(get_embedding(dictionary))}"

            # filling of the dynamic columns
            for dynamic_col in current_dynamic_columns:
                value = metrics_dict.get(dynamic_col, '')

                # decide wich score logic to apply
                if dynamic_col in metrics_minimum_score:
                    # apply lower is better using dict lower_dict
                    if value == lower_dict[dynamic_col][0]:  # Primo valore in lower_dict
                        table_string += f" & \\textbf{{{round(value, round_to) if value is not None else ''}}}"
                    elif value == lower_dict[dynamic_col][1]:  # Secondo valore in lower_dict
                        table_string += f" & \\underline{{{round(value, round_to) if value is not None else ''}}}"
                    else:
                        table_string += f" & {round(value, round_to) if value is not None else ''}"
                else:
                    # apply higher is better with dict higher_dict
                    if value == higher_dict[dynamic_col][0]:  # Primo valore in higher_dict
                        table_string += f" & \\textbf{{{round(value, round_to) if value is not None else ''}}}"
                    elif value == higher_dict[dynamic_col][1]:  # Secondo valore in higher_dict
                        table_string += f" & \\underline{{{round(value, round_to) if value is not None else ''}}}"
                    else:
                        table_string += f" & {round(value, round_to) if value is not None else ''}"

            table_string += " \\\\\n"

        table_string += "    \\bottomrule\n"
        table_string += "   \\end{tabular}\n"
        table_string += "\\end{adjustwidth}\n"
        table_string += "\\end{table}\n\n"

    return table_string


# Funzione che crea il dizionario contenente le metriche come chiavi e per ogni metrica una coppia di valori
# che sono il minimo e il secondo minimo, !!! LA FUNZIONE È UTILIZZATA DA generate_table_for_comparison() !!!
# per evidenziare nella tabella questi valori come grassetto e sottolineato
def best_lower_dict(list_of_dicts):
    """
        The function  use a list of dictionary to create a new dictionary that has for key the name of all
        the metrics retrived from all the dictionaries in input and for value the lowest result find for each metric
        and the second-lowest results.

        Parameters:
        - list_of_dicts (list): The dictionary contained will be used to create a new dictionary.

        Returns:
        - dict(key_values) (dict): dictionary created.

        Author:
        - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
    """
    key_values = defaultdict(list)

    # get all key present in the dictionaries
    all_keys = set()
    for d in list_of_dicts:
        all_keys.update(d.keys())

    # find minimum for each metrics
    for key in all_keys:
        values = [d.get(key, float('inf')) for d in list_of_dicts]
        values.sort()
        min_value = values[0]
        second_min_value = values[1] if len(values) > 1 else float('inf')
        key_values[key] = (min_value, second_min_value)

    return dict(key_values)


# Funzione che crea il dizionario contenente le metriche come chiavi e per ogni metrica una coppia di valori
# # che sono il massimo e il secondo massimo, !!! LA FUNZIONE È UTILIZZATA DA generate_table_for_comparison() !!!
# per evidenziare nella tabella questi valori come grassetto e sottolineato
def best_higher_dict(list_of_dicts):
    """
        The function  use a list of dictionary to create a new dictionary that has for key the name of all
        the metrics retrieved from all the dictionaries in input and for value the highest result find for each metric
        and the second-highest results.

        Parameters:
        - list_of_dicts (list): The dictionary contained will be used to create a new dictionary.

        Returns:
        - dict(key_values) (dict): dictionary created.

        Author:
        - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
    """
    key_values = defaultdict(list)

    # get all key present in the dictionaries
    all_keys = set()
    for d in list_of_dicts:
        all_keys.update(d.keys())

    # find maximum values
    for key in all_keys:
        values = [d.get(key, float('-inf')) for d in list_of_dicts]
        values.sort(reverse=True)
        max_value = values[0]
        second_max_value = values[1] if len(values) > 1 else float('-inf')
        key_values[key] = (max_value, second_max_value)

    return dict(key_values)


# Questa versione ritorna un dizionario anziché una lista di tuple che racchiudano il nome della metrica e
# il valore associato dal dizionario estratto. La funzione è utilizzata da generate_table_for_comparison()
def get_metrics(data_dict):
    """
        The function  use a dictonary to extract a subdictionary containing the information on the metrics.

        Parameters:
        - list_of_dicts (list): The dictionary contained will be used to create a new dictionary.

        Returns:
        - result (dict): dict containing the names of metrics for keys and their values as values.

        Author:
        - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
    """
    def extract_metrics(d, prefix=''):
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result.update(extract_metrics(value, f'{prefix}{key} - '))
            else:
                result[f'{prefix}{key}'] = value
        return result

    for key, value in data_dict.items():
        if key == 'sys - mean':
            return extract_metrics(value)
        elif isinstance(value, dict):
            nested_result = get_metrics(value)
            if nested_result:
                return nested_result

    return {}


# Questa funzione accede a un dizionario e recupera andando a formattare la stringa creata accedendo ai valori
# contenuti nella chiave 'embedding_combiner'. La funzione è utilizzata da generate_table_for_comparison()
def get_embedding(data_dict):
    """
        The function  extract a specific value from the dictionary, in particular it extract the value associated
        to the embedding key.

        Parameters:
        - data_dict (dict): The dictionary used to extract the values needed.

        Returns:
        - result (str): sting that represent the values retrieved formatted.

        Author:
        - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
    """
    # search for 'embedding_combiner' key
    def find_embedding_combiner(d):
        for key, value in d.items():
            if key == 'embedding_combiner':
                return value
            elif isinstance(value, dict):
                result = find_embedding_combiner(value)
                if result is not None:
                    return result
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        result = find_embedding_combiner(item)
                        if result is not None:
                            return result
        return None

    # retrieve data needed from key 'embedding_combiner'
    embedding_data = find_embedding_combiner(data_dict)

    # Stampa di debug per vedere il tipo di embedding_data
    # print(f"Tipo di embedding_data: {type(embedding_data)}")

    # formatting the value based on what value is found associated to the key
    if embedding_data is None:
        return ""

    elif isinstance(embedding_data, str):
        return embedding_data

    elif isinstance(embedding_data, (int, float)):
        return str(embedding_data)

    elif isinstance(embedding_data, list):
        return " + ".join(map(str, embedding_data))

    elif isinstance(embedding_data, dict):
        return " + ".join(embedding_data.keys())

    # if value found different from formatting case return an empty string
    return ""


# Questa funzione recupera sotto forma di stringa il contenuto utilizzato per l'esperimento ovvero tutti i field usati
# e li restituisce come stringa concatenata con il +. La funzione è utilizzata da generate_table_for_comparison()
def get_content(data_dict):
    """
        The function  extract a specific value from the dictionary, in particular it extracts the value associated
        to the item_field key unifing the sub key presented.

        Parameters:
        - data_dict (dict): The dictionary used to extract the values needed.

        Returns:
        - content (str): sting that represent the values retrieved formatted.

        Author:
        - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
    """
    # check on keys
    if 'algorithm' in data_dict:
        algorithm_keys = [key for key, value in data_dict['algorithm'].items() if 'item_field' in value]

        # retrieve data needed
        if algorithm_keys:
            # get to the key wanted and extract all its primary sub keys
            algorithm_key = algorithm_keys[0]
            item_field_data = data_dict['algorithm'][algorithm_key].get('item_field', {})
            content_keys = [key for key, value in item_field_data.items() if isinstance(value, (list, dict)) and value]
            content = " + ".join(map(str, content_keys))

            return content

    # if key not in data_dict
    return ""


# Questa funzione recupera sotto forma di stringa quello che sarà utilizzato
# per riempire la casella sotto la colonna Repr.
# Della rappresentazione utilizzata per processare i dati con il content analyzer e che è stata in seguito
# utilizzata per addestrare il recsys istanziato
def get_representation(data_dict):
    """
        The function  extract a specific value from the dictionary, in particular it extracts the data
        that describe the representation used for the field.

        Parameters:
        - data_dict (dict): The dictionary used to extract the values needed.

        Returns:
        - representation (str): sting that represent the values retrieved formatted and processed.

        Author:
        - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
    """
    # check keys
    if 'algorithm' in data_dict:
        # retrieve key under algorithm
        algorithm_keys = [key for key, value in data_dict['algorithm'].items() if 'item_field' in value]

        # retrieve values
        if algorithm_keys:
            algorithm_key = algorithm_keys[0]
            item_field_data = data_dict['algorithm'][algorithm_key].get('item_field', {})
            item_field_values = [value for key, value in item_field_data.items() if isinstance(value, list) and value]
            representation_strings = [" + ".join(map(str, values)) for values in item_field_values]
            representation = " + ".join(representation_strings)

            return representation

    # return an empty string if key is not found
    return ""


# Funzione di supporto per ripulire le stringhe e adattarle al formato latex
def sanitize_latex_string(input_str):
    return input_str.replace("_", "\\_").replace("&", "\\&").replace("#", "\\#")


# funzioni usate come supporto nel momento in cui get_metrics restituisce una lista di tuple e non un dizionario
"""
def extract_first_elements(tuple_list):
    return [tup[0] if len(tup) > 0 else None for tup in tuple_list]

# Prepara le tuple contenenti i dati per la renderizzazione della tabella OK
def make_tuple_for_table(data_dict, start, end):
    # Ottieni i valori per la tupla
    representation = get_representation(data_dict)
    content = get_content(data_dict)
    embedding = get_embedding(data_dict)
    metrics_list = get_metrics(data_dict)

    # Controlla se start ed end sono validi
    if start < 0:
        start = 0
    if end >= len(metrics_list):
        end = len(metrics_list)

    # Prendi solo il secondo elemento dalle tuple, convertilo in numero e approssimalo a tre cifre decimali
    metrics_values = [f"{float(tup[1]):.3f}" for tup in metrics_list[start:end]]

    # Costruisci la tupla
    result_tuple = (
        sanitize_latex_string(representation),
        sanitize_latex_string(content),
        sanitize_latex_string(embedding),
        *metrics_values
    )

    return result_tuple
"""

# Funzione di supporto per generate_latex_table, serve per trovare i massimi da evidenziare nella tabella, la
# seconda versione è capace di lavorare con la tipologia di dizionari che andremo a usare
"""
def find_highest_bests(dictionaries, keys, decimal_places):
    result = {}

    for key in keys:
        maximum = float('-inf')
        second_maximum = float('-inf')

        for d in dictionaries:
            if key in d:
                value = d[key]

                if value > maximum:
                    second_maximum = maximum
                    maximum = value
                elif value > second_maximum and value != maximum:
                    second_maximum = value

        # Rounding the values
        maximum = round(maximum, decimal_places) if maximum != float('-inf') else None
        second_maximum = round(second_maximum, decimal_places) if second_maximum != float('-inf') else None

        result[key] = (maximum, second_maximum)

    return result
"""


# ----------------------- End use of support function for generate_table_for_comparison ------------------------------


# ----------------------- generate_latex_table() + support functions used -------------------------------------------
# La funzione è utilizzata per recuperare un dizionario che contenga per chiavi le metriche e per valori una coppia
# (x, y) ove x è il valore più alto per quella metrica trovato nella lista dei dizionari ricevuta in ingresso e y è il
# secondo valore più alto trovato per quella metrica. È utilizzata dalla funzione generate_latex_table() per
# evidenziare in grassetto e sottolineato tali risultati
def find_highest_bests(dictionaries, keys, decimal_places):
    """
       The function  use a list of dictionary and a list of wich keys will be used from these dictionaries
       to create a new dictionary that has for key the name of all the metrics presented in keys retrieved
       and for value the highest result find for each metric and the second-highest results.

       Parameters:
       - dictionaries (list): The dictionary contained will be used to create a new dictionary.
       - keys (list): list of string containing the name of the metrics.
       - decimal_places (int): the approximation for the values found.

       Returns:
       - result (dict): dictionary created with best value for each metric.

       Author:
       - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
   """
    result = {}

    # search and retrieve the best two maximum values
    for key in keys:
        maximum = float('-inf')
        second_maximum = float('-inf')

        for d in dictionaries:
            for top_level_key, inner_dict in d.items():
                if key in inner_dict:
                    value = inner_dict[key]

                    if value > maximum:
                        second_maximum = maximum
                        maximum = value
                    elif value > second_maximum and value != maximum:
                        second_maximum = value

        # Rounding the values
        maximum = round(maximum, decimal_places) if maximum != float('-inf') else None
        second_maximum = round(second_maximum, decimal_places) if second_maximum != float('-inf') else None

        result[key] = (maximum, second_maximum)

    return result


# La funzione è utilizzata per recuperare un dizionario che contenga per chiavi le metriche e per valori una coppia
# (x, y) ove x è il valore più basso per quella metrica trovato nella lista dei dizionari ricevuta in ingresso e y è il
# secondo valore più basso trovato per quella metrica. È utilizzata dalla funzione generate_latex_table() per
# evidenziare in grassetto e sottolineato tali risultati
def find_lowest_bests(dictionaries, keys_list, decimal_places):
    """
       The function  use a list of dictionary and a list of wich keys will be used from these dictionaries
       to create a new dictionary that has for key the name of all the metrics presented in keys retrieved
       and for value the lowest result find for each metric and the second-lowest results.

       Parameters:
       - dictionaries (list): The dictionary contained will be used to create a new dictionary.
       - keys (list): list od string containing the manes of the metrics.
       - decimal_places (int): approximation for the values found.

       Returns:
       - result (dict): dictionary created with best value for each metric.

       Author:
       - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
   """
    result = {}
    # search and find the best two minimum
    for key in keys_list:
        minimum = float('inf')
        second_minimum = float('inf')

        for d in dictionaries:
            for top_level_key, inner_dict in d.items():
                if key in inner_dict:
                    value = inner_dict[key]

                    if value < minimum:
                        second_minimum = minimum
                        minimum = value
                    elif value < second_minimum and value != minimum:
                        second_minimum = value

        # Rounding the values
        minimum = round(minimum, decimal_places) if minimum != float('inf') else None
        second_minimum = round(second_minimum, decimal_places) if second_minimum != float('inf') else None

        result[key] = (minimum, second_minimum)

    return result


# funzione per il confronto tra algoritmi PRONTA E FUNZIONANTE OK usa le funzioni di supporto find_highest_bests()
# e find_lowest_best()
def generate_latex_table(algorithms, decimal_place=3, column_width=3.0,
                         max_columns_per_part=5, caption_for_table="Comparison between algorithms"):
    """
       The function  use a list of dictionaries and a bunch of other parameter to make a table able to compare
       the result of different algorithm of recommendation displaying the results of the metrics used to evaluate them.

       Parameters:
       - algorithms (list): The list of dictionaries used to fill and make the table of comparison.
       - decimal_place=3 (int): approximation of the values represented in the table.
       - column_width=3.0 (int): formatting value used for resizing of the table.
       - max_columns_per_part=5 (int): number of columns to be shown per table.
       - caption_for_table="Comparison between algorithms" (str): title of the table.

       Returns:
       - latex_code (str): table formatted in latex syntax.

       Author:
       - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
   """
    # check on max number of columns to show per table
    if max_columns_per_part > 10:
        max_columns_per_part = 10

    # get the names for the columns
    first_algorithm = algorithms[0]
    column_names = list(next(iter(first_algorithm.values())).keys())
    num_columns = len(column_names)

    # list of metrics for lower score best logic  NEED TO BE SET AS PARAMETER
    metrics_minimum_score = ['RMSE', 'MSE', 'MAE', 'Gini']

    # Dictionary used to apply the logic for selection of best result
    best_metrics = {}

    # creating the two dictionries to apply the logic of best higher score and best lower score
    highest_best_metrics = find_highest_bests(algorithms, column_names, decimal_place)
    # print(f"il dizionario con i migliori risultati per metrica crescenti {highest_best_metrics}")
    lowest_best_metrics = find_lowest_bests(algorithms, column_names, decimal_place)
    # print(f"il dizionario con le migliori metriche più basse è {lowest_best_metrics}")

    # calculating the number of table needed
    num_parts = -(-num_columns // max_columns_per_part)

    latex_code = ""

    # creating the tables
    for part_index in range(num_parts):
        # get index for the part of the table needed
        start_col_index = part_index * max_columns_per_part
        end_col_index = (part_index + 1) * max_columns_per_part
        current_column_names = column_names[start_col_index:end_col_index]

        # evaluate the size of the table
        total_width = len(current_column_names) * column_width + 1
        latex_code += "\\begin{table}[ht]\n"
        latex_code += "\\centering\n"
        latex_code += "\\resizebox{\\textwidth}{!}{%\n"
        latex_code += "\\begin{tabular}{@{}c" + " *{" + str(len(current_column_names)) + "}{" + "p{" + str(
            column_width) + "cm}}@{}}\n"
        latex_code += "\\toprule\n"
        latex_code += "\\multirow{2}{*}{Algorithms} & \\multicolumn{" + str(
            len(current_column_names)) + "}{c}{Columns} \\\\\n"
        latex_code += "\\cmidrule{2-" + str(len(current_column_names) + 1) + "}\n"

        # add columns names
        for col_index, column_name in enumerate(current_column_names):
            latex_code += "& \\multirow{2}{*}{\\makecell{" + column_name.replace("_", "\\_") + "}} "

        latex_code += "\\\\\n"
        latex_code += "\\addlinespace[5pt]\n"
        latex_code += "\\cmidrule{2-" + str(len(current_column_names) + 1) + "}\n"

        # fill with data
        for algorithm in algorithms:
            algorithm_name = list(algorithm.keys())[0]
            # print(f"algorithm name is {algorithm_name}")
            values = list(algorithm.values())[0]
            # print(f"values is {values}")
            latex_code += algorithm_name
            for column_name in current_column_names:
                # print(f"The name of the column is {column_name}")
                column_value = values.get(column_name, '')
                # conversion to (float) before rounding it
                try:
                    column_value = float(column_value)
                    rounded_value = round(column_value, decimal_place)
                except (ValueError, TypeError):
                    # if conversion impossible keep the starting value
                    rounded_value = column_value

                # select which dictionary to use
                best_metrics = highest_best_metrics if column_name not in metrics_minimum_score else lowest_best_metrics

                # get the best values for the column=metric
                best_values = best_metrics[column_name]

                # Formatting based on comparison with the best values
                if rounded_value == best_values[0]:
                    latex_code += " & \\textbf{" + str(rounded_value) + "}"
                elif rounded_value == best_values[1]:
                    latex_code += " & \\underline{" + str(rounded_value) + "}"
                else:
                    latex_code += " & " + str(rounded_value)

            latex_code += " \\\\\n"
            latex_code += "\\addlinespace[5pt]\n"
            latex_code += "\\midrule\n"

        # closing the table
        latex_code += "\\bottomrule\n"
        latex_code += "\\end{tabular}}\n"
        latex_code += "\\caption{" + caption_for_table + " (Part " + str(part_index + 1) + ")}\n"
        latex_code += "\\end{table}\n"

        # leave space between the table parts
        if part_index < num_parts - 1:
            latex_code += "\\vspace{10pt}\n"

    return latex_code


# ------------------------------- end generate_latex_table + support functions --------------------------------

####################################################################################################################
# TABELLA PER LA GENERAZIONE DEI CONFRONTI TRA ALGORITMI INTEGRA LE INFO STATISTICHE
def generate_latex_table_pvalued(algorithms, stats_rel, comparison="", treshold_pvalue=0.5,
                                 decimal_place=3, column_width=3.0,
                                 max_columns_per_part=5,
                                 caption_for_table="Comparison between algorithms"):
    """
          The function  use a list of dictionaries, a data frame pandas and a bunch of other parameter
           to make a table able to compare the result of different algorithm of recommendation displaying
           the results of the metrics used to evaluate them. The best results will be shown in bold the best and
           underlined the second-best, could be possible that some values will have an asterisk which notify that
           the result obtained is statistically significant.

          Parameters:
          - algorithms (list): The list of dictionaries used to fill and make the table of comparison.
          - stats_rel (pandas data frame): data frame containing the relevance for statistic test.
          - comparison="" (str): .
          - treshold_pvalue=0.5 (float): reference value for the p-values.
          - decimal_place=3 (int): approximation of the values represented in the table.
          - column_width=3.0 (int): formatting value used for resizing of the table.
          - max_columns_per_part=5 (int): number of columns to be shown per table.
          - caption_for_table="Comparison between algorithms" (str): title of the table.

          Returns:
          - latex_code (str): table formatted in latex syntax.

          Author:
          - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
    """
    # check on number of columns to show
    if max_columns_per_part > 10:
        max_columns_per_part = 10

    # get names of the columns
    first_algorithm = algorithms[0]
    column_names = list(next(iter(first_algorithm.values())).keys())
    num_columns = len(column_names)

    # list of metrics where to apply the lower score is best NEED TO BE USED AS PARAMETER
    metrics_minimum_score = ['RMSE', 'MSE', 'MAE', 'Gini']

    # dictionary used for applying the logic of highlighting score
    best_metrics = {}

    # create the 2 dictonaries containing the best results for both logics high and low
    highest_best_metrics = find_highest_bests(algorithms, column_names, decimal_place)
    # print(f"il dizionario con i migliori risultati per metrica crescenti {highest_best_metrics}")
    lowest_best_metrics = find_lowest_bests(algorithms, column_names, decimal_place)
    # print(f"il dizionario con le migliori metriche più basse è {lowest_best_metrics}")

    # calculating the number of part which will be used to create the table
    num_parts = -(-num_columns // max_columns_per_part)  # Divisione arrotondata per eccesso

    latex_code = ""
    # DEBUG CODE
    # controllo degli indici del dataframe stats_ref
    # print(f"gli indici sono:\n {stats_rel.index}\n")

    # controllo dei multi-indici di colonna del df stats_ref
    # print(f"I multi indici di colonna sono_\n {stats_rel.columns}")

    for part_index in range(num_parts):
        # get index for the part
        start_col_index = part_index * max_columns_per_part
        end_col_index = (part_index + 1) * max_columns_per_part
        current_column_names = column_names[start_col_index:end_col_index]

        # evaluating the size of all columns and create header of the table part
        total_width = len(current_column_names) * column_width + 1
        latex_code += "\\begin{table}[H]\n"  # changed [ht] to [H]
        latex_code += "\\centering\n"
        latex_code += "\\resizebox{\\columnwidth}{!}{%\n"  # textwidth changed with columnwitdth
        latex_code += "\\begin{tabular}{@{}c" + " *{" + str(len(current_column_names)) + "}{" + "p{" + str(
            column_width) + "cm}}@{}}\n"
        latex_code += "\\toprule\n"
        latex_code += "\\multirow{2}{*}{Algorithms} & \\multicolumn{" + str(
            len(current_column_names)) + "}{c}{Columns} \\\\\n"
        latex_code += "\\cmidrule{2-" + str(len(current_column_names) + 1) + "}\n"

        # Add columns names
        for col_index, column_name in enumerate(current_column_names):
            latex_code += "& \\multirow{2}{*}{\\makecell{" + column_name.replace("_", "\\_") + "}} "

        latex_code += "\\\\\n"
        latex_code += "\\addlinespace[12pt]\n"
        latex_code += "\\cmidrule{2-" + str(len(current_column_names) + 1) + "}\n"
        latex_code += " & & & \\\\\n"

        # fill with data the table
        for algorithm in algorithms:
            algorithm_name = list(algorithm.keys())[0]
            # print(f"algorithm name is {algorithm_name}")
            values = list(algorithm.values())[0]
            # print(f"values is  {values}")
            latex_code += algorithm_name
            for column_name in current_column_names:
                # print(f"column value is {column_name}")
                column_value = values.get(column_name, '')
                # Conversion to (float) before rounding it
                try:
                    column_value = float(column_value)
                    rounded_value = round(column_value, decimal_place)
                except (ValueError, TypeError):
                    # if conversion is impossible keep it as string
                    rounded_value = column_value

                # Choose which dictionary to use for applying the logic to mark the best scores
                best_metrics = highest_best_metrics if column_name not in metrics_minimum_score else lowest_best_metrics

                # get best values for the column=metric
                best_values = best_metrics[column_name]

                # Formatting based on best score
                if rounded_value == best_values[0]:
                    latex_code += " & \\textbf{" + str(rounded_value) + "}"
                elif rounded_value == best_values[1]:
                    latex_code += " & \\underline{" + str(rounded_value) + "}"
                else:
                    latex_code += " & " + str(rounded_value)

                # usage of module statistic_test_table used as stt in order to create an index
                # to gain accesss to dataframe
                access_index = stt.set_access_index(comparison, algorithm_name, column_name)
                # DEBUG CODE
                """
                print(f"access_index is {access_index} \n "
                      f"il primo elemento dell'indice è {access_index[0][0]} \n "
                      f"mentre il secondo è {access_index[1]}")

                # stampe per il controllo degli indici matchati
                if access_index[0][0] in stats_rel.index:
                    print(f"INDICE DI RIGA TROVATO OOOOK")

                if access_index[1] in stats_rel.columns:
                    print(f"MULTI INDICE DI COLONNA TROVATO OOOK")
                """

                # This part opf the code will mark the statistic significant results with an asterisk
                # use the access_index to get access to stats_rel data frame and compare the p-value for marking
                if access_index[1] in stats_rel.columns and access_index[0][0] in stats_rel.index:
                    val_retrieved = stats_rel.loc[access_index]
                    if not val_retrieved.empty:
                        float_pvalue = val_retrieved.iloc[0]
                        if float_pvalue < treshold_pvalue:
                            latex_code += "*"
                else:
                    # if access_index is not present invert the name of the algorithm to create a new index and retry
                    # to gain access for checking the p-value
                    access_index = stt.set_access_index(algorithm_name, comparison, column_name)
                    if access_index[1] in stats_rel.columns and access_index[0][0] in stats_rel.index:
                        val_retrieved = stats_rel.loc[access_index]
                        if not val_retrieved.empty:
                            float_pvalue = val_retrieved.iloc[0]
                            if float_pvalue < treshold_pvalue:
                                latex_code += "*"
                    else:
                        pass  # keep on

            latex_code += " \\\\\n"
            latex_code += "\\addlinespace[5pt]\n"
            latex_code += "\\midrule\n"

        # closing the table part
        latex_code += "\\bottomrule\n"
        latex_code += "\\end{tabular}}\n"
        latex_code += "\\caption{" + caption_for_table + " (Part " + str(part_index + 1) + ")}\n"
        latex_code += "\\end{table}\n"

        # add space between the parts of the table
        if part_index < num_parts - 1:
            latex_code += "\\vspace{10pt}\n"

    return latex_code


# Questa funzione è di supporto e serve per creare il mapping tra i nomi generi syste_n usati dai test statistici
# con i nomi degli algoritmi effettivamente utilizzati in tali test, in modo da andare a modificare il dataframe
# contenente i risultati dei test statistici con questi nomi di algoritmi che forniranno i nuovi indici di
# accesso per riga
def list_to_dict_system_map(string_list):
    system_map = {}
    for i, string in enumerate(string_list, start=1):
        key = f'system_{i}'
        system_map[key] = string
    return system_map


################################################## working on ########################################################


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

    def explore_dictionary(current_dict, curnt_parent):
        if curnt_parent == main_key and isinstance(current_dict, dict):
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


def merge_dicts(*dicts, merge_key=None):
    """
    Merge two or more dictionaries, nesting them under a specified key.

    Parameters:
    - *dicts: Two or more dictionaries to merge.
    - merge_key (str): The key under which dictionaries should be nested. If None, a new key will be created.

    Returns:
    - dict: Merged dictionary.
    """
    if len(dicts) < 2:
        raise ValueError("merge_dicts requires at least two dictionaries as input.")

    if merge_key is None:
        merged_dict = {}
        for i, d in enumerate(dicts):
            key = str(i)
            merged_dict[key] = d
    else:
        merged_dict = {merge_key: {}}
        for d in dicts:
            for k, v in d.items():
                if k == merge_key:
                    merged_dict[merge_key].update(v)
                else:
                    merged_dict[merge_key][k] = v

    return merged_dict


# Esegui lo script
if __name__ == "__main__":
    # COMPLETE TEST per generate_latex_table_pvalued
    # qui andremo prima a definire e organizzare i parametri che serviranno alla funzione
    # in particolare quelli che caricano il data frame contenente i test statistici con i
    # p-value di riferimento che saranno usati per mettere in evidenzia i valori nella tabella
    # generata
    eva_yaml_paths = ["./../data/data_for_test_two/eva_report_amarSingleSource.yml",
                      "./../data/data_for_test_two/eva_report_centroidVector.yml",
                      "./../data/data_for_test_two/eva_report_classifierRecommender.yml",
                      "./../data/data_for_test_two/eva_report_indexQuery.yml",
                      "./../data/data_for_test_two/eva_report_linearPredictor.yml"]

    recsys_yaml_paths = ["./../data/data_for_test_two/rs_report_amarSingleSource.yml",
                         "./../data/data_for_test_two/rs_report_centroidVector.yml",
                         "./../data/data_for_test_two/rs_report_classifierRecommender.yml",
                         "./../data/data_for_test_two/rs_report_indexQuery.yml",
                         "./../data/data_for_test_two/rs_report_linearPredictor.yml"]

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
    keys = get_algorithm_keys(dictyonary_list)  # lista dei nomi degli algoritmi usati
    print(f"\nLa LISTA CONTENENTE I NOMI DEGLI ALGORITMI ESTRATTI DAI YML FILE\n {keys} \n\n")

    # show the dictonary extracted after processing them
    result_dictionary = nest_dictionaries(keys, my_dictio)
    print("I DIZIONARI USATI PER CREARE LA LISTA DA DARE IN INPUT PER LA CREAZIONE DELLA TABELLA:")
    for r in result_dictionary:
        print(r)

    # andiamo a caricare il dataframe che sarà usato per il confronto dei p-value
    # partiamo con la creazione del dizionario di mapping per i nomi dei sistemi
    system_map = list_to_dict_system_map(keys)
    print(f"\n\nIL DIZIONARIO USATO PER IL MAPPING DEI SISTEMI \n SYSTEM MAP: {system_map} \n\n")

    # procediamo con il recupero del dataframe
    file_ref_df = 'ttest_expand.xlsx'
    p_value_ref_df = pd.read_excel(file_ref_df, header=[0, 1], index_col=0)
    # controlliamo che il dataframe caricato non presenti problemi
    print(f"IL DATAFRAME CARICATO CHE UTILIZZEREMO PER LE REFERENZE DOPO LE OPPORTUNE MODIFICHE\n {p_value_ref_df}")
    # effettuiamo le modifiche degli indici di accesso per riga al dataframe caricato p_value_ref_df
    p_value_ref_df = stt.change_system_name(p_value_ref_df, system_map)
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
    pv_ref = 1.0
    print(f"\n4. treshold_pvalue=0.5 di default in questo caso settato con:\n"
          f"pv_ref = {pv_ref} \n")
    print(f"\n5. i restanti paremetri saranno usati di default o cambiati all'interno della chiamata.\n\n")

    # QUI TEST SULLA TABELLA
    latex_table = generate_latex_table_pvalued(result_dictionary, p_value_ref_df, reference_alg,
                                               pv_ref, max_columns_per_part=3)
    print(latex_table)

    """
    # COMPLETE TEST SECTION TO GENERATE A LATEX TABLE with function generate_table_for_comparison()
    # prova per la nuova funzione generate_table_for_comparison(), e delle funzioni di supporto
    #  best_higher_dict(), best_lower_dict(), get_metrics(), get_representation(), get_content(),
    # get_embedding(), sanitize_text()

    list_of_dicts = [
        {'algorithm': {
            'CentroidVector': {'item_field': {'plot': ['tfidf_sk']}, 'similarity': 'CosineSimilarity', 'threshold': 4,
                               'embedding_combiner': 'Centroid'},
            'sys - mean': {'Precision - macro': 0.10, 'Recall - macro': 0.30, 'F1 - macro': 0.210, 'Gini': 5.0,
                           'NDCG': 0.40,
                           'R-Precision - macro': 0.320, 'RMSE': 3.9, 'MSE': 9.0, 'MAE': 3.45, 'MRR': 0.430,
                           'MAP': 0.550,
                           'PredictionCoverage': 32.67, 'Precision@5 - macro': 0.40, 'Recall@5 - macro': 0.340,
                           'F1@5 - micro': 0.20, 'MRR@5': 0.30, 'NDCG@5': 0.120}}},
        {'algorithm': {
            'ClassifierRecommender': {'item_field': {'plot': ['tfidf_sk']}, 'classifier': 'SkKNN', 'threshold': None,
                                      'embedding_combiner': 'Centroid'},
            'sys - mean': {'Precision - macro': 1.0, 'Recall - macro': 0.980, 'F1 - macro': 0.970, 'Gini': 0.2,
                           'NDCG': 0.90,
                           'R-Precision - macro': 0.890, 'RMSE': 0.75, 'MSE': 0.25, 'MAE': 1.5, 'MRR': 0.960,
                           'MAP': 0.970,
                           'PredictionCoverage': 11.67, 'Precision@5 - macro': 0.990, 'Recall@5 - macro': 0.890,
                           'F1@5 - micro': 1.0, 'MRR@5': 0.770, 'NDCG@5': 0.80}}},
        {'algorithm': {'IndexQuery': {'item_field': {'plot': ['search_i']}, 'classic_similarity': True, 'threshold': 4},
                       'sys - mean': {'Precision - macro': 0.30, 'Recall - macro': 0.230, 'F1 - macro': 0.420,
                                      'Gini': 11.0,
                                      'NDCG': 0.4220, 'R-Precision - macro': 0.4650, 'RMSE': 63.65394315890862,
                                      'MSE': 4051.8244796775693, 'MAE': 63.65394315890862, 'MRR': 0.3510, 'MAP': 0.3770,
                                      'PredictionCoverage': 1.67, 'Precision@5 - macro': 0.330,
                                      'Recall@5 - macro': 0.330,
                                      'F1@5 - micro': 0.120, 'MRR@5': 0.2310, 'NDCG@5': 0.4230}}},
        {'algorithm': {
            'LinearPredictor': {'item_field': {'plot': ['tfidf_sk']}, 'regressor': 'SkLinearRegression',
                                'only_greater_eq': None, 'embedding_combiner': 'Centroid'},
            'sys - mean': {'Precision - macro': 0.875, 'Recall - macro': 1.0,
                           'F1 - macro': 0.9166666666666666, 'Gini': 0.8, 'NDCG': 0.830,
                           'R-Precision - macro': 0.850, 'RMSE': 0.8134293935347402, 'MSE': 0.6996958397430165,
                           'MAE': 0.7616526982381033, 'MRR': 0.830, 'MAP': 0.8110, 'PredictionCoverage': 50.0,
                           'Precision@5 - macro': 0.875, 'Recall@5 - macro': 0.860,
                           'F1@5 - micro': 0.9285714285714286, 'MRR@5': 0.8730, 'NDCG@5': 0.8220}}},
    ]

    n_col = 7  # Numero totale di colonne desiderate (4 colonne fisse + 3 colonne dinamiche)
    table_output = generate_table_for_comparison(list_of_dicts, n_col, alg_type="CentroidVector", set_title="Different setting evaluation")
    print(table_output)
    """

    # prova della funzione che estrae un dizionario con i migliori 2 valori best_higher_dict(list_of_dicts)
    # è un test sulla funzione di supporto usata da generate_table_for_comparison()
    """
    list_of_dicts = [ # in questa lista ci sono dei dizionari che non  possono essere usati direttamente da  
        # best_higher_dict o da best_lower_dict
        {'algorithm': {
            'AmarDoubleSource': {
            'network': "<class 'clayrs.recsys.network_based_algorithm.amar.amar_network.AmarNetworkBasic'>",
            'item_fields': [{'plot': ['tfidf_sk']}], 'user_fields': [{}], 'batch_size': 512, 'epochs': 5,
            'threshold': 4, 'additional_opt_parameters': {'batch_size': 512},
            'train_loss': '<function binary_cross_entropy at 0x00000234EC9A3760>',
            'optimizer_class': "<class 'torch.optim.adam.Adam'>", 'device': 'cuda:0',
            'embedding_combiner': {'Centroid': {}}, 'seed': None, 'additional_dl_parameters': {}},
             'sys - mean': {'Precision - macro': 0.875, 'Recall - macro': 1.0,
                                      'F1 - macro': 0.9166666666666666, 'Gini': 0.0, 'NDCG': 1.0,
                                      'R-Precision - macro': 1.0, 'RMSE': 2.170059972267377, 'MSE': 5.561319090821989,
                                      'MAE': 2.141366135329008, 'MRR': 1.0, 'MAP': 1.0, 'PredictionCoverage': 50.0,
                                      'pearson': 1.0, 'Precision@5 - macro': 0.875, 'Recall@5 - macro': 1.0,
                                      'F1@5 - micro': 0.9285714285714286, 'MRR@5': 1.0, 'NDCG@5': 1.0}}},
        {'algorithm': {
            'CentroidVector': {'item_field': {'plot': ['tfidf_sk']}, 'similarity': 'CosineSimilarity', 'threshold': 4,
                               'embedding_combiner': 'Centroid'},
            'sys - mean': {'Precision - macro': 1.0, 'Recall - macro': 1.0, 'F1 - macro': 1.0, 'Gini': 0.0, 'NDCG': 1.0,
                           'R-Precision - macro': 1.0, 'RMSE': 3.0, 'MSE': 9.0, 'MAE': 3.0, 'MRR': 1.0, 'MAP': 1.0,
                           'PredictionCoverage': 16.67, 'Precision@5 - macro': 1.0, 'Recall@5 - macro': 1.0,
                           'F1@5 - micro': 1.0, 'MRR@5': 1.0, 'NDCG@5': 1.0}}},
        {'algorithm': {
            'ClassifierRecommender': {'item_field': {'plot': ['tfidf_sk']}, 'classifier': 'SkKNN', 'threshold': None,
                                      'embedding_combiner': 'Centroid'},
            'sys - mean': {'Precision - macro': 1.0, 'Recall - macro': 1.0, 'F1 - macro': 1.0, 'Gini': 0.0, 'NDCG': 1.0,
                           'R-Precision - macro': 1.0, 'RMSE': 1.5, 'MSE': 2.25, 'MAE': 1.5, 'MRR': 1.0, 'MAP': 1.0,
                           'PredictionCoverage': 16.67, 'Precision@5 - macro': 1.0, 'Recall@5 - macro': 1.0,
                           'F1@5 - micro': 1.0, 'MRR@5': 1.0, 'NDCG@5': 1.0}}},
        {'algorithm': {'IndexQuery': {'item_field': {'plot': ['search_i']}, 'classic_similarity': True, 'threshold': 4},
                       'sys - mean': {'Precision - macro': 1.0, 'Recall - macro': 1.0, 'F1 - macro': 1.0, 'Gini': 0.0,
                                      'NDCG': 1.0, 'R-Precision - macro': 1.0, 'RMSE': 63.65394315890862,
                                      'MSE': 4051.8244796775693, 'MAE': 63.65394315890862, 'MRR': 1.0, 'MAP': 1.0,
                                      'PredictionCoverage': 16.67, 'Precision@5 - macro': 1.0, 'Recall@5 - macro': 1.0,
                                      'F1@5 - micro': 1.0, 'MRR@5': 1.0, 'NDCG@5': 1.0}}},
        {'algorithm': {
            'LinearPredictor': {'item_field': {'plot': ['tfidf_sk']}, 'regressor': 'SkLinearRegression',
                                           'only_greater_eq': None, 'embedding_combiner': 'Centroid'},
            'sys - mean': {'Precision - macro': 0.875, 'Recall - macro': 1.0,
                                      'F1 - macro': 0.9166666666666666, 'Gini': 0.0, 'NDCG': 1.0,
                                      'R-Precision - macro': 1.0, 'RMSE': 0.8134293935347402, 'MSE': 0.6996958397430165,
                                      'MAE': 0.7616526982381033, 'MRR': 1.0, 'MAP': 1.0, 'PredictionCoverage': 50.0,
                                      'Precision@5 - macro': 0.875, 'Recall@5 - macro': 1.0,
                                      'F1@5 - micro': 0.9285714285714286, 'MRR@5': 1.0, 'NDCG@5': 1.0}}}
    ]  # se in un dizionario fosse presente una metrica e quindi una chiave non presente negli altri nel dizionario
    # ritornato avremmo tutte le chiavi e anche quella presente in un solo dizionario questa però, avrà per valore
    # l'unico valore associato al dizionario che la conteneva l'altro sarà il valore di default data dalla funzione
    # che ritrova i massimi
    
    # qui ci facciamo aiutare da get_metrics che recupera i dizionari delle sole metriche corrispondenti ai dizionari
    # presenti in list_of_dicts
    processed_dict = [get_metrics(d) for d in list_of_dicts]
    for p in processed_dict:
        print(p)
        
    # una volta che i dizionari sono nella forma utile per le funzioni passiamo tali dizionari
    result = best_higher_dict(processed_dict)
    print(result)
    result = best_lower_dict(processed_dict)
    print(result)
    """

    # Example usage to test the function def find_highest_bests(dictionaries, keys, decimal_places):
    """
    list_of_dictionaries = [
        {'pri': {'a': 10, 'b': -20, 'c': 30}},
        {'sec': {'a': 15, 'b': 25, 'c': 35}},
        {'ter': {'a': 5, 'b': 15, 'c': 25}}
    ]

    list_of_keys = ['a', 'b', 'c']
    decimal_places = 2

    result = find_highest_bests(list_of_dictionaries, list_of_keys, decimal_places)
    print(result)

    result = find_lowest_bests(list_of_dictionaries, list_of_keys, decimal_places)
    print(result)
    """

    # prova per la funzione merge_dicts(*dicts, merge_key=None)
    """
    dict1 = {"a": "Centroid", "Repr.": "TF-IDF", "Content": "T", "F1": 0.5667}
    dict2 = {"b": "Word2Vec", "Repr.": "TF-IDF", "Content": "T", "F1": 0.5612}
    dict3 = {"c": "LSA", "Repr.": "TF-IDF", "Content": "T", "F1": 0.5536}

    A = {
        'alg': {
            'a': {
                's': 5,
                'k': 2
            },
            'b': {
                'd': 4
            },
            'c': {
                'z': 1
            }
        }
    }

    B = {'bet':
        {
            'a': {
                's': 52,
                'k': 21
            },
            't': {
                'p': 41
            },
            's': {
                'o': 11
            }
        }
    }

    C = {'oppet': {
        'h': {
            's': 92,
            'k': 210
        }
    }
    }

    merged_result = merge_dicts(dict1, dict2, dict3)
    print(merged_result)

    merge_2 = merge_dicts(A, B, C, merge_key="alg")
    print(merge_2)

    XX = {'0': {'a': 'Centroid', 'Repr.': 'TF-IDF', 'Content': 'T', 'F1': 0.5667},
          '1': {'b': 'Word2Vec', 'Repr.': 'TF-IDF', 'Content': 'T', 'F1': 0.5612},
          '2': {'c': 'LSA', 'Repr.': 'TF-IDF', 'Content': 'T', 'F1': 0.5536}}

    k_def = {'alg': {
                        'a': {'s': 5, 'k': 2},
                        'b': {'d': 4},
                        'c': {'z': 1},
                        'bet': {'a': {'s': 52, 'k': 21}, 't': {'p': 41}, 's': {'o': 11}},
                        'oppet': {'h': {'s': 92, 'k': 210}}
    }
    }

    giusto = { 'alg': {
                    'a': {'s': 5, 'k': 2},
                    'b': {'d': 4},
                    'c': {'z': 1},
                    'bet': {
                        'a': {'s': 52, 'k': 21},
                        't': {'p': 41},
                        's': {'o': 11}
                    },
                    'oppet': {
                        'h': {'s': 92, 'k': 210}
                    }
                }
             }
    """

    # COMPLETE TEST SECTION FOR TABLE GENERATION with function generate_latex_table()
    # codice per la generazione della tabella dei confronti, qui implementiamo tutte le operazione
    # per creare i dizionari di cui necessitiamo per poi passarli alla funzione generate_latex_table,
    # questo codice rappresenta un sezione di test per le funzioni generate_latex_table(), find_highest_best(),
    # find_lowest_best() e delle funzioni definite per preparare i dizionari da passare alle funzioni testate.
    """
    eva_yaml_paths = ["./../data/data_for_test_two/eva_report_amarSingleSource.yml",
                      "./../data/data_for_test_two/eva_report_centroidVector.yml",
                      "./../data/data_for_test_two/eva_report_classifierRecommender.yml",
                      "./../data/data_for_test_two/eva_report_indexQuery.yml",
                      "./../data/data_for_test_two/eva_report_linearPredictor.yml"]

    recsys_yaml_paths = ["./../data/data_for_test_two/rs_report_amarSingleSource.yml",
                         "./../data/data_for_test_two/rs_report_centroidVector.yml",
                         "./../data/data_for_test_two/rs_report_classifierRecommender.yml",
                         "./../data/data_for_test_two/rs_report_indexQuery.yml",
                         "./../data/data_for_test_two/rs_report_linearPredictor.yml"]

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
    keys = get_algorithm_keys(dictyonary_list)  # lista dei nomi degli algoritmi usati

    # show the dictonary extracted after processing them
    result_dictionary = nest_dictionaries(keys, my_dictio)
    for r in result_dictionary:
        print(r)

    # with the dictnory processed create the latex table
    latex_table = generate_latex_table(result_dictionary, max_columns_per_part=3)
    print(latex_table)
    """

    # prova per get_algorithm_keys ovvero per recuperare una lista di chiavi da dei dizionari
    """
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

    # prova per il recupero di una lista di dizionari a partire da una lista di file yaml
    """
    eva_yaml_paths = ["./../data/data_to_test/eva_report_amarSingleSource.yml",
                      "./../data/data_to_test/eva_report_centroidVector.yml",
                      "./../data/data_to_test/eva_report_classifierRecommender.yml",
                      "./../data/data_to_test/eva_report_indexQuery.yml",
                      "./../data/data_to_test/eva_report_linearPredictor.yml"]
    result = from_yaml_list_to_dict_list(eva_yaml_paths)
    for d in result:
        print(d)
    """

    # prova per la funzione che recupera un sottodizionario innestato da un dizionario fatto
    # di dizionari innestati
    """
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

    # esempio di dizionari su cui usare la funzione enerate_latex_table_based_on_representation(data_list)
    """
    {'algorithm': {'AmarDoubleSource': {'network': "<class 'clayrs.recsys.network_based_algorithm.amar.amar_network.AmarNetworkBasic'>", 'item_fields': [{'plot': ['tfidf_sk']}], 'user_fields': [{}], 'batch_size': 512, 'epochs': 5, 'threshold': 4, 'additional_opt_parameters': {'batch_size': 512}, 'train_loss': '<function binary_cross_entropy at 0x00000234EC9A3760>', 'optimizer_class': "<class 'torch.optim.adam.Adam'>", 'device': 'cuda:0', 'embedding_combiner': {'Centroid': {}}, 'seed': None, 'additional_dl_parameters': {}}, 'sys - mean': {'Precision - macro': 0.875, 'Recall - macro': 1.0, 'F1 - macro': 0.9166666666666666, 'Gini': 0.0, 'NDCG': 1.0, 'R-Precision - macro': 1.0, 'RMSE': 2.170059972267377, 'MSE': 5.561319090821989, 'MAE': 2.141366135329008, 'MRR': 1.0, 'MAP': 1.0, 'PredictionCoverage': 50.0, 'pearson': 1.0, 'Precision@5 - macro': 0.875, 'Recall@5 - macro': 1.0, 'F1@5 - micro': 0.9285714285714286, 'MRR@5': 1.0, 'NDCG@5': 1.0}}}
    {'algorithm': {'CentroidVector': {'item_field': {'plot': ['tfidf_sk']}, 'similarity': 'CosineSimilarity', 'threshold': 4, 'embedding_combiner': 'Centroid'}, 'sys - mean': {'Precision - macro': 1.0, 'Recall - macro': 1.0, 'F1 - macro': 1.0, 'Gini': 0.0, 'NDCG': 1.0, 'R-Precision - macro': 1.0, 'RMSE': 3.0, 'MSE': 9.0, 'MAE': 3.0, 'MRR': 1.0, 'MAP': 1.0, 'PredictionCoverage': 16.67, 'Precision@5 - macro': 1.0, 'Recall@5 - macro': 1.0, 'F1@5 - micro': 1.0, 'MRR@5': 1.0, 'NDCG@5': 1.0}}}
    {'algorithm': {'ClassifierRecommender': {'item_field': {'plot': ['tfidf_sk']}, 'classifier': 'SkKNN', 'threshold': None, 'embedding_combiner': 'Centroid'}, 'sys - mean': {'Precision - macro': 1.0, 'Recall - macro': 1.0, 'F1 - macro': 1.0, 'Gini': 0.0, 'NDCG': 1.0, 'R-Precision - macro': 1.0, 'RMSE': 1.5, 'MSE': 2.25, 'MAE': 1.5, 'MRR': 1.0, 'MAP': 1.0, 'PredictionCoverage': 16.67, 'Precision@5 - macro': 1.0, 'Recall@5 - macro': 1.0, 'F1@5 - micro': 1.0, 'MRR@5': 1.0, 'NDCG@5': 1.0}}}
    {'algorithm': {'IndexQuery': {'item_field': {'plot': ['search_i']}, 'classic_similarity': True, 'threshold': 4}, 'sys - mean': {'Precision - macro': 1.0, 'Recall - macro': 1.0, 'F1 - macro': 1.0, 'Gini': 0.0, 'NDCG': 1.0, 'R-Precision - macro': 1.0, 'RMSE': 63.65394315890862, 'MSE': 4051.8244796775693, 'MAE': 63.65394315890862, 'MRR': 1.0, 'MAP': 1.0, 'PredictionCoverage': 16.67, 'Precision@5 - macro': 1.0, 'Recall@5 - macro': 1.0, 'F1@5 - micro': 1.0, 'MRR@5': 1.0, 'NDCG@5': 1.0}}}
    {'algorithm': {'LinearPredictor': {'item_field': {'plot': ['tfidf_sk']}, 'regressor': 'SkLinearRegression', 'only_greater_eq': None, 'embedding_combiner': 'Centroid'}, 'sys - mean': {'Precision - macro': 0.875, 'Recall - macro': 1.0, 'F1 - macro': 0.9166666666666666, 'Gini': 0.0, 'NDCG': 1.0, 'R-Precision - macro': 1.0, 'RMSE': 0.8134293935347402, 'MSE': 0.6996958397430165, 'MAE': 0.7616526982381033, 'MRR': 1.0, 'MAP': 1.0, 'PredictionCoverage': 50.0, 'Precision@5 - macro': 0.875, 'Recall@5 - macro': 1.0, 'F1@5 - micro': 0.9285714285714286, 'MRR@5': 1.0, 'NDCG@5': 1.0}}}
"""
