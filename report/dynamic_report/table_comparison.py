from collections import defaultdict

import yaml

"""
def generate_table(list_of_dicts, n_col_desired, width=3.0, alg_type="killer"):
    # Estrai le colonne fisse (Alg., Repr., Content, Emb.)
    fixed_columns = ['Alg.', 'Repr.', 'Content', 'Emb.']

    # estrai i dizionari che contengono i nomi delle metriche e i valori associati da list_of_dicts
    metrics_dicts = [get_metrics(d) for d in list_of_dicts]
    print(metrics_dicts)

    # Estrai le colonne dinamiche dal dizionario restituito da best_higher_dict
    higher_dict = best_higher_dict(metrics_dicts)
    print(higher_dict)
    dynamic_columns = list(higher_dict.keys())

    # Calcola il numero totale di colonne che saranno usate
    total_columns = len(fixed_columns) + len(dynamic_columns)

    # verifica che le colonne volute non superino il limite
    if n_col_desired > 8:
        n_col_desired = 8

    # Inizializza la stringa della tabella LaTeX
    table_string = ""

    # Calcola il numero di tabelle da generare in base al numero di colonne dinamiche
    # Calcola il risultato e il resto della divisione
    num_tables, remainder = divmod(len(dynamic_columns), n_col_desired - len(fixed_columns))

    # Aggiungi una tabella aggiuntiva se c'è un resto
    if remainder > 0:
        num_tables += 1

    for i in range(num_tables):
        start_idx = i * (n_col_desired - len(fixed_columns))
        end_idx = (i + 1) * (n_col_desired - len(fixed_columns))
        current_dynamic_columns = dynamic_columns[start_idx:end_idx]

        # Costruisci l'header della tabella
        table_string += "\\begin{table}\n"
        table_string += "\\begin{adjustwidth}{-1 in}{-1 in}\n"
        table_string += "  \\centering\n"
        table_string += f"   \\caption{{Risultati delle metriche - Tabella {i + 1}}}\n"
        table_string += f"  \\begin{{tabular}}{{{'l' + 'c' * (len(fixed_columns) + len(current_dynamic_columns))}}}\n"
        table_string += "    \\toprule\n"
        table_string += "    " + " & ".join(fixed_columns + current_dynamic_columns) + " \\\\\n"
        table_string += "    \\midrule\n"
        table_string += "    \\midrule\n"

        # Riempi le righe della tabella con i valori dei dizionari
        for dictionary in list_of_dicts:
            table_string += "    "
            table_string += f"\\multirow{{{len(list_of_dicts)}}}{{*}}{{{alg_type}}}"

            for col in ['Repr.', 'Content', 'Emb.']:
                table_string += f" & {globals()['get_' + col.lower()](dictionary)}"

            for dynamic_col in current_dynamic_columns:
                table_string += f" & {dictionary.get(dynamic_col, '')}"

            table_string += " \\\\\n"

        table_string += "    \\bottomrule\n"
        table_string += "   \\end{tabular}\n"
        table_string += "\\end{adjustwidth}\n"
        table_string += "\\end{table}\n\n"

    return table_string
"""
# seconda versione
"""
def generate_table(list_of_dicts, n_col_desired, width=3.0, alg_type="killer"):
    # Estrai le colonne fisse (Alg., Repr., Content, Emb.)
    fixed_columns = ['Alg.', 'Repr.', 'Content', 'Emb.']

    # estrai i dizionari che contengono i nomi delle metriche e i valori associati da list_of_dicts
    metrics_dicts = [get_metrics(d) for d in list_of_dicts]
    print(metrics_dicts)

    # Estrai le colonne dinamiche dal dizionario restituito da best_higher_dict
    higher_dict = best_higher_dict(metrics_dicts)
    print(higher_dict)
    dynamic_columns = list(higher_dict.keys())

    # Calcola il numero totale di colonne che saranno usate
    total_columns = len(fixed_columns) + len(dynamic_columns)

    # verifica che le colonne volute non superino il limite
    if n_col_desired > 8:
        n_col_desired = 8

    # Inizializza la stringa della tabella LaTeX
    table_string = ""

    # Calcola il numero di tabelle da generare in base al numero di colonne dinamiche
    # Calcola il risultato e il resto della divisione
    num_tables, remainder = divmod(len(dynamic_columns), n_col_desired - len(fixed_columns))

    # Aggiungi una tabella aggiuntiva se c'è un resto
    if remainder > 0:
        num_tables += 1

    for i in range(num_tables):
        start_idx = i * (n_col_desired - len(fixed_columns))
        end_idx = (i + 1) * (n_col_desired - len(fixed_columns))
        current_dynamic_columns = dynamic_columns[start_idx:end_idx]

        # Costruisci l'header della tabella
        table_string += "\\begin{table}\n"
        table_string += "\\begin{adjustwidth}{-1 in}{-1 in}\n"
        table_string += "  \\centering\n"
        table_string += f"   \\caption{{Risultati delle metriche - Tabella {i + 1}}}\n"
        table_string += f"  \\begin{{tabular}}{{{'l' + 'c' * (len(fixed_columns) + len(current_dynamic_columns))}}}\n"
        table_string += "    \\toprule\n"
        table_string += "    " + " & ".join(fixed_columns + current_dynamic_columns) + " \\\\\n"
        table_string += "    \\midrule\n"
        table_string += "    \\midrule\n"

        # Riempi le righe della tabella con i valori dei dizionari
        for dictionary, metrics_dict in zip(list_of_dicts, metrics_dicts):
            table_string += "    "
            table_string += f"\\multirow{{{len(list_of_dicts)}}}{{*}}{{{alg_type}}}"

            # Riempimento delle colonne fisse
            for col in fixed_columns[1:]:  # Parti da 'Repr.' per evitare di ripetere 'Alg.'
                if col == 'Repr.':
                    table_string += f" & {get_representation(dictionary)}"
                elif col == 'Content':
                    table_string += f" & {get_content(dictionary)}"
                elif col == 'Emb.':
                    table_string += f" & {get_embedding(dictionary)}"

            # Riempimento delle colonne dinamiche
            for dynamic_col in current_dynamic_columns:
                value = metrics_dict.get(dynamic_col, '')
                table_string += f" & {value if value is not None else ''}"

            table_string += " \\\\\n"

        table_string += "    \\bottomrule\n"
        table_string += "   \\end{tabular}\n"
        table_string += "\\end{adjustwidth}\n"
        table_string += "\\end{table}\n\n"

    return table_string
"""
# terza versione
"""
def generate_table(list_of_dicts, n_col_desired, width=3.0, alg_type="killer"):
    # Estrai le colonne fisse (Alg., Repr., Content, Emb.)
    fixed_columns = ['Alg.', 'Repr.', 'Content', 'Emb.']

    # estrai i dizionari che contengono i nomi delle metriche e i valori associati da list_of_dicts
    metrics_dicts = [get_metrics(d) for d in list_of_dicts]
    print(metrics_dicts)

    # Estrai le colonne dinamiche dal dizionario restituito da best_higher_dict
    higher_dict = best_higher_dict(metrics_dicts)
    print(higher_dict)
    dynamic_columns = list(higher_dict.keys())

    # Calcola il numero totale di colonne che saranno usate
    total_columns = len(fixed_columns) + len(dynamic_columns)

    # verifica che le colonne volute non superino il limite
    if n_col_desired > 8:
        n_col_desired = 8

    # Inizializza la stringa della tabella LaTeX
    table_string = ""

    # Calcola il numero di tabelle da generare in base al numero di colonne dinamiche
    # Calcola il risultato e il resto della divisione
    num_tables, remainder = divmod(len(dynamic_columns), n_col_desired - len(fixed_columns))

    # Aggiungi una tabella aggiuntiva se c'è un resto
    if remainder > 0:
        num_tables += 1

    for i in range(num_tables):
        start_idx = i * (n_col_desired - len(fixed_columns))
        end_idx = (i + 1) * (n_col_desired - len(fixed_columns))
        current_dynamic_columns = dynamic_columns[start_idx:end_idx]

        # Costruisci l'header della tabella
        table_string += "\\begin{table}\n"
        table_string += "\\begin{adjustwidth}{-1 in}{-1 in}\n"
        table_string += "  \\centering\n"
        table_string += f"   \\caption{{Risultati delle metriche - Tabella {i + 1}}}\n"
        table_string += f"  \\begin{{tabular}}{{{'l' + 'c' * (len(fixed_columns) + len(current_dynamic_columns))}}}\n"
        table_string += "    \\toprule\n"
        table_string += "    " + " & ".join(fixed_columns + current_dynamic_columns) + " \\\\\n"
        table_string += "    \\midrule\n"
        table_string += "    \\midrule\n"

        # Riempi le righe della tabella con i valori dei dizionari
        for dictionary, metrics_dict in zip(list_of_dicts, metrics_dicts):
            # Costruisci la parte dell'header multirow solo per la prima riga di ogni gruppo
            if list_of_dicts.index(dictionary) % len(list_of_dicts) == 0:
                table_string += f"    \\multirow{{{len(list_of_dicts)}}}{{*}}{{{alg_type}}}"

            # Riempimento delle colonne fisse
            for col in fixed_columns[1:]:  # Parti da 'Repr.' per evitare di ripetere 'Alg.'
                if col == 'Repr.':
                    table_string += f" & {get_representation(dictionary)}"
                elif col == 'Content':
                    table_string += f" & {get_content(dictionary)}"
                elif col == 'Emb.':
                    table_string += f" & {get_embedding(dictionary)}"

            # Riempimento delle colonne dinamiche
            for dynamic_col in current_dynamic_columns:
                value = metrics_dict.get(dynamic_col, '')
                table_string += f" & {value if value is not None else ''}"

            table_string += " \\\\\n"

        table_string += "    \\bottomrule\n"
        table_string += "   \\end{tabular}\n"
        table_string += "\\end{adjustwidth}\n"
        table_string += "\\end{table}\n\n"

    return table_string
"""
# quarta versione al momento funzionante BISOGNA IMPLEMNETARE I MECCANISMI PER EVIDENZIARE I MIGLIORI RISULTATI
"""
def generate_table(list_of_dicts, n_col_desired, width=3.0, alg_type="killer"):
    # Estrai le colonne fisse (Alg., Repr., Content, Emb.)
    fixed_columns = ['Alg.', 'Repr.', 'Content', 'Emb.']

    # estrai i dizionari che contengono i nomi delle metriche e i valori associati da list_of_dicts
    metrics_dicts = [get_metrics(d) for d in list_of_dicts]
    print(metrics_dicts)

    # Estrai le colonne dinamiche dal dizionario restituito da best_higher_dict
    higher_dict = best_higher_dict(metrics_dicts)
    print(higher_dict)
    dynamic_columns = list(higher_dict.keys())

    # Calcola il numero totale di colonne che saranno usate
    total_columns = len(fixed_columns) + len(dynamic_columns)

    # verifica che le colonne volute non superino il limite
    if n_col_desired > 8:
        n_col_desired = 8

    # Inizializza la stringa della tabella LaTeX
    table_string = ""

    # Calcola il numero di tabelle da generare in base al numero di colonne dinamiche
    # Calcola il risultato e il resto della divisione
    num_tables, remainder = divmod(len(dynamic_columns), n_col_desired - len(fixed_columns))

    # Aggiungi una tabella aggiuntiva se c'è un resto
    if remainder > 0:
        num_tables += 1

    for i in range(num_tables):
        start_idx = i * (n_col_desired - len(fixed_columns))
        end_idx = (i + 1) * (n_col_desired - len(fixed_columns))
        current_dynamic_columns = dynamic_columns[start_idx:end_idx]

        # Costruisci l'header della tabella
        table_string += "\\begin{table}\n"
        table_string += "\\begin{adjustwidth}{-1 in}{-1 in}\n"
        table_string += "  \\centering\n"
        table_string += f"   \\caption{{Risultati delle metriche - Tabella {i + 1}}}\n"
        table_string += f"  \\begin{{tabular}}{{{'l' + 'c' * (len(fixed_columns) + len(current_dynamic_columns))}}}\n"
        table_string += "    \\toprule\n"
        table_string += "    " + " & ".join(fixed_columns + current_dynamic_columns) + " \\\\\n"
        table_string += "    \\midrule\n"
        table_string += "    \\midrule\n"

        # Riempi le righe della tabella con i valori dei dizionari
        for i, (dictionary, metrics_dict) in enumerate(zip(list_of_dicts, metrics_dicts)):
            # Costruisci la parte dell'header multirow solo per la prima riga di ogni gruppo
            if i % len(list_of_dicts) == 0:
                table_string += f"    \\multirow{{{len(list_of_dicts)}}}{{*}}{{{alg_type}}}"

            # Riempimento delle colonne fisse
            for col in fixed_columns[1:]:  # Parti da 'Repr.' per evitare di ripetere 'Alg.'
                if col == 'Repr.':
                    table_string += f" & {get_representation(dictionary)}"
                elif col == 'Content':
                    table_string += f" & {get_content(dictionary)}"
                elif col == 'Emb.':
                    table_string += f" & {get_embedding(dictionary)}"

            # Riempimento delle colonne dinamiche
            for dynamic_col in current_dynamic_columns:
                value = metrics_dict.get(dynamic_col, '')
                table_string += f" & {value if value is not None else ''}"

            table_string += " \\\\\n"

        table_string += "    \\bottomrule\n"
        table_string += "   \\end{tabular}\n"
        table_string += "\\end{adjustwidth}\n"
        table_string += "\\end{table}\n\n"

    return table_string
"""
# quinta versione
def generate_table(list_of_dicts, n_col_desired, width=3.0, alg_type="killer", round_to=3):
    # Estrai le colonne fisse (Alg., Repr., Content, Emb.)
    fixed_columns = ['Alg.', 'Repr.', 'Content', 'Emb.']

    # Lista di metriche il cui punteggio migliore è quello minimo
    metrics_minimum_score = ['RMSE', 'MSE', 'MAE', 'Gini']

    # estrai i dizionari che contengono i nomi delle metriche e i valori associati da list_of_dicts
    metrics_dicts = [get_metrics(d) for d in list_of_dicts]
    print(metrics_dicts)

    # Estrai le colonne dinamiche dal dizionario restituito da best_higher_dict
    # higher_dict risulta essere il dizionario che contiene i massimo e il secondo massimo per ogni colonna della
    # tabella
    higher_dict = best_higher_dict(metrics_dicts)
    print(higher_dict)
    dynamic_columns = list(higher_dict.keys())

    # andiamo a creare il dizionario che contiene il minimo e il secondo minimo per ogni colonna della tabella
    lower_dict = best_lower_dict(metrics_dicts)
    print(lower_dict)

    # Calcola il numero totale di colonne che saranno usate
    total_columns = len(fixed_columns) + len(dynamic_columns)

    # verifica che le colonne volute non superino il limite
    if n_col_desired > 8:
        n_col_desired = 8

    # Inizializza la stringa della tabella LaTeX
    table_string = ""

    # Calcola il numero di tabelle da generare in base al numero di colonne dinamiche
    # Calcola il risultato e il resto della divisione
    num_tables, remainder = divmod(len(dynamic_columns), n_col_desired - len(fixed_columns))

    # Aggiungi una tabella aggiuntiva se c'è un resto
    if remainder > 0:
        num_tables += 1

    for i in range(num_tables):
        start_idx = i * (n_col_desired - len(fixed_columns))
        end_idx = (i + 1) * (n_col_desired - len(fixed_columns))
        current_dynamic_columns = dynamic_columns[start_idx:end_idx]

        # Costruisci l'header della tabella
        table_string += "\\begin{table}\n"
        table_string += "\\begin{adjustwidth}{-1 in}{-1 in}\n"
        table_string += "  \\centering\n"
        table_string += f"   \\caption{{Risultati delle metriche - Tabella {i + 1}}}\n"
        table_string += f"  \\begin{{tabular}}{{{'l' + 'c' * (len(fixed_columns) + len(current_dynamic_columns))}}}\n"
        table_string += "    \\toprule\n"
        table_string += "    " + " & ".join(fixed_columns + current_dynamic_columns) + " \\\\\n"
        table_string += "    \\midrule\n"
        table_string += "    \\midrule\n"

        # Riempi le righe della tabella con i valori dei dizionari
        for i, (dictionary, metrics_dict) in enumerate(zip(list_of_dicts, metrics_dicts)):
            # Costruisci la parte dell'header multirow solo per la prima riga di ogni gruppo
            if i % len(list_of_dicts) == 0:
                table_string += f"    \\multirow{{{len(list_of_dicts)}}}{{*}}{{{alg_type}}}"

            # Riempimento delle colonne fisse
            for col in fixed_columns[1:]:  # Parti da 'Repr.' per evitare di ripetere 'Alg.'
                if col == 'Repr.':
                    table_string += f" & {get_representation(dictionary)}"
                elif col == 'Content':
                    table_string += f" & {get_content(dictionary)}"
                elif col == 'Emb.':
                    table_string += f" & {get_embedding(dictionary)}"

            # Riempimento delle colonne dinamiche
            for dynamic_col in current_dynamic_columns:
                value = metrics_dict.get(dynamic_col, '')

                # Verifica se la colonna dinamica è presente nella lista metrics_minimum_score
                if dynamic_col in metrics_minimum_score:
                    # Se presente, confronta con il dizionario lower_dict
                    if value == lower_dict[dynamic_col][0]:  # Primo valore in lower_dict
                        table_string += f" & \\textbf{{{round(value, round_to) if value is not None else ''}}}"
                    elif value == lower_dict[dynamic_col][1]:  # Secondo valore in lower_dict
                        table_string += f" & \\underline{{{round(value, round_to) if value is not None else ''}}}"
                    else:
                        table_string += f" & {round(value, round_to) if value is not None else ''}"
                else:
                    # Se non presente, confronta con il dizionario higher_dict
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


def best_lower_dict(list_of_dicts):
    key_values = defaultdict(list)

    # Estrai tutte le chiavi dai dizionari
    all_keys = set()
    for d in list_of_dicts:
        all_keys.update(d.keys())

    # Trova i minimi e i secondi minimi per ogni chiave
    for key in all_keys:
        values = [d.get(key, float('inf')) for d in list_of_dicts]
        values.sort()
        min_value = values[0]
        second_min_value = values[1] if len(values) > 1 else float('inf')
        key_values[key] = (min_value, second_min_value)

    return dict(key_values)


def best_higher_dict(list_of_dicts):
    key_values = defaultdict(list)

    # Estrai tutte le chiavi dai dizionari
    all_keys = set()
    for d in list_of_dicts:
        all_keys.update(d.keys())

    # Trova i massimi e i secondi massimi per ogni chiave
    for key in all_keys:
        values = [d.get(key, float('-inf')) for d in list_of_dicts]
        values.sort(reverse=True)
        max_value = values[0]
        second_max_value = values[1] if len(values) > 1 else float('-inf')
        key_values[key] = (max_value, second_max_value)

    return dict(key_values)


# FUNZIONE DI CREAZIONE TABELLA CORRETTA
def set_table_complete(columns, name, list_tuple, algo, title="comparison of results"):
    # Verifica che columns sia uguale alla lunghezza di name
    if columns != len(name) + 4:  # 4 colonne fisse iniziali (Alg., Repr., Content, Emb.)
        return ""

    # Costruisci la stringa per la dichiarazione delle colonne
    column_declaration = 'l' + ('c' * (columns - 1))

    # Costruisci la stringa per l'intestazione della tabella
    add_to_table = "\\begin{table}\n"
    add_to_table += "\\begin{adjustwidth}{-1 in}{-1 in}\n"  # adjust margin
    add_to_table += "  \\centering\n"
    add_to_table += f"   \\caption{{{title}}}\n"
    add_to_table += f"  \\begin{{tabular}}{{{column_declaration}}}\n"
    add_to_table += "    \\toprule\n"
    add_to_table += f"    Alg. & Repr. & Content & Emb. & {' & '.join(name)} \\\\\n"
    add_to_table += "    \\midrule\n"
    add_to_table += "    \\midrule\n"  # to make double line

    # Aggiungi le righe dalla lista di tuple
    for i, row_tuple in enumerate(list_tuple):
        add_to_table += "    "  # Aggiungi questa riga
        add_to_table += f"    \\multirow{{{len(list_tuple)}}}{{*}}{{{algo}}} & " if i == 0 else " & "  # Modifica questa riga
        for value in row_tuple:
            add_to_table += f"{value} & "
        add_to_table = add_to_table[:-2]  # Rimuovi l'ultimo "& "
        add_to_table += " \\\\\n"

    # Completa la stringa della tabella
    add_to_table += "    \\bottomrule\n"
    add_to_table += "   \\end{tabular}\n"
    add_to_table += "\\end{adjustwidth}\n"
    add_to_table += "\\end{table}"

    return add_to_table


# usate per supporto non più necessaria
"""
def set_table(columns, name):
    # Verifica che columns sia uguale alla lunghezza di name
    if columns != len(name) + 4:  # 4 colonne fisse iniziali (Alg., Repr., Content, Emb.)
        return ""

    # Costruisci la stringa per la dichiarazione delle colonne
    column_declaration = 'l' + ('c' * (columns - 1))

    # Costruisci la stringa per l'intestazione della tabella
    table_header = "\\begin{table}\n"
    table_header += f"\\begin{{tabular}}{{{column_declaration}}}\n"
    table_header += "    \\toprule\n"
    table_header += f"    Alg. & Repr. & Content & Emb. & {' & '.join(name)} \\\\\n"
    table_header += "    \\midrule\n"

    # Completa la stringa della tabella
    table_header += "    \\bottomrule\n"
    table_header += "\\end{tabular}\n"
    table_header += "\\end{table}"

    return table_header
"""

"""
def get_metrics(data_dict):
    def extract_metrics(d, prefix=''):
        result = []
        for key, value in d.items():
            if isinstance(value, dict):
                result.extend(extract_metrics(value, f'{prefix}{key} - '))
            else:
                result.append((f'{prefix}{key}', value))
        return result

    for key, value in data_dict.items():
        if key == 'sys - mean':
            return extract_metrics(value)
        elif isinstance(value, dict):
            nested_result = get_metrics(value)
            if nested_result:
                return nested_result

    return []
"""


# questa versione ritorna un dizionario anzicché una lista di tuple che racchiudano il nome della metrica e
# il valore associato dal dizionario estratto.
def get_metrics(data_dict):
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


def get_embedding(data_dict):
    # Cerca ricorsivamente la chiave 'embedding_combiner' nel dizionario
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

    # Trova il valore associato a 'embedding_combiner'
    embedding_data = find_embedding_combiner(data_dict)

    # Stampa di debug per vedere il tipo di embedding_data
    print(f"Tipo di embedding_data: {type(embedding_data)}")

    # Se non c'è nessun valore associato, restituisci una stringa vuota
    if embedding_data is None:
        return ""

    # Se è una stringa, restituisci direttamente quella stringa
    elif isinstance(embedding_data, str):
        return embedding_data

    # Se è un numero, trasformalo in stringa e restituisci
    elif isinstance(embedding_data, (int, float)):
        return str(embedding_data)

    # Se è una lista, restituisci gli elementi come stringhe concatenate da '+'
    elif isinstance(embedding_data, list):
        return " + ".join(map(str, embedding_data))

    # Se è un dizionario, restituisci la concatenazione delle chiavi di primo livello
    elif isinstance(embedding_data, dict):
        return " + ".join(embedding_data.keys())

    # In tutti gli altri casi, restituisci una stringa vuota
    return ""


# questa funzione recupera sotto forma di stringa il contenuto utilizzato per l'esperimento ovvero tutti i field usati
# e li restituisce come stringa concatenata con il +
def get_content(data_dict):
    # Verifica se la chiave 'algorithm' è presente nel dizionario
    if 'algorithm' in data_dict:
        # Cerca la chiave specifica sotto 'algorithm' che contiene 'item_field'
        algorithm_keys = [key for key, value in data_dict['algorithm'].items() if 'item_field' in value]

        # Se troviamo una chiave, recuperiamo i valori associati
        if algorithm_keys:
            # Scegliamo la prima chiave (potrebbe esserci solo una)
            algorithm_key = algorithm_keys[0]

            # Recupera i valori sotto 'item_field'
            item_field_data = data_dict['algorithm'][algorithm_key].get('item_field', {})

            # Estrai chiavi di primo livello da 'item_field'
            content_keys = [key for key, value in item_field_data.items() if isinstance(value, (list, dict)) and value]

            # Unisci le chiavi ottenute
            content = " + ".join(map(str, content_keys))

            return content

    # Restituisci una stringa vuota se la chiave 'algorithm' non è presente
    return ""


# questa funzione recupera sotto forma di stringa quello che sarà utilizzato
# per riempirre la casella sotto la colonna Repr.
# della rapresentazione utilizzata
def get_representation(data_dict):
    # Verifica se la chiave 'algorithm' è presente nel dizionario
    if 'algorithm' in data_dict:
        # Cerca la chiave specifica sotto 'algorithm' che contiene 'item_field'
        algorithm_keys = [key for key, value in data_dict['algorithm'].items() if 'item_field' in value]

        # Se troviamo una chiave, recuperiamo i valori associati
        if algorithm_keys:
            # Scegliamo la prima chiave (potrebbe esserci solo una)
            algorithm_key = algorithm_keys[0]

            # Recupera i valori sotto 'item_field'
            item_field_data = data_dict['algorithm'][algorithm_key].get('item_field', {})

            # Estrai i valori da 'item_field' ignorando liste vuote
            item_field_values = [value for key, value in item_field_data.items() if isinstance(value, list) and value]

            # Unisci i valori ottenuti
            representation_strings = [" + ".join(map(str, values)) for values in item_field_values]

            # Unisci le stringhe ottenute
            representation = " + ".join(representation_strings)

            return representation

    # Restituisci una stringa vuota se la chiave 'algorithm' non è presente
    return ""


def sanitize_latex_string(input_str):
    return input_str.replace("_", "\\_").replace("&", "\\&").replace("#", "\\#")


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


# funzione usata per creare le tabelle con i risultati su un determinato algoritmo che usa differenti rappresentazioni
# dei dati OK
def generate_latex_table_based_on_representation(data_list, num_columns, title="results", alg_column_value="WWW"):
    result = ""
    number_first_static_colomns = 4
    # Ottieni il numero di righe per ogni tabella
    num_raw = len(data_list)
    # print(f"il numero di righe num_raw è {num_raw}")

    # Ottieni il numero di colonne da posizionare sotto Metrics
    num_metrics_columns = num_columns - number_first_static_colomns
    # print(f"il numero di colonne dedicato alle metriche è {num_metrics_columns}")

    # ottieni il numero di metriche uguale al numero di elementi presenti nella lista di tuple
    # restituita da get_metrics
    num_all_metrics = len(get_metrics(data_list[0]))
    # print(f"il numero di tutte le metriche è {num_all_metrics}")

    # stabiliamo il numero di tabelle da utilizzare
    num_tables = 0
    # Assicurati che num_metrics_columns sia <= rispetto a num_all_metrics
    if num_metrics_columns >= num_all_metrics:
        num_tables = 1
        # print("coooool I\'m here")
        # abbiamo una sola tabella e il numero di metriche sarà proprio nume_metrics_all
        num_metrics_columns = num_all_metrics
        # print(f"num_metrics_columns is {num_metrics_columns}")
        culomns_metrics_name = extract_first_elements(get_metrics(data_list[0]))
        # print(culomns_metrics_name)
        tuple_format = [make_tuple_for_table(data, 0, num_all_metrics) for data in data_list]
        # print(tuple_format)
        adjust_colomns = number_first_static_colomns + num_metrics_columns
        result = set_table_complete(adjust_colomns, culomns_metrics_name, tuple_format, alg_column_value, title)
    else:
        # Calcola il numero di tabelle necessarie in base al numero di colonne Metrics
        num_tables = num_all_metrics // num_metrics_columns
        colomn_for_last_table = num_all_metrics % num_metrics_columns
        first = 0
        last = num_metrics_columns
        metrics_name = extract_first_elements(get_metrics(data_list[0]))
        # print(f"il nome di tutte le metriche è {metrics_name}")
        # print("well enough")
        # Se la divisione ha resto, aggiungi 1 al numero di tabelle
        if colomn_for_last_table != 0:
            num_tables += 1
            # print(f"il numero di tabelle da è {num_tables}")
            for i in range(num_tables - 1):
                # estri i nomi per le colonne della tabella i
                metrics_name_chuck = metrics_name[first:last]
                # print(metrics_name_chuck)
                # print(f"first is {first} and last is {last}")
                # prepara i dati da inserire per la tabella i
                tuple_format_for_table = [make_tuple_for_table(data, first, last) for data in data_list]
                # print(tuple_format_for_table)
                # aggiungi i la tabella a result
                result += set_table_complete(num_columns,
                                             metrics_name_chuck, tuple_format_for_table, alg_column_value, title)
                # print(result)
                result += " \n\n"
                first = last
                last = first + num_metrics_columns
                # print(f"first is {first} and last is {last}")
            # ora recuperiamo i nomi delle ultime metriche rimaste per l'ultima colonna
            # print(f"colomn for last table {colomn_for_last_table}") # debug
            # print(metrics_name[- colomn_for_last_table:]) # debug
            name_chunk = metrics_name[- colomn_for_last_table:]
            # print(name_chunk)
            last = first + colomn_for_last_table
            # print(f"first is {first} and last is {last}")
            tuple_format_for_table = [make_tuple_for_table(data, first, last) for data in data_list]
            # print(tuple_format_for_table)
            new_number_of_colmns = number_first_static_colomns + colomn_for_last_table
            # print(new_number_of_colmns)
            result += set_table_complete(new_number_of_colmns,
                                         name_chunk, tuple_format_for_table, alg_column_value, title)
        else:
            # print("in the else side")
            for i in range(num_tables):
                # print(f"first is {first} and last is {last}")
                # estri i nomi per le colonne della tabella i
                metrics_name_chuck = metrics_name[first:last]
                # print(metrics_name_chuck)
                # prepara i dati da inserire per la tabella i
                tuple_format_for_table = [make_tuple_for_table(data, first, last) for data in data_list]
                # print(tuple_format_for_table)
                # aggiungi i la tabella a result
                result += set_table_complete(num_columns,
                                             metrics_name_chuck, tuple_format_for_table, alg_column_value, title)
                result += "\n\n"
                # print(result)
                first = last
                last = first + num_metrics_columns
                # print(f"first is {first} and last is {last}")

    return result


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


def find_highest_bests(dictionaries, keys, decimal_places):
    result = {}

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


# funzione speculare alla precedente utilizzata per trovare i due minimi come valori da evidenziare nella tabella
# la funzione offre supporto alla funzione generate_latex_table
def find_lowest_bests(dictionaries, keys_list, decimal_places):
    result = {}

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


# funzione per il confronto tra algoritmi PRONTA E FUNZIONANTE OK
"""
def generate_latex_table(algorithms, decimal_place=3, column_width=3.0, max_columns_per_part=5):
    # Estrai le chiavi (nomi di colonne) dal primo dizionario
    first_algorithm = algorithms[0]
    column_names = list(next(iter(first_algorithm.values())).keys())
    print(column_names)  # debug to check what it is inside
    num_columns = len(column_names)

    # genera i due dizionari che terranno traccia dei due punteggi più alti migliori e dei
    # 2 punteggi più bassi migliori
    # highest_best_metrics = find_highest_bests(first_algorithm, column_names, decimal_place)
    # print(highest_best_metrics)
    # lowest_best_metrics = find_lowest_bests(first_algorithm, column_names, decimal_place)
    # print(lowest_best_metrics)

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
        latex_code += "\\addlinespace[5pt]\n"
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
                    rounded_value = round(column_value, decimal_place)
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
"""

"""
# work in progress
def generate_latex_table(algorithms, decimal_place=3, column_width=3.0, max_columns_per_part=5):
    # Estrai le chiavi (nomi di colonne) dal primo dizionario
    first_algorithm = algorithms[0]
    column_names = list(next(iter(first_algorithm.values())).keys())
    print(f" i nomi delle metriche con le quali abbiamo a che fare {column_names}")
    num_columns = len(column_names)

    # lista di metriche il cui punteggio migliore è queelo minimo
    metrics_minimum_score = ['RMSE', 'MSE', 'MAE', 'Gini']

    # genera i due dizionari che terranno traccia dei due punteggi più alti migliori e dei
    # 2 punteggi più bassi migliori
    highest_best_metrics = find_highest_bests(algorithms, column_names, decimal_place)
    print(f"il dizionario dei migliori risultati per le metriche{highest_best_metrics}")

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
        latex_code += "\\addlinespace[5pt]\n"
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
                    rounded_value = round(column_value, decimal_place)
                except (ValueError, TypeError):
                    # Se la conversione non è possibile, mantieni il valore come stringa
                    rounded_value = column_value

                # Estrai i valori migliori per la colonna corrente
                highest_best_values = highest_best_metrics[column_name]

                # Formatta il valore in base ai risultati migliori
                if rounded_value == highest_best_values[0]:
                    latex_code += " & \\textbf{" + str(rounded_value) + "}"
                elif rounded_value == highest_best_values[1]:
                    latex_code += " & \\underline{" + str(rounded_value) + "}"
                else:
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
"""


def generate_latex_table(algorithms, decimal_place=3, column_width=3.0,
                         max_columns_per_part=5, caption_for_table="Comparison between algorithms"):
    # Controllo per assicurarsi che max_columns_per_part non superi mai 10
    if max_columns_per_part > 10:
        max_columns_per_part = 10

    # Estrai le chiavi (nomi di colonne) dal primo dizionario
    first_algorithm = algorithms[0]
    column_names = list(next(iter(first_algorithm.values())).keys())
    num_columns = len(column_names)

    # Lista di metriche il cui punteggio migliore è quello minimo
    metrics_minimum_score = ['RMSE', 'MSE', 'MAE', 'Gini']

    # Dizionario dinamico per determinare i migliori punteggi
    best_metrics = {}

    # genera i due dizionari che terranno traccia dei due punteggi più alti migliori e dei
    # 2 punteggi più bassi migliori
    highest_best_metrics = find_highest_bests(algorithms, column_names, decimal_place)
    print(f"il dizionario con i migliori risultati per metrica crescenti {highest_best_metrics}")
    lowest_best_metrics = find_lowest_bests(algorithms, column_names, decimal_place)
    print(f"il dizionario con le migliori metriche più basse è {lowest_best_metrics}")

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
            len(current_column_names)) + "}{c}{Columns} \\\\\n"
        latex_code += "\\cmidrule{2-" + str(len(current_column_names) + 1) + "}\n"

        # Aggiungi i nomi delle colonne
        for col_index, column_name in enumerate(current_column_names):
            latex_code += "& \\multirow{2}{*}{\\makecell{" + column_name.replace("_", "\\_") + "}} "

        latex_code += "\\\\\n"
        latex_code += "\\addlinespace[5pt]\n"
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
                    rounded_value = round(column_value, decimal_place)
                except (ValueError, TypeError):
                    # Se la conversione non è possibile, mantieni il valore come stringa
                    rounded_value = column_value

                # Determina quale dizionario utilizzare per ottenere i migliori punteggi
                best_metrics = highest_best_metrics if column_name not in metrics_minimum_score else lowest_best_metrics

                # Estrai i valori migliori per la colonna corrente
                best_values = best_metrics[column_name]

                # Formatta il valore in base ai risultati migliori
                if rounded_value == best_values[0]:
                    latex_code += " & \\textbf{" + str(rounded_value) + "}"
                elif rounded_value == best_values[1]:
                    latex_code += " & \\underline{" + str(rounded_value) + "}"
                else:
                    latex_code += " & " + str(rounded_value)

            latex_code += " \\\\\n"
            latex_code += "\\addlinespace[5pt]\n"
            latex_code += "\\midrule\n"

        # Aggiungi la parte finale del codice LaTeX
        latex_code += "\\bottomrule\n"
        latex_code += "\\end{tabular}}\n"
        latex_code += "\\caption{" + caption_for_table + " (Part " + str(part_index + 1) + ")}\n"
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
        latex_code += "\\caption{Comparison of the algorithms (Part " + str(part_index + 1) + ")}\n"
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

    list_of_dicts = [
        {'algorithm': {
            'CentroidVector': {'item_field': {'plot': ['tfidf_sk']}, 'similarity': 'CosineSimilarity', 'threshold': 4,
                               'embedding_combiner': 'Centroid'},
            'sys - mean': {'Precision - macro': 0.10, 'Recall - macro': 0.30, 'F1 - macro': 0.210, 'Gini': 5.0, 'NDCG': 0.40,
                           'R-Precision - macro': 0.320, 'RMSE': 3.9, 'MSE': 9.0, 'MAE': 3.45, 'MRR': 0.430, 'MAP': 0.550,
                           'PredictionCoverage': 32.67, 'Precision@5 - macro': 0.40, 'Recall@5 - macro': 0.340,
                           'F1@5 - micro': 0.20, 'MRR@5': 0.30, 'NDCG@5': 0.120}}},
        {'algorithm': {
            'ClassifierRecommender': {'item_field': {'plot': ['tfidf_sk']}, 'classifier': 'SkKNN', 'threshold': None,
                                      'embedding_combiner': 'Centroid'},
            'sys - mean': {'Precision - macro': 1.0, 'Recall - macro': 0.980, 'F1 - macro': 0.970, 'Gini': 0.2, 'NDCG': 0.90,
                           'R-Precision - macro': 0.890, 'RMSE': 0.75, 'MSE': 0.25, 'MAE': 1.5, 'MRR': 0.960, 'MAP': 0.970,
                           'PredictionCoverage': 11.67, 'Precision@5 - macro': 0.990, 'Recall@5 - macro': 0.890,
                           'F1@5 - micro': 1.0, 'MRR@5': 0.770, 'NDCG@5': 0.80}}},
        {'algorithm': {'IndexQuery': {'item_field': {'plot': ['search_i']}, 'classic_similarity': True, 'threshold': 4},
                       'sys - mean': {'Precision - macro': 0.30, 'Recall - macro': 0.230, 'F1 - macro': 0.420, 'Gini': 11.0,
                                      'NDCG': 0.4220, 'R-Precision - macro': 0.4650, 'RMSE': 63.65394315890862,
                                      'MSE': 4051.8244796775693, 'MAE': 63.65394315890862, 'MRR': 0.3510, 'MAP': 0.3770,
                                      'PredictionCoverage': 1.67, 'Precision@5 - macro': 0.330, 'Recall@5 - macro': 0.330,
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
    
    n_col = 7 # Numero totale di colonne desiderate (4 colonne fisse + 3 colonne dinamiche)
    table_output = generate_table(list_of_dicts, n_col)
    print(table_output)


    # prova della funzione che estrae un dizionario con i migliori 2 valori best_higher_dict(list_of_dicts)
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

    # prova per generate_latex_table_based_on_representation
    """
    # Lista di dizionari
    d = [
        {'algorithm': {
            'CentroidVector': {'item_field': {'plot': ['tfidf_sk']}, 'similarity': 'CosineSimilarity', 'threshold': 4,
                               'embedding_combiner': 'Centroid'},
            'sys - mean': {'Precision - macro': 1.0, 'Recall - macro': 1.0, 'F1 - macro': 1.0}}},
        {'algorithm': {
            'ClassifierRecommender': {'item_field': {'plot': ['tfidf_sk']}, 'classifier': 'SkKNN', 'threshold': None,
                                      'embedding_combiner': 'Centroid'},
            'sys - mean': {'Precision - macro': 1.0, 'Recall - macro': 1.0, 'F1 - macro': 1.0}}},
        {'algorithm': {'IndexQuery': {'item_field': {'plot': ['search_i']}, 'classic_similarity': True, 'threshold': 4},
                       'sys - mean': {'Precision - macro': 1.0, 'Recall - macro': 1.0, 'F1 - macro': 1.0}}},
        {'algorithm': {'LinearPredictor': {'item_field': {'plot': ['tfidf_sk']}, 'regressor': 'SkLinearRegression',
                                           'only_greater_eq': None, 'embedding_combiner': 'Centroid'},
                       'sys - mean': {'Precision - macro': 0.875,
                                      'Recall - macro': 1.0,
                                      'F1 - macro': 0.9166666666666666
                                      }}}
    ]

    # Numero di colonne per tabella
    columns = 8  # Modifica il numero di colonne secondo le tue esigenze

    # Titolo della tabella
    table_title = "Risultati delle metriche"

    # Nome dell'algoritmo (da inserire nella colonna "Alg.")
    algorithm_name = "Killer"

    # Chiamata alla funzione
    latex_table = generate_latex_table_based_on_representation(d, columns, title=table_title,
                                                               alg_column_value=algorithm_name)

    # Stampa il risultato
    print(latex_table)
    """

    # test per usare list comprension con la funzione make_tuple_for_table
    """
    # Lista di dizionari
    list_of_dicts = [
        {'algorithm': {
            'CentroidVector': {'item_field': {'plot': ['tfidf_sk']}, 'similarity': 'CosineSimilarity', 'threshold': 4,
                               'embedding_combiner': 'Centroid'},
            'sys - mean': {'Precision - macro': 1.0, 'Recall - macro': 1.0, 'F1 - macro': 1.0, 'Gini': 0.0,
                           'NDCG': 1.0}}},
        {'algorithm': {
            'ClassifierRecommender': {'item_field': {'plot': ['tfidf_sk']}, 'classifier': 'SkKNN', 'threshold': None,
                                      'embedding_combiner': 'Centroid'},
            'sys - mean': {'Precision - macro': 1.0, 'Recall - macro': 1.0, 'F1 - macro': 1.0, 'Gini': 0.0,
                           'NDCG': 1.0}}},
        {'algorithm': {'IndexQuery': {'item_field': {'plot': ['search_i']}, 'classic_similarity': True, 'threshold': 4},
                       'sys - mean': {'Precision - macro': 1.0, 'Recall - macro': 1.0, 'F1 - macro': 1.0, 'Gini': 0.0,
                                      'NDCG': 1.0}}},
        {'algorithm': {'LinearPredictor': {'item_field': {'plot': ['tfidf_sk']}, 'regressor': 'SkLinearRegression',
                                           'only_greater_eq': None, 'embedding_combiner': 'Centroid'},
                       'sys - mean': {'Precision - macro': 0.875, 'Recall - macro': 1.0,
                                      'F1 - macro': 0.9166666666666666, 'Gini': 0.0, 'NDCG': 1.0}}}
    ]

    # Definisci start ed end a piacere
    start = 0
    end = 5  # Ad esempio, puoi scegliere tu i valori

    # Usa list comprehension per creare la lista di tuple
    tuple_format = [
        make_tuple_for_table(data_dict, start, end)
        for data_dict in list_of_dicts
    ]
    
    # Stampa la lista di tuple
    print(tuple_format)
    """

    # test per le funzioni di supporto a generate_latex_table_based_on_representation
    """
    data_dict = {
        'algorithm': {
            'caccca': {
                'item_field': {
                    'ploter': ['tfidf_sk'],
                    'mark': ['vera', 'gold'],
                    'apple': ['a', 'kal', '88'],
                    'gufo': ['rep', 'ferry'],
                    'razzo': { 'e': 5,
                               'tre':[],
                               'pera':"rape"
                               }
                },
                'regressor': 'SkLinearRegression',
                'only_greater_eq': None,
                'embedding_combiner': {
                        'Centroid': {},
                        'Retro': {}
                }
            },
            'sys - mean': {
                'Precision - macro': 0.875,
                'Recall - macro': 1.0,
                'F1 - macro': 0.9166666666666666,
                'Gini': 0.0,
                'NDCG': 1.0,
                'R-Precision - macro': 1.0,
                'RMSE': 0.8134293935347402,
                'MSE': 0.6996958397430165,
                'MAE': 0.7616526982381033,
                'MRR': 1.0,
                'MAP': 1.0,
                'PredictionCoverage': 50.0,
                'Precision@5 - macro': 0.875,
                'Recall@5 - macro': 1.0,
                'F1@5 - micro': 0.9285714285714286,
                'MRR@5': 1.0,
                'NDCG@5': 1.0
            }
        }
    }

    result = get_representation(data_dict)
    print(sanitize_latex_string(result))
    print(result)
    print()
    result = get_content(data_dict)
    print(sanitize_latex_string(result))
    print(result)
    print()
    result = get_embedding(data_dict)
    print(sanitize_latex_string(result))
    print(result)
    print()
    result = get_metrics(data_dict)
    print(result)
    """

    # creazione lista di dizionari pronti per essere forniti a generate_latex_table_based_on_representation(data_list)
    """
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

    dictionary_eva_sys = from_yaml_list_to_dict_list(eva_yaml_paths)
    dict_sys_mean = []  # lista di dizionari contenenti le metriche calcolate dal recsys associato
    for e in dictionary_eva_sys:
        tmp = extract_subdictionary(e, "sys - mean")
        dict_sys_mean.append(tmp)

    for sys_mean in dict_sys_mean:
        if 'sys - mean' in sys_mean:
            sys_mean['sys - mean'].pop('CatalogCoverage (PredictionCov)', None)
    # dict_sys_mean = [{k.replace("sys - mean", "algorithm"): v for k, v in d.items()} for d in dict_sys_mean]

    dictionary_rec_sys = from_yaml_list_to_dict_list(recsys_yaml_paths)
    dict_algo = []  # lista di dizionari contenenti le info sul reccomender system usato
    for t in dictionary_rec_sys:
        tmp = extract_subdictionary(t, "algorithm")
        dict_algo.append(tmp)

    if len(dict_algo) != len(dict_sys_mean):
        raise ValueError("Le liste devono avere la stessa lunghezza.")

    dict_ready = []

    for algo, sys_mean in zip(dict_algo, dict_sys_mean):
        merged_result = merge_dicts(algo, sys_mean, merge_key='algorithm')
        dict_ready.append(merged_result)

    # Ora dict_ready contiene i risultati delle fusioni
    for merged_result in dict_ready:
        print(merged_result)

    data_x = {'algorithm': {
        'LinearPredictor': {'item_field': {'plot': ['tfidf_sk']},
                            'regressor': 'SkLinearRegression',
                            'only_greater_eq': None,
                            'embedding_combiner': 'Centroid'},
        'sys - mean': {'Precision - macro': 0.875,
                       'Recall - macro': 1.0,
                       'F1 - macro': 0.9166666666666666,
                       'Gini': 0.0,
                       'NDCG': 1.0,
                       'R-Precision - macro': 1.0,
                       'RMSE': 0.8134293935347402,
                       'MSE': 0.6996958397430165,
                       'MAE': 0.7616526982381033,
                       'MRR': 1.0,
                       'MAP': 1.0,
                       'PredictionCoverage': 50.0,
                       'Precision@5 - macro': 0.875,
                       'Recall@5 - macro': 1.0,
                       'F1@5 - micro': 0.9285714285714286,
                       'MRR@5': 1.0,
                       'NDCG@5': 1.0
                       }
    }}

    print()
    print()
    table = generate_latex_table_based_on_representation(dict_ready, 8, "Confronto su diversi impostazioni")
    print(table)
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

    """
    eva_yaml_paths = ["./../data/data_to_test/eva_report_amarSingleSource.yml",
                      "./../data/data_to_test/eva_report_centroidVector.yml",
                      "./../data/data_to_test/eva_report_classifierRecommender.yml",
                      "./../data/data_to_test/eva_report_indexQuery.yml",
                      "./../data/data_to_test/eva_report_linearPredictor.yml"]

    dictionary_res_sys = from_yaml_list_to_dict_list(eva_yaml_paths)
    dict_alg = []
    for e in dictionary_res_sys:
        tmp = extract_subdictionary(e, "sys - mean")
        dict_alg.append(tmp)

    for d in dict_alg:
        print(d)
    """

    # codice per la generazione della tabella dei confronti
    # qui implementiamo tutte le operazione per creare i dizionari di cui
    # necessitiamo per poi passarli alla funzione generate_latex_table
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
    keys = get_algorithm_keys(dictyonary_list) # lista dei nomi degli algoritmi usati

    # show the dictonary extracted after processing them
    result = nest_dictionaries(keys, my_dictio)
    for r in result:
        print(r)

    # with the dictnory processed create the latex table
    latex_table = generate_latex_table(result, max_columns_per_part=3)
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

    # Qui testiamo la funzione generate_latex_table
    # prova per la formatazione e creazione della tabella di comparazione tra algortitmi
    """
    algorithms_data = [
        {'CentroidVector': {'F1@1 - macro': 0.85, 'MRR': 0.92, 'MAE': 0.12, 'Recall': 0.78, 'Precision': 0.91, 'R-Precision - macro': 1.0, 'RMSE': 3.0, 'MSE': 9.0, 'MAE': 3.0, 'MRR': 1.0, 'MAP': 1.0, 'PredictionCoverage': 16.67, 'Precision@5 - macro': 1.0, 'Recall@5 - macro': 1.0, 'F1@5 - micro': 1.0, 'MRR@5': 1.0, 'NDCG@5': 1.0}},
        {'AmarDoubleSource': {'F1@1 - macro': 0.92, 'MRR': 0.88, 'MAE': 0.14, 'Recall': 0.85, 'Precision': 0.89, 'R-Precision - macro': 1.0, 'RMSE': 63.65, 'MSE': 4051.82, 'MAE': 63.65, 'MRR': 1.0, 'MAP': 1.0, 'PredictionCoverage': 16.67, 'Precision@5 - macro': 1.0, 'Recall@5 - macro': 1.0, 'F1@5 - micro': 1.0, 'MRR@5': 1.0, 'NDCG@5': 1.0}},
        {'LinearPredictor': {'F1@1 - macro': 0.88, 'MRR': 0.91, 'MAE': 0.11, 'Recall': 0.82, 'Precision': 0.90, 'R-Precision - macro': 1.0, 'RMSE': 0.81, 'MSE': 0.69, 'MAE': 0.76, 'MRR': 1.0, 'MAP': 1.0, 'PredictionCoverage': 50.0, 'Precision@5 - macro': 0.87, 'Recall@5 - macro': 1.0, 'F1@5 - micro': 0.92, 'MRR@5': 1.0, 'NDCG@5': 1.0}},
    ]

    latex_table = generate_latex_table(algorithms_data)
    print(latex_table)
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
