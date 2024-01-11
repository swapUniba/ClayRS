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


def extract_subdictionary(data, key):
    if key in data:
        return {key: data[key]}
    for value in data.values():
        if isinstance(value, dict):
            result = extract_subdictionary(value, key)
            if result:
                return result
    return None


# Esegui lo script
if __name__ == "__main__":
    """
    algorithms_data = [
        {'CentroidVector': {'F1@1 - macro': 0.85, 'MRR': 0.92, 'MAE': 0.12, 'Recall': 0.78, 'Precision': 0.91}},
        {'AmarDoubleSource': {'F1@1 - macro': 0.92, 'MRR': 0.88, 'MAE': 0.14, 'Recall': 0.85, 'Precision': 0.89}},
        {'LinearPredictor': {'F1@1 - macro': 0.88, 'MRR': 0.91, 'MAE': 0.11, 'Recall': 0.82, 'Precision': 0.90}},
    ]

    latex_table = generate_latex_table(algorithms_data)
    print(latex_table)
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