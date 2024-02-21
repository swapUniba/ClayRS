# This script has been created to deal with the creation of a latex table from
# the result retrived from the statistics test conducted on the recsys

import pandas as pd
import openpyxl

from report.dynamic_report.support_reporting_lib import add_to_latex_file, replace_in_latex_file


# funzione di supporto per la formattazione della tabella che mostra la rilevanza statistica
# tra due sistemi messi a confronto
def format_scientific_notation(value):
    # Formatta l'esponente scientifico in modo più leggibile
    formatted_value = "{:.2e}".format(value)
    parts = formatted_value.split("e")
    return f"{parts[0]} \\times 10^{{{int(parts[1])}}}"


# Questa funzione preleva le informazioni inerenti ai sistemi messi a confronto e le stampa invertendo la disposizione
# righe colonne e ottenendo una tabella con 2 colonne metrics e il nome dei sistemi messi a confronto
def stats_relevance_tab(df, row_index, tab_title="", scientific_notation=True, round_to=4):
    # Verifica se l'indice passato è presente in df.index
    if row_index not in df.index:
        # Se l'indice non è presente, crea un nuovo indice scambiando le sottostringhe
        parts = row_index.strip("()").split(",")
        new_index = f"('{parts[1].strip()}', '{parts[0].strip()}')"

        # Verifica nuovamente se il nuovo indice è presente in df.index
        if new_index not in df.index:
            return ""
        else:
            row_index = new_index

    # Aggiungi \ prima di ogni _
    formatted_row_index = row_index.replace("_", "\\_")

    # Estrai le informazioni dalla riga corrispondente all'indice dato
    row_data = df.loc[row_index]

    # Costruisci la tabella LaTeX con header e titolo
    latex_table = "\\begin{table}[H]\n"
    latex_table += f"\\caption{{\\textbf{{{tab_title}}}}}\n"
    latex_table += "\\center\n"
    latex_table += "\\begin{tabular}{ll}\n"
    latex_table += "\\textbf{Metrics} & \\textbf{" + formatted_row_index + "} \\\\\n"
    for col_name, value in row_data.items():
        # Estrai i nomi completi delle colonne dai multi-indici
        full_col_name = " - ".join(col_name)

        # Formatta il valore
        if scientific_notation:
            formatted_value = format_scientific_notation(value)
        else:
            formatted_value = str(round(value, round_to))
        latex_table += f"  {full_col_name} & {formatted_value} \\\\\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}"

    return latex_table


# Questa funzione è di supporto, infatti è capace di modificare un dataframe prodotto da una delle due funzioni
# che calcolano i test statistici e modifica il dataframe creandone uno nuovo, nel quale le colonne multi-indice
# vengono eliminate in base a removal_idx optando per la rimozione delle colonne 'statistics' oppure di quelle 'pvalue'
def remove_stats_from_df(dataf, removal_idx):
    # Copia il DataFrame originale
    new_dataf = dataf.copy()

    # Filtra i livelli di colonna che soddisfano il criterio
    filtered_columns = [col for col in dataf.columns if col[1] != removal_idx]

    # Seleziona solo le colonne che non soddisfano il criterio
    new_dataf = new_dataf.loc[:, filtered_columns]

    return new_dataf


# funzione per la creazione di un indice di accesso al dataframe
def set_access_index(sys1, sys2, metric, type_val='pvalue'):
    # Costruisci la tupla per l'indice della riga
    row_index = ('({}, {})'.format(sys1, sys2),)
    # Costruisci la tupla per l'indice della colonna
    column_index = (metric, type_val)
    # Ritorna la tupla completa
    return (row_index, column_index)


# Questa funzione permette di sostituire gli indici del dataframe restituito da una delle
# funzioni di test statistico in modo da cambiare gli indici con i nomi di un dizionario
# che abbia per chiavi i nomi scelti dalla funzione dei test statistici e che sono nomi del
# tipo systm_n con n incrementale intero da 1 in poi a seconda di quanti systemi vengono messi
# a confronto questo permetterà in seguito l'accesso al dataframe modificato con questi indici
def change_system_name(df, sys_name):
    # Funzione per sostituire gli indici con i valori del dizionario
    def replace_system_names(index_str):
        # Estrai i token dalla stringa dell'indice
        tokens = index_str.strip("()").replace("'", "").split(', ')
        # Sostituisci i token con i valori associati nel dizionario
        new_tokens = [sys_name.get(token, token) for token in tokens]
        return '(' + ', '.join(new_tokens) + ')'

    # Applica la funzione replace_system_names agli indici del livello dell'indice
    df.index = df.index.map(replace_system_names)

    return df


# funziona in modo fantastico BISOGNA TESTARLA prima implementazione
"""
def from_dataframe_to_latex_table(df, col, title=""):
    # Verifica se il DataFrame è vuoto
    if df.empty:
        return ""

    # Assicurati che il numero di colonne sia minore o uguale a 3
    col = min(col, 3)

    # Estrai le colonne del primo livello del header
    header_cols = df.columns.get_level_values(0).unique().tolist()

    # Inizializza la stringa LaTeX con il titolo
    latex_str = f"\\begin{{table}}[H]\n\\centering\n\\caption{{{title}}}\n"

    # Itera sulle colonne richieste
    for i in range(0, len(header_cols), col):
        # Seleziona un sottoinsieme di colonne
        selected_cols = header_cols[i:i + col]

        # Filtra il DataFrame per le colonne selezionate
        df_selected = df[selected_cols]

        # Aggiungi un sottotitolo alle tabelle (basato sulle colonne selezionate)
        subtitle = ', '.join(selected_cols)
        table_title = f"{title} - {subtitle}"

        # Concatena il codice LaTeX per la tabella con sottotitolo
        latex_str += f"\\subsubsection*{{{table_title}}}\n"
        latex_str += df_selected.to_latex()

        # Aggiungi una nuova riga LaTeX per separare le tabelle
        latex_str += "\n\\vspace{0.5cm}\n"

    # Concludi la stringa LaTeX
    latex_str += "\\end{table}"

    return latex_str
"""

# seconda implementazione
# Questa funzione è stat aggiunta per sostituire l'iniziale che ha un problema nel tagliare le tabelle nel caso
# non entrino in un unica pagina, va testata ance in altri scenari prima di sostituirla definitivamente
"""
def from_dataframe_to_latex_table(df, col, title=""):
    # Verifica se il DataFrame è vuoto
    if df.empty:
        return ""

    # Assicurati che il numero di colonne sia minore o uguale a 3
    col = min(col, 3)

    # Estrai le colonne del primo livello del header
    header_cols = df.columns.get_level_values(0).unique().tolist()

    # Inizializza la stringa LaTeX vuota
    latex_str = ""

    # Itera sulle colonne richieste
    for i in range(0, len(header_cols), col):
        # Seleziona un sottoinsieme di colonne
        selected_cols = header_cols[i:i + col]

        # Filtra il DataFrame per le colonne selezionate
        df_selected = df[selected_cols]

        # Aggiungi un sottotitolo alle tabelle (basato sulle colonne selezionate)
        subtitle = ', '.join(selected_cols)
        table_title = f"{title} - {subtitle}"

        # Aggiungi un'istruzione LaTeX per una nuova tabella indipendente con il titolo
        latex_str += f"\\begin{{table}}[H]\n\\centering\n\\caption{{{table_title}}}\n"
        latex_str += df_selected.to_latex()
        latex_str += "\\end{table}\n\n"

    return latex_str
"""


#  nuova versione da testare terza implementazione
def from_dataframe_to_latex_table(df, col, title=""):
    # Verifica se il DataFrame è vuoto
    if df.empty:
        return ""

    # Assicurati che il numero di colonne sia minore o uguale a 3
    col = min(col, 3)

    # Estrai le colonne del primo livello del header
    header_cols = df.columns.get_level_values(0).unique().tolist()

    # Inizializza la stringa LaTeX vuota
    latex_str = ""

    # Itera sulle colonne richieste
    for i in range(0, len(header_cols), col):
        # Seleziona un sottoinsieme di colonne
        selected_cols = header_cols[i:i + col]

        # Filtra il DataFrame per le colonne selezionate
        df_selected = df[selected_cols]

        # Aggiungi un sottotitolo alle tabelle (basato sulle colonne selezionate)
        subtitle = ', '.join(selected_cols)
        table_title = f"{title} - {subtitle}"

        # Genera la tabella LaTeX come stringa
        table_latex = df_selected.to_latex()

        # Aggiungi un'istruzione LaTeX per una nuova tabella indipendente con il titolo
        latex_str += f"\\begin{{table}}[H]\n\\centering\n\\caption{{{table_title}}}\n"
        latex_str += "\\resizebox{\\columnwidth}{!}{%\n"
        latex_str += table_latex
        latex_str += "}\n\\end{table}\n\n"

    return latex_str


# Esegui lo script
if __name__ == "__main__":
    """
    # Specifica il percorso del tuo file Excel
    file_excel = 'Ttest_.xlsx'

    # Carica il DataFrame da Excel, specificando la riga di intestazione e le colonne da unire
    df = pd.read_excel(file_excel, header=[0, 1], index_col=0)

    # Ora puoi lavorare con il DataFrame 'df'
    print(df)

    # Salva il DataFrame come un nuovo file Excel
    file_excel_nuovo = 'nuovo_ttest.xlsx'
    df.to_excel(file_excel_nuovo)

    # Carica il DataFrame dal nuovo file Excel
    df_nuovo = pd.read_excel(file_excel_nuovo, header=[0, 1], index_col=0)

    # Controlla se i due DataFrame sono uguali
    sono_uguali = df.equals(df_nuovo)

    if sono_uguali:
        print("I due DataFrame sono uguali.")
    else:
        print("I due DataFrame sono diversi.")

    # check the new data frame
    print(df_nuovo)

    # Esempio di utilizzo
    # Sostituisci 'df_originale' con il tuo DataFrame
    latex_table_code = from_dataframe_to_latex_table(df_nuovo, col=2, title="statistics comparison")

    # Stampa il codice LaTeX risultante
    print(latex_table_code)

    # dizionario usato per la sostituzione degli indici di riga del dataframe
    rec_name = {
        'system_1': 'CentroidVector',
        'system_2': 'IndexQuery',
        'system_3': 'Amar_single_source'
    }

    # stampa il dataframe modificato con la sostituzione degli indici
    df = change_system_name(df, rec_name)
    print(df)
    """

    # Test on the function remove_stats_from_df(dataf, removal_idx)
    # Specifica il percorso del tuo file Excel
    file_excel = 'ttest_expand.xlsx'
    # Carica il DataFrame da Excel, specificando la riga di intestazione e le colonne da unire
    df = pd.read_excel(file_excel, header=[0, 1], index_col=0)

    # dizionario usato per la sostituzione degli indici di riga del dataframe
    rec_name = {
        'system_1': 'Amar_single_source',
        'system_2': 'CentroidVector',
        'system_3': 'ClassifierRecommender',
        'system_4': 'IndexQuery',
        'system_5': 'LinearPredictor'
    }

    # stampa il dataframe modificato con la sostituzione degli indici
    df = change_system_name(df, rec_name)
    print(df)

    # apportiamo le modifiche sul dataframe per rimuovere le colonne che contengono le statistiche
    p_value_only_df = remove_stats_from_df(df, 'statistic')
    # print(p_value_only_df)

    # Adesso chiamiamo la funzione di stampa per la tabella latex
    latex_table_pvalue = from_dataframe_to_latex_table(p_value_only_df, col=2, title="p-value results")
    print(latex_table_pvalue)

    # Questa parte andrà a testare il funzionamento delle funzioni
    # stats_relevance_tab(df, row_index, tab_title="", scientific_notation=True, round_to=4) e la sua funzione ausiliare
    #  format_scientific_notation(value) usata per la formattazione dei dati apportati nella tabella latex
    idx_1 = set_access_index('CentroidVector', 'IndexQuery', 'Precision - macro', type_val='pvalue')
    print(f"\n\nl'indice di accesso alla riga che forniremo alla funzione: {idx_1[0][0]}")
    latex_stats_rel_tab_for_idx_1 = stats_relevance_tab(df, idx_1[0][0], tab_title="CentroidVector and IndexQuery",
                                                        scientific_notation=True, round_to=4)
    print(f"\n\nEcco la tabella della rilevanza statistica estratta per l'indice {idx_1[0][0]}: \n"
          f"{latex_stats_rel_tab_for_idx_1}")

    idx_2 = set_access_index('IndexQuery', 'ClassifierRecommender', 'Precision - macro', type_val='pvalue')
    print(f"\n\nl'indice di accesso alla riga che forniremo alla funzione: {idx_1[0][0]}")
    latex_stats_rel_tab_for_idx_2 = stats_relevance_tab(p_value_only_df, idx_1[0][0],
                                                        tab_title="IndexQuery and ClassifierRecommender",
                                                        scientific_notation=True, round_to=4)
    print(f"\n\nEcco la tabella della rilevanza statistica estratta per l'indice {idx_1[0][0]}: \n"
          f"{latex_stats_rel_tab_for_idx_2}")

    # test per aggiunta di testo al documento latex in particolare qui stiamo aggiungendo
    # la tabella latex generata al file desiderato
    # add_to_latex_file("./documento_latex.tex", latex_stats_rel_tab_for_idx_2)
    # replace_in_latex_file("./documento_latex.tex", r"\PPX", latex_stats_rel_tab_for_idx_2)
