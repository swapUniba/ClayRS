# This script has been created to deal with the creation of a latex table from
# the result retrived from the statistics test conducted on the recsys

import pandas as pd
import openpyxl


# funziona in modo fantastico BISOGNA TESSTARLA
def from_dataframe_to_latex_table(df, col, title=""):
    # Verifica se il DataFrame è vuoto
    if df.empty:
        return ""

    # Assicurati che il numero di colonne sia minore o uguale a 3
    col = min(col, 3)

    # Estrai le colonne del primo livello del header
    header_cols = df.columns.get_level_values(0).unique().tolist()

    # Inizializza la stringa LaTeX con il titolo
    latex_str = f"\\begin{{table}}[h]\n\\centering\n\\caption{{{title}}}\n"

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


# Esegui lo script
if __name__ == "__main__":
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
