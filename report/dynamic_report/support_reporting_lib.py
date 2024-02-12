# This script contains some function that support all the process of generating an automatized latex report

import os


def replace_in_latex_file(latex_file_path, placeholder, substitution):
    try:
        # Apre il file LaTeX in modalità di lettura
        with open(latex_file_path, 'r') as file:
            # Legge il contenuto del file
            file_content = file.read()

            # Cerca il placeholder nel contenuto del file
            if placeholder not in file_content:
                print(f"Placeholder '{placeholder}' non trovato nel file.")
                return

            # Sostituisce il placeholder con la stringa di sostituzione
            updated_content = file_content.replace(placeholder, substitution)

        # Scrive il contenuto aggiornato nel file
        with open(latex_file_path, 'w') as file:
            file.write(updated_content)

        print(f"Sostituzione effettuata con successo nel file {latex_file_path}.")
    except FileNotFoundError:
        print(f"File non trovato: {latex_file_path}.")
    except Exception as e:
        print(f"Si è verificato un errore: {str(e)}")


def add_to_latex_file(latex_file_path, text_in):
    try:
        # Ricerca del file LaTeX nel percorso specificato e nelle sottocartelle
        file_found = False
        for root, dirs, files in os.walk(os.path.dirname(latex_file_path)):
            if os.path.basename(latex_file_path) in files:
                latex_file = os.path.join(root, os.path.basename(latex_file_path))
                file_found = True
                break

        # Se il file esiste, aggiunge il testo
        if file_found:
            with open(latex_file, 'a') as file:
                file.write(text_in + '\n')
                file.write(r'\hfill\break' + '\n' + r'\hfill\break' + '\n')
                file.write('\n')  # Aggiunge una riga vuota
            print(f"Testo aggiunto con successo al file {latex_file}.")
        else:
            # Se il file non esiste, lo crea nella cartella specificata
            with open(latex_file_path, 'w') as file:
                file.write(text_in + '\n')
                file.write(r'\hfill\break' + '\n' + r'\hfill\break' + '\n')
                file.write('\n')  # Aggiunge una riga vuota
            print(f"File {latex_file_path} creato e testo aggiunto.")
    except Exception as e:
        print(f"Si è verificato un errore: {str(e)}")


# Esegui lo script
if __name__ == "__main__":
    # Esempio d'uso:
    testo = r"PLACEHOLDER"
    cartella_iniziale = "./"
    nome_file_latex = "documento_latex.tex"
    percorso_completo_file = os.path.join(cartella_iniziale, nome_file_latex)
    add_to_latex_file(percorso_completo_file, testo)

    # qui vediamo un caso d'uso della funzione replace_in_latex_file()
    placeholder = r"\PPX"
    substitution = r"\section{Nuova sezione}\nArriverò fino alla fine e vincerò!!"
    replace_in_latex_file(percorso_completo_file, placeholder, substitution)
