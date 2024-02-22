# This script contains some function that support all the process of generating an automatized latex report

import os

# TODO creare nuove funzionni modifiers in caso di necessita nella formatazione


def format_latex(input_string):
    """
         It will replace the special simbol indicated in order to not give problem to the latex compiler.

         Parameters:
         - input_string (str): text to be processed and changed if needed.

        Returns:
        - formatted_string (str):text with the substitution indicated by the dictionary.

         Author:
         - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
     """
    special_chars = {
        '\\': r'\textbackslash ',
        '{': r'\{',
        '}': r'\}',
        '$': r'\$',
        '&': r'\&',
        '#': r'\#',
        '_': r'\_',
        '^': r'\^{}',
        '~': r'\textasciitilde ',
        '%': r'\%',
    }

    formatted_string = ''
    for char in input_string:
        if char in special_chars:
            formatted_string += special_chars[char]
        else:
            formatted_string += char
    return formatted_string


def modifier_string(input_string, modifier_func=None):
    if modifier_func:
        return modifier_func(input_string)
    else:
        return input_string


# Funzione make_latex_string che chiama la funzione modifier_string
def make_latex_string(input_string, modifier_func=None):
    return modifier_string(input_string, modifier_func)


def replace_in_latex_file(latex_file_path, placeholder, substitution):
    """
        It will replace a placeholder inside a file with a text given.

        Parameters:
        - latex_file_path (str): path to find the file needed.
        - placeholder (str): placeholder to be substituted.
        - substitution (str): text to be inserted.

        Author:
        - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
    """
    try:
        # Apre il file LaTeX in modalità di lettura
        with open(latex_file_path, 'r') as file:
            # Legge il contenuto del file
            file_content = file.read()

            # Cerca il placeholder nel contenuto del file
            if placeholder not in file_content:
                print(f"Placeholder '{placeholder}' not finded in the file.")
                return

            # Sostituisce il placeholder con la stringa di sostituzione
            updated_content = file_content.replace(placeholder, substitution)

        # Scrive il contenuto aggiornato nel file
        with open(latex_file_path, 'w') as file:
            file.write(updated_content)

        print(f"Substitution succeded in the file {latex_file_path}.")
    except FileNotFoundError:
        print(f"File not finded: {latex_file_path}.")
    except Exception as e:
        print(f"Error occured: {str(e)}")


def add_to_latex_file(latex_file_path, text_in):
    """
        It will add text to the file given in input.

        Parameters:
        - latex_file_path (str): path to find the file needed.
        - text_in (str): text to be added to the file.

        Author:
        - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
    """
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
                # file.write(r'\hfill\break' + '\n' + r'\hfill\break' + '\n')
                file.write('\n')  # Aggiunge una riga vuota
            print(f"Text added with success to the file {latex_file}.")
        else:
            # Se il file non esiste, lo crea nella cartella specificata
            with open(latex_file_path, 'w') as file:
                file.write(text_in + '\n')
                # file.write(r'\hfill\break' + '\n' + r'\hfill\break' + '\n')
                file.write('\n')  # Aggiunge una riga vuota
            print(f"File {latex_file_path} created and text added to it.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")


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

    # test sulla funzione che permette di formattare una stringa e prepararla per
    # poi utilizzarla con le precedenti viste
    stringa_da_modificare = "Arriveranno_giorni migliori_per_noi"
    add_to_latex_file(percorso_completo_file,
                      make_latex_string(stringa_da_modificare, format_latex))
