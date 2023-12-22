import os
import re
import shutil


def read_file_latex(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        contenuto = file.read()
    return contenuto


def copy_file_latex(origine, destinazione):
    shutil.copyfile(origine, destinazione)


def get_text_from_latex(contenuto_latex):
    # Utilizza un'espressione regolare per estrarre il testo desiderato
    pattern = re.compile(r'###(.*?)###', re.DOTALL)
    match = pattern.search(contenuto_latex)
    if match:
        return match.group(1)
    else:
        return None


def write_on_file_latex(testo, destinazione):
    with open(destinazione, 'w', encoding='utf-8') as file:
        file.write(testo)


def main():
    # Specifica i percorsi dei tuoi file LaTeX
    file_origine = 'templates_chunks/intro.tex'
    file_destinazione = 'final_report.tex'

    # Leggi il contenuto del file LaTeX di origine
    contenuto_origine = read_file_latex(file_origine)

    # Estrai il testo desiderato dal contenuto LaTeX
    testo_estraito = get_text_from_latex(contenuto_origine)

    if testo_estraito:
        # Scrivi il testo estratto nel nuovo file LaTeX di destinazione
        write_on_file_latex(testo_estraito, file_destinazione)
        print(f"Testo estratto con successo e scritto in {file_destinazione}")
    else:
        print("Impossibile estrarre il testo dal documento LaTeX di origine.")


if __name__ == "__main__":
    main()