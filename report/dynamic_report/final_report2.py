import os
import re
import shutil


# legge il contenuto ddel file in modo da estrarlo
def read_file_latex(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        contenuto = file.read()
    return contenuto


# effettua la copia del file
#def copy_file_latex(origine, destinazione):
#   shutil.copyfile(origine, destinazione)


# definisce la una espressione regolare capace di catturare tutto il testo
# presente tra le delimitazioni indicate nel caso specifico ### .... ###
def get_text_from_latex(contenuto_latex):
    pattern = re.compile(r'###(.*?)###', re.DOTALL)
    match = pattern.search(contenuto_latex)
    if match:
        return match.group(1)
    else:
        return None


# permette di scrivere sul file indicato il testo andando ad appenderlo
def write_on_file_latex(testo, destinazione):
    with open(destinazione, 'a', encoding='utf-8') as file:
        file.write(testo)


"""
    Questa funzione si occupa di andare a creare il nuopvo file latex per farlo 
    attualmente usa una lista contenente il percorso verso i singoli pezzi del report
    un dizionario=customizations che attualmente serve per effettuare particolari 
    sostituzioni e sarà utile quando sostituiremo le parti dinamiche con le variabili jinja
    infine vi è il terzo argomento usato per andare a creare  il file latex dovce desiderato
"""
def build_final_latex_file(file_list, customizations, file_destinazione):
    for file_path in file_list:
        # Leggi il contenuto del file LaTeX
        contenuto_file = read_file_latex(file_path)

        # Applica eventuali personalizzazioni dal dizionario
        # questa parte la utilizzeremo per sostituire le chiavi che vogliamo inserire
        if file_path in customizations:
            personalizzazione = customizations[file_path]
            contenuto_file = contenuto_file.replace('XXX', personalizzazione)

        # Estrai il testo desiderato dal contenuto LaTeX
        testo_estratto = get_text_from_latex(contenuto_file)

        if testo_estratto:
            # Scrivi il testo estratto nel nuovo file LaTeX di destinazione
            write_on_file_latex(testo_estratto, file_destinazione)
            print(f"Contenuto di {file_path} aggiunto con successo a {file_destinazione}")
        else:
            print(f"Impossibile estrarre il testo dal documento LaTeX: {file_path}")



if __name__ == "__main__":
    # Specifica la lista di percorsi dei file LaTeX
    file_list = ['templates_chunks/intro.tex',
                 'templates_chunks/content_analyzer_section.tex',
                 'templates_chunks/recsys_section.tex',
                 'templates_chunks/evaluation_section.tex']

    # Specifica le personalizzazioni da applicare a ciascun file (se necessario)
    customizations = {'templates_chunks/intro.tex': 'DIEGO',
                      'templates_chunks/content_analyzer_section.tex': 'MARCO',
                      'templates_chunks/recsys_section.tex': 'GIULIO',
                      'templates_chunks/evaluation_section.tex': 'FAY'
                      }

    # Specifica il percorso del file LaTeX finale
    file_destinazione = 'final_report.tex'

    # Costruisci il file LaTeX finale
    build_final_latex_file(file_list, customizations, file_destinazione)