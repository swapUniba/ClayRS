import os
import yaml

from report.test_generate import get_data  # òa funzione get_data() è stato definita appositamente per la funzione di unificazione
                                           # non dovrebbe essere cambiato nulla su di essa


def unify_yaml_files_new(*file_paths):
    # Assicurati che almeno un percorso di file sia stato fornito
    if not file_paths:
        raise ValueError("Devi fornire almeno un percorso di file come argomento.")

    # Inizializza il dizionario vuoto
    unified_data = {}

    # Processa ogni file in file_paths
    for file_path in file_paths:
        # Leggi i dati dal file YAML
        data = get_data(file_path)

        # Se i dati non sono vuoti, aggiornali nel dizionario unificato
        if data is not None:
            unified_data.update(data)

    # Restituisci il dizionario unificato
    return unified_data

"""
# Scenario di utilizzo plausibile per i nostri scopi
file_ca_path = "path/al/file_ca.yml"
file_ev_path = "path/al/file_ev.yml"
file_rs_path = "path/al/file_rs.yml"

# Chiamata alla funzione con i percorsi dei file come argomenti
resulting_dict = unify_yaml_files_new(file_ca_path, file_ev_path, file_rs_path)

# Esempio di utilizzo con un numero diverso di file
resulting_dict_dynamic = unify_yaml_files_new("path/al/file1.yml", "path/al/file2.yml", "path/al/file3.yml",
                                              "path/al/file4.yml")
"""