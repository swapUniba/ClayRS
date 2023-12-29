import jinja2
import yaml
import os
import shutil
import subprocess
import pdfkit
import yaml_dump as yaml_dump
from flask import render_template
import json
from pathlib import Path
import pandas as pd
from jinja2 import Undefined

# VARIABILI GLOBALI NELLO SCRIPT

# template non indentato
# TEMPLATE_FILE = "report_template3.html"
# template indentato
# TEMPLATE_FILE = "report_template2.html"
# TEMPLATE_FILE = "report_template.html"
# TEMPLATE_FILE = "templates/report_template.html"
# TEMPLATE_FILE = "report_template.tex"
# il report del modulo evaluation
# DATA_FILE = "data/eva_report.yml"
# il report del modulo recsys
# DATA_FILE = "data/rs_report.yml"
# il report del modulo content analyzer
# DATA_FILE = "data/ca_report.yml"
OUTPUT_TEX = "output/report.TEX"
OUTPUT_PATH = "output/report.pdf"
LIST_YAML_FILES = ["data/ca_report.yml", "data/rs_report.yml", "data/eva_report.yml"]
# TEMPLATE_FILE = "report_templateNew.tex"
TEMPLATE_FILE = "dynamic_fin_rep.tex"

# setting enviroment based on latex needs
LATEX_JINJA_ENV = jinja2.Environment(
    block_start_string="\BLOCK{",
    block_end_string="}",
    variable_start_string="\VAR{",
    variable_end_string="}",
    comment_start_string="\#{",
    comment_end_string="}",
    trim_blocks=True,
    autoescape=False,
    loader=jinja2.FileSystemLoader(searchpath="templates"),
)


def safe_text(text: str) -> str:
    special_chars = ['&', '%', '$', '_', '{', '}', '#']
    for char in special_chars:
        text = str(text)
        text = text.replace(char, "\\" + char)
    return text


def truncate(text: str) -> str:
    number = float(text)
    number = round(number, 5)
    text = str(number)
    print(text)
    return text


# adding filter to the enviroment
LATEX_JINJA_ENV.filters["safe_text"] = safe_text
LATEX_JINJA_ENV.filters["truncate"] = truncate


def merge_yaml_files(input_paths_list, output_folder, output_filename):
    """
    Merge multiple YAML files into a single YAML file.

    Parameters:
    - input_paths (list): List of paths to input YAML files.
    - output_folder (str): Path to the folder where the output YAML file will be created.
    - output_filename (str): Name of the output YAML file.

    Returns:
    - str: Path to the merged YAML file.

    Author:
    - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
    """
    if not input_paths_list:
        raise ValueError("The list of input files is empty. It must contain at least one path.")

    merged_data = {}

    # read from input file yaml and write data in dict merged_data
    for input_path in input_paths_list:
        data = get_data(input_path)
        if data is not None:
            merged_data.update(data)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # create and adjust path to yaml merged file
    output_path = os.path.join(output_folder, output_filename)
    output_path = os.path.normpath(output_path)

    # write data in file unified YAML
    with open(output_path, 'w') as output_file:
        yaml.dump(merged_data, output_file, default_flow_style=False)

    return output_path


"""
def unify_yaml_files():
    file_ca_path = DATA_FILES_LIST[0]
    file_ev_path = DATA_FILES_LIST[1]
    file_rs_path = DATA_FILES_LIST[2]

    data_ca = get_data(file_ca_path)
    data_ev = get_data(file_ev_path)
    data_rs = get_data(file_rs_path)

    if (data_ev is not None):
        data_ca.update(data_ev)
    if (data_rs is not None):
        data_ca.update(data_rs)

    return data_ca
    # TODO  to write the unified YAML file
    # with open(DATA_FILE, 'w') as file:
    #    yaml.dump(data_ca, file)
"""


# La funzione riceve in input un file yaml e ne restituisce il
# corrispettivo dizionario
def get_data(rendering_file):
    with open(rendering_file, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)


def recursive_print_dict(d, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print("\t" * indent, f"{k}:")
            recursive_print_dict(v, indent + 1)
        else:
            print("\t" * indent, f"{k}:{v}")


# sembra che questa funzione sia abbastanza inutile essendo che abbiamo già un ambiente jinja
# definito ad inizio di questo file di script @ Diego
def get_latex_template():
    '''template_loader = jinja2.FileSystemLoader(searchpath="templates")
    template_env = jinja2.Environment(loader=template_loader)
    # define new delimiters to avoid TeX conflicts
    template_env.block_start_string = '\BLOCK{'
    template_env.block_end_string = '}'
    template_env.variable_start_string = '\VAR{'
    template_env.variable_end_string = '}'
    template_env.comment_start_string = '\#{'
    template_env.comment_end_string = '=}'
    template_env.line_statement_prefix = '%%'
    template_env.line_comment_prefix = '%#'
    template_env.trim_blocks = True
    template_env.autoescape = True,
    return template_env.get_template(TEMPLATE_FILE)'''
    return LATEX_JINJA_ENV.get_template(TEMPLATE_FILE)


def generate_tex_output(path_data_in, output_tex_path):
    my_dict = {}
    data = get_data(path_data_in)

    if isinstance(data, dict):
        print("yes")
    else:
        print("no")

    # Verifica se i dati sono già un dizionario
    try:
        my_dict = dict(data)
    except TypeError as e:
        print(f"Impossible to convert data to dictionary: {e}")
        my_dict = {}  # o un altro valore di default a tua scelta

    type(my_dict)
    print(my_dict)

    # Carica il template LaTeX
    template = get_latex_template()
    print(template)

    # Rendi il dizionario disponibile al template usando il nome "data"
    # my_dict['data'] = data

    # Renderizza il template con i dati
    output_text = template.render(my_dict=data)
    print(output_text)

    try:
        # Estrai il percorso della directory dal percorso del file LaTeX
        output_directory = os.path.dirname(output_tex_path)

        # Verifica se la directory di output esiste, altrimenti crea la directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Scrivi il contenuto sul file LaTeX
        with open(output_tex_path, 'w') as ofile:
            ofile.write(output_text)

        # Restituisci il percorso del file scritto
        return output_tex_path

    except Exception as e:
        print(f"Error during writing LaTeX file: {e}")
        return None


"""
def generate_tex_output():
    # TODO SINGLE FILE
    # data = get_data(DATA_FILE) #dict

    # TODO MULTIPLE FILE TO REVIEW
    data = unify_yaml_files()
    template = get_template()

    output_text = template.render(dict=data)

    # print(output_text)
    with open(OUTPUT_TEX, 'w') as ofile:
        ofile.write(output_text)
"""


def load_file(*args: str):
    with open(Path(*args)) as f:
        return f.read()


def generate_pdf_output(latex_file_path, output_folder=None):
    try:
        # Estrai il nome del file LaTeX e la sua estensione
        latex_file_name, _ = os.path.splitext(os.path.basename(latex_file_path))

        # Costruisci il percorso del file PDF nella cartella di output o nella stessa cartella del file LaTeX
        pdf_file_path = os.path.join(output_folder,
                                     f"tex_to_pdf_report.pdf") if output_folder else f"{latex_file_name}_to_pdf_report.pdf"

        # Copia il file LaTeX nella cartella di output se è specificata
        if output_folder:
            output_latex_path = os.path.join(output_folder, f"{latex_file_name}_copied.tex")
            shutil.copy2(latex_file_path, output_latex_path)
        else:
            output_latex_path = latex_file_path

        # Compila il file LaTeX
        subprocess.run(['pdflatex', '-interaction=nonstopmode', output_latex_path])

        # Sposta il file PDF nella cartella di output con il nome specificato
        os.rename(f"{output_latex_path[:-4]}.pdf", pdf_file_path)

        # Restituisci il percorso del file PDF
        return pdf_file_path

    except Exception as e:
        print(f"Error during PDF generation: {e}")
        return None


"""
def generate_pdf_output(output_path):
    # generate_tex_output()
    '''data = unify_yaml_files()  # dict
    #source_file = data['source_file']
    source_file = {k: v for k, v in data.items() if 'source_file' in k}
    field_representations = {k: v for k, v in data.items() if 'field_representations' in k}
    print(source_file)
    template = get_template()
    rendered_template = template.render(
        field_representations = field_representations
    )'''
    # TeX source filename
    # TODO ORIGINAL
    tex_filename = OUTPUT_TEX
    # tex_filename = "report.pdf"
    # TODO ORIGINAL
    filename, ext = os.path.splitext(tex_filename)
    # the corresponding PDF filename
    # TODO ORIGINAL
    pdf_filename = filename + '.pdf'
    # TODO ORIGINAL print(pdf_filename)
    # compile TeX file
    subprocess.run(['pdflatex', '-interaction=nonstopmode', str(tex_filename)])
"""


def main():
    input_yaml = merge_yaml_files(LIST_YAML_FILES, "data", "finale_unify_report.yml")
    latex_file_to_compile = generate_tex_output(input_yaml, "output/report_new_latex.tex")
    # generate_pdf_output(latex_file_to_compile, output_folder=None)
    print("Generating Report PDF File...")
    print(latex_file_to_compile)


if __name__ == "__main__":
    main()
