import jinja2
import yaml
import os
import subprocess
import pdfkit
import yaml_dump as yaml_dump
from flask import render_template
import json
from pathlib import Path
import pandas as pd

from jinja2 import Undefined

TEMPLATE_FILE = "report_template.tex"
#todo UNIFIEDI DATA_FILE = "data/unified_report.yml"
DATA_FILE = "data/ca_report.yml"
#DATA_FILES_LIST = ["data/ca_report.yml", "data/eva_report.yml", "data/rs_report.yml"]
#DATA_FILES_LIST = ["/home/vincenzo/PycharmProjects/ClayRS/ca_report.yml",
                #   "/home/vincenzo/PycharmProjects/ClayRS/eva_report.yml",
                #   "/home/vincenzo/PycharmProjects/ClayRS/rs_report.yml"]
DATA_FILES_LIST = ["/home/vincenzo/PycharmProjects/ClayRS/ca_report.yml", "data/eva_report.yml", "data/rs_report.yml"]


OUTPUT_TEX = "output/report.tex"
OUTPUT_PATH = "output/report.pdf"


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

LATEX_JINJA_ENV.filters["safe_text"] = safe_text



def unify_yaml_files():
    file_ca_path = DATA_FILES_LIST[0]
    file_ev_path = DATA_FILES_LIST[1]
    file_rs_path = DATA_FILES_LIST[2]

    data_ca = get_data(file_ca_path)
    data_ev = get_data(file_ev_path)
    data_rs = get_data(file_rs_path)

    if(data_ev is not None):
        data_ca.update(data_ev)
    if(data_rs is not None):
        data_ca.update(data_rs)

    return data_ca
    #TODO  to write the unified YAML file
    # with open(DATA_FILE, 'w') as file:
    #    yaml.dump(data_ca, file)


def get_data(DATA_FILE):
    with open(DATA_FILE, 'r') as stream:
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


def get_template():
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



def generate_tex_output():
    #TODO SINGLE FILE
    # data = get_data(DATA_FILE) #dict

    #TODO MULTIPLE FILE DA RIVEDERE
    data = unify_yaml_files()
    #print(data)
    template = get_template()

    #print(template)
    #output_text = template.render(data['source_file'])
    #output_text = template.render(**data)
    output_text = template.render(dict = data)

    #print(output_text)
    with open(OUTPUT_TEX, 'w') as ofile:
        ofile.write(output_text)

def load_file(*args: str):
    with open(Path(*args)) as f:
        return f.read()

def generate_pdf_output():
    generate_tex_output()
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
    #tex_filename = "report.pdf"
    #TODO ORIGINAL
    filename, ext = os.path.splitext(tex_filename)
    # the corresponding PDF filename
    #TODO ORIGINAL
    pdf_filename = filename + '.pdf'
    # TODO ORIGINAL print(pdf_filename)
    # compile TeX file
    subprocess.run(['pdflatex', '-interaction=nonstopmode', str(tex_filename)])


def main():
    generate_pdf_output()
    print("Generating Report PDF File...")
    print("Test process data")
    #unify_yaml_files()


if __name__ == "__main__":
    main()
