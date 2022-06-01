import jinja2
import yaml
import os
import subprocess
import pdfkit
from flask import render_template
import json

TEMPLATE_FILE = "report_template.tex"
DATA_FILE = "data/ca_report.yml"
DATA_FILE_LIST = ["data/ca_report.yml", "data/eva_report.yml", "rs_report"]
OUTPUT_TEX = "output/report.tex"
OUTPUT_PATH = "output/report.pdf"

def unify_yaml_files(DATA_FILES_LIST):
    return 0

def get_data():
    with open(DATA_FILE, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            print(data)
            #print("GET_DATA type: " + str(type(data)))
            # return yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)

def process_data():
    #TODO
    data = get_data() #dict
    print(data['source_file'])


def get_template():
    template_loader = jinja2.FileSystemLoader(searchpath="templates")
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
    return template_env.get_template(TEMPLATE_FILE)


def generate_tex_output():
    data = get_data() #dict
    template = get_template()
    #print(template)
    output_text = template.render(**data)
    #output_text = template.render(**data)
    #print(output_text)
    with open(OUTPUT_TEX, 'w') as ofile:
        ofile.write(output_text)


def generate_pdf_output():
    generate_tex_output()

    # TeX source filename
    tex_filename = OUTPUT_TEX
    filename, ext = os.path.splitext(tex_filename)
    # the corresponding PDF filename
    pdf_filename = filename + '.pdf'
    print(pdf_filename)  # TODO
    # compile TeX file
    subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_filename])


def main():
    generate_pdf_output()
    print("Generating Report PDF File...")
    print("Test process data")
    process_data()


if __name__ == "__main__":
    main()
