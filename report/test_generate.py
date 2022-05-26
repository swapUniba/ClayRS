import jinja2
import yaml
import pdfkit
from flask import render_template
import json

TEMPLATE_FILE = "report_template.html"
DATA_FILE = "data/rs_report.yml"
OUTPUT_HTML = "output/report.html"
OUTPUT_PATH = "output/report.pdf"


def get_data():
    with open(DATA_FILE, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def get_template():
    template_loader = jinja2.FileSystemLoader(searchpath="templates")
    template_env = jinja2.Environment(loader=template_loader)
    return template_env.get_template(TEMPLATE_FILE)


def generate_html_output():
    data = get_data()
    template = get_template()
    output_text = template.render(**data)
    with open(OUTPUT_HTML, 'w') as ofile:
        ofile.write(output_text)


def generate_pdf_output():
    generate_html_output()
    options = {
        'dpi': 600,
        'page-size': 'A4',
        'margin-top': '0.2in',
        'margin-right': '0.2in',
        'margin-bottom': '0.2in',
        'margin-left': '0.2in',
        'encoding': "UTF-8",
        'custom-header': [
            ('Accept-Encoding', 'gzip')
        ],
        'no-outline': None,
    }
    pdfkit.from_file(input=OUTPUT_HTML,
                     output_path=OUTPUT_PATH,
                     options=options)


def main():
    generate_pdf_output()
    print("Generating Report PDF File...")


if __name__ == "__main__":
    main()
