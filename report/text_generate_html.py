import jinja2
import yaml
import pdfkit
import os
from flask import render_template
import json
# import weasyprint
# from weasyprint import html
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from bs4 import BeautifulSoup
from formForDoc import HTMLToPDFParser, HTMLToPDF
from report.formForDoc import HTMLToPDFParser

# this template do not present indentation this allows to have a better rendering
# TEMPLATE_FILE = "report_template_not_indented.html"
# this is the template indented to clearly understand how jinja statement works
TEMPLATE_FILE = "report_template_indented.html"
# TEMPLATE_FILE = "report_template.html"
# TEMPLATE_FILE = "templates/report_template.html"
# TEMPLATE_FILE = "report_template.tex"

# report of eva module, recsys module and content analyzer module
DATA_FILE = "data/eva_report.yml"
# DATA_FILE = "data/rs_report.yml"
# DATA_FILE = "data/ca_report.yml"
# path used to direct the creation of the report
OUTPUT_HTML = "output/report.html"
OUTPUT_PATH = "output/report.pdf"

# used to deal with yaml file, another similar list can be used with more yaml file
LIST_YAML_FILES = [ "data/ca_report.yml", "data/rs_report.yml", "data/eva_report.yml" ]


# We define the environment for jinja in order to state which are the statement
LATEX_JINJA_ENV = jinja2.Environment(
    block_start_string="{%",
    block_end_string="%}",
    variable_start_string="{{",
    variable_end_string="}}",
    comment_start_string="{#",
    comment_end_string="#}",
    trim_blocks=True,
    autoescape=False,
    loader=jinja2.FileSystemLoader(searchpath="personal_change_dir"),
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


# adding the filter created to the environment of jinja
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

    # write data in file YAML
    with open(output_path, 'w') as output_file:
        yaml.dump(merged_data, output_file, default_flow_style=False)

    return output_path

# First function for the conversion from HTML to PDF, in this case we managed to generate the PDF
# with the content of the HTML, but there is no formatting in the document, making it look horrible
def html_to_pdf(input_html_path, output_pdf_path):
    """
       Convert the text in a file HTML to a text in a file PDF.

       Parameters:
        - param input_html_path: Path to file HTML where the text would be extract.
        - type input_html_path: str
        - param output_pdf_path: Path where the file PDF would be created and saved.
        - type output_pdf_path: str

       Exceptions:
        - raises FileNotFoundError: If the file HTML is not present in the folder indicated.
        - raises Exception: If a general unspecified error occur during the process of conversion.

       Example of use:
           html_file_path = 'path_to_your_file.html'
           pdf_output_path = 'path_to_your_file.pdf'
           html_to_pdf(html_file_path, pdf_output_path)

       Author:
        - Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
       """
    try:
        # read file HTML
        with open(input_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # get text from HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        text_content = soup.get_text()

        # Create file PDF and add text extracted using BeautifulSoup to PDF
        pdf_document = SimpleDocTemplate(output_pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph(text_content, styles["Normal"])]

        pdf_document.build(story)
        print(f"Conversion from {input_html_path} to {output_pdf_path} completed.")
    except FileNotFoundError:
        print(f"Error: file HTML {input_html_path} impossible to locate.")

    except Exception as e:
        print(f"Error inspected during conversion: {e}")


# This function is the second for converting from HTML to PDF and uses the formForDoc module,
# in which there are classes for handling HTML tags for document conversion to PDF
# [HORRIBLE RESULTS FOR FORMATTING].
def html_to_pdf2(html_path, pdf_path):
    pdf = HTMLToPDF()
    pdf.add_page()

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    parser = HTMLToPDFParser(pdf)
    parser.feed(html_content)

    pdf.output(pdf_path)


def create_html_output(path_data_in, output_html_path):
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
    template = get_template()
    print(template)
    output_text = template.render(my_dict=data)
    print(output_text)

    try:
        # Extract the directory path from the HTML file path
        output_directory = os.path.dirname(output_html_path)

        # Check if the output directory exists; otherwise, create the directory.
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # write the content on the HTML file
        with open(output_html_path, 'w') as ofile:
            ofile.write(output_text)

        # Return the path of the written file.
        return output_html_path

    except Exception as e:
        print(f"Error during writing file HTML: {e}")
        return None


def get_data(path_file):
    with open(path_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def get_template():
    # searchpath necessita la cartella ove si trova il file di interesse
    # template_loader = jinja2.FileSystemLoader(searchpath="personal_change_dir")
    # template_env = jinja2.Environment(loader=template_loader)
    return LATEX_JINJA_ENV.get_template(TEMPLATE_FILE)



"""
def generate_html_output(path_data_in):
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
        print(f"Impossibile convertire i dati in un dizionario: {e}")
        my_dict = {}  # o un altro valore di default a tua scelta

    type(my_dict)
    print(my_dict)
    template = get_template()
    print(template)
    output_text = template.render(my_dict=data)
    print(output_text)
    with open(OUTPUT_HTML, 'w') as ofile:
        ofile.write(output_text)
"""

"""
def generate_pdf_output(path_data_in):
    generate_html_output(path_data_in)
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
"""


def main():
    input_file = merge_yaml_files(LIST_YAML_FILES, "report/personal_change_dir", "final_report_yaml")
    input_html = create_html_output(input_file, OUTPUT_HTML)
    html_to_pdf2(input_html, OUTPUT_PATH )
    print("Generating Report PDF File...")


if __name__ == "__main__":
    main()
