"""
html_to_pdf.py

This module provides classes for rendering HTML content into a PDF document.

Classes:
    HTMLToPDFParser: Subclass of HTMLParser, parses HTML tags and translates them into PDF formatting.
    HTMLToPDF: Subclass of FPDF, handles the overall PDF document and formatting.

Usage:
    1. Create an instance of HTMLToPDFParser, passing the target PDF document (an instance of HTMLToPDF) as an argument.
    2. Feed the HTML content to the HTMLToPDFParser using its feed() method.
    3. Call the generate_pdf() method on the HTMLToPDF instance to produce the final PDF.

Example:
    from fpdf import FPDF
    from html.parser import HTMLParser

    # Import your module
    from html_to_pdf import HTMLToPDF, HTMLToPDFParser

    # Create an instance of HTMLToPDF
    pdf_document = HTMLToPDF()

    # Create an instance of HTMLToPDFParser, passing the PDF document as an argument
    html_parser = HTMLToPDFParser(pdf_document)

    # Feed HTML content to the parser
    html_content = "<h1>Hello, World!</h1><p>This is a sample HTML content.</p>"
    html_parser.feed(html_content)

    # Generate the final PDF
    pdf_document.generate_pdf("output.pdf")

Author:
     Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
"""

from fpdf import FPDF
from html.parser import HTMLParser


class HTMLToPDFParser(HTMLParser):
    """
       HTMLToPDFParser class extends HTMLParser to parse HTML tags and translate them into PDF formatting.

       Attributes:
           pdf_document (HTMLToPDF): An instance of HTMLToPDF to which the parsed content is applied.
           in_table (bool): A flag indicating whether the parser is currently inside a table.
           in_list (bool): A flag indicating whether the parser is currently inside an unordered list (<ul>).
           in_list_item (bool): A flag indicating whether the parser is currently inside a list item (<li>).
           table_data (list): A list to store table data during parsing.

       Methods:
           handle_starttag(tag, attrs): Handle the start of an HTML tag and apply corresponding PDF formatting.
           handle_endtag(tag): Handle the end of an HTML tag and perform necessary actions.
           handle_data(data): Handle text data and apply appropriate formatting based on the parser's state.
           draw_table(): Handle drawing the table using the stored table data.

       Author:
           Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
    """
    def __init__(self, pdf_document):
        super().__init__()
        self.pdf_document = pdf_document
        self.in_table = False
        self.in_list = False  # attribute to deal with tag <ul>
        self.in_list_item = False
        self.table_data = []

    def handle_starttag(self, tag, attrs):
        """
            Handle the start of an HTML tag and apply corresponding PDF formatting.

            Args:
                tag (str): The HTML tag.
                attrs (list): List of attributes for the tag.
        """
        if tag == 'p':
            self.pdf_document.set_font("Arial", size=12)
            self.pdf_document.ln(10)
        elif tag == 'h1':
            self.pdf_document.set_font("Arial", 'B', size=24)
            self.pdf_document.ln(10)
        elif tag == 'h2':
            self.pdf_document.set_font("Arial", 'B', size=20)
            self.pdf_document.ln(10)
        elif tag == 'h3':
            self.pdf_document.set_font("Arial", 'B', size=18)
            self.pdf_document.ln(10)
        elif tag == 'table':
            self.in_table = True
        elif tag == 'tr':
            self.table_data.append([])
        elif tag == 'strong':
            self.pdf_document.set_font("Arial", 'B', size=12)
        elif tag == 'ul':
            self.in_list = True
        elif tag == 'li':
            self.in_list_item = True

    def handle_endtag(self, tag):
        """
           Handle the end of an HTML tag and perform necessary actions.

           Args:
               tag (str): The HTML tag.
        """
        if tag == 'p' or tag == 'h1' or tag == 'h2' or tag == 'h3':
            self.pdf_document.ln(10)  # Add some spacing after paragraphs and headers
        elif tag == 'table':
            self.in_table = False
            self.draw_table()
        elif tag == 'ul':
            self.in_list = False
            self.pdf_document.ln(10)  # Add spacing after the unordered list
        elif tag == 'li':
            self.in_list_item = False
            self.pdf_document.ln(5)  # Add spacing between list items

    def handle_data(self, data):
        """
            Handle text data and apply appropriate formatting based on the parser's state.

            Args:
                data (str): Text data.
        """
        if self.in_table:
            # Check list self.table_data is initialized
            if not self.table_data:
                self.table_data.append([])
            self.table_data[-1].append(data)
        else:
            self.pdf_document.multi_cell(0, 10, data)

    def draw_table(self):
        # Handle drawing the table using self.table_data
        # Implement the logic to draw rows and columns in the PDF
        pass


class HTMLToPDF(FPDF):
    """
      HTMLToPDF class extends FPDF to handle the overall PDF document and formatting.

      Methods:
          header(): Placeholder for the header method.
          footer(): Placeholder for the footer method.
          normalize_text(txt): Normalize text before rendering in the PDF.
          multi_cell(w, h, txt, border=0, align='L', fill=False): Override multi_cell method for custom text handling.

      Author:
           Diego Miccoli (Kozen88) <d.miccoli13@studenti.uniba>
    """
    def header(self):
        pass

    def footer(self):
        pass

    def normalize_text(self, txt):
        return txt

    def multi_cell(self, w, h, txt, border=0, align='L', fill=False):
        """
            Override multi_cell method for custom text handling.

            Args:
                w (float): Cell width.
                h (float): Cell height.
                txt (str): Text to be inserted into the cell.
                border (int, optional): Cell border. Defaults to 0.
                align (str, optional): Text alignment. Defaults to 'L'.
                fill (bool, optional): Fill cell background. Defaults to False.
        """
        self.set_font("Arial", size=12)  # set the font before of using it
        txt = self.normalize_text(txt)
        super().multi_cell(w, h, txt, border, align, fill)
