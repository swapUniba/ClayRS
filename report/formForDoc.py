from fpdf import FPDF
from html.parser import HTMLParser


class HTMLToPDFParser(HTMLParser):
    def __init__(self, pdf_document):
        super().__init__()
        self.pdf_document = pdf_document
        self.in_table = False
        self.in_list = False  # Nuovo attributo per gestire i tag <ul>
        self.in_list_item = False
        self.table_data = []

    def handle_starttag(self, tag, attrs):
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
        if self.in_table:
            # Assicurati che la lista self.table_data sia inizializzata
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
    def header(self):
        pass

    def footer(self):
        pass

    def normalize_text(self, txt):
        return txt

    def multi_cell(self, w, h, txt, border=0, align='L', fill=False):
        self.set_font("Arial", size=12)  # Imposta il font prima di utilizzarlo
        txt = self.normalize_text(txt)
        super().multi_cell(w, h, txt, border, align, fill)