from collections import defaultdict
from pathlib import Path
from typing import List, Any

import jinja2

TEMPLATE_DIR = Path(Path(__file__).parent, "templates")

HOMOGLYPHS = {
    "Ø": "o",
    "Ö": "o",
    "Ç": "c",
    "Ş": "s",
    "Š": "s",
    "Á": "a"
}

def load_file(*args: str):
    with open(Path(*args)) as f:
        return f.read()

def homoglyph(char: str) -> str:
    return HOMOGLYPHS.get(char, char.lower())

LATEX_JINJA_ENV = jinja2.Environment(
    block_start_string="\BLOCK{",
    block_end_string="}",
    variable_start_string="\VAR{",
    variable_end_string="}",
    comment_start_string="\#{",
    comment_end_string="}",
    trim_blocks=True,
    autoescape=False,
    loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
)

'''LATEX_JINJA_ENV.globals.update(
    load_file=load_file,
    join_names=join_names,
    group_by_last_name=group_by_last_name,
    to_string_sorting_by_last_name=to_string_sorting_by_last_name,
    program_date=program_date,
    session_times=session_times,
    join_page_numbers=join_page_numbers,
    index_author=index_author,
)'''

def load_template(template: str) -> jinja2.Template:
    return LATEX_JINJA_ENV.get_template(f"{template}.tex")
