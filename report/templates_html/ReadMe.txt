
Currently, there are 2 HTML templates for testing, plus a copy named "prova" for changes. The reason for having
2 templates, namely report_template2 and report_template3, lies in the rendering process. The first template is
more readable because it includes indentation for both HTML tags and Jinja statements. However, during rendering
by Jinja, these spaces may introduce confusion in the generated HTML report document, making it always readable
but more complex to read or modify for various needs. This is why the second template file is also used. It lacks
indentation in both HTML code and Jinja statements. When Jinja renders it, it produces a more readable result.