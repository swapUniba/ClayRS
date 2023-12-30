WARNINGS AND INFORMATION:

This 3 latex files are used to create dynamically the latex template with jinja
statement to render the report file on the experiment conducted with ClayRs
framework.

In particular end_eva.tex and intro_eva_all_metrics.tex are latex file used
in static and fixed way, in fact will be jinja to deal with logic of rendering
after when the final report will be created.

The third one sys_result_on_fold_eva.tex is more general and used dynamically
to retrieve key from the dictionary made from the eva_report_yml file and once
the keys are retrieved they will be replaced in this template from the python
code in charge to generate the dynamic report template in latex.


Project: ClayRS branch report
==============================================================================

Author: Diego Miccoli
GitHub nickname: Kozen88
Email: <d.miccoli13@studenti.uniba>
Date: 30/12/2023

