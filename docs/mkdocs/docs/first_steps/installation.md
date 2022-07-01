# Installation

## Via PIP <small>recommended</small> { data-toc-label="Via PIP" }

*ClayRS* requires Python **3.7** or later, while package dependencies are in `requirements.txt` and are all installable
via `pip`, as *ClayRS* itself.

To install it execute the following command:

=== "Latest"

    ``` sh
    pip install clayrs
    ```

This will automatically install compatible versions of all dependencies.

---
**Tip**: We suggest installing ClayRS (or any python package, for that matters) in a virtual environment

!!! quote ""
    *Virtual environments are special isolated environments where all the packages and versions you install only 
    apply to that specific environment. It’s like a private island! — but for code.*

Read this [Medium article][medium] for understanding all the advantages and the [official python guide] [venv]
on how to set up one

[medium]: https://towardsdatascience.com/why-you-should-use-a-virtual-environment-for-every-python-project-c17dab3b0fd0
[venv]: https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/