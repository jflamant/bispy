"""Sphinx configuration for the BiSPy documentation.

Build locally with ``uv run sphinx-build -b html docs docs/_build/html``.
CI builds with ``-W``: any warning fails the build.
"""

# bispy is imported from the installed package (`uv sync --group docs`), so its
# version, and the code autodoc documents, match what users install.
import bispy

# -- Project -------------------------------------------------------------------

project = "BiSPy"
author = "Julien Flamant"
copyright = "2017-2026, Julien Flamant"
release = bispy.__version__  # read from pyproject.toml via the package metadata
version = ".".join(release.split(".")[:2])

# -- General -------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",  # publish HTML docs on GitHub Pages
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",  # rendered in the browser, no LaTeX install needed
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_nb",  # Markdown pages (README, CHANGELOG) and notebooks (tutorials)
]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = ["_build"]
language = "en"
pygments_style = "sphinx"

# autosummary writes one page per listed function into generated/ (not versioned)
autosummary_generate = True

# numpydoc would add a "Methods" table to each class, linking to pages that are
# never generated; autodoc's :members: already documents the methods.
numpydoc_show_class_members = False

# The tutorials are notebooks run at every build, so their outputs always come from
# the current code, and an error in a tutorial fails the build. Cells tagged
# "skip-execution" (those needing the external wind dataset) keep their stored outputs.
nb_execution_mode = "force"
nb_execution_timeout = 600
nb_execution_raise_on_error = True
nb_execution_show_tb = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- HTML output ---------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
