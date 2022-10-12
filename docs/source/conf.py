# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import ci_lib

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'WIPAR'
copyright = '2022, Damin Kühn, Jonas Colve'
author = 'Damin Kühn, Jonas Colve'
release = '0.0.9'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    #'sphinx.ext.doctest',
    'autoapi.extension',
    'numpydoc',
    #'sphinx.ext.autodoc',
    #'sphinx.ext.autosummary',
    #'sphinx.ext.intersphinx',  # Link to other project's documentation (see mapping below)
    #'sphinx.ext.viewcode',  # Add a link to the Python source code for classes, functions etc.
    #'myst_parser'
]

autoapi_dirs = ['../../ci_lib']

#autosummary_generate = True

htmlhelp_basename = 'SphinxAutoAPIdoc'

#templates_path = ['_templates']
#exclude_patterns = ['snakemake']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ['_static']
html_logo = "_images/wipar_logo.png"
html_theme_options = {
    'logo_only': True
}

#def setup(app):
#    app.add_css_file('theme.css')
