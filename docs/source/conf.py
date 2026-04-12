# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.

import os
import subprocess
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "_ext")))

def generate_single_header(app, exception):
    # Destination folder relative to conf.py
    single_header_path = os.path.abspath(os.path.join(app.builder.outdir))
    os.makedirs(single_header_path, exist_ok=True)

    # Path to your script
    script_path = os.path.abspath(os.path.join(app.srcdir, '..', '..', 'script', 'create-single-header.sh'))

    # Call the script with the destination folder as argument
    subprocess.run([script_path, single_header_path], check=True)
    print(f"Generated single header in {single_header_path}")

def build_doxygen():
    subprocess.call("cd ..; doxygen", shell=True)
    subprocess.call("cd ..; doxygen Doxyfile_dev", shell=True)

def copy_doxygen(dst, src):
    print("Copying from:", src)
    print("Copying to:", dst)
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        print("Doxygen HTML not found at:", src)

def copy_doxygen_html(app, exception):
    # confdir = …/repo-root/docs/source
    confdir = os.path.dirname(app.confdir)

    # --------------------------------------------------------------
    # USER documentation (docs/doxygen/html)
    # --------------------------------------------------------------
    src = os.path.abspath(os.path.join(confdir, '..', 'docs/doxygen', 'html'))
    dst = os.path.join(app.builder.outdir, 'doxygen')
    copy_doxygen(dst, src)

    # --------------------------------------------------------------
    # DEVELOPER documentation (docs/doxygen_dev/html)
    # --------------------------------------------------------------
    src = os.path.abspath(os.path.join(confdir, '..', 'docs/doxygen_dev', 'html'))
    dst = os.path.join(app.builder.outdir, 'doxygen_dev')
    copy_doxygen(dst, src)

def setup(app):
    # Hook into the 'builder-inited' event to run the function before the build starts
    if not "ALPAKA_NO_SINGLE_HEADER" in os.environ:
        app.connect('build-finished', generate_single_header)
    app.connect('build-finished', copy_doxygen_html)

# -- Project information -----------------------------------------------------

project = "alpaka"
copyright = "Documentation under CC-BY 4.0"
author = "The alpaka team."
# The short X.Y version.
version = "3.0.0"
# The full version, including alpha/beta/rc tags.
release = "3.0.0-dev"

# The master toctree document.
master_doc = "index"

# -- General configuration ---------------------------------------------------

highlight_language = 'c++'
on_rtd = os.environ.get("READTHEDOCS", None) == "True"

show_authors = True

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    #    'sphinx.ext.napoleon',
    "breathe",
    "filtered_literalinclude",
    "sphinx_rtd_theme",
    "sphinxcontrib.programoutput",
    #    'matplotlib.sphinxext.plot_directive'
    #    "sphinx.ext.autosectionlabel",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["Thumbs.db",
                    ".DS_Store",
                    ]

source_suffix = [".rst"]
master_doc = "index"
language = "en"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"  #'default'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# modifies the HTML Sphinx Doc layout
html_css_files = ["custom.css"]

html_logo = "../logo/alpaka.svg"
html_theme_options = {
    "logo_only": True,
    "collapse_navigation": False,
    "navigation_depth": 2,
}

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "alpakadoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    "papersize": "a4paper",
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    "preamble": r"\setcounter{tocdepth}{2}",
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}
latex_logo = "../logo/alpaka.png"

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "alpaka-doc.tex", "alpaka Documentation", "The alpaka Community", "manual"),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "alpaka", "alpaka Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "alpaka",
        "alpaka Documentation",
        author,
        "alpaka",
        "Abstraction Library for Parallel Kernel Acceleration",
        """
     The alpaka library is a header-only C++20 abstraction library for
     accelerator development. Its aim is to provide performance portability
     across accelerators through the abstraction (not hiding!) of the underlying
     levels of parallelism.
     """,
    ),
]

# -- Options for Epub output -------------------------------------------------

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


# -- Extension configuration -------------------------------------------------

breathe_projects = {"alpaka": "../doxygen/xml"}
breathe_default_project = "alpaka"

breathe_domain_by_extension = {"cpp": "cpp", "h": "cpp", "hpp": "cpp", "tpp": "cpp"}

# define alpaka attributes
# breath has problems to parse C++ attributes
cpp_id_attributes = [
    "ALPAKA_FN_ACC",
    "ALPAKA_FN_HOST",
    "ALPAKA_FN_HOST_ACC",
    "ALPAKA_FN_INLINE",
    "ALPAKA_NO_HOST_ACC_WARNING",
    "ALPAKA_FORWARD",
    "ALPAKA_TYPEOF",
]

# -- processing --

if on_rtd:
    build_doxygen()
    #subprocess.call(
    #    "cd ../cheatsheet; rst2pdf -s cheatsheet.style ../source/basic/cheatsheet.rst -o cheatsheet.pdf", shell=True
    #)
else:
    import sphinx_rtd_theme
    from sphinx.util import logging

    logger = logging.getLogger(__name__)

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

    logger.info("single header build can be skipped with the environment variable 'ALPAKA_NO_SINGLE_HEADER=1'")

    if shutil.which("doxygen"):
        if not "ALPAKA_NO_DOXYGEN" in os.environ:
            build_doxygen()
        logger.info("doxygen build can be skipped with the environment variable 'ALPAKA_NO_DOXYGEN=1'")
    else:
        logger.warning("could not find 'doxygen' executable. Skip building doxygen documentation.")
