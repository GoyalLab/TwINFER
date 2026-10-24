# Sphinx configuration of the TwINFER documentation (same setup as the squidpy docs: Read the Docs theme, notebooks rendered with myst-nb).
# Build locally from the repository root:  sphinx-build -b html docs docs/_build/html
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "package"))  # so autodoc imports twinfer without installing it


# The notebooks live in tutorials/ (one copy, run from there); copy them into docs/tutorials/ for the site (git-ignored).
# Links written for Jupyter / GitHub are rewritten in the copy: Jupyter heading anchors (#1.-The-output) to the site's
# heading anchors (#1-the-output), and ../README.md#installation to the installation page.
def _site_anchor(jupyter_anchor):
    return re.sub(r"[^\w\-]", "", jupyter_anchor.lower())


def _copy_tutorials():
    out_dir = Path(__file__).parent / "tutorials"
    for nb_path in sorted((REPO_ROOT / "tutorials").glob("*.ipynb")):
        nb = json.loads(nb_path.read_text())
        for cell in nb["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            text = "".join(cell["source"])
            text = re.sub(r"\]\(#([^)]+)\)", lambda m: "](#" + _site_anchor(m.group(1)) + ")", text)
            text = text.replace("](../README.md#installation)", "](../installation.md)")
            cell["source"] = text
        new_text = json.dumps(nb, indent=1, ensure_ascii=False)
        out_path = out_dir / nb_path.name
        if not out_path.exists() or out_path.read_text() != new_text:  # unchanged copies keep their mtime, so Sphinx does not rebuild them
            out_path.write_text(new_text)


_copy_tutorials()


project = "TwINFER"
author = "Keerthana M Arun, Yuval Scher, Yuhan D Zhang"
copyright = "2026, " + author
release = (REPO_ROOT / "package" / "__version__").read_text().strip()
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# Notebooks are shown with the outputs saved in them; they are not re-run when the site is built.
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]
myst_heading_anchors = 2

autosummary_generate = True
autodoc_default_options = {"members": False}
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": 4}
html_title = "TwINFER"

html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_context = {
    "display_github": True,
    "github_user": "GoyalLab",
    "github_repo": "TwINFER",
    "github_version": "TwINFER_v2",
    "conf_py_path": "/docs/",
}


# Sphinx wraps every multi-line display equation in \begin{split}...\end{split}, where \tag is not allowed
# ("\tag not allowed in split environment"). Equations with their own \tag are passed to MathJax as written.
def _tagged_math_as_written(app, doctree):
    from docutils import nodes
    for node in doctree.findall(nodes.math_block):
        text = node.astext()
        if r"\tag" in text and not node.get("nowrap"):
            node["nowrap"] = True
            node.children = [nodes.Text(r"\[" + text + r"\]")]


def setup(app):
    app.connect("doctree-read", _tagged_math_as_written)
