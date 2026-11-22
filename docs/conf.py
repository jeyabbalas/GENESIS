"""Sphinx configuration for the GENESIS documentation.

Build locally with::

    pip install -e ".[docs]"
    sphinx-build -W --keep-going -b html docs docs/_build/html

Hand-written pages are MyST Markdown; the API reference (``docs/api``) is
reStructuredText because it consists of autodoc directives.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
from typing import Any

import genesis_gwas

# -- Project information ------------------------------------------------------
project = "GENESIS"
author = "Jeya Balaji Balasubramanian"
copyright = (
    "2026, Jeya Balaji Balasubramanian. Method and original R package: "
    "Yan Zhang, Guanghao Qi, Ju-Hyun Park and Nilanjan Chatterjee"
)
release = genesis_gwas.__version__
version = ".".join(release.split(".")[:2])

REPO_URL = "https://github.com/jeyabbalas/GENESIS"
SITE_URL = "https://jeyabbalas.github.io/GENESIS/"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "sphinxarg.ext",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "_scripts", "Thumbs.db", ".DS_Store", "**/.ipynb_checkpoints"]
language = "en"
nitpicky = False
default_role = None
pygments_style = "friendly"
pygments_dark_style = "monokai"
numfig = True
numfig_format = {"figure": "Figure %s", "table": "Table %s", "code-block": "Listing %s"}

# -- MyST ---------------------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "substitution",
]
if importlib.util.find_spec("linkify_it") is not None:
    myst_enable_extensions.append("linkify")
myst_heading_anchors = 3
myst_dmath_double_inline = True
myst_fence_as_directive = ["mermaid"]
myst_linkify_fuzzy_links = False
myst_substitutions = {
    "version": release,
    "repo": REPO_URL,
    "install_spec": '"genesis-gwas @ git+https://github.com/jeyabbalas/GENESIS.git"',
}

# -- Mathematics --------------------------------------------------------------
# Macros are conveniences only: every page also renders with plain LaTeX.
mathjax3_config = {
    "tex": {
        "macros": {
            "bhat": r"\hat{\beta}",
            "pic": r"\pi_c",
            "Nstar": r"N^{*}",
            "E": r"\operatorname{E}",
            "var": r"\operatorname{var}",
            "cov": r"\operatorname{cov}",
            "Normal": r"\mathcal{N}",
            "tr": r"\operatorname{tr}",
        }
    }
}

# -- Diagrams -----------------------------------------------------------------
mermaid_init_js = (
    "mermaid.initialize({startOnLoad: true, securityLevel: 'strict', "
    "theme: 'neutral', flowchart: {useMaxWidth: true, htmlLabels: true}});"
)

# -- autodoc / autosummary / napoleon ------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": False,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_class_signature = "mixed"
autodoc_preserve_defaults = True
autodoc_mock_imports = [] if importlib.util.find_spec("torch") is not None else ["torch"]
autosummary_generate = False
python_maximum_signature_line_length = 88
add_module_names = True
toc_object_entries_show_parents = "hide"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_use_ivar = False
napoleon_preprocess_types = False
napoleon_attr_annotations = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}
intersphinx_timeout = 20

# -- copybutton ---------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_exclude = ".linenos, .gp, .go"

# -- HTML output --------------------------------------------------------------
html_theme = "furo"
html_title = "GENESIS"
html_baseurl = SITE_URL
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_last_updated_fmt = "%Y-%m-%d"
html_theme_options = {
    "source_repository": f"{REPO_URL}/",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": ["view", "edit"],
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#1f5a8a",
        "color-brand-content": "#1f5a8a",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7fb8e6",
        "color-brand-content": "#7fb8e6",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": REPO_URL,
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 '
                "3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-"
                "2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 "
                "1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-"
                ".89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2"
                ".82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 "
                "1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25"
                ".54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 "
                '16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}


# -- Cross-references to re-exported names -------------------------------------
# The public objects are documented under the names users import
# (``genesis_gwas.fit``, ``genesis_gwas.ld.build_reference``) while docstrings also
# refer to the defining modules (``genesis_gwas.fit.fit``,
# ``genesis_gwas.ld.build.build_reference``). Resolve a reference that Sphinx cannot
# find by importing the target and looking up every name under which the same
# object may have been documented.
def _import_dotted(target: str) -> tuple[list[Any], list[str]] | None:
    """Resolve ``a.b.c`` to the chain of objects reached after the module part."""
    parts = target.split(".")
    for i in range(1, len(parts) + 1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:i]))
        except Exception:
            continue
        chain = [obj]
        for attr in parts[i:]:
            try:
                obj = inspect.getattr_static(obj, attr)
            except AttributeError:
                break
            if isinstance(obj, (staticmethod, classmethod)):
                obj = obj.__func__
            chain.append(obj)
        else:
            return chain, parts[i:]
    return None


def _candidate_names(target: str) -> list[str]:
    resolved = _import_dotted(target)
    if resolved is None:
        return []
    chain, attrs = resolved
    # Last element of the chain that knows where it was defined (class or function).
    for depth in range(len(chain) - 1, 0, -1):
        owner = chain[depth]
        module = getattr(owner, "__module__", None)
        qualname = getattr(owner, "__qualname__", None)
        if isinstance(module, str) and isinstance(qualname, str):
            rest = "".join(f".{a}" for a in attrs[depth:])
            names = [f"{module}.{qualname}{rest}"]
            pieces = module.split(".")
            for j in range(len(pieces) - 1, 0, -1):
                names.append(f"{'.'.join(pieces[:j])}.{qualname}{rest}")
            return names
    return []


_PREFIX_ALIASES = {"np.": "numpy.", "pd.": "pandas.", "plt.": "matplotlib.pyplot."}


def _resolve_alias(app: Any, env: Any, node: Any, contnode: Any) -> Any:
    if node.get("refdomain") != "py" or node.get("reftype") == "mod":
        return None
    target = node.get("reftarget", "")

    # ``np.ndarray`` in an annotation -> ``numpy.ndarray`` in the NumPy inventory.
    for short, full in _PREFIX_ALIASES.items():
        if target.startswith(short):
            from sphinx.ext.intersphinx import missing_reference

            node["reftarget"] = full + target[len(short) :]
            try:
                return missing_reference(app, env, node, contnode)
            finally:
                node["reftarget"] = target

    from sphinx.util.nodes import make_refnode

    objects = env.get_domain("py").objects

    def link(name: str) -> Any:
        entry = objects[name]
        return make_refnode(
            app.builder, node["refdoc"], entry.docname, entry.node_id, contnode, name
        )

    tries = [target]
    for key in ("py:class", "py:module"):
        ctx = node.get(key)
        if ctx:
            tries.append(f"{ctx}.{target}")
            if key == "py:class" and node.get("py:module"):
                tries.append(f"{node['py:module']}.{ctx}.{target}")
    for name in tries:
        if not name.startswith("genesis_gwas"):
            continue
        for cand in _candidate_names(name):
            if cand in objects:
                return link(cand)

    # A bare name used inside a docstring that is rendered under a re-exported name
    # (``default_data_dir`` in the docstring of ``genesis_gwas.load_reference``): accept
    # it if exactly one documented object ends with that name.
    if "." not in target:
        suffix = f".{target}"
        matches = [name for name in objects if name.endswith(suffix)]
        # A class documented under a re-exported name is also registered under its
        # canonical name; both entries point to the same anchor.
        anchors = {(objects[name].docname, objects[name].node_id) for name in matches}
        if len(anchors) == 1:
            return link(min(matches, key=len))
    return None


class _InventoryFailureFilter(logging.Filter):
    """Do not let an unreachable third-party inventory fail a ``-W`` build.

    Links into the NumPy/SciPy/... documentation are a convenience; if one of those
    sites is down when the documentation is built, the affected references simply
    render as plain text.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "failed to reach any of the inventories" in message:
            print(f"[intersphinx] skipped: {message.splitlines()[0]}")
            return False
        return True


def _quiet_argparse_domain() -> None:
    """Keep sphinx-argparse's ``commands`` domain out of Markdown link resolution.

    MyST resolves ``[text](#target)`` by asking every domain for the target. The
    ``commands`` domain does not implement Sphinx's ``resolve_any_xref`` interface, so
    MyST warns about that and then falls back to the domain's ``resolve_xref``, which
    itself logs a warning for every target that is not a command — that is, for every
    link to the bibliography. Commands are never the intended target of such links
    here, so the domain simply answers "no match".
    """
    try:
        from sphinxarg.ext import ArgParseDomain
    except ImportError:  # the extension is listed above; without it Sphinx stops earlier
        return
    if "resolve_any_xref" not in vars(ArgParseDomain):

        def resolve_any_xref(self: Any, *args: Any, **kwargs: Any) -> list[Any]:
            return []

        ArgParseDomain.resolve_any_xref = resolve_any_xref


def setup(app: Any) -> dict[str, Any]:
    """Register the alias resolver and the two warning-hygiene shims."""
    _quiet_argparse_domain()
    app.connect("missing-reference", _resolve_alias)
    logging.getLogger("sphinx.sphinx.ext.intersphinx").addFilter(_InventoryFailureFilter())
    return {"parallel_read_safe": True, "parallel_write_safe": True}
