"""Settings shared by the English and Chinese Sphinx projects (each <lang>/conf.py imports this)."""

project = "MQDEsys"
author = "MQDEsys contributors"
copyright = "2026, MQDEsys contributors"
release = "0.1"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_status_tables.md"]

myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence", "deflist", "attrs_inline"]

html_theme = "furo"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "lang_switch.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ]
}
html_theme_options = {
    "source_repository": "",
    "navigation_with_keys": True,
}
html_css_files = ["custom.css"]
copybutton_prompt_text = r"^\$ "
copybutton_prompt_is_regexp = True
