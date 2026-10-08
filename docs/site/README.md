# MQDEsys documentation site

Sphinx project with two language editions of identical structure, `en/` and `zh/`, modelled on the layout of the EvalScope docs (one Sphinx project per language, a shared configuration, a language link in the sidebar). Both editions are Markdown (MyST) built with the Furo theme.

```text
docs/site/
  conf_common.py        settings shared by both languages (extensions, theme, sidebar)
  requirements.txt      docs build only: sphinx, myst-parser, furo, sphinx-copybutton, sphinx-design
  Makefile              make html-en | html-zh | html | status | clean
  en/  zh/              the two projects: conf.py, index.md, get_started/, user_guides/, advanced/, faq.md, status.md
  _shared/img/          screenshots, referenced from both languages
  landing/index.html    root page with links to both languages (copied to _build/html/index.html)
  scripts/gen_status.py generates en/_status_tables.md and zh/_status_tables.md from ../TASKS.md
  pages-workflow.proposal.yml   proposed GitHub Pages workflow (not active, see below)
```

## Build

```bash
python3 -m venv build/venv-docs                      # from the repository root
build/venv-docs/bin/pip install -r docs/site/requirements.txt
export PATH=$PWD/build/venv-docs/bin:$PATH
cd docs/site
make html                # both languages + landing page into _build/html/{en,zh,index.html}
make html-en             # or just one
make html-zh
```

The Makefile runs `sphinx-build -W --keep-going`, so any warning (broken link, missing image, bad heading level) fails the build. Open `_build/html/index.html` in a browser.

## Language switching

The two editions are separate Sphinx projects with the same file names. Each project's sidebar contains `_templates/lang_switch.html`, which links the current page to `../<other-language>/<same-page>.html`. That is why the file structure of `en/` and `zh/` must stay identical: adding a page to one language without the other leaves a dead switch link. The built site is served as `/en/...` and `/zh/...` next to a root `index.html` that offers both.

## Keeping it truthful

- Every command, flag and endpoint in the pages was checked against the code or run. When you change `llm_job.py`, `server.cpp` or the CLI, update both languages.
- Each page marks features as implemented, partial or planned; do not remove those labels without checking `docs/TASKS.md`.
- Numbers come from `docs/PROGRESS.md` or real jobs shown in the UI; do not add numbers you have not measured.
- Do not write internal addresses, host names or absolute server paths; use `<REPO>`, `<MODEL_DIR>` and similar placeholders.
- Regenerate the status tables after `docs/TASKS.md` changes: `make status` (add an English title to the `EN` map in `scripts/gen_status.py` for new task IDs).
- Screenshots live in `_shared/img/`. They come from the web UI (whose text is Chinese); crop out model path fields.

## Publishing

`pages-workflow.proposal.yml` is a GitHub Actions workflow that builds both languages and deploys `_build/html` to GitHub Pages. It is a proposal only: CI files need maintainer approval, so it is not under `.github/`. To adopt it, copy it to `.github/workflows/docs.yml` and enable Pages (source: GitHub Actions) in the repository settings.
