# MQDEsys documentation site

This directory contains the Sphinx documentation for MQDEsys, with two language editions of identical structure, `en/` and `zh/`. Both editions use Markdown (MyST), the Furo theme, and a shared configuration module. The layout follows the conventions used by the EvalScope documentation: one Sphinx project per language, a shared configuration, and a language switch in the sidebar.

```text
docs/site/
  conf_common.py        configuration shared by both languages
  requirements.txt      documentation build dependencies
  Makefile              make html-en | html-zh | html | clean
  en/  zh/              Sphinx projects: conf.py, index.md, get_started/, user_guides/, advanced/, faq.md
  _shared/img/          screenshots referenced by both languages
  landing/index.html    landing page linking to both editions
```

## Build

```bash
python3 -m venv build/venv-docs
build/venv-docs/bin/pip install -r docs/site/requirements.txt
export PATH=$PWD/build/venv-docs/bin:$PATH
cd docs/site
make html                # both editions and the landing page into _build/html
make html-en             # or a single edition
make html-zh
```

The Makefile runs `sphinx-build -W --keep-going`, so any warning (broken link, missing image, invalid heading level) fails the build. Open `_build/html/index.html` in a browser to review the result.

## Language switching

The two editions are separate Sphinx projects with matching file names. Each project's sidebar includes `_templates/lang_switch.html`, which links the current page to `../<other-language>/<same-page>.html`. The file structure of `en/` and `zh/` must therefore remain identical. The built site is served as `/en/...` and `/zh/...` alongside a root `index.html`.

## Content guidelines

- Every command, flag and endpoint must be verified against the source or a running instance. Changes to `llm_job.py`, `server.cpp` or the CLI must be reflected in both languages.
- Documentation describes capabilities of the current release. Internal planning material, completion tracking and task identifiers belong outside this directory.
- Measured numbers must come from reproducible runs. State the model, dataset, hardware and software versions alongside them.
- Do not include internal host names, addresses or absolute server paths. Use placeholders such as `<REPO>`, `<MODEL_DIR>` and `<JOB_ID>`.
- Screenshots live in `_shared/img/` and are captured from the web workspace. Crop model path fields where they would expose local paths.

## Publishing

Documentation can be published to GitHub Pages by building both editions and deploying `_build/html` as a static site. The build output is self-contained and requires no server-side runtime.
