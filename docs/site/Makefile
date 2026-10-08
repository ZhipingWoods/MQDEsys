# Build both language editions into _build/html/{en,zh} plus a language landing page.
#   make html-en | html-zh | html | clean | status
SPHINXBUILD ?= sphinx-build
SPHINXOPTS  ?= -W --keep-going
BUILD       := _build/html

.PHONY: html html-en html-zh landing status clean

html: html-en html-zh landing

html-en:
	$(SPHINXBUILD) $(SPHINXOPTS) -b html en $(BUILD)/en

html-zh:
	$(SPHINXBUILD) $(SPHINXOPTS) -b html zh $(BUILD)/zh

landing:
	mkdir -p $(BUILD)
	cp landing/index.html $(BUILD)/index.html

# Regenerate the status tables from ../TASKS.md (run after TASKS.md changes)
status:
	python3 scripts/gen_status.py

clean:
	rm -rf _build
