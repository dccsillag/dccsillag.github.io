MARKDOWNS = $(shell find . -name '*.md')
HTMLS = $(MARKDOWNS:%.md=%.html)

BASE_STYLE = $(abspath style/base-style.css)
PANDOC_OPTIONS = --mathjax -H style/head.html


.PHONY: default
default: all


.PHONY: all
all: $(HTMLS)

.PHONY: clean
clean:
	rm -f $(HTMLS)


%.html: %.md
	@echo "pandoc: $< -> $@"
	@pandoc --css $(shell realpath --relative-to="$(@D)" $(BASE_STYLE)) --resource-path $(@D):`pwd` $(PANDOC_OPTIONS) -o $@ $<
