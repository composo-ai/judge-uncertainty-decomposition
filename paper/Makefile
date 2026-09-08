# Build the workshop paper. Requires a TeX Live install with latex-recommended,
# latex-extra and fonts-recommended (see FIT_REPORT.md for the apt line).

.PHONY: all clean pages proof

all: main.pdf

main.pdf: main.tex refs.bib neurips_2026.sty
	pdflatex -interaction=nonstopmode -file-line-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -file-line-error main.tex
	pdflatex -interaction=nonstopmode -file-line-error main.tex
	@echo
	@grep -E "Output written" main.log
	@echo "--- overfull boxes (0 is the target) ---"
	@grep -c "Overfull" main.log || true

# Where each section starts, and where the page-limit-exempt references begin.
pages: main.pdf
	@python3 pagemap.py

# Render each page to PNG for a visual check of the port.
proof: main.pdf
	@mkdir -p proof && pdftoppm -png -r 110 main.pdf proof/page && ls proof/

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf build*.log bib.log
	rm -rf proof
