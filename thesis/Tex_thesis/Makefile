.PHONY: all clean

all:
	xelatex -synctex=1 -interaction=nonstopmode Thesis.tex
	bibtex Thesis
	xelatex -synctex=1 -interaction=nonstopmode Thesis.tex
	xelatex -synctex=1 -interaction=nonstopmode Thesis.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.synctex.gz
