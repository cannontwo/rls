.PHONY: root.pdf all clean
	all: root.pdf root_opt.pdf

root.pdf: src/*.tex Makefile
	latexmk -pdf -bibtex -pdflatex="pdflatex -interaction=nonstopmode -file-line-error -shell-escape" -use-make src/root.tex
	cp root.pdf writeup.pdf

# Optimize the paper for submission
root_opt.pdf: root.pdf
	ghostscript -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -sOutputFile=root_opt.pdf root.pdf

clean:
	latexmk -CA -f root.pdf
	rm root*
