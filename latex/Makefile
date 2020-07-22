TEX_FILES:=$(shell find -type f -name '*.tex')
PNG_FILES:=$(shell find -type f -name '*.png')
BIB_FILE:=bibliography.bib
SRCS:=${TEX_FILES} ${PNG_FILES} ${BIB_FILE}

all: main.pdf
	
main.pdf: ${SRCS}
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

clean:
	rm main.pdf

lint:
	chktex -n 12 main.tex

.PHONY: clean lint
