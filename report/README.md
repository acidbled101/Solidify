# Lab report

KAUST Academy lab report for the TRELLIS.2 Apple-Silicon print-prep project.

- `report.tex` — the report. Self-contained (compiles with `pdflatex` + `biber`).
  To use the official KAUST Academy template, paste the body (from
  `\section{Introduction}` to `\printbibliography`) into the template `.tex`
  and keep `bibliography.bib`.
- `bibliography.bib` — only the works actually used by the project.

## Compile

```bash
cd report
pdflatex report.tex
biber report
pdflatex report.tex
pdflatex report.tex
```

## Numbers

Every quantitative figure in the report comes from running the real pipeline on
a real generated asset (`test/output_3d.glb`), captured July 2026 on the machine
in this repo. The validation self-check (exact 0.00% deviation on an identity
transform) and the overhang cross-check (measured 14.9% vs. analytic 14.6% on a
sphere) were reproduced with the current `make_printable.py`.
