"""Emit the weekly report as LaTeX from the same CONTENT the PDF is built from.

Single source of truth: build_weekly_report.py renders CONTENT to PDF, this
renders the identical CONTENT to .tex, so the two cannot drift apart.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_weekly_report import CONTENT, ABSTRACT

SPECIAL = {'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#',
           '_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
def esc(t):
    t = ''.join(SPECIAL.get(c, c) for c in t)
    return t.replace(' - ', ' --- ').replace('"', "''")

PRE = r"""\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\setmainfont{TeX Gyre Pagella}
\setsansfont{Carlito}
\setmonofont{DejaVu Sans Mono}
\usepackage[a4paper,top=2.4cm,bottom=2.4cm,left=2.3cm,right=2.3cm,
            headheight=34pt,headsep=14pt]{geometry}
\usepackage{graphicx}\usepackage{fancyhdr}\usepackage{caption}
\usepackage{titlesec}\usepackage{booktabs}\usepackage{xcolor}
\usepackage{enumitem}\usepackage{amsmath}\usepackage{parskip}
\usepackage[hidelinks]{hyperref}
\definecolor{good}{HTML}{157A3C}
\titleformat{\section}{\sffamily\large\bfseries}{\thesection}{0.8em}{}
\titleformat{\subsection}{\sffamily\normalsize\bfseries}{\thesubsection}{0.8em}{}
\captionsetup[figure]{labelfont={sf,bf},textfont=small,labelsep=space}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\includegraphics[height=30pt]{weekly_figs/header_banner.png}}
\fancyfoot[L]{\sffamily\small Weekly Report}\fancyfoot[R]{\sffamily\small\thepage}
\renewcommand{\headrulewidth}{0pt}\setlength{\parindent}{0pt}
\begin{document}
\thispagestyle{empty}
\begin{center}
  \includegraphics[height=62pt]{weekly_figs/logo_kaust_academy.jpg}\hspace{28pt}
  \includegraphics[height=62pt]{weekly_figs/logo_kaust.jpg}
\end{center}
\vspace{18pt}
\begin{center}
  {\sffamily\bfseries\LARGE Weekly Report}\\[10pt]
  {\large Ali Alhulaimi}\\[3pt]{\ttfamily\small alialhulaimi2005@gmail.com}
\end{center}
\vspace{16pt}
\begin{center}\begin{minipage}{0.88\linewidth}
{\sffamily\bfseries\large Abstract}\\[4pt]\small
%%ABSTRACT%%
\end{minipage}\end{center}
\vspace{20pt}
\begin{center}\begin{minipage}{0.88\linewidth}\small
\begin{tabular}{@{}p{2.2cm}l@{}}
\textbf{Date:} & August 2026\\[3pt]
\textbf{Explainer:} & {\ttfamily\footnotesize\url{https://claude.ai/public/artifacts/c52594ab-8c5c-404c-8999-68e62b4054ee}}\\[3pt]
\textbf{Service:} & {\ttfamily\footnotesize\url{https://scifablabs-mac-mini.tailfc1a5e.ts.net/}}\\
\end{tabular}
\end{minipage}\end{center}
\newpage
\tableofcontents
\newpage
"""

FIGW = {1:'\\linewidth', 2:'0.85\\linewidth', 3:'0.23\\linewidth', 4:'\\linewidth'}

def emit(content, abstract, dest, figw=None):
    FIGW = figw or globals()['FIGW']
    out = [PRE.replace('%%ABSTRACT%%', esc(abstract.strip()))]
    for it in content:
        k = it[0]
        if k == 'h1':
            out.append('' if it[1] else r'\section*{%s}\addcontentsline{toc}{section}{%s}' % (it[2], it[2]))
            if it[1]: out.append(r'\section{%s}' % esc(it[2]))
        elif k == 'h2':
            out.append(r'\subsection{%s}' % esc(it[2]))
        elif k == 'p':
            out.append(esc(it[1].strip()) + '\n')
        elif k == 'eq':
            out.append(r'\[ \text{%s} \]' % esc(it[1]))
        elif k == 'fig':
            files, _w, n, cap = it[1], it[2], it[3], it[4]
            imgs = '\\hfill\n  '.join(r'\includegraphics[width=%s]{weekly_figs/%s}' % (FIGW[n], f) for f in files)
            out.append('\\begin{figure}[htbp]\n  \\centering\n  %s\n  \\caption{%s}\n\\end{figure}' % (imgs, esc(cap)))
        elif k == 'table':
            hdr, rows = it[1], it[2]
            out.append(r'\begin{table}[htbp]\centering\small')
            out.append(r'\begin{tabular}{@{}lrr@{}}\toprule')
            out.append(' & '.join(r'\textbf{%s}' % esc(c) for c in hdr) + r' \\ \midrule')
            for r in rows:
                cells = [(r'\textcolor{good}{\textbf{%s}}' % esc(c[1:])) if c.startswith('*') else esc(c) for c in r]
                out.append(' & '.join(cells) + r' \\')
            out.append(r'\bottomrule\end{tabular}\end{table}')
        elif k == 'refs':
            out.append(r'\begin{enumerate}[leftmargin=*,itemsep=3pt]')
            for t in it[1]:
                t = esc(t)
                t = re.sub(r'(https?://\S+)', r'\\url{\1}', t)
                out.append(r'  \item ' + t)
            out.append(r'\end{enumerate}')
    out.append(r'\end{document}')
    open(dest, 'w').write('\n'.join(x for x in out if x is not None) + '\n')
    return dest


if __name__ == '__main__':
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weekly_report_ali.tex')
    print('wrote', emit(CONTENT, ABSTRACT, dest))
