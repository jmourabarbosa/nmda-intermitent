"""Convert the paper's Google Doc (a documents.get JSON) into LaTeX, keeping the doc's text, formatting and figures.

Pending suggestions are resolved as not accepted: suggested insertions are dropped, suggested deletions kept.
Usage: python doc_to_latex.py DOC.json [OUT_DIR]   -> OUT_DIR/main.tex, OUT_DIR/figures/figNN.png
Build: pdflatex main.tex
"""
import json, os, re, sys, urllib.request

ESC = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '$': r'\$', '&': r'\&', '#': r'\#', '^': r'\^{}',
       '_': r'\_', '%': r'\%', '~': r'\textasciitilde{}'}
PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,textcomp,amsmath,amssymb}
\usepackage{graphicx}
\usepackage{setspace}
\usepackage{titlesec}
\usepackage[hidelinks]{hyperref}
\setstretch{1.15}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.6em}
\titleformat{\section}{\Large\bfseries}{}{0pt}{}
\titleformat{\subsection}{\large\bfseries}{}{0pt}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{}{0pt}{}
\begin{document}
"""


GREEK = dict(zip('αβγδεζηθικλμνξπρστυφχψωΔΣΓΘΛΞΠΦΨΩ',
                 ['alpha', 'beta', 'gamma', 'delta', 'varepsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu',
                  'nu', 'xi', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega', 'Delta', 'Sigma',
                  'Gamma', 'Theta', 'Lambda', 'Xi', 'Pi', 'Phi', 'Psi', 'Omega']))
MATH = {'−': '-', '±': r'\pm', '×': r'\times', '√': r'\surd', '·': r'\cdot', '∈': r'\in', '≥': r'\geq',
        '≤': r'\leq', '≈': r'\approx', '∼': r'\sim', '→': r'\rightarrow'}
SUP = dict(zip('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻', '0123456789+-'))
OK = set('–—’‘“”éèêëáàâäíìîïóòôöúùûüçñÉÁÖÜ°')
unknown = set()


def esc(s):
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        if i + 1 < len(s) and s[i + 1] == '\u0304':            # combining macron: x̄
            out.append(r'\ensuremath{\bar{%s}}' % ch); i += 2; continue
        if ch in ESC:
            out.append(ESC[ch])
        elif ch in GREEK:
            out.append(r'\ensuremath{\%s}' % GREEK[ch])
        elif ch in MATH:
            out.append(r'\ensuremath{%s}' % MATH[ch])
        elif ch in SUP:
            out.append(r'\textsuperscript{%s}' % ('\\ensuremath{-}' if SUP[ch] == '-' else SUP[ch]))
        elif ch == '°':
            out.append(r'\textdegree{}')
        elif ord(ch) > 127 and ch not in OK:
            unknown.add(ch); out.append(ch)
        else:
            out.append(ch)
        i += 1
    return ''.join(out)


def run_tex(text, st):
    t = esc(text)
    if not t:
        return ''
    off = st.get('baselineOffset')
    if off == 'SUBSCRIPT':
        t = r'\textsubscript{' + t + '}'
    elif off == 'SUPERSCRIPT':
        t = r'\textsuperscript{' + t + '}'
    if st.get('italic'):
        t = r'\textit{' + t + '}'
    if st.get('bold'):
        t = r'\textbf{' + t + '}'
    return t


def convert(doc, out_dir):
    tab = doc['tabs'][0]['documentTab'] if 'tabs' in doc else doc
    inline = tab.get('inlineObjects', {})
    body = tab['body']['content']
    os.makedirs(os.path.join(out_dir, 'figures'), exist_ok=True)
    out, nfig, n_pending = [PREAMBLE], 0, 0
    for el in body:
        p = el.get('paragraph')
        if not p:
            continue
        style = p['paragraphStyle'].get('namedStyleType', 'NORMAL_TEXT')
        align = p['paragraphStyle'].get('alignment', '')
        parts = []
        for r in p['elements']:
            if 'inlineObjectElement' in r:
                ie = r['inlineObjectElement']
                if ie.get('suggestedInsertionIds'):
                    n_pending += 1
                    continue
                obj = inline[ie['inlineObjectId']]['inlineObjectProperties']['embeddedObject']
                url = obj.get('imageProperties', {}).get('sourceUri') or obj.get('imageProperties', {}).get('contentUri')
                nfig += 1
                fn = 'figures/fig%02d.png' % nfig
                urllib.request.urlretrieve(url, os.path.join(out_dir, fn))
                w = obj.get('size', {}).get('width', {}).get('magnitude', 468)
                parts.append('\\includegraphics[width=%.3f\\textwidth]{%s}' % (min(w / 468, 1), fn))
                continue
            tr = r.get('textRun')
            if not tr:
                continue
            if tr.get('suggestedInsertionIds'):
                n_pending += 1
                continue
            if tr.get('suggestedDeletionIds'):
                n_pending += 1
            parts.append(run_tex(tr['content'].replace('\n', ''), tr.get('textStyle', {})))
        text = ''.join(parts).strip()
        if not text:
            continue
        plain = re.sub(r'\\text(bf|it|subscript|superscript)\{|\}', '', text)
        if style == 'TITLE':
            out.append('{\\centering\\LARGE\\bfseries %s\\par}\n' % text)
        elif style == 'SUBTITLE':
            out.append('{\\centering\\large %s\\par}\n' % text)
        elif style == 'HEADING_1':
            out.append('\\section*{%s}\n' % plain)
        elif style == 'HEADING_2':
            out.append('\\subsection*{%s}\n' % text)
        elif style.startswith('HEADING'):
            out.append('\\subsubsection*{%s}\n' % text)
        elif align == 'CENTER' or text.startswith('\\includegraphics'):
            out.append('\\begin{center}%s\\end{center}\n' % text)
        else:
            out.append(text + '\n')
    out.append('\\end{document}\n')
    open(os.path.join(out_dir, 'main.tex'), 'w').write('\n'.join(out))
    return nfig, n_pending


if __name__ == '__main__':
    d = json.load(open(sys.argv[1]))
    d = d.get('content', d)
    nfig, npend = convert(d, sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__)))
    if unknown: print('UNMAPPED characters:', ''.join(sorted(unknown)))
    print('figures', nfig, '| pending suggestion runs resolved as not accepted:', npend, '| revision', d.get('revisionId', '')[:20])
