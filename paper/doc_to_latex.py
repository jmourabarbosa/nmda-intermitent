"""Convert the paper's Google Doc (documents.get JSON) into a typeset LaTeX paper.

The doc is the only source of text. The converter adds the layout the doc cannot hold:
- figures: each image + its "Figure N." / "Supplementary Figure N." paragraph -> a float with a caption;
- citations: "(Author et al., 2024)", "Author (2024)" -> natbib \\citep/\\citet with refs.bib
  (a citation or doc reference without a bib entry stops the build);
- references: generated from refs.bib (only cited entries); the doc's reference list is checked, not copied;
- equations: centred equation paragraphs whose text matches EQUATIONS are typeset from their LaTeX version;
  any other centred paragraph is kept as text and reported, so an edited equation is never replaced silently.
Pending suggestions count as not accepted (insertions dropped, deletions kept).

Usage: python doc_to_latex.py DOC.json [OUT_DIR] [--exclude 'AI methodology'] [--name main_noAI]   then: pdflatex main && bibtex main && pdflatex main && pdflatex main
"""
import json, os, re, sys, urllib.request

ESC = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '$': r'\$', '&': r'\&', '#': r'\#', '^': r'\^{}',
       '_': r'\_', '%': r'\%', '~': r'\textasciitilde{}'}
GREEK = dict(zip('αβγδεζηθικλμνξπρστυφχψωΔΣΓΘΛΞΠΦΨΩ',
                 ['alpha', 'beta', 'gamma', 'delta', 'varepsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu',
                  'nu', 'xi', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega', 'Delta', 'Sigma',
                  'Gamma', 'Theta', 'Lambda', 'Xi', 'Pi', 'Phi', 'Psi', 'Omega']))
MATH = {'−': '-', '±': r'\pm', '×': r'\times', '√': r'\surd', '·': r'\cdot', '∈': r'\in', '≥': r'\geq',
        '≤': r'\leq', '≈': r'\approx', '∼': r'\sim', '→': r'\rightarrow'}
SUP = dict(zip('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻', '0123456789+-'))
OK = set('–—’‘“”éèêëáàâäíìîïóòôöúùûüçñÉÁÖÜ')

# LaTeX for the doc's display equations, keyed by the equation's plain text in the doc
EQUATIONS = {
    'τr dri/dt = −ri + φ(xi)': r'\tau_r \frac{dr_i}{dt} = -r_i + \varphi(x_i)',
    'xi = ws si + wa ri − wI rI + I0 m(t) − wff q + cuei + σ ξi':
        r'x_i = \bm{w_s} s_i + w_a r_i - w_I r_I + I_0\, m(t) - w_{\mathit{ff}}\, q + \mathrm{cue}_i + \bm{\sigma} \xi_i',
    'τN dsi/dt = −si + β ri (1 − si)': r'\tau_N \frac{ds_i}{dt} = -s_i + \beta\, r_i (1 - s_i)',
    'τI drI/dt = −rI + wEI (rcued + runcued)': r'\tau_I \frac{dr_I}{dt} = -r_I + w_{EI} \left(r_\mathrm{cued} + r_\mathrm{uncued}\right)',
    'τff dq/dt = −q + m(t)': r'\tau_{\mathit{ff}} \frac{dq}{dt} = -q + m(t)',
    'Cm dV/dt = −gL (V − EL) − Iext − INMDA − IGABA + Icue,':
        r'C_m \frac{dV}{dt} = -g_L (V - E_L) - I_\mathrm{ext} - I_\mathrm{NMDA} - I_\mathrm{GABA} + I_\mathrm{cue},',
    'Iext = gext sAMPA (V − EE),   INMDA = gNMDA B(V) · (V − EE) Σj Wij sNMDA,j,   IGABA = gGABA (V − EI) Σj sGABA,j,':
        r'\begin{gathered} I_\mathrm{ext} = g_\mathrm{ext}\, s_\mathrm{AMPA} (V - E_E), \\ I_\mathrm{NMDA} = g_\mathrm{NMDA}\, B(V) \cdot (V - E_E) '
        r'\textstyle\sum_j W_{ij}\, s_{\mathrm{NMDA},j}, \\ I_\mathrm{GABA} = g_\mathrm{GABA} (V - E_I) \textstyle\sum_j s_{\mathrm{GABA},j}, \end{gathered}',
    'dx/dt = −x/τx + Σk δ(t − tk),   dsNMDA/dt = −sNMDA/τNMDA + α x (1 − sNMDA),':
        r'\frac{dx}{dt} = -\frac{x}{\tau_x} + \sum_k \delta(t - t_k), \qquad \frac{ds_\mathrm{NMDA}}{dt} = '
        r'-\frac{s_\mathrm{NMDA}}{\tau_\mathrm{NMDA}} + \alpha\, x (1 - s_\mathrm{NMDA}),',
    '0 = −si + β (ui/U) ri (1 − si)': r'0 = -s_i + \beta\, (u_i/U)\, r_i (1 - s_i)',
    'τf dui/dt = −(ui − U) + τf U ri (1 − ui)': r'\tau_f \frac{du_i}{dt} = -(u_i - U) + \tau_f U r_i (1 - u_i)',
    'Wij = J− + (J+ − J−) exp(−(θi − θj)2/2σ2),':
        r'W_{ij} = J^- + (J^+ - J^-) \exp\!\left(-\frac{(\theta_i - \theta_j)^2}{2\sigma^2}\right),',
}
PREAMBLE = r"""%% Generated from the Google Doc by doc_to_latex.py -- do not edit by hand.
\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,textcomp}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb,bm}
\usepackage{graphicx}
\usepackage{xcolor}
\definecolor{natblue}{RGB}{0,107,182}
\usepackage{caption}
\DeclareCaptionLabelSeparator{pipe}{ $|$ }
\captionsetup{labelfont=bf,labelsep=pipe,font=small}
\usepackage[super,sort&compress,comma]{natbib}
\usepackage{microtype}
\usepackage[colorlinks,linkcolor=natblue,citecolor=natblue,urlcolor=natblue]{hyperref}
\renewcommand{\figurename}{Fig.}
\setcounter{secnumdepth}{0}
\renewcommand{\refname}{References}
"""
unknown, warnings = set(), []


def esc(s, bold=False):
    B = (lambda x: r'\bm{%s}' % x) if bold else (lambda x: x)   # \textbf does not reach math
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        if i + 1 < len(s) and s[i + 1] == '\u0304':
            out.append(r'\ensuremath{%s}' % B(r'\bar{%s}' % ch)); i += 2; continue
        if ch in ESC:
            out.append(ESC[ch])
        elif ch in GREEK:
            out.append(r'\ensuremath{%s}' % B('\\' + GREEK[ch]))
        elif ch in MATH:
            out.append(r'\ensuremath{%s}' % MATH[ch])
        elif ch in SUP:
            out.append(r'\textsuperscript{%s}' % (r'\ensuremath{-}' if SUP[ch] == '-' else SUP[ch]))
        elif ch == '°':
            out.append(r'\textdegree{}')
        else:
            if ord(ch) > 127 and ch not in OK:
                unknown.add(ch)
            out.append(ch)
        i += 1
    return ''.join(out)


def run_tex(text, st):
    t = esc(text, bold=bool(st.get('bold')))
    if not t.strip():
        return t
    if st.get('baselineOffset') == 'SUBSCRIPT':
        t = r'\textsubscript{' + t + '}'
    elif st.get('baselineOffset') == 'SUPERSCRIPT':
        t = r'\textsuperscript{' + t + '}'
    if st.get('italic'):
        t = r'\emph{' + t + '}'
    if st.get('bold'):
        t = r'\textbf{' + t + '}'
    return t


# ---------------------------------------------------------------- citations
def bib_field(body, name):
    m = re.search(r'\b%s\s*=\s*' % name, body)
    if not m:
        return None
    i = m.end()
    if body[i] != '{':
        return re.match(r'[^,}]*', body[i:]).group(0).strip()
    depth, j = 0, i
    while True:
        depth += {'{': 1, '}': -1}.get(body[j], 0)
        if depth == 0:
            return body[i + 1:j]
        j += 1


def bib_index(path):
    idx = {}
    for chunk in open(path).read().split('@')[1:]:
        key = chunk[chunk.index('{') + 1:chunk.index(',')].strip()
        au, yr = bib_field(chunk, 'author'), bib_field(chunk, 'year')
        if not au or not yr:
            continue
        names = [a.strip() for a in re.split(r'\s+and\s+', au.replace('\n', ' '))]
        first = names[0].split(',')[0].strip('{} ')
        second = names[1].split(',')[0].strip('{} ') if len(names) > 1 else ''
        n = 1 if len(names) == 1 else 2 if len(names) == 2 else 3
        idx[(norm(first), yr.strip('{} '), n, norm(second) if n == 2 else '')] = key
    return idx


def norm(s):
    return re.sub(r'[^a-z]', '', s.lower().replace('é', 'e').replace('ö', 'o').replace('ü', 'u'))


AUTH = r"[A-Z][A-Za-zÀ-ÿ\-]+"
ONE = rf"({AUTH})(?:\s+(et al\.)|\s+and\s+({AUTH}))?"


def keys_for(m_first, etal, second, years, idx):
    n, sec = (3, '') if etal else (2, norm(second)) if second else (1, '')
    out = []
    for y in re.findall(r'\d{4}', years):
        k = idx.get((norm(m_first), y, n, sec))
        if not k:   # two-word surnames ("Rahnama Rad"): the citation regex keeps only the last word
            ks = [v for (f, yy, nn, ss), v in idx.items() if f.endswith(norm(m_first)) and (yy, nn, ss) == (y, n, sec)]
            k = ks[0] if len(ks) == 1 else None
        if not k:
            raise KeyError(f'no bib entry for citation "{m_first}{" et al." if etal else ""}{" and " + second if second else ""}, {y}"')
        out.append(k)
    return out


def cite(text, idx, used):
    def paren(m):
        items = [x.strip() for x in m.group(1).split(';')]
        conv, any_cite = [], False
        for it in items:
            mm = re.fullmatch(rf"(.*?)(?:^|(?<=,\s)|(?<=\s)){ONE},\s*(\d{{4}}(?:,\s*\d{{4}})*)", it)
            if mm and re.fullmatch(rf"{ONE},\s*\d{{4}}(?:,\s*\d{{4}})*", it[len(mm.group(1)):]):
                ks = keys_for(mm.group(2), mm.group(3), mm.group(4), mm.group(5), idx); used.update(ks); any_cite = True
                conv.append((mm.group(1), ks))
            else:
                conv.append((it, None))
        if not any_cite:
            return m.group(0)
        if all(k is not None and not pre for pre, k in conv):
            return r'\citep{' + ','.join(k for _, ks in conv for k in ks) + '}'
        return '(' + '; '.join(pre + (r'\citealp{' + ','.join(ks) + '}' if ks else '') for pre, ks in conv) + ')'

    def narr(m):
        ks = keys_for(m.group(1), m.group(2), m.group(3), m.group(4), idx); used.update(ks)
        return m.group(0)[:m.group(0).rindex('(')].rstrip() + r'\citep{' + ','.join(ks) + '}'   # 'Author et al.' + superscript
    text = re.sub(r'\(([^()]*\d{4}[^()]*)\)', paren, text)
    text = re.sub(rf"{ONE}\s+\((\d{{4}}(?:,\s*\d{{4}})*)\)", narr, text)
    return text


# ---------------------------------------------------------------- figure references (Nature style)
PANELS = r"(?:[a-z](?![a-z])(?:[,–-][a-z](?![a-z]))*)?"


def figrefs(text):
    r"""'Fig. 1c,d' -> Fig.~\hyperref[fig:1]{1c,d}; 'Supplementary Figs. 2 and 3' likewise (labels fig:S2, ...)."""
    def one(sup, num, pan):
        return r'\hyperref[fig:%s%s]{%s%s}' % ('S' if sup else '', num, num, pan)

    def rep(m):
        sup, figs, num, pan, rest = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5) or ''
        rest = re.sub(r'(\s(?:and|to)\s|,\s)(\d+)(' + PANELS + ')', lambda r: r.group(1) + one(sup, r.group(2), r.group(3)), rest)
        return '%s%s.~%s%s' % (sup or '', figs, one(sup, num, pan), rest)
    return re.sub(r'(Supplementary )?(Figs?)\.\s(\d+)(' + PANELS + r')((?:(?:\s(?:and|to)\s|,\s)\d+' + PANELS + r'(?![\d.]))*)', rep, text)


def bold_panels(cap):
    """Caption panel labels in bold: 'a, ...' at the start of a sentence, and '(b)' references."""
    cap = re.sub(r'(^|(?<=\.\s)|(?<=\.\}\s)|(?<=\.\s\}))([a-l](?:(?:,\s?|[–-])[a-l])*),(?=\s)', lambda m: m.group(1) + r'\textbf{' + m.group(2) + '},', cap)
    return re.sub(r'\(([a-l](?:[,–-][a-l])*)\)', r'(\\textbf{\1})', cap)


# ---------------------------------------------------------------- paragraphs
def paragraphs(tab):
    inline = tab.get('inlineObjects', {})
    for el in tab['body']['content']:
        p = el.get('paragraph')
        if not p:
            continue
        tex, plain, imgs = [], [], []
        for r in p['elements']:
            if 'inlineObjectElement' in r:
                ie = r['inlineObjectElement']
                if ie.get('suggestedInsertionIds'):
                    continue
                obj = inline[ie['inlineObjectId']]['inlineObjectProperties']['embeddedObject']
                imgs.append(obj.get('imageProperties', {}).get('sourceUri') or obj.get('imageProperties', {}).get('contentUri'))
                continue
            tr = r.get('textRun')
            if not tr or tr.get('suggestedInsertionIds'):
                continue
            c = tr['content'].replace('\n', '')
            plain.append(c)
            tex.append(run_tex(c, tr.get('textStyle', {})))
        yield dict(style=p['paragraphStyle'].get('namedStyleType', 'NORMAL_TEXT'),
                   align=p['paragraphStyle'].get('alignment', ''), tex=''.join(tex).strip(),
                   plain=''.join(plain).strip(), imgs=imgs)


def convert(doc, out_dir, bib, exclude=(), name='main'):
    tab = doc['tabs'][0]['documentTab']
    idx = bib_index(bib)
    used = set()
    P = list(paragraphs(tab))
    os.makedirs(os.path.join(out_dir, 'figures'), exist_ok=True)
    # title block: paragraphs before "Abstract"
    ia = next(i for i, p in enumerate(P) if p['plain'] == 'Abstract')
    head = [p for p in P[:ia] if p['plain']]
    title, author, affil = head[0]['tex'], head[1]['tex'], ' '.join(p['tex'] for p in head[2:])
    out = [PREAMBLE, '\\title{\\Large %s}\n\\author{%s\\\\[4pt]\\parbox{0.9\\textwidth}{\\centering\\small %s}}\n\\date{}\n\\begin{document}\n\\maketitle\n'
           % (title, author, affil)]
    nfig, section, pending_img, doc_refs, eq_text = 0, '', None, [], []
    i = ia
    skipping = False
    while i < len(P):
        p = P[i]; i += 1
        is_title = p['style'].startswith('HEADING') or (p['tex'].startswith('\\textbf{') and p['tex'].endswith('}')
                                                       and 0 < len(p['plain']) < 60 and p['tex'].count('\\textbf{') == 1)
        if is_title:
            skipping = p['plain'] in exclude
        if skipping:
            continue
        if p['style'].startswith('HEADING') and p['plain']:
            if section == 'Abstract':
                out.append('\\end{abstract}\n')
            section = p['plain']
            if section == 'Abstract':
                out.append('\\begin{abstract}'); continue
            if section == 'References':
                continue
            if section == 'Supplementary Figures':
                out.append('\\clearpage\n\\section{Supplementary Figures}\n\\setcounter{figure}{0}\n'
                           '\\renewcommand{\\figurename}{Supplementary Fig.}\n'); continue
            out.append('\\%s{%s}\n' % ('section' if p['style'] == 'HEADING_1' else 'subsection', p['tex']))
            continue
        if p['imgs']:
            for url in p['imgs']:
                nfig += 1
                fn = 'figures/fig%02d.png' % nfig
                urllib.request.urlretrieve(url, os.path.join(out_dir, fn))
                pending_img = fn
            continue
        if not p['plain']:
            continue
        if section == 'References':
            doc_refs.append(p['plain']); continue
        m = re.match(r'(Supplementary )?Figure (\d+)\.\s*', p['plain'])
        if m and pending_img:
            cap = re.sub(r'^\\textbf\{(Supplementary )?Figure \d+\.\s*', r'\\textbf{', p['tex'])
            cap = re.sub(r'^(Supplementary )?Figure \d+\.\s*', '', cap)
            cap = figrefs(bold_panels(cite(cap, idx, used)))
            lab = 'fig:%s%s' % ('S' if m.group(1) else '', m.group(2))
            part2 = None
            from PIL import Image
            w, h = Image.open(os.path.join(out_dir, pending_img)).size
            fits = lambda c: 16 * h / w + 0.42 * len(re.sub(r'\\[a-zA-Z]+\{|\}|\\[a-zA-Z]+', '', c)) / 115 + 1.2 <= 23.5
            if not fits(cap):          # caption too long for the page: continue it on the next page, at a panel boundary
                cuts = [mm.start() for mm in re.finditer(r'\\textbf\{[a-l](?:[,–-]\s?[a-l])*\},', cap)]
                ok = [c for c in cuts if fits(cap[:c])]
                if ok:
                    cap, part2 = cap[:ok[-1]].rstrip(), cap[ok[-1]:]
                else:
                    warnings.append('caption too long for one page and no panel boundary to split at: ' + lab)
            out.append('\\begin{figure}[%s]\n\\centering\n\\includegraphics[width=\\textwidth]{%s}\n\\caption{%s}\\label{fig:%s%s}\n\\end{figure}\n'
                       % ('p' if m.group(1) or part2 else 'tbp', pending_img, cap, 'S' if m.group(1) else '', m.group(2)))
            if part2:
                out.append('\\begin{figure}[p]\n\\ContinuedFloat\n\\caption[]{\\textit{(continued)} %s}\n\\end{figure}\n' % part2)
            pending_img = None
            continue
        if p['align'] == 'CENTER' and section != '':
            if p['plain'] in EQUATIONS:
                out.append('\\begin{equation*}\n%s\n\\end{equation*}\n' % EQUATIONS[p['plain']])
            else:
                warnings.append('equation not in EQUATIONS (kept as text): ' + p['plain'])
                out.append('\\begin{center}%s\\end{center}\n' % p['tex'])
            continue
        out.append(figrefs(cite(p['tex'], idx, used)) + '\n')
    if pending_img:
        warnings.append('image without a caption: ' + pending_img)
    # the doc's reference list must match the cited bib entries
    ref_keys = set()
    for r in doc_refs:
        mm = re.match(r"([^,]+),.*?\((\d{4})\)", r)
        if not mm:
            continue
        n_auth = r.split('(')[0].count(',') // 2 + 1
        cand = [k for (a, y, n, s), k in idx.items() if a == norm(mm.group(1)) and y == mm.group(2)]
        if not cand:
            raise KeyError('reference in the doc without a bib entry: ' + r[:80])
        ref_keys.update(cand if len(cand) == 1 else [k for (a, y, n, s), k in idx.items()
                                                      if a == norm(mm.group(1)) and y == mm.group(2) and n == min(n_auth, 3)] or cand)
    if used - ref_keys:
        warnings.append('cited but missing from the doc reference list: ' + ', '.join(sorted(used - ref_keys)))
    if ref_keys - used:
        warnings.append('in the doc reference list but never cited: ' + ', '.join(sorted(ref_keys - used)))
    out.append('\\bibliographystyle{unsrtnat}\n\\bibliography{refs}\n\\end{document}\n')
    tex = '\n'.join(out)
    tex = tex.replace('\\section{Supplementary Figures}', '\\section{Supplementary Figures}')
    k = tex.index('\\bibliographystyle')
    j = tex.find('\\clearpage\n\\section{Supplementary Figures}')
    if j > 0:   # references before the supplementary figures, as in the doc
        tex = tex[:j] + tex[k:tex.index('\\end{document}')] + tex[j:k] + '\\end{document}\n'
    open(os.path.join(out_dir, name + '.tex'), 'w').write(tex)
    return nfig, len(used), len(doc_refs)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('doc'); ap.add_argument('out_dir', nargs='?', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--exclude', action='append', default=[], help='leave out a section by its title, e.g. "AI methodology"')
    ap.add_argument('--name', default='main', help='output name (NAME.tex)')
    a = ap.parse_args()
    d = json.load(open(a.doc))
    d = d.get('content', d)
    out_dir = a.out_dir
    nfig, ncit, nref = convert(d, out_dir, os.path.join(out_dir, 'refs.bib'), set(a.exclude), a.name)
    for w in warnings:
        print('WARNING:', w)
    if unknown:
        print('WARNING: unmapped characters:', ''.join(sorted(unknown)))
    print(f'figures {nfig} | cited bib entries {ncit} | doc references {nref} | revision {d.get("revisionId", "")[:20]}')
