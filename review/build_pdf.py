"""Build review/REVIEW.pdf: REVIEW.md, followed by an appendix with the static call graphs (review/callgraph.py).

usage: python review/callgraph.py && python review/build_pdf.py      (needs pandoc and the pre-installed Chromium)
"""
import html
import json
import os
import subprocess

import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CG = os.path.join(HERE, 'out', 'callgraph')
TITLES = dict(fig1='Figure 1', fig2='Figure 2', fig3='Figure 3', fig4='Figure 4', figS1='Supplementary Figure 1',
              figS2='Supplementary Figure 2', figS3='Supplementary Figure 3')

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
@page wide { size: A4 landscape; margin: 10mm 10mm 10mm 10mm; }
body { font-family: 'DejaVu Sans', Helvetica, Arial, sans-serif; font-size: 9.6pt; line-height: 1.38; color: #1a1a1a; }
h1 { font-size: 17pt; margin: 0 0 6pt 0; line-height: 1.2; }
h2 { font-size: 13pt; margin: 16pt 0 5pt 0; border-bottom: 1px solid #bbb; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt 0; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
ul, ol { padding-left: 18pt; margin: 4pt 0; }
li { margin: 2pt 0; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.4pt; background: #f3f3f3; padding: 0 2px; border-radius: 2px; }
table { border-collapse: collapse; margin: 6pt 0 8pt 0; font-size: 8.2pt; break-inside: avoid; }
th, td { border: 1px solid #c8c8c8; padding: 2pt 4pt; vertical-align: top; }
th { background: #eef1f5; }
hr { border: 0; border-top: 1px solid #ddd; margin: 12pt 0; }
img { max-width: 100%; }
.meta { color: #555; font-size: 8.5pt; }
.appendix-title { break-before: page; }
.wide { page: wide; break-before: page; }
.wide h3 { margin-top: 0; }
.graph { width: 100%; text-align: center; }
.graph img { max-width: 100%; max-height: 108mm; }
.tabs { display: flex; gap: 10pt; align-items: flex-start; }
.tabs table { margin: 2pt 0; }
.tall img { width: 100%; height: 232mm; max-height: none; object-fit: contain; }
.steps { font-size: 7pt; width: 100%; }
.steps td { padding: 1pt 3pt; }
.opt { color: #777; }
.legend { font-size: 8pt; color: #333; }
"""


def md_to_html(path):
    out = subprocess.run(['pandoc', '-f', 'gfm', '-t', 'html5', path], capture_output=True, text=True, check=True)
    return out.stdout


def appendix():
    d = json.load(open(os.path.join(CG, 'calls.json')))
    n_fn, n_e = len(d['nodes']), sum(len(v) for v in d['edges'].values())
    parts = [f"""
<h2 class="appendix-title">Appendix A. Call graphs of the code</h2>
<p>Built by static analysis of the source (<code>review/callgraph.py</code>, Python <code>ast</code>; nothing is run):
{n_fn} functions and methods of <code>nmdaint/</code>, <code>scripts/</code> and the <code>draw()</code> entry points of
<code>plots/</code>, with {n_e} call edges. A call is resolved through the package's import aliases
(<code>from . import onoff</code>, <code>from nmdaint import readout</code>), local names, <code>self.</code> methods,
instances built in the same function (<code>S = recordings.Session(...)</code>), a method name that is unique in its
module (<code>net.cue_current</code>), class instantiation (<code>Class()</code>), and functions passed by reference
(<code>joblib.delayed(f)</code>). Nested functions and lambdas count as their enclosing function, and the
<code>if __name__ == '__main__'</code> block of a script is shown as <code>__main__</code>. Calls to
<code>numpy</code>, <code>scipy</code>, etc. are left out. Limitations: calls through variables holding functions
(other than <code>delayed</code>), <code>getattr</code> and dictionary dispatch are not seen; the plotting internals
are omitted.</p>
<p class="legend">Colours: one per <code>nmdaint</code> module; white component boxes are <code>scripts/</code>;
folded-corner boxes are <code>plots/</code>. An arrow A &rarr; B means A calls B.</p>

<h3>A.1 Module dependencies</h3>
<p>Edge labels give the number of distinct (caller, callee) function pairs between two modules.</p>
<div class="graph"><img src="out/callgraph/callgraph_modules.svg"></div>
<p>Three shared cores: <code>onoff</code> (state labelling and statistics, used by the recordings, both models'
read-outs and the spiking read-out), <code>readout</code> (the rate-model read-out, used by the fit, the memory tests,
the synthetic validation and the tau sweep) and <code>ratemodel</code>. The recordings pipeline
(<code>recordings</code>, <code>decoder</code>, <code>population</code>) is reached only from
<code>scripts/recordings.py</code> and from <code>synthetic</code>, whose validation runs the recordings' decoder on
model-driven spikes.</p>

<div style="break-before: page">
<h3>A.2 Functions: rate model, read-out, fit, behaviour (and the scripts that use them)</h3>
<p class="legend">Dashed grey boxes: functions of other modules (A.3). <code>plots/*.draw</code> take data and call no
package function, except <code>plots.fig3.draw</code> (Wilson intervals); <code>behaviour.held_curves</code> is called
only by the Fig. 3 notebook.</p>
<div class="graph tall"><img src="out/callgraph/callgraph_model.svg"></div>
</div>
<div style="break-before: page">
<h3>A.3 Functions: recordings, decoders, population geometry, spiking network (and their scripts)</h3>
<div class="graph tall"><img src="out/callgraph/callgraph_data.svg"></div>
<p class="legend">The single graph of all functions is <code>review/out/callgraph/callgraph_all.svg</code>; the edges
as JSON: <code>review/out/callgraph/calls.json</code>.</p>
</div>

<h2 class="appendix-title">Appendix B. What each figure runs, in order</h2>
<p>For each figure notebook (<code>figure_notebooks/</code>): the package functions the notebook calls, numbered in
execution order (first call of each function; within a statement, arguments before the call), each expanded into
everything it calls. Cells under a "Play" or "optional" heading are exploration, not part of the figure; they are
listed separately in grey. Dashed: the <code>results/</code> files the figure loads (directly or inside a library
function such as <code>readout.load_data</code>) and the <code>scripts/</code> functions that wrote them, with what
those scripts call. These are the heavy, precomputed steps behind the figure.</p>
"""]
    for fig, info in d['figures'].items():
        rows = []
        main = [s for s in info['steps'] if not s['optional']]
        opt = [s for s in info['steps'] if s['optional']]
        for i, s in enumerate(main, 1):
            rows.append(f"<tr><td>{i}</td><td><code>{html.escape(s['call'])}</code></td><td>{s['cell']}</td></tr>")
        for i, s in enumerate(opt, 1):
            rows.append(f"<tr class='opt'><td>P{i}</td><td><code>{html.escape(s['call'])}</code></td><td>{s['cell']}</td></tr>")
        files = ''.join(f"<tr><td><code>results/{html.escape(f)}</code></td><td><code>{html.escape(str(v['written_by']))}</code></td>"
                        f"<td>{'optional' if v['optional'] else 'figure'}</td></tr>" for f, v in info['files'].items())
        parts.append(f"""
<div class="wide">
<h3>{TITLES.get(fig, fig)} (<code>figure_notebooks/{fig}.ipynb</code>)</h3>
<div class="graph"><img src="out/callgraph/fig_{fig}.svg"></div>
<div class="tabs"><table class="steps" style="width:auto">
<tr><th>#</th><th>function called by the notebook</th><th>cell</th></tr>{''.join(rows)}</table>
{'<table class="steps" style="width:auto"><tr><th>loaded file</th><th>written by</th><th>used by</th></tr>' + files + '</table>' if files else ''}</div>
</div>""")
    return '\n'.join(parts)


def main():
    body = md_to_html(os.path.join(HERE, 'REVIEW.md'))
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Review: NMDA intermittent coding</title>
<style>{CSS}</style></head><body>{body}{appendix()}</body></html>"""
    src = os.path.join(HERE, 'REVIEW.html')
    open(src, 'w').write(doc)
    pdf = os.path.join(HERE, 'REVIEW.pdf')
    js = """const {chromium} = require('playwright');
(async () => {
  const b = await chromium.launch();
  const pg = await b.newPage();
  await pg.goto('file://' + process.argv[2], {waitUntil: 'networkidle'});
  await pg.pdf({path: process.argv[3], preferCSSPageSize: true, printBackground: true, displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:7pt;color:#777;width:100%;text-align:center;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>'});
  await b.close();
})();"""
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
        f.write(js)
    subprocess.run(['node', f.name, src, pdf], check=True)
    os.remove(f.name)
    os.remove(src)
    print('wrote', os.path.join(HERE, 'REVIEW.pdf'))


if __name__ == '__main__':
    main()
