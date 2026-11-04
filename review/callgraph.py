"""Static call graphs of the code (no execution), drawn with Graphviz.

1. The whole code base: every function / method of nmdaint/, scripts/ and the draw() entry points of plots/, and
   which of them each calls (resolved through the `from . import x` / `from nmdaint import x` aliases, local names,
   `self.` methods and class instantiation; nested functions and lambdas count as their parent).
2. One graph per figure: the calls a figure notebook makes, in execution order (numbered), each expanded into the
   nmdaint functions it reaches, plus the results/ files the notebook loads and the scripts/ functions that write them.

usage: python review/callgraph.py      writes review/out/callgraph/*.dot, *.svg, *.png, calls.json
"""
import ast
import glob
import json
import os
import re
import subprocess
from collections import defaultdict, OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'review', 'out', 'callgraph')
PKGS = ('nmdaint', 'plots', 'scripts')
COLORS = dict(onoff='#fde0dd', decoder='#e5f5e0', recordings='#deebf7', behaviour='#fee6ce', ratemodel='#efedf5',
              readout='#f2f0f7', fit='#fff7bc', memory='#fdd0a2', spiking='#c7e9c0', spiking_readout='#d9f0d3',
              population='#d4b9da', synthetic='#c6dbef', paths='#f0f0f0')


def module_name(path):
    rel = os.path.relpath(path, ROOT)[:-3]
    return rel.replace(os.sep, '.')


def import_aliases(tree, mod):
    """{local name: module} for the package modules imported in a module."""
    al = {}
    pkg = mod.split('.')[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = node.module or ''
            if node.level == 1:
                base = pkg + ('.' + base if base else '')
            if base.split('.')[0] in PKGS:
                for a in node.names:
                    full = f'{base}.{a.name}' if base in PKGS else base
                    al[a.asname or a.name] = (full, None if base in PKGS else a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split('.')[0] in PKGS:
                    al[a.asname or a.name] = (a.name, None)
    return al


class Collector(ast.NodeVisitor):
    """Definitions (qualified names) and raw calls of one module."""

    def __init__(self, mod):
        self.mod, self.stack, self.defs, self.calls, self.classes = mod, [], [], defaultdict(list), {}
        self.inst = defaultdict(dict)        # owner -> {variable: (module alias or None, class name)}

    def visit_Assign(self, node):
        v = node.value
        if isinstance(v, ast.Call) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            f = v.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.attr[:1].isupper():
                self.inst[self.owner()][node.targets[0].id] = (f.value.id, f.attr)
            elif isinstance(f, ast.Name) and f.id[:1].isupper():
                self.inst[self.owner()][node.targets[0].id] = (None, f.id)
        self.generic_visit(node)

    def owner(self):
        # nested functions belong to the outermost function (or Class.method)
        if not self.stack:
            return f'{self.mod}.__main__' if self.mod.startswith('scripts.') else f'{self.mod}.<module>'
        if self.stack[0][0] == 'class':
            return f'{self.mod}.{self.stack[0][1]}' + (f'.{self.stack[1][1]}' if len(self.stack) > 1 else '')
        return f'{self.mod}.{self.stack[0][1]}'

    def visit_ClassDef(self, node):
        self.classes[node.name] = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        self.stack.append(('class', node.name))
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        top = len(self.stack) == 0 or (len(self.stack) == 1 and self.stack[0][0] == 'class')
        self.stack.append(('def', node.name))
        if top:
            self.defs.append(self.owner())
        self.generic_visit(node)
        self.stack.pop()

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and isinstance(node.ctx, ast.Load):
            self.calls[self.owner()].append(('attr', node.value.id, node.attr))
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):            # a local function passed by name, e.g. delayed(run_session)
            self.calls[self.owner()].append(('name', node.id))

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Name):
            self.calls[self.owner()].append(('name', f.id))
        elif isinstance(f, ast.Attribute):
            v = f.value
            if isinstance(v, ast.Name):
                self.calls[self.owner()].append(('attr', v.id, f.attr))
        self.generic_visit(node)


def parse_package():
    mods = {}
    for pkg in PKGS:
        for path in sorted(glob.glob(os.path.join(ROOT, pkg, '*.py'))):
            if path.endswith('__init__.py'):
                continue
            mod = module_name(path)
            tree = ast.parse(open(path).read())
            c = Collector(mod)
            c.visit(tree)
            mods[mod] = dict(c=c, alias=import_aliases(tree, mod))
    return mods


def resolve(mods, mod, call, owner):
    """Qualified name of the package function a raw call refers to, or None."""
    c, al = mods[mod]['c'], mods[mod]['alias']
    defs = set(c.defs)
    if call[0] == 'name':
        n = call[1]
        if f'{mod}.{n}' in defs:
            return f'{mod}.{n}'
        if n in c.classes:
            return f'{mod}.{n}.__init__' if '__init__' in c.classes[n] else None
        if n in al and al[n][1]:                                  # from .x import f
            tgt = f'{al[n][0]}.{al[n][1]}'
            return tgt
        return None
    _, base, attr = call
    if base in c.inst.get(owner, {}):                          # method of an instance built in this function
        amod, cls = c.inst[owner][base]
        tmod = mod if amod is None else al.get(amod, (None,))[0]
        if tmod in mods and attr in mods[tmod]['c'].classes.get(cls, []):
            return f'{tmod}.{cls}.{attr}'
        return None
    owners = [k for k, ms in c.classes.items() if attr in ms and not attr.startswith('__')]
    if base not in al and base != 'self' and len(owners) == 1:
        return f'{mod}.{owners[0]}.{attr}'
    if base == 'self':
        cls = owner.split('.')[2] if owner.count('.') >= 3 else None
        return f'{mod}.{cls}.{attr}' if cls and attr in c.classes.get(cls, []) else None
    if base in al and al[base][1] is None:
        tmod = al[base][0]
        if tmod in mods:
            tc = mods[tmod]['c']
            if f'{tmod}.{attr}' in tc.defs:
                return f'{tmod}.{attr}'
            if attr in tc.classes:
                return f'{tmod}.{attr}.__init__'
    return None


def build_graph(mods):
    edges = defaultdict(set)
    nodes = set()
    for mod, d in mods.items():
        c = d['c']
        nodes |= {x for x in c.defs if not (mod.startswith('plots') and not x.endswith('.draw'))}
        if f'{mod}.__main__' in c.calls:
            nodes.add(f'{mod}.__main__')
        for owner, calls in c.calls.items():
            if owner.endswith('<module>'):
                continue
            for call in calls:
                tgt = resolve(mods, mod, call, owner)
                if tgt and tgt != owner:
                    edges[owner].add(tgt)
    # keep plots internals out; everything else in
    keep = lambda n: not n.startswith('plots.') or n.endswith('.draw')
    edges = {a: {b for b in bs if keep(b)} for a, bs in edges.items() if keep(a)}
    return sorted(nodes), edges


def short(q):
    return q.replace('nmdaint.', '').replace('.__init__', '()')


def node_attrs(q):
    m = q.split('.')
    pkg, mod = m[0], m[1]
    fill = COLORS.get(mod, '#ffffff') if pkg == 'nmdaint' else ('#ffffff' if pkg == 'scripts' else '#eeeeee')
    shape = 'box' if pkg == 'nmdaint' else ('component' if pkg == 'scripts' else 'note')
    return f'shape={shape}, style="filled,rounded", fillcolor="{fill}"'


def write_dot(name, body, engine='dot'):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    open(p + '.dot', 'w').write(body)
    for fmt in ('svg', 'png'):
        subprocess.run([engine, f'-T{fmt}', p + '.dot', '-o', f'{p}.{fmt}'], check=True)


def full_graph(nodes, edges):
    used = set(edges) | {b for bs in edges.values() for b in bs}
    lines = ['digraph G {', 'rankdir=LR; newrank=true; nodesep=0.12; ranksep=0.9; fontname="Helvetica";',
             'node [fontname="Helvetica", fontsize=10, height=0.25]; edge [color="#00000066", arrowsize=0.5];']
    bymod = defaultdict(list)
    for n in nodes:
        if n in used or (n.startswith('nmdaint.') and n.split('.')[1] != 'paths') or n.startswith('scripts.') and not n.startswith('plots.'):
            bymod['.'.join(n.split('.')[:2])].append(n)
    for mod, ns in sorted(bymod.items()):
        lines.append(f'subgraph "cluster_{mod}" {{ label="{mod}.py"; fontsize=12; style=rounded; color="#999999";')
        for n in ns:
            lab = n[len(mod) + 1:]
            lines.append(f'"{n}" [label="{lab}", {node_attrs(n)}];')
        lines.append('}')
    for a, bs in sorted(edges.items()):
        for b in sorted(bs):
            lines.append(f'"{a}" -> "{b}";')
    lines.append('}')
    write_dot('callgraph_all', '\n'.join(lines))


PARTS = {
    'model': ('nmdaint.ratemodel', 'nmdaint.readout', 'nmdaint.onoff', 'nmdaint.fit', 'nmdaint.memory',
              'nmdaint.behaviour', 'nmdaint.synthetic', 'scripts.tau_sweep', 'scripts.fit', 'plots'),
    'data': ('nmdaint.recordings', 'nmdaint.decoder', 'nmdaint.population', 'nmdaint.spiking',
             'nmdaint.spiking_readout', 'scripts.recordings', 'scripts.spiking', 'nmdaint.paths'),
}


def part_graph(name, nodes, edges):
    """The functions of some modules with all their calls; callees in other modules drawn as grey ghosts."""
    mods = PARTS[name]
    inside = lambda n: n.startswith(mods)
    used = {a for a, bs in edges.items() if bs} | {b for bs in edges.values() for b in bs}
    L = ['digraph P {', 'rankdir=LR; newrank=true; nodesep=0.12; ranksep=0.7; fontname="Helvetica";',
         'node [fontname="Helvetica", fontsize=10, height=0.25]; edge [color="#00000066", arrowsize=0.5];']
    bymod = defaultdict(list)
    for n in nodes:
        if inside(n) and (n in used or n.startswith(('nmdaint.', 'scripts.'))) and not n.startswith('nmdaint.paths') \
                or (n.startswith(('nmdaint.paths', 'plots.')) and inside(n) and n in used):
            bymod['.'.join(n.split('.')[:2])].append(n)
    for mod, ns in sorted(bymod.items()):
        L.append(f'subgraph "cluster_{mod}" {{ label="{mod}.py"; fontsize=12; style=rounded; color="#999999";')
        L += [f'"{n}" [label="{n[len(mod) + 1:]}", {node_attrs(n)}];' for n in ns]
        L.append('}')
    shown = {n for ns in bymod.values() for n in ns}
    ghosts = set()
    for a, bs in sorted(edges.items()):
        if a not in shown:
            continue
        for b in sorted(bs):
            if b not in shown:
                ghosts.add(b)
            L.append(f'"{a}" -> "{b}";')
    for g in sorted(ghosts):
        L.append(f'"{g}" [label="{short(g)}", shape=box, style="rounded,dashed", color="#999999", fontcolor="#777777"];')
    L.append('}')
    write_dot(f'callgraph_{name}', '\n'.join(L))


def module_graph(edges):
    """Module-level summary: which module calls which (edge label = number of distinct function pairs)."""
    cnt = defaultdict(int)
    for a, bs in edges.items():
        for b in bs:
            ma, mb = '.'.join(a.split('.')[:2]), '.'.join(b.split('.')[:2])
            if ma != mb:
                cnt[ma, mb] += 1
    ms = sorted({m for k in cnt for m in k})
    lines = ['digraph M {', 'rankdir=TB; nodesep=0.3; ranksep=0.5; fontname="Helvetica";',
             'node [fontname="Helvetica", fontsize=11]; edge [fontsize=9, color="#00000088"];']
    for m in ms:
        lines.append(f'"{m}" [label="{m.replace("nmdaint.", "")}", {node_attrs(m + ".x")}];')
    for (a, b), k in sorted(cnt.items()):
        lines.append(f'"{a}" -> "{b}" [label="{k}", penwidth={0.6 + 0.35 * k:.2f}];')
    lines.append('}')
    write_dot('callgraph_modules', '\n'.join(lines))


# ------------------------------------------------------------------------------------------------ figures
def results_key(args):
    return '/'.join(ast.literal_eval(x) if isinstance(x, ast.Constant) else ast.unparse(x) for x in args)


def notebook_calls(nb_path, mods):
    """Ordered (qualified function, cell index, optional) of the package calls a notebook makes (execution order:
    a call's arguments run before the call), and the results/ files it loads. Cells under a markdown heading with
    'Play' or 'optional' are optional explorations, not part of the figure."""
    nb = json.load(open(nb_path))
    src_all, seq, files, optional = '', [], [], False
    for k, cell in enumerate(nb['cells']):
        src = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            head = src.strip().splitlines()[0] if src.strip() else ''
            if head.startswith('#'):
                optional = bool(re.search(r'play|optional', head, re.I))
            continue
        src_all += src + '\n'
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        al = import_aliases(ast.parse(src_all), 'nmdaint.nb')
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and not any(
                    isinstance(p, ast.Call) and p.func is node for p in ast.walk(tree)):
                b, a = node.value.id, node.attr              # a function passed by reference, e.g. delayed(f)
                if b in al and al[b][1] is None and al[b][0] in mods and f'{al[b][0]}.{a}' in mods[al[b][0]]['c'].defs \
                        and not al[b][0].endswith(('paths', 'style')):
                    calls.append((node.end_lineno, node.end_col_offset, f'{al[b][0]}.{a}'))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                b, a = node.func.value.id, node.func.attr
                if b == 'paths' and a == 'results':
                    files.append((results_key(node.args), optional))
                    continue
                if b in al and al[b][1] is None and al[b][0] in mods and not al[b][0].endswith(('paths', 'style')):
                    tc = mods[al[b][0]]['c']
                    q = f'{al[b][0]}.{a}' if f'{al[b][0]}.{a}' in tc.defs else (
                        f'{al[b][0]}.{a}.__init__' if a in tc.classes else None)
                    if q:
                        calls.append((node.end_lineno, node.end_col_offset, q))
        seq += [(q, k, optional) for _, _, q in sorted(calls)]
    seen = OrderedDict()
    for q, k, opt in seq:                         # first call of each function; main use wins over optional
        if q not in seen or (seen[q][1] and not opt):
            seen[q] = (k, opt)
    steps = sorted(((q, k, o) for q, (k, o) in seen.items()), key=lambda x: (x[2], x[1]))
    fl = OrderedDict()
    for f, o in files:
        fl[f] = fl.get(f, True) and o
    return steps, fl


def library_reads():
    """{nmdaint function: [results files it reads]}"""
    out = defaultdict(list)
    for path in glob.glob(os.path.join(ROOT, 'nmdaint', '*.py')):
        mod = module_name(path)
        for fn in [n for n in ast.parse(open(path).read()).body if isinstance(n, ast.FunctionDef)]:
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'results':
                    out[f'{mod}.{fn.name}'].append(results_key(node.args))
    return out


def script_writers(mods):
    """{results file: script function writing it} from paths.results(...) calls inside scripts/."""
    out = {}
    for path in glob.glob(os.path.join(ROOT, 'scripts', '*.py')):
        mod = module_name(path)
        tree = ast.parse(open(path).read())
        for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'results':
                    out.setdefault(results_key(node.args), f'{mod}.{fn.name}')
    return out


def reach(edges, roots):
    seen, todo = set(), list(roots)
    while todo:
        q = todo.pop()
        for b in edges.get(q, ()):
            if b not in seen:
                seen.add(b)
                todo.append(b)
    return seen


def figure_graph(fig, steps, files, writers, edges, reads):
    L = ['digraph F {', 'rankdir=LR; newrank=true; nodesep=0.1; ranksep=0.45; fontname="Helvetica";',
         'node [fontname="Helvetica", fontsize=10, height=0.25]; edge [color="#00000055", arrowsize=0.5];',
         f'"nb" [label="figure_notebooks/{fig}.ipynb", shape=folder, style=filled, fillcolor="#ffffcc", fontsize=12];']
    drawn, seen_e = set(), set()

    def edge(a, b, attr=''):
        if (a, b) not in seen_e:
            seen_e.add((a, b))
            q_ = lambda x, side: ('"{}":{}:'.format(*x.split(':', 1)) + side) if x.startswith(('steps:', 'opts:')) else f'"{x}"'
            if a.split(':')[0] in ('steps', 'opts') and b.split(':')[0] in ('steps', 'opts'):
                return                                     # a step calling another step: shown by the order
            L.append(f'{q_(a, "e")} -> {q_(b, "w")}' + (f' [{attr}]' if attr else '') + ';')

    def node(n, extra=''):
        if n not in drawn:
            drawn.add(n)
            L.append(f'"{n}" [label="{short(n)}", {node_attrs(n)}{extra}];')

    def expand(roots, src_of):
        """edges from each root through everything it reaches (package functions, not paths/plots internals)"""
        ok = lambda q: not q.startswith(('nmdaint.paths', 'plots.'))
        for r in roots:
            for n in {r} | reach(edges, [r]):
                if not ok(n):
                    continue
                for b in edges.get(n, ()):
                    if ok(b):
                        if b not in src_of:
                            node(b)
                        edge(src_of.get(n, n), src_of.get(b, b))

    # files read inside library functions reached by the figure
    hidden = OrderedDict()
    for q, _, opt in steps:
        for n in {q} | reach(edges, [q]):
            for f in reads.get(n, ()):
                hidden.setdefault(f, []).append(n)
                files[f] = files.get(f, True) and opt
    late = []
    # heavy steps: scripts that wrote the loaded results, with what they call
    if files:
        L.append('subgraph cluster_res { label="precomputed: scripts/ write results/, the notebook loads them"; '
                 'style="dashed,rounded"; color="#888888"; fontsize=10;')
        ws = []
        for f, opt in files.items():
            L.append(f'"res:{f}" [label="results/{f}", shape=cylinder, style=filled, fillcolor="#f7f7f7", fontsize=9];')
            w = writers.get(f)
            if w and w not in ws:
                ws.append(w)
                L.append(f'"{w}" [label="{w}", {node_attrs(w)}];')
            if w:
                L.append(f'"{w}" -> "res:{f}" [style=dashed, color="#888888"];')
        L.append('}')
        for f, opt in files.items():
            if f in hidden:
                late += [(f, n) for n in dict.fromkeys(hidden[f])]
            else:
                L.append(f'"res:{f}" -> "nb" [style=dashed, color="#888888"];')
    # the figure's steps, in execution order, as one table (one row = one port), so the order is fixed;
    # optional explorations ('Play' sections) in a second, grey table
    src_of = {}
    for tab, rows, head, col in (('steps', [x for x in steps if not x[2]], f'{fig}: calls in order', '#fde0dd'),
                                 ('opts', [x for x in steps if x[2]], 'optional (Play / exploration cells)', '#f2f2f2')):
        if not rows:
            continue
        cells = [f'<tr><td bgcolor="{col}"><b>{head}</b></td></tr>']
        for i, (q, cell, opt) in enumerate(rows, 1):
            mod = q.split('.')[1]
            bg = COLORS.get(mod, '#ffffff') if q.startswith('nmdaint') else '#eeeeee'
            tag = f'P{i}' if opt else f'{i}'
            cells.append(f'<tr><td port="p{i}" align="left" bgcolor="{bg}">{tag}. {short(q)}  <font point-size="8" color="#666666">cell {cell}</font></td></tr>')
            src_of[q] = f'{tab}:p{i}'
        L.append(f'"{tab}" [shape=plaintext, label=<<table border="1" cellborder="1" cellspacing="0" cellpadding="3">'
                 + ''.join(cells) + '</table>>];')
        L.append(f'"nb" -> "{tab}" [color="#d62728", penwidth=1.5' + (', style=dotted' if tab == 'opts' else '') + '];')
    expand([q for q, _, _ in steps], src_of)
    for f, n in late:
        if n not in src_of:
            node(n)
        edge(f'res:{f}', src_of.get(n, n), 'style=dashed, color="#888888"')
    if files:
        ws = list(dict.fromkeys(w for w in (writers.get(f) for f in files) if w))
        expand(ws, src_of)
    L.append('}')
    write_dot(f'fig_{fig}', '\n'.join(L))


def main():
    mods = parse_package()
    nodes, edges = build_graph(mods)
    full_graph(nodes, edges)
    for part in PARTS:
        part_graph(part, nodes, edges)
    module_graph(edges)
    writers = script_writers(mods)
    reads = library_reads()
    figs = {}
    for nb in sorted(glob.glob(os.path.join(ROOT, 'figure_notebooks', '*.ipynb'))):
        fig = os.path.basename(nb)[:-6]
        steps, files = notebook_calls(nb, mods)
        figure_graph(fig, steps, files, writers, edges, reads)
        figs[fig] = dict(steps=[dict(call=short(q), cell=c, optional=o) for q, c, o in steps],
                         files={f: dict(optional=o, written_by=writers.get(f)) for f, o in files.items()})
    json.dump(dict(edges={short(a): sorted(short(b) for b in bs) for a, bs in sorted(edges.items())},
                   nodes=[short(n) for n in nodes], figures=figs),
              open(os.path.join(OUT, 'calls.json'), 'w'), indent=1)
    print(f'{len(nodes)} functions, {sum(len(b) for b in edges.values())} call edges; figures: {", ".join(figs)}')
    for f, d in figs.items():
        print(f'\n{f}:', ' -> '.join(x['call'] + ('*' if x['optional'] else '') for x in d['steps']))
        print('   loads:', d['files'])


if __name__ == '__main__':
    main()
