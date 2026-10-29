"""Figure style shared by all figures (Nature-like: Arial-metric font, thin axes with outward ticks)."""
import matplotlib.pyplot as plt
import numpy as np

C_ON, C_OFF, INK, GREY, LIGHT, C_UN = '#e8474a', '#3cb4c9', '#000000', '#7f7f7f', '#d9d9d9', '#8c8c8c'
C_STP, C_FIT, C_NET = '#2b83ba', '#d7301f', '#2b7bba'
RC = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
      'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
      'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
      'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
      'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
      'axes.titlesize': 8, 'axes.titleweight': 'normal', 'lines.linewidth': 0.9,
      'legend.frameon': False, 'legend.fontsize': 7.5, 'mathtext.fontset': 'custom',
      'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic', 'pdf.fonttype': 42, 'svg.fonttype': 'none'}
plt.rcParams.update(RC)


def letter(ax, txt, x=-0.08, y=1.06):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')


def runs(mask):
    a = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(a == 1)[0], np.where(a == -1)[0]))


def shade(ax, tz, state, step=10):
    """On (red) and Off (cyan) states as shaded time spans."""
    for lab, c in ((1, C_ON), (-1, C_OFF)):
        for a, b in runs(state == lab):
            ax.axvspan(tz[a] - step / 2, tz[b - 1] + step / 2, color=c, alpha=0.15, lw=0)


def plain_log(ax, bottom):
    """Log y axis from `bottom`, with plain tick labels at 1, 3, 10, 30, ..."""
    ax.set_ylim(bottom=bottom)
    top = ax.get_ylim()[1]
    ticks = [t for t in (1, 3, 10, 30, 100, 300, 1000, 3000, 10000) if bottom <= t <= top]
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())


def save(fig, path, dpi=600):
    """PNG and SVG (editable text)."""
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.rsplit('.', 1)[0] + '.svg')


def limits(ax, axis, lo, hi, values, ticks=None, labels=None):
    """Set the paper's axis limits (lo, hi) and ticks; where the data fall outside them, widen the limits (5% margin)
    and let matplotlib choose the ticks, so that changed parameters never put data off the axes."""
    v = np.asarray(values, float).ravel()
    v = v[np.isfinite(v)]
    out = v.size > 0 and (v.min() < lo or v.max() > hi)
    if out:
        pad = 0.05 * (max(hi, v.max()) - min(lo, v.min()))
        lo, hi = (v.min() - pad if v.min() < lo else lo), (v.max() + pad if v.max() > hi else hi)
    (ax.set_xlim if axis == 'x' else ax.set_ylim)(lo, hi)
    if ticks is not None and not out:
        (ax.set_xticks if axis == 'x' else ax.set_yticks)(ticks, labels)
    return lo, hi
