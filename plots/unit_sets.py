"""On/Off statistics of the seven decoded populations, data vs models (layout of Figs. 2, S2 and S3)."""
import numpy as np

from nmdaint import readout
from .style import GREY, INK, LIGHT, letter, limits, plain_log

SETS = list(readout.SETS)
FIT = SETS.index('all')
PANELS = (('off_per_trial', 'n_off', 'Off states per trial', 'off'), ('off_median_ms', 'off_median', 'Off length, median (ms)', 'off'),
          ('frac_off', 'frac_off', 'Fraction of delay Off', 'off'), ('on_per_trial', 'n_on', 'On states per trial', 'on'),
          ('on_median_ms', 'on_median', 'On length, median (ms)', 'on'), ('frac_on', 'frac_on', 'Fraction of delay On', 'on'))


def panels(axs, ses, models, tops, span=(0.45, 0.45), fitted_dx=0.0, letter_y=1.02, mew=0):
    """axs: 6 axes; models: list of (stats table, marker, x offset, colour or {'on': c, 'off': c}); mew: marker edge
    width of the models (None: matplotlib's default)."""
    x = np.arange(len(SETS))
    jit = np.random.default_rng(0).uniform(-0.18, 0.18, 25)
    for k, (a, (dk, mk, name, state)) in enumerate(zip(axs, PANELS)):
        a.axvspan(FIT - span[0], FIT + span[1], color=LIGHT, lw=0, zorder=0)
        for i, lab in enumerate(SETS):
            v = ses[lab][dk]
            a.plot(i + jit[:len(v)], v, '.', ms=2.5, color=GREY, alpha=0.5, mew=0, zorder=1)
        v = [ses[lab][dk] for lab in SETS]
        a.errorbar(x, [s.mean() for s in v], yerr=[2 * s.std(ddof=1) / np.sqrt(len(s)) for s in v], fmt='o', ms=3,
                   color=INK, elinewidth=0.6, capsize=0, zorder=3)
        for st, marker, dx, col in models:
            a.plot(x + dx, st.loc[SETS, mk].values, marker, ms=3, color=col[state] if isinstance(col, dict) else col, mew=mew, zorder=2)
        a.axvline(2.5, color=GREY, ls=(0, (3, 2)), lw=0.5); a.axvline(5.5, color=GREY, ls=(0, (3, 2)), lw=0.5)
        a.set_xlim(-0.6, len(SETS) - 0.4)
        a.set_xticks(x); a.set_xticklabels(SETS, rotation=60, ha='right', rotation_mode='anchor', fontsize=7)
        if tops[k]:                       # the paper's limit, widened if the data or a model exceed it
            values = np.r_[np.concatenate(v), np.concatenate([st.loc[SETS, mk].values for st, *_ in models])]
            limits(a, 'y', 0, tops[k], values, *(([0, 0.5, 1], ['0', '', '1']) if tops[k] == 1 else ()))
        else:
            a.set_ylim(0, np.ceil(a.get_ylim()[1]))
        a.set_ylabel(name)
        letter(a, 'abcdef'[k], x=-0.5, y=letter_y)
        a.text(FIT + fitted_dx, 0.97, 'Fitted', transform=a.get_xaxis_transform(), ha='center', va='top', rotation=90, fontsize=7)
        a.text(2.5, 1.1, 'Predicted', transform=a.get_xaxis_transform(), ha='center', va='bottom', fontsize=7)
        a.plot([0, 0, 5, 5], [1.06, 1.09, 1.09, 1.06], color=INK, lw=0.5, transform=a.get_xaxis_transform(), clip_on=False)


def text_legend(a, items, x, y0, va, dy=0.09, fontsize=7):
    """Coloured words stacked inside the axes, from y0 upwards (va 'bottom') or downwards (va 'top')."""
    for n, (txt, c) in enumerate(items):
        y = y0 + (len(items) - 1 - n) * dy if va == 'bottom' else y0 - n * dy
        a.text(x, y, txt, color=c, transform=a.transAxes, va=va, fontsize=fontsize)


def loss_panels(fig, gs, conds, ylim_g, xt_h, fontsize_h):
    """g: fit loss per decoded population; h: summed over the populations. conds: list of (name, losses, colour)."""
    x = np.arange(len(SETS))
    g = fig.add_subplot(gs[2, :2])
    w = 0.78 / len(conds) if len(conds) == 3 else 0.36
    for m, (name, loss, c) in enumerate(conds):
        g.bar(x + (m - (len(conds) - 1) / 2) * w, loss, width=w, color=c, lw=0)
    g.set_yscale('log'); g.set_xticks(x); g.set_xticklabels(SETS, rotation=60, ha='right', rotation_mode='anchor', fontsize=7)
    g.axvline(2.5, color=GREY, ls=(0, (3, 2)), lw=0.5); g.axvline(5.5, color=GREY, ls=(0, (3, 2)), lw=0.5)
    g.set_ylabel('Fit loss (log)'); g.set_xlim(-0.6, len(SETS) - 0.4)
    g.set_ylim(ylim_g[0], max(ylim_g[1], 3 * max(np.max(loss) for _, loss, _ in conds))); plain_log(g, 1)
    for n, (name, _, c) in enumerate(conds):
        g.text(0.02, 0.97 - n * 0.11, name, color=c, transform=g.transAxes, ha='left', va='top', fontsize=7)
    letter(g, 'g', x=-0.2, y=1.02)
    h = fig.add_subplot(gs[2, 2])
    tot = [np.sum(loss) for _, loss, _ in conds]
    h.bar(range(len(conds)), tot, color=[c for *_, c in conds], lw=0, width=0.6)
    for i, v in enumerate(tot):
        h.text(i, v * 1.15, f'{v:.0f}', ha='center', va='bottom', fontsize=7)
    h.set_yscale('log'); h.set_ylim(10, max(tot) * 6)
    plain_log(h, 10)
    if len(conds) < 3:
        h.set_xlim(-0.7, len(conds) - 0.3)
    h.set_xticks(range(len(conds)), xt_h, fontsize=fontsize_h)
    h.set_ylabel('Fit loss, sum over\n7 populations (log)')
    letter(h, 'h', x=-0.45, y=1.02)
