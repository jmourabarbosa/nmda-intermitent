"""Figure 2: On/Off statistics against decoding strength, every session and decoded population, and the NMDA model
read out at varying gain."""
import matplotlib.pyplot as plt
import numpy as np

from nmdaint import readout
from .style import C_OFF, C_ON, INK, LIGHT, letter
from .unit_sets import PANELS

MARKERS = {'top': ('^', 'Most selective'), 'random': ('o', 'Random'), 'all': ('s', 'All neurons')}
DASHES = ('-', (0, (3, 2)))


def draw(ses, sweeps):
    """ses: the data per decoded population (readout.load_data); sweeps: {rho: readout.gain_sweep table}, one model
    line each (at most two)."""
    fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.2), gridspec_kw=dict(hspace=0.55, wspace=0.55, left=0.09, right=0.98,
                                                                       top=0.94, bottom=0.11))
    mu = ses['all']['mu_on']
    band = mu.mean() + np.array([-2, 2]) * mu.std(ddof=1) / np.sqrt(len(mu))       # the fitted decoder, mean +- 2 SEM
    for k, (a, (dk, mk, name, state)) in enumerate(zip(axs.ravel(), PANELS)):
        a.axvspan(*band, color=LIGHT, lw=0, zorder=0)
        a.text(np.sqrt(band[0] * band[1]), 0.98, 'Fitted', transform=a.get_xaxis_transform(), rotation=90, ha='center',
               va='top', fontsize=7)
        for lab, d in ses.items():                                                 # one dot per session and population
            a.scatter(d['mu_on'], d[dk], s=14, marker=MARKERS[readout.SETS[lab][0]][0], lw=0.4, alpha=0.85,
                      facecolor=C_ON if state == 'on' else C_OFF, edgecolor=INK, zorder=2)
        for sw, ls in zip(sweeps.values(), DASHES):
            a.plot(sw.mu_on, sw[mk], color=INK, lw=1.2, ls=ls, zorder=3)
        a.set_xscale('log'); a.set_xlim(0.8, 8); a.set_xticks([1, 2, 4, 8], ['1', '2', '4', '8']); a.minorticks_off()
        a.set_ylim(0, 1 if 'frac' in dk else 8 if 'per_trial' in dk else 300)
        if 'frac' in dk:
            a.set_yticks([0, 0.5, 1], ['0', '', '1'])
        a.set_ylabel(name)
        letter(a, 'abcdef'[k], x=-0.38, y=1.02)
    for a in axs[1]:
        a.set_xlabel('Decoding strength')
    h = [plt.Line2D([], [], ls='', marker=m, mfc='w', mec=INK, mew=0.6, ms=4, label=t) for m, t in MARKERS.values()]
    h += [plt.Line2D([], [], color=INK, lw=1.2, ls=ls, label=rf'Model, $\rho$ = {rho:.2f}') for rho, ls in zip(sweeps, DASHES)]
    axs[0, 0].legend(handles=h, loc='upper right', bbox_to_anchor=(1.12, 1.05), fontsize=6, handletextpad=0.4,
                     labelspacing=0.25, borderaxespad=0.1)
    return fig
