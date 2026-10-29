"""Supplementary Figure 3: two models (the NMDA and the STP model), both fitted to the all-units read-out."""
import matplotlib.pyplot as plt
import numpy as np

from .style import C_OFF, C_ON, GREY, INK
from .unit_sets import SETS, loss_panels, panels, text_legend

BAR, MARKERS = ('#4d4d4d', '#b3b3b3'), (('o', '●'), ('s', '■'))


def draw(ses, models):
    """models: {name: statistics per population (readout.summary)} of two models."""
    fig = plt.figure(figsize=(6.4, 6.2))
    gs = fig.add_gridspec(3, 3, hspace=0.95, wspace=0.6, left=0.11, right=0.98, top=0.95, bottom=0.08, height_ratios=[1, 1, 0.9])
    axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
    col = {'off': C_OFF, 'on': C_ON}
    panels(axs.ravel(), ses, [(s, mk, dx, col) for s, (mk, _), dx in zip(models.values(), MARKERS, (0.2, 0.36))],
           (None, 250, 1, None, 250, 1), span=(0.45, 0.5), fitted_dx=0.03)
    for a, c, y0, va in ((axs[0, 0], C_OFF, 0.05, 'bottom'), (axs[1, 0], C_ON, 0.97, 'top')):
        text_legend(a, [('Data', INK)] + [(f'{sym} {name}', c) for name, (_, sym) in zip(models, MARKERS)] + [('Sessions', GREY)],
                    0.5, y0, va)
    loss_panels(fig, gs, [(f'{name} model', s.loc[SETS, 'loss'].values, c) for (name, s), c in zip(models.items(), BAR)],
                (1, 100), list(models), 7)
    return fig
