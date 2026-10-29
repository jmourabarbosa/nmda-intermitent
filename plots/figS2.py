"""Supplementary Figure 2: the fitted model without dips, read out with the same and with a recalibrated gain."""
import matplotlib.pyplot as plt
import numpy as np

from .style import GREY, INK
from .unit_sets import SETS, loss_panels, panels

COLOURS = ('#d7301f', '#2b83ba', '#7b3294')


def draw(ses, conds):
    """conds: list of (legend label, bar label, model statistics per population from readout.summary), the fitted
    model first."""
    fig = plt.figure(figsize=(6.4, 6.2))
    gs = fig.add_gridspec(3, 3, hspace=0.95, wspace=0.6, left=0.11, right=0.98, top=0.95, bottom=0.08, height_ratios=[1, 1, 0.9])
    axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
    panels(axs.ravel(), ses, [(s, 'o', 0.2 + 0.14 * m, col) for m, ((_, _, s), col) in enumerate(zip(conds, COLOURS))],
           (None, 600, 1, None, 1500, 1), span=(0.45, 0.55), fitted_dx=0.05)
    items = [('Data', INK)] + [(name, col) for (name, _, _), col in zip(conds, COLOURS)] + [('Sessions', GREY)]
    for a, y0 in ((axs[0, 1], 0.95), (axs[1, 1], 0.8)):
        for n, (txt, c) in enumerate(items):
            a.text(0.06, y0 - n * 0.09, txt, color=c, transform=a.transAxes, va='center', fontsize=6.5)
    loss_panels(fig, gs, [(name, s.loc[SETS, 'loss'].values, col) for (name, _, s), col in zip(conds, COLOURS)], (1, 30000),
                [short for _, short, _ in conds], 6)
    return fig
