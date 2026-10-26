"""Supplementary Figure 3: the NMDA model and the STP model, both fitted to the all-units read-out."""
import matplotlib.pyplot as plt
import numpy as np

from nmdaint import readout
from .style import C_OFF, C_ON, GREY, INK
from .unit_sets import loss_panels, model_stats, panels, text_legend

BAR = {'NMDA model': '#4d4d4d', 'STP model': '#b3b3b3'}


def plot():
    ses, _ = readout.load_data()
    fig = plt.figure(figsize=(6.4, 6.2))
    gs = fig.add_gridspec(3, 3, hspace=0.95, wspace=0.6, left=0.11, right=0.98, top=0.95, bottom=0.08, height_ratios=[1, 1, 0.9])
    axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
    nmda, stp = model_stats('nmda'), model_stats('stp')
    col = {'off': C_OFF, 'on': C_ON}
    panels(axs.ravel(), ses, [(nmda, 'o', 0.2, col), (stp, 's', 0.36, col)], (None, 250, 1, None, 250, 1),
           span=(0.45, 0.5), fitted_dx=0.03)
    for a, c, y0, va in ((axs[0, 0], C_OFF, 0.05, 'bottom'), (axs[1, 0], C_ON, 0.97, 'top')):
        text_legend(a, (('Data', INK), ('● NMDA', c), ('■ STP', c), ('Sessions', GREY)), 0.5, y0, va)
    loss_panels(fig, gs, [(name, s['loss'].values, BAR[name]) for name, s in (('NMDA model', nmda), ('STP model', stp))],
                (1, 100), ['NMDA', 'STP'], 7)
    return fig
