"""Figure 2: On/Off statistics of the seven decoded populations, data and the NMDA model."""
import matplotlib.pyplot as plt

from nmdaint import readout
from .style import C_OFF, C_ON, GREY, INK
from .unit_sets import model_stats, panels, text_legend


def plot():
    ses, _ = readout.load_data()
    fig, axs = plt.subplots(2, 3, figsize=(6.2, 4.0), gridspec_kw=dict(hspace=0.8, wspace=0.6, left=0.11, right=0.98,
                                                                       top=0.92, bottom=0.12))
    panels(axs.ravel(), ses, [(model_stats('nmda'), 'o', 0.25, {'off': C_OFF, 'on': C_ON})], (None, 250, 1, None, 250, 1))
    for a, col, y0, va in ((axs[0, 0], C_OFF, 0.05, 'bottom'), (axs[1, 0], C_ON, 0.97, 'top')):
        text_legend(a, (('Data', INK), ('Model', col), ('Sessions', GREY)), 0.5, y0, va)
    return fig
