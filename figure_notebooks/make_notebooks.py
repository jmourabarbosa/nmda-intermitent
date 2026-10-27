"""Writes the figure notebooks (python figure_notebooks/make_notebooks.py; then run them with
jupyter nbconvert --to notebook --execute --inplace figure_notebooks/fig*.ipynb)."""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
SETUP = """import os, sys
sys.path.insert(0, os.path.abspath('..'))
os.makedirs('../figures', exist_ok=True)
import numpy as np, pandas as pd
from nmdaint import paths"""

NOTEBOOKS = {
    'fig1': ("Figure 1. Input dips create On and Off states, and NMDA carries the memory",
             "Computed by `scripts/ratemodel.py` (`example_trial`): the fitted NMDA rate model (`nmdaint/ratemodel.py`) "
             "read out like the data (`nmdaint/readout.py`, labelling in `nmdaint/onoff.py`).",
             """e = np.load(paths.results('ratemodel', 'example_trial.npz'))
for g in ('high', 'mid', 'low'):
    print(f"{g} gain: {float(e[g + '_n_off']):.2f} Off states and {float(e[g + '_n_on']):.2f} On states per trial")"""),
    'fig2': ("Figure 2. The model predicts how On/Off statistics change with the decoded population",
             "Data: `scripts/recordings.py` (`nmdaint/recordings.py`). Model: `scripts/ratemodel.py` "
             "(`population_stats`, `nmdaint/readout.evaluate`).",
             """s = pd.read_csv(paths.results('ratemodel', 'population_stats.csv'))
s[s.condition == 'nmda'].set_index('set').round(3)"""),
    'fig3': ("Figure 3. Errors increase with time Off, and the decline in performance points to a trace of about 120 ms",
             "a: `scripts/ratemodel.py` (`nmdaint/memory.single_dip`). b-d: `scripts/behaviour.py` "
             "(`nmdaint/behaviour.py`: `glmm_single`, `fit_tau`), with the tau sweep of `nmdaint/memory.tau_sweep`.",
             """print(pd.read_csv(paths.results('behaviour', 'glmm.csv'))[['predictor', 'odds_ratio_per_sd', 'or_ci_lo', 'or_ci_hi', 'p_lrt']])
t = np.load(paths.results('behaviour', 'tau_fit.npz'))
print(f"tau_N {float(t['tau']):.0f} ms, 95% bootstrap interval {np.percentile(t['boot'], 2.5):.0f}-{np.percentile(t['boot'], 97.5):.0f} ms; "
      f"{int(t['n'].sum())} trials")"""),
    'fig4': ("Figure 4. Intermittent rate coding in a spiking ring-attractor network with NMDA currents",
             "a-c: `scripts/spiking.py` (`nmdaint/spiking.py`, `nmdaint/spiking_readout.py`). d, e: "
             "`scripts/recordings.py` and `scripts/spiking.py` (`nmdaint/population.py`).",
             """print(pd.read_csv(paths.results('spiking', 'sessions.csv')).mean().round(3))
for f in ('data_procrustes.csv', 'net_procrustes.csv'):
    d = pd.read_csv(paths.results('population', f))
    print(f, d[d.space == 'pcs'].groupby('period').normalised.agg(['mean', 'sem', 'count']).round(3), sep='\\n')"""),
    'figS1': ("Supplementary Figure 1. Reading the model out like the data",
              "a, c, d: `scripts/ratemodel.py` (`readout_example`). b, c: `scripts/recordings.py`. e-i: "
              "`scripts/synthetic.py` (`nmdaint/synthetic.py`).",
              """acf = pd.read_csv(paths.results('recordings', 'precue_acf.csv'))
print('pre-cue z autocorrelation at 200 ms (the read-out noise share rho):')
print(acf[acf.lag_ms == 200].groupby('set').acf.mean().round(3))
print(pd.read_csv(paths.results('synthetic', 'sessions.csv')).groupby('level').mean(numeric_only=True).round(3).T)"""),
    'figS2': ("Supplementary Figure 2. Off states are not produced by the read-out noise alone",
              "`scripts/ratemodel.py` (`population_stats`): the fitted model with a constant common input, read out "
              "with the fitted gain and with a recalibrated gain.",
              """s = pd.read_csv(paths.results('ratemodel', 'population_stats.csv'))
print(s.groupby('condition').loss.sum().round(0))
s[s.condition != 'stp'].pivot(index='set', columns='condition', values='n_off').round(2)"""),
    'figS3': ("Supplementary Figure 3. The STP model fits the On/Off statistics about as well as the NMDA model",
              "`scripts/ratemodel.py` (`population_stats`); the STP model is `ratemodel.params('stp')`.",
              """s = pd.read_csv(paths.results('ratemodel', 'population_stats.csv'))
print(s[s.condition.isin(['nmda', 'stp'])].groupby('condition').loss.sum().round(0))"""),
}


for name, (title, source, numbers) in NOTEBOOKS.items():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(f'# {title}\n\n{source} The plotting code is `plots/{name}.py`.'),
                nbf.v4.new_code_cell(SETUP),
                nbf.v4.new_code_cell(f"from plots import {name}, style\n"
                                     f"fig = {name}.plot()\n"
                                     f"style.save(fig, '../figures/{name}.png', dpi=getattr({name}, 'DPI', 600))"),
                nbf.v4.new_markdown_cell('Numbers quoted in the text and caption:'),
                nbf.v4.new_code_cell(numbers)]
    nb.metadata['kernelspec'] = dict(name='python3', display_name='Python 3', language='python')
    nbf.write(nb, os.path.join(HERE, f'{name}.ipynb'))
