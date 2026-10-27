"""Writes the two model notebooks (python model_notebooks/make_model_notebooks.py)."""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SETUP = """import os, sys
sys.path.insert(0, os.path.abspath('..'))
import numpy as np, matplotlib.pyplot as plt"""

RATE = [
    md("# The rate model\n\nTwo memory pools (cued, uncued) with slow recurrent excitation and shared inhibition, "
       "driven by a common input that dips (`nmdaint/ratemodel.py`). Parameters are the fitted ones "
       "(`ratemodel.FIT`); change any of them with `ratemodel.params(model, name=value)`."),
    code(SETUP + "\nfrom nmdaint import ratemodel, readout, onoff\nratemodel.params('nmda')"),
    md("## One trial\nDips silence both pools; when the input returns, the pool that kept more NMDA gating wins."),
    code("""p = ratemodel.params('nmda')
sim = ratemodel.simulate(p, n_tr=5, seed=1)
k = 0
fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True, height_ratios=(3, 1))
ax[0].plot(sim['t'], sim['rc'][k], 'k', label='cued pool rate (Hz)')
ax[0].plot(sim['t'], sim['ru'][k], 'grey', label='uncued pool rate (Hz)')
ax[0].plot(sim['t'], 100 * sim['sc'][k], 'k:', label='cued NMDA gating x 100')
ax[1].plot(sim['t'], sim['m'][k], 'k'); ax[1].set_ylabel('common input')
ax[0].legend(fontsize=8); ax[1].set_xlabel('time from cue onset (ms)')
print('memory held at the end of the delay:', ratemodel.memory_correct(sim))"""),
    md("## Read out like the data\n`z = c dr + eta`: the rate difference in 100 ms windows times a gain, plus noise "
       "(`nmdaint/readout.py`); On/Off labels as for the recordings (`nmdaint/onoff.py`). The gain is the one "
       "calibrated to the all-units decoder in the paper."),
    code("""import pandas as pd
from nmdaint import paths
stats = pd.read_csv(paths.results('ratemodel', 'population_stats.csv'))
gain = stats[(stats.condition == 'nmda') & (stats.set == 'all')].gain.item()
sim = ratemodel.simulate(p, n_tr=200, seed=2)
dw, tz = readout.rate_diff(sim)
rng = np.random.default_rng(0)
z = gain * dw + readout.noise(dw.shape, readout.RHO['all'], rng)
state = onoff.label(z, readout.null_threshold(dw.shape, readout.RHO['all'], rng), (tz >= 0) & (tz <= 1400))
k = 0
plt.figure(figsize=(8, 2.5))
plt.plot(tz, z[k], 'k')
for v, c in ((1, 'r'), (-1, 'c')):
    for a, b in onoff.runs(state[k] == v):
        plt.axvspan(tz[a], tz[b - 1], color=c, alpha=0.2, lw=0)
plt.axhline(1.96, color='r', ls='--', lw=0.6); plt.axhline(0.84, color='c', ls='--', lw=0.6)
plt.xlabel('time from cue onset (ms)'); plt.ylabel('decoding z')
s = onoff.stats(state, (tz >= 500) & (tz <= 1400))
print(f"{s['n_off']:.2f} Off and {s['n_on']:.2f} On states per trial; delay {s['frac_off']:.2f} Off, {s['frac_on']:.2f} On")"""),
    md("## One silence: NMDA vs STP (Fig. 3a)\nOne dip of fixed length, no other dips. The NMDA trace decays with "
       "tau_N, so long silences erase the memory; facilitation (tau_f = 1.5 s) bridges them."),
    code("""lengths = np.arange(0, 1001, 100)
for model, col in (('nmda', 'k'), ('stp', 'tab:blue')):
    p = ratemodel.params(model)
    held = [ratemodel.memory_correct(ratemodel.simulate(p, 50, seed=i, m=ratemodel.single_dip(p, 50, 500, L))).mean()
            for i, L in enumerate(lengths)]
    plt.plot(lengths, held, 'o-', color=col, label=model)
plt.xlabel('silence duration (ms)'); plt.ylabel('fraction memory held'); plt.legend()"""),
    md("## Try it\nShorter NMDA time constant: the same dips now erase the memory more often."),
    code("""for tau in (60, 100, 150):
    sim = ratemodel.simulate(ratemodel.params('nmda', tau_N=tau), 200, seed=3)
    print(f'tau_N {tau} ms: memory held in {ratemodel.memory_correct(sim).mean():.2f} of trials')"""),
]

SPIKING = [
    md("# The spiking network\n\nA ring of 2,048 excitatory and 512 inhibitory integrate-and-fire neurons with "
       "NMDA-only recurrent excitation (tau_NMDA = 150 ms) and a common input that dips and overshoots "
       "(`nmdaint/spiking.py`). Simulating one 2 s trial takes a few seconds per core."),
    code(SETUP + "\nfrom nmdaint import spiking\nnet = spiking.Network(spiking.params())"),
    md("## Trials with and without dips\nWithout dips the bump of activity persists at the cue; with dips it is "
       "silenced and reignites at the cued location when the input returns."),
    code("""cues = [0, 90, 180, 270]          # deg
sims = {name: spiking.simulate(spiking.Network(spiking.params(dip_rate=rate)), cues, seed=1, raster=True)
        for name, rate in (('no dips', 1e-9), ('dips', 7.0))}
fig, ax = plt.subplots(2, 2, figsize=(10, 4.5), sharex=True, height_ratios=(3, 1))
for j, (name, sim) in enumerate(sims.items()):
    k = 2                             # the trial with the cue at 180 deg
    t, cell = np.nonzero(sim['raster'][k])
    ax[0, j].plot(sim['t1'][t], cell * 360 / net.p['N_E'], ',k', alpha=0.3)
    ax[0, j].set_title(f"{name}: memory held {spiking.memory(sim, net)}", fontsize=9)
    ax[1, j].plot(sim['t1'], sim['m_eff'][k], 'k')
    ax[1, j].set_xlabel('time from cue onset (ms)')
ax[0, 0].set_ylabel('preferred location (deg)'); ax[1, 0].set_ylabel('common input')"""),
    md("## Where is the bump?\nPopulation-vector angle of the E spikes in 50 ms bins: it stays at the cue between "
       "silences."),
    code("""sim = sims['dips']
ras = sim['raster'].reshape(len(cues), -1, 50, net.p['N_E']).sum(2)          # 50 ms bins
vec = ras @ np.exp(1j * net.theta_E)
t50 = sim['t1'][::50]
for k, c in enumerate(cues):
    active = np.abs(vec[k]) > 20
    plt.plot(t50[active], np.rad2deg(np.angle(vec[k][active])) % 360, '.', label=f'cue {c} deg')
plt.xlabel('time from cue onset (ms)'); plt.ylabel('bump location (deg)'); plt.legend(fontsize=8)"""),
    md("## Read out like the data\nDecoding needs whole sessions (480 trials, ~10 min each); the paper's example "
       "(Fig. 4a) is stored in `results/spiking/`. To run one: `d = spiking.session(net, 100, 0)` and "
       "`nmdaint.spiking_readout.read_session(d, seed=0)`."),
    code("""from nmdaint import paths
e = np.load(paths.results('spiking', 'example_trial.npz'))
plt.figure(figsize=(8, 2.5)); plt.plot(e['tz'], e['z'], 'k')
plt.axhline(1.96, color='r', ls='--', lw=0.6); plt.axhline(0.84, color='c', ls='--', lw=0.6)
plt.xlabel('time from cue onset (ms)'); plt.ylabel('decoding z (all units)')"""),
]

for name, cells in (('rate_model', RATE), ('spiking_network', SPIKING)):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata['kernelspec'] = dict(name='python3', display_name='Python 3', language='python')
    nbf.write(nb, os.path.join(HERE, f'{name}.ipynb'))
