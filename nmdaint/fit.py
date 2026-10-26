"""CMA-ES fit of the rate model's dynamics (ratemodel.FIT) to the On/Off statistics of the all-units read-out.

Loss (readout.loss on 792 simulated trials) + 2e4 x max(0, 0.98 - memory correct)^2, plus 10 x the squared
distance outside the bounds. Seven parameters, scaled to [0, 1] within BOUNDS: the slow time constant (NMDA tau_N;
for the STP model tau_f is fixed at 1.5 s), the recurrent strength w_s and the noise sigma, and the rate, mean
length, gamma shape and depth of the dips. The STP model is fitted with 1 s before the cue and a penalty
2e4 x max(0, P(spontaneous activity before the cue) - 0.02)^2, which excludes parameters where facilitation at
rest ignites the pools without a cue.
"""
import json
import os

import cma
import numpy as np

from . import ratemodel, readout

BOUNDS = dict(nmda=dict(tau_N=(50.0, 150.0), w_s=(1.30, 1.70), dip_rate=(0.5, 12.0), dip_mean=(20.0, 300.0),
                        dip_shape=(0.7, 10.0), dip_depth=(0.3, 1.0), sigma=(0.005, 0.04)),
              stp=dict(w_s=(0.5, 1.7), dip_rate=(0.5, 12.0), dip_mean=(20.0, 300.0), dip_shape=(0.7, 10.0),
                       dip_depth=(0.0, 1.0), sigma=(0.005, 0.04)))
FIXED = dict(nmda={}, stp=dict(tau_f=1500.0, t_pre=1000.0))
N_TRIALS = 792


def objective(model, theta, ses, hist, seed):
    out = readout.evaluate(ratemodel.params(model, **FIXED[model], **theta), ses, n_trials=N_TRIALS, seed=seed,
                           sets=('all',))
    L = readout.loss(out['sets']['all'], ses['all'], {s: hist['all', s] for s in ('on', 'off')})
    L += 2e4 * max(0.0, 0.98 - out['mem']) ** 2
    if model == 'stp':
        sim = out['sim']
        pre = (sim['t'] >= 100 - FIXED['stp']['t_pre']) & (sim['t'] < 0)     # from 100 ms after the start to the cue
        spont = ((sim['rc'][:, pre] > 10) | (sim['ru'][:, pre] > 10)).any(1).mean()
        L += 2e4 * max(0.0, spont - 0.02) ** 2
    return L


def fit(model, ses, hist, out_dir, seed=0, maxfev=1000):
    """Runs CMA-ES; the best parameters so far are kept in out_dir/best.json."""
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = np.array(list(BOUNDS[model].values())).T
    theta = lambda u: dict(zip(BOUNDS[model], lo + (hi - lo) * np.clip(u, 0, 1)))
    best = [np.inf]

    def f(u):
        L = objective(model, theta(u), ses, hist, seed)
        L += 10 * np.sum(np.clip(u - 1, 0, None) ** 2 + np.clip(-u, 0, None) ** 2)
        if L < best[0]:
            best[0] = L
            json.dump(dict(theta=theta(u), loss=L), open(os.path.join(out_dir, 'best.json'), 'w'), indent=1)
        return L

    u0 = np.random.default_rng(seed).uniform(0.25, 0.75, len(lo))
    es = cma.CMAEvolutionStrategy(u0, 0.25, dict(seed=seed + 1, maxfevals=maxfev, verbose=-9, popsize=12))
    while not es.stop():
        U = es.ask()
        es.tell(U, [f(u) for u in U])
        print(f'{model} seed {seed}: {es.countevals:5d} evaluations, best loss {best[0]:.3f}', flush=True)
    return json.load(open(os.path.join(out_dir, 'best.json')))
