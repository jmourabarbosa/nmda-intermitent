"""Rate-model results (Figs. 1, 2, 3a, S1a-d, S2, S3); about 1 h on 4 cores. Needs results/recordings/.

results/ratemodel/
  population_stats.csv  On/Off statistics, gain and fit loss of each decoded population, for the fitted NMDA and
                        STP models and the NMDA model without dips (same gain / recalibrated gain)  (Figs. 2, S2, S3)
  example_trial.npz     the example trial of Fig. 1a and the On/Off length counts at three gains (Fig. 1b)
  readout_example.npz   a model trial through the read-out, and the gain calibration (Fig. S1a, c, d)
  single_dip.csv        memory held against the length of one dip, NMDA and STP models (Fig. 3a)
  tau_sweep.csv         memory held per bin of the fraction of the delay Off, for each tau_N (Fig. 3c, d)

usage: python scripts/ratemodel.py [--fit]      (--fit: also refit both models, ~2 h per run, in results/fits/)
"""
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from nmdaint import behaviour, fit, memory, onoff, paths, ratemodel as RM, readout as R

N_TRIALS = 2112
GAINS = {'high': 2.0, 'mid': 1.0, 'low': 0.5}       # Fig. 1: read-out gain relative to the fitted one
EXAMPLE = dict(sim_seed=12, noise_seed=11, trial=226, quiet=1000, dips=1500, cue_dur=75)   # Fig. 1a; BUGS.md


def population_stats(ses, hist):
    """Model statistics of the seven decoded populations in four conditions."""
    nmda = R.evaluate(RM.params('nmda'), ses, n_trials=N_TRIALS, seed=1)
    gains = {lab: r['gain'] for lab, r in nmda['sets'].items()}
    no_dips = RM.params('nmda', dip_depth=0.0)
    conds = {'nmda': nmda, 'stp': R.evaluate(RM.params('stp'), ses, n_trials=N_TRIALS, seed=1),
             'no_dips_same_gain': R.evaluate(no_dips, ses, seed=2, sim=RM.simulate(no_dips, N_TRIALS, seed=101), gains=gains),
             'no_dips_recalibrated': R.evaluate(no_dips, ses, n_trials=N_TRIALS, seed=1)}
    rows = [dict(condition=c, set=lab, memory=o['mem'], gain=r['gain'],
                 loss=R.loss(r, ses[lab], {s: hist[lab, s] for s in ('on', 'off')}),
                 **{k: r[k] for k in ('n_off', 'off_median', 'off_mean', 'frac_off', 'n_on', 'on_median', 'on_mean', 'frac_on')})
            for c, o in conds.items() for lab, r in o['sets'].items()]
    return pd.DataFrame(rows), nmda


def example_trial(nmda):
    """Fig. 1: one long trial (1 s of delay without dips, then dips) read out with the fitted all-units gain, and
    the On/Off length counts of the 2,112 simulated trials read out at three gains (same noise)."""
    r, e = nmda['sets']['all'], EXAMPLE
    p = RM.params(cue_dur=e['cue_dur'], t_delay=e['quiet'] + e['dips'] - e['cue_dur'])
    sim = RM.simulate(p, 300, seed=e['sim_seed'], quiet=e['quiet'])
    rng = np.random.default_rng(e['noise_seed'])
    dw, tz = R.rate_diff(sim, t_from=-250)
    eta = R.noise(dw.shape, R.RHO['all'], rng)
    state = onoff.label(r['gain'] * dw + eta, R.null_threshold(dw.shape, R.RHO['all'], rng), tz >= 0)
    k = e['trial']
    out = dict(t=sim['t'], tz=tz, dw=dw[k], eta=eta[k], state=state[k], gain=r['gain'], held=RM.memory_correct(sim)[k],
               **{v: sim[v][k] for v in ('rc', 'ru', 'sc', 'su', 'm')})
    dw, tz = R.rate_diff(nmda['sim'])                 # length distributions at each gain
    eta = r['z'] - r['gain'] * dw
    thr = R.null_threshold(dw.shape, R.RHO['all'], np.random.default_rng(7))
    delay, valid = (tz >= 500) & (tz <= 1400), (tz >= 0) & (tz <= 1400)
    for name, f in GAINS.items():
        st = r['state'] if f == 1.0 else onoff.label(f * r['gain'] * dw + eta, thr, valid)
        s = onoff.stats(st, delay)
        out.update({f'{name}_h_on': s['h_on'], f'{name}_h_off': s['h_off'], f'{name}_n_on': s['n_on'], f'{name}_n_off': s['n_off']})
    np.savez(paths.results('ratemodel', 'example_trial.npz'), **out)


def readout_example(nmda, ses):
    """Fig. S1a: one model trial through the read-out; S1c, d: the model's delay z at the fitted gain, and its
    decoding strength as a function of the gain."""
    c, rho = nmda['sets']['all']['gain'], R.RHO['all']
    sim = RM.simulate(RM.params(), 30, seed=5)
    dw, tz = R.rate_diff(sim)
    rng = np.random.default_rng(4)
    eta = R.noise(dw.shape, rho, rng)
    valid = (tz >= 0) & (tz <= 1400)
    st = onoff.label(c * dw + eta, R.null_threshold(dw.shape, rho, rng), valid)
    k = int(np.argmax([(s[valid] == -1).sum() * (s[valid] == 1).sum() for s in st]))     # a trial with both states
    simL = RM.simulate(RM.params(), 600, seed=21)
    dwL, tzL = R.rate_diff(simL)
    dl = (tzL >= 500) & (tzL <= 1400)
    etaL = R.noise(dwL.shape, rho, np.random.default_rng(22))
    grid = 10 ** np.linspace(-2.6, -0.6, 14)
    zm = (c * dwL + etaL)[:, dl].ravel()
    np.savez(paths.results('ratemodel', 'readout_example.npz'), t=sim['t'], rc=sim['rc'][k], ru=sim['ru'][k],
             m=sim['m'][k], tz=tz, dw=dw[k], eta=eta[k], state=np.where(valid, st[k], 0), gain=c, rho=rho,
             delay_z=zm, mu_on_model=onoff.mu_on(zm[None], np.random.default_rng(1)), grid=grid,
             mu_on_grid=[onoff.mu_on((g * dwL + etaL)[:, dl], np.random.default_rng(1)) for g in grid],
             mu_on_data=ses['all']['mu_on'].mean(),
             gains={lab: nmda['sets'][lab]['gain'] for lab in ('rand 20', 'all', 'top 20')})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', action='store_true')
    a = ap.parse_args()
    ses, hist = R.load_data()
    stats, nmda = population_stats(ses, hist)
    stats.to_csv(paths.results('ratemodel', 'population_stats.csv'), index=False)
    example_trial(nmda)
    readout_example(nmda, ses)
    rows = []
    for model in ('nmda', 'stp'):
        L, held = memory.single_dip(RM.params(model))
        rows += [dict(model=model, dip_ms=x, n=int((L == x).sum()), held=int(held[L == x].sum())) for x in np.unique(L)]
    pd.DataFrame(rows).to_csv(paths.results('ratemodel', 'single_dip.csv'), index=False)
    gain = memory.reference_gain(RM.params(), ses)
    sweep = Parallel(n_jobs=4)(delayed(memory.tau_sweep)(RM.params(), tau, gain) for tau in memory.TAUS)
    pd.DataFrame([dict(tau_ms=tau, bin=j, n=int((behaviour.bins(f) == j).sum()), held=int(h[behaviour.bins(f) == j].sum()))
                  for tau, (f, h) in zip(memory.TAUS, sweep) for j in range(behaviour.NB)]
                 ).to_csv(paths.results('ratemodel', 'tau_sweep.csv'), index=False)
    if a.fit:
        Parallel(n_jobs=4)(delayed(fit.fit)(m, ses, hist, paths.results('fits', f'{m}_seed{s}'), seed=s)
                           for m in ('nmda', 'stp') for s in (0, 1, 2))


if __name__ == '__main__':
    main()
