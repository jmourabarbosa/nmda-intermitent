"""Fig. 2: how specific is the 'prediction' of the six decoded populations?

(1) Leakage: at a FIXED read-out gain, does the decoding strength mu_on (used to calibrate the gain of each
    population) depend on how often the model is Off? If it does, calibrating c to each population's mu_on already
    encodes part of its On/Off statistics, and the 'prediction' is partly a fit.
(2) Specificity: models with clearly wrong dynamics (dip rate halved, dips twice as long, shallower dips), read out
    with the same per-population calibration. If they also produce the top-vs-random shift, the shift is a
    property of the read-out (z = c dr + eta with c set per population), not a test of the dynamics.
usage: python review/synth_readout_generic.py   (~10 min on 4 cores)
"""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from nmdaint import onoff, ratemodel, readout

ses, hist = readout.load_data()
P = ratemodel.params('nmda')


def leak(rate, c=0.05, n=1056):
    p = {**P, 'dip_rate': rate}
    sim = ratemodel.simulate(p, n, seed=3)
    dw, tz = readout.rate_diff(sim)
    rng = np.random.default_rng(4)
    delay, valid = (tz >= 500) & (tz <= 1400), (tz >= 0) & (tz <= 1400)
    eta = readout.noise(dw.shape, readout.RHO['all'], rng)
    st = onoff.label(c * dw + eta, readout.null_threshold(dw.shape, readout.RHO['all'], rng), valid)
    s = onoff.stats(st, delay)
    return dict(dip_rate=rate, gain=c, mu_on=onoff.mu_on((c * dw + eta)[:, delay], rng), n_off=s['n_off'],
                frac_off=s['frac_off'], frac_on=s['frac_on'])


VARIANTS = {'fitted': {}, 'dip_rate/2': dict(dip_rate=P['dip_rate'] / 2), 'dip_mean x2': dict(dip_mean=P['dip_mean'] * 2),
            'dip_depth 0.4': dict(dip_depth=0.4), 'dip_rate x1.5': dict(dip_rate=P['dip_rate'] * 1.5)}


def variant(name):
    out = readout.evaluate({**P, **VARIANTS[name]}, ses, n_trials=1056, seed=0)
    s = readout.summary(out, ses, hist)
    s['variant'] = name
    return s.reset_index()


if __name__ == '__main__':
    pd.set_option('display.width', 220)
    L = pd.DataFrame(Parallel(n_jobs=4)(delayed(leak)(r) for r in (2, 5, 10.2, 15, 20)))
    print('== (1) fixed gain 0.05: decoding strength against dip rate')
    print(L.round(3).to_string(index=False))
    V = pd.concat(Parallel(n_jobs=4)(delayed(variant)(v) for v in VARIANTS))
    V.to_csv('review/out/synth_readout_generic.csv', index=False)
    print('\n== (2) per-population read-out of models with other dynamics')
    print(V[['variant', 'set', 'memory', 'gain', 'n_off', 'frac_off', 'n_on', 'frac_on', 'loss']].round(3).to_string(index=False))
    d = {lab: {k: ses[lab][k].mean() for k in ('off_per_trial', 'frac_off', 'on_per_trial', 'frac_on')} for lab in readout.SETS}
    print('\ndata:', pd.DataFrame(d).T.round(3).to_string())
    agg = V.groupby('variant').apply(lambda g: pd.Series(dict(
        loss_all=g.loc[g.set == 'all', 'loss'].iloc[0], loss_6_predicted=g.loc[g.set != 'all', 'loss'].sum(),
        shift_off_top_lt_rand=all(g.set_index('set').loc[f'top {k}', 'frac_off'] < g.set_index('set').loc[f'rand {k}', 'frac_off'] for k in (5, 20, 50)),
        shift_on_top_gt_rand=all(g.set_index('set').loc[f'top {k}', 'frac_on'] > g.set_index('set').loc[f'rand {k}', 'frac_on'] for k in (5, 20, 50)))))
    L.to_csv('review/out/synth_readout_leak.csv', index=False)
    print('\n', agg.round(2).to_string())
