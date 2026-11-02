"""On-state threshold: the data use, per trial, the 95th percentile of the LARGEST cluster mass of each label
shuffle (a max statistic); the rate model uses the 95th percentile of ALL cluster masses of pure noise. How different
are the two thresholds on the same noise, and how much does that change the On statistics of the fitted model?"""
import numpy as np
from nmdaint import onoff, ratemodel, readout

rng = np.random.default_rng(0)
ses, hist = readout.load_data()
for rho in (0.09, 0.35):
    eta = readout.noise((3000, 146), rho, rng)
    thr_all = onoff.null_threshold(eta)
    mx = [max([r[a:b].sum() for a, b in onoff.runs(r > onoff.ON_Z)], default=0.0) for r in eta]
    print(f'rho {rho}: threshold all clusters {thr_all:.1f}  vs  max-per-trace {np.percentile(mx, 95):.1f}')
out = readout.evaluate(ratemodel.params('nmda'), ses, n_trials=1056, seed=0, sets=('all',))
r = out['sets']['all']; tz = out['tz']
valid, delay = (tz >= 0) & (tz <= 1400), (tz >= 500) & (tz <= 1400)
eta = readout.noise((3000, len(tz)), 0.35, rng)
mx = np.percentile([max([x[a:b].sum() for a, b in onoff.runs((x > onoff.ON_Z) & valid)], default=0.0) for x in eta], 95)
st2 = onoff.label(r['z'], mx, valid)
s1, s2 = onoff.stats(r['state'], delay), onoff.stats(st2, delay)
d = ses['all']
print('all units          paper-threshold  max-threshold  data')
for k, dk in (('n_on', 'on_per_trial'), ('frac_on', 'frac_on'), ('on_median', 'on_median_ms'), ('n_off', 'off_per_trial')):
    print(f'  {k:10s} {s1[k]:12.3f} {s2[k]:14.3f} {np.nanmean(d[dk]):8.3f}')
print('loss (all units): paper threshold', round(readout.loss(r, d, {s: hist['all', s] for s in ('on', 'off')}), 2),
      ' max threshold', round(readout.loss({**r, **s2}, d, {s: hist['all', s] for s in ('on', 'off')}), 2))
