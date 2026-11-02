"""Reproduce headline numbers of the paper from results/ (no simulations)."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from nmdaint import behaviour, paths, readout

ses, hist = readout.load_data()
print('== Fig 2: top vs random, paired Wilcoxon over sessions')
for k in (5, 20, 50):
    for col in ('frac_on', 'frac_off', 'off_per_trial', 'on_per_trial'):
        a, b = ses[f'top {k}'][col], ses[f'rand {k}'][col]
        print(f'k={k:2d} {col:14s} top {np.nanmean(a):.3f} rand {np.nanmean(b):.3f}  p={wilcoxon(a, b).pvalue:.1e}')
print('== all-units data stats (paper: Off/trial 3.3, Off median 112, On/trial 2.1, On median 134)')
for c in ('off_per_trial', 'off_median_ms', 'on_per_trial', 'on_median_ms', 'mu_on', 'frac_off', 'frac_on'):
    print(f'  {c:14s} {np.nanmean(ses["all"][c]):.3f}')

print('== Does decoding strength (mu_on) predict Off frequency across sessions? (all units)')
for c in ('off_per_trial', 'frac_off', 'on_per_trial'):
    r = spearmanr(ses['all']['mu_on'], ses['all'][c]); print(f'  mu_on vs {c}: rho={r.statistic:.2f} p={r.pvalue:.1e}')

trials = pd.read_csv(paths.results('recordings', 'per_trial.csv'))
print('== behaviour: trials', len(trials), 'errors', int((1 - trials.correct).sum()))
print('error rate by monkey:', trials.groupby('monkey').correct.apply(lambda c: 1 - c.mean()).round(3).to_dict())
print('trials without Off state, error rate:', round(1 - trials[trials.n_off == 0].correct.mean(), 3),
      'with Off:', round(1 - trials[trials.n_off > 0].correct.mean(), 3))
print('corr between per-trial stats (Spearman):')
print(trials[list(behaviour.STATS)].corr('spearman').round(2))
# Off time of error vs correct trials: is the error effect driven by a subset of 'all-Off' trials?
for c in (0, 1):
    f = trials[trials.correct == c].frac_off
    print(f'  correct={c}: frac_off mean {f.mean():.3f}, median {f.median():.3f}, P(frac_off>0.5) {np.mean(f > 0.5):.3f}')
sweep = pd.read_csv(paths.results('ratemodel', 'tau_sweep.csv'))
taus = np.sort(sweep.tau_ms.unique())
cnt = lambda c: sweep.pivot(index='tau_ms', columns='bin', values=c).loc[taus].values
fine, Hf = behaviour.held_curves(taus, cnt('held'), cnt('n'))
t = behaviour.fit_tau(trials, fine, Hf)
print(f"== tau fit: {t['tau']:.0f} ms  CI {np.percentile(t['boot'], 2.5):.0f}-{np.percentile(t['boot'], 97.5):.0f}  p0 {t['p0']:.3f}")
print('data P(correct) per Off bin:', np.round(t['k'] / t['n'], 3), 'n', t['n'])
print('model P(held) per bin @ tau=150 (sweep):', np.round(cnt('held')[taus == 150][0] / cnt('n')[taus == 150][0], 3))
print('model n per bin @ tau=150:', cnt('n')[taus == 150][0])
