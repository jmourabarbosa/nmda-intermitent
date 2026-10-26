"""Off states and errors (Fig. 3b-d); a few minutes. Needs results/recordings/per_trial.csv and
results/ratemodel/tau_sweep.csv.

results/behaviour/
  glmm.csv              fixed effect of each statistic on the log-odds of an error (Fig. 3b)
  glmm_sessions.csv     session weights (Fig. 3b, grey dots)
  tau_fit.npz           fraction correct per Off bin, the best-fitting model curve and the bootstrap (Fig. 3c, d)

usage: python scripts/behaviour.py
"""
import numpy as np
import pandas as pd

from nmdaint import behaviour as B, paths


def main():
    trials = pd.read_csv(paths.results('recordings', 'per_trial.csv'))
    fixed, weights = B.glmm_single(trials)
    fixed.to_csv(paths.results('behaviour', 'glmm.csv'), index=False)
    weights.to_csv(paths.results('behaviour', 'glmm_sessions.csv'), index=False)
    sweep = pd.read_csv(paths.results('ratemodel', 'tau_sweep.csv'))
    taus = np.sort(sweep.tau_ms.unique())
    count = lambda c: sweep.pivot(index='tau_ms', columns='bin', values=c).loc[taus].values
    fine, Hf = B.held_curves(taus, count('held'), count('n'))
    r = B.fit_tau(trials, fine, Hf)
    np.savez(paths.results('behaviour', 'tau_fit.npz'), bin_ms=900 * (B.EDGES[:-1] + np.minimum(B.EDGES[1:], 1)) / 2,
             p_correct_model=r['p0'] * (r['held'] + B.G * (1 - r['held'])), **r)
    print(fixed.round(4).to_string())
    print(f"tau_N {r['tau']:.1f} ms, p0 {r['p0']:.3f}, bootstrap 95% interval {np.percentile(r['boot'], 2.5):.0f}-"
          f"{np.percentile(r['boot'], 97.5):.0f} ms")


if __name__ == '__main__':
    main()
