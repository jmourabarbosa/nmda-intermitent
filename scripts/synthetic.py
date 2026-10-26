"""Validation of the model read-out on synthetic data (Fig. S1e-i); about 15 min on 4 cores.

results/synthetic/sessions.csv     one row per synthetic session (3 tuning strengths x 8 sessions)
results/synthetic/example.npz      example trial (medium strength, first session)

usage: python scripts/synthetic.py
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from nmdaint import paths, synthetic as SY


def main():
    jobs = [(lev, k, 1000 * i + j) for i, (lev, k) in enumerate(SY.LEVELS.items()) for j in range(SY.N_SESSIONS)]
    res = Parallel(n_jobs=4, verbose=5)(delayed(SY.session)(k, seed) for _, k, seed in jobs)
    pd.DataFrame([dict(level=lev, **r) for (lev, _, _), (r, _) in zip(jobs, res)]).to_csv(
        paths.results('synthetic', 'sessions.csv'), index=False)
    np.savez(paths.results('synthetic', 'example.npz'), **res[[s for _, _, s in jobs].index(1000)][1])


if __name__ == '__main__':
    main()
