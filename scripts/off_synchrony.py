"""Are Off states shared between two halves of the units? In each session, the cue is decoded separately from the
two random halves of the units (population.unit_halves, the split of Fig. 4), and the delay time points at which
one half is Off are compared with those of the other half, in the same trial and in another trial with the same cue
(population.off_synchrony). About 40 min on 4 cores; sessions are cached in cache/off_synchrony/.
Writes results/recordings/off_synchrony.csv.

usage: python scripts/off_synchrony.py [--jobs 4]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from nmdaint import paths, population, recordings as REC


def run_session(path):
    f = paths.cache('off_synchrony', os.path.basename(path).replace('.mat', '.npz'))
    if not os.path.exists(f):
        S = REC.Session(path)
        dsw = S.delay[S.sw]
        st = [S.states(S.confidence((H, H)))[1][:, dsw] for H in population.unit_halves(S.n_units)]
        same, other = population.off_synchrony(st[0], st[1], S.cue, np.random.default_rng(0))
        np.savez(f, session=S.name, same_trial=same, other_trial=other, frac_off=[(s == -1).mean() for s in st])
    d = np.load(f)
    return dict(session=str(d['session']), same_trial=float(d['same_trial']), other_trial=float(d['other_trial']),
                frac_off_half1=float(d['frac_off'][0]), frac_off_half2=float(d['frac_off'][1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=4)
    files = sorted(glob.glob(os.path.join(paths.DATA, '*.mat')), key=lambda f: -os.path.getsize(f))
    rows = Parallel(n_jobs=ap.parse_args().jobs, verbose=5)(delayed(run_session)(f) for f in files)
    pd.DataFrame(rows).sort_values('session').to_csv(paths.results('recordings', 'off_synchrony.csv'), index=False)


if __name__ == '__main__':
    main()
