"""CMA-ES fit of the rate model to the all-units read-out (nmdaint/fit.py), 3 seeds per model; ~2 h per run.
The figures use the published parameters (ratemodel.FIT); BUGS.md, A2 compares them with these refits.

usage: python scripts/fit.py [nmda|stp]       (writes cache/fits/<model>_seed<s>/best.json)
"""
import sys

from joblib import Parallel, delayed

from nmdaint import fit, paths, readout

if __name__ == '__main__':
    models = sys.argv[1:] or ['nmda', 'stp']
    ses, hist = readout.load_data()
    Parallel(n_jobs=4)(delayed(fit.fit)(m, ses, hist, paths.cache('fits', f'{m}_seed{s}'), seed=s)
                       for m in models for s in (0, 1, 2))
