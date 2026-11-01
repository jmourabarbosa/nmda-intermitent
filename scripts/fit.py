"""CMA-ES fit of the rate model to the all-units read-out (nmdaint/fit.py), 3 seeds per model; ~2 h per run.
The figures use the published parameters (ratemodel.FIT), from the original fits (archive best losses 8.99 NMDA,
9.23 STP, with an earlier version of the loss or data table; 9.43 and 9.05 with the current one). Refits with this
code reach 9.10-9.48 (NMDA) and 8.96 (STP seed 0) with slightly different parameters; STP seeds 1 and 2 return the
archived parameters.

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
