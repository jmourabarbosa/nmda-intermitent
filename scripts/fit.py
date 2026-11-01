"""CMA-ES fit of the rate model to the all-units read-out (nmdaint/fit.py), 3 seeds per model; ~2 h per run.
The figures use the published parameters (ratemodel.FIT), from the original fits (archive best losses 8.99 NMDA,
9.23 STP, with an earlier version of the loss or data table). The original NMDA fit also let tau_N vary (50-150 ms);
it settled at 150 ms, the value now fixed. Refits with this code (seeds 0-2) reach losses of 8.96, 8.74 and 8.69 for
the NMDA model (published parameters on the same trials: 9.38, 9.30, 9.01), with w_s at 1.29-1.30 even when allowed
down to 0.9, and deeper dips (0.74-0.83); on new trials they fit the all-units read-out no better (9.4-10.1 against
9.8). STP: seed 0 reaches 8.96 with slightly different parameters; seeds 1 and 2 return the archived parameters.

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
