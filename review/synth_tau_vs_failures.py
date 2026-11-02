"""Fig. 3c,d: in a world where Off states never cause errors (STP model, memory lost at a random time in a fraction F
of trials, independent of the dips), the paper's tau fit returns a value set by F, not by any trace time constant.
usage: python review/synth_tau_vs_failures.py   (~10 min on 4 cores)
"""
import os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(__file__))
import common
import synth_behaviour as sb
from nmdaint import ratemodel, readout


def run(F, seed):
    sb.F_ERR = F
    name, df, held = sb.world('stp_random_loss', GAINS, seed=seed)
    fine, Hf = common.sweep_curves()
    r = common.tau_fits(df, fine, Hf, n_boot=100)
    return dict(F=F, seed=seed, err=1 - df.correct.mean(), **r)


ses, _ = readout.load_data()
GAINS = {'stp': readout.evaluate(ratemodel.params('stp'), ses, n_trials=1056, seed=0, sets=('all',))['sets']['all']['gain']}
if __name__ == '__main__':
    res = pd.DataFrame(Parallel(n_jobs=4)(delayed(run)(F, s) for F in (0.01, 0.02, 0.04, 0.08, 0.16) for s in (1, 2)))
    res.to_csv('review/out/synth_tau_vs_failures.csv', index=False)
    print(res.round(3).to_string(index=False))
