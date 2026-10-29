"""Memory held against time Off for each NMDA time constant (Fig. 3c, d): 17 tau_N x 30,000 trials of the fitted
model with its own dips, read out like the data (memory.tau_sweep); about 20 min on 4 cores. Needs
results/recordings/. Writes results/ratemodel/tau_sweep.csv (trials and memory held per bin of the fraction of the
delay Off).

usage: python scripts/tau_sweep.py
"""
import pandas as pd
from joblib import Parallel, delayed

from nmdaint import behaviour, memory, paths, ratemodel, readout


def main():
    ses, _ = readout.load_data()
    p = ratemodel.params('nmda')
    gain = memory.reference_gain(p, ses)
    sweep = Parallel(n_jobs=4)(delayed(memory.tau_sweep)(p, tau, gain) for tau in memory.TAUS)
    rows = []
    for tau, (frac, held) in zip(memory.TAUS, sweep):
        b = behaviour.bins(frac)
        rows += [dict(tau_ms=tau, bin=j, n=int((b == j).sum()), held=int(held[b == j].sum())) for j in range(behaviour.NB)]
    pd.DataFrame(rows).to_csv(paths.results('ratemodel', 'tau_sweep.csv'), index=False)


if __name__ == '__main__':
    main()
