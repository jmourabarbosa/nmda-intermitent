"""Everything computed from the recordings (about 4 h on 4 cores; sessions are cached in cache/recordings/).

results/recordings/
  readout_per_session.csv   On/Off statistics per session, decoded population and random draw (Figs. 2, S2-S4)
  readout_histograms.csv    state-length counts pooled over sessions, per decoded population (model fit)
  per_trial.csv             per-trial Off statistics and cue of correct and error trials, all units (Fig. 3b-d)
  precue_acf.csv            autocorrelation of the pre-cue z of each decoded population, random draws averaged (Fig. S1b)
  example_delay_z.npz       delay z of the session closest to the mean decoding strength, with its mixture fit (Fig. S1c)
  procrustes.csv, pca_plane.csv  Procrustes similarity and the PCA plane of an example session (Fig. 4d, e)

usage: python scripts/recordings.py [--jobs 4]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture

from nmdaint import onoff, paths, population, recordings as REC

EXAMPLE_PCA = '221024'


def acf(z, max_lag=30):
    x = z - z.mean()
    return np.array([(x[:, :x.shape[1] - k] * x[:, k:]).mean() / (x * x).mean() for k in range(max_lag + 1)])


def set_label(key):
    """The read-out's name of a decoded population (readout.SETS); None for the sizes the paper does not use."""
    kind, k, _ = key
    if kind == 'all':
        return 'all'
    return f"{'top' if kind == 'top' else 'rand'} {k}" if k in (5, 20, 50) else None


def run_session(path):
    f = paths.cache('recordings', os.path.basename(path).replace('.mat', '.npz'))
    if os.path.exists(f):
        return dict(np.load(f, allow_pickle=True))['res'].item()
    S = REC.Session(path)
    pre = S.t < 0
    res = dict(name=S.name, rows=[], hists={}, acf={})
    for key, units in S.sets.items():
        conf = S.confidence(units)
        z, st = S.states(conf)
        row, h = S.readout_row(key, conf, z, st)
        res['rows'].append(row)
        res['hists'][key] = h
        lab = set_label(key)
        if lab:
            res['acf'].setdefault(lab, []).append(acf(z[:, pre]))
        if key[0] == 'all':
            res['z_delay'] = z[:, S.delay].astype(np.float32)
            st_all = st
    res['acf'] = {lab: np.mean(v, 0) for lab, v in res['acf'].items()}      # random draws averaged
    # per trial: correct trials (leave-one-out decoders) and error trials (decoders of all correct trials)
    _, st_err = S.states(S.error_confidence())
    trials = []
    for idx, s, cue, correct in ((S.trial_index, st_all, S.cue, 1), (S.error_index, st_err, S.cue_err, 0)):
        trials.append(pd.DataFrame(dict(session=S.name, trial_index=idx, cue=cue, correct=correct,
                                        **onoff.trial_stats(s, S.delay[S.sw]))))
    res['trials'] = pd.concat(trials).sort_values('trial_index')
    # population geometry: states from all units of one half of the units, activity of the other half
    H1, H2 = population.unit_halves(S.n_units)
    _, st = S.states(S.confidence((H1, H1)))
    res['procrustes'], res['pca'] = population.analyse(S.Z[:, S.delay][:, :, H2], S.Z[:, S.t < 0][:, :, H2], S.cue,
                                                       st[:, S.delay[S.sw]], plane=S.name == EXAMPLE_PCA)
    res['cue_ang'] = S.cue_ang
    np.savez(f, res=np.array(res, dtype=object))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(paths.DATA, '*.mat')), key=lambda f: -os.path.getsize(f))
    res = sorted(Parallel(n_jobs=a.jobs, verbose=5)(delayed(run_session)(f) for f in files), key=lambda r: r['name'])

    rows = pd.DataFrame([r for s in res for r in s['rows']])
    rows.loc[rows.set == 'all', 'n'] = 0
    rows.to_csv(paths.results('recordings', 'readout_per_session.csv'), index=False)
    hist = []
    for key in sorted({(k[0], 0 if k[0] == 'all' else k[1]) for s in res for k in s['hists']}):
        for st in ('on', 'off'):
            c = sum(h[st] for s in res for k, h in s['hists'].items() if (k[0], 0 if k[0] == 'all' else k[1]) == key)
            hist += [dict(set=key[0], n=key[1], state=st, bin_lo_ms=20 * b, count=int(c[b])) for b in range(31)]
    pd.DataFrame(hist).to_csv(paths.results('recordings', 'readout_histograms.csv'), index=False)

    trials = pd.concat([s['trials'] for s in res])
    trials['monkey'] = trials.session.map(REC.monkey)
    trials.to_csv(paths.results('recordings', 'per_trial.csv'), index=False)
    pd.DataFrame([dict(session=s['name'], set=lab, lag_ms=10 * k, acf=v[k]) for s in res for lab, v in s['acf'].items()
                  for k in range(len(v))]).to_csv(paths.results('recordings', 'precue_acf.csv'), index=False)
    mu = rows[rows.set == 'all'].set_index('session').mu_on
    ex = min(res, key=lambda s: abs(mu[s['name']] - mu.mean()))
    z = ex['z_delay'].ravel()
    gm = GaussianMixture(2, random_state=0).fit(np.random.default_rng(1).choice(z, min(40000, len(z)), replace=False)[:, None])
    np.savez(paths.results('recordings', 'example_delay_z.npz'), session=ex['name'], z=z, mu_on_mean=mu.mean(),
             gmm_weights=gm.weights_, gmm_means=gm.means_.ravel(), gmm_sds=np.sqrt(gm.covariances_.ravel()))

    pd.DataFrame([dict(session=s['name'], **r) for s in res for r in s['procrustes']]).to_csv(
        paths.results('recordings', 'procrustes.csv'), index=False)
    ex = next(s for s in res if s['pca'])
    pd.DataFrame(ex['pca']).assign(session=ex['name'], angle_deg=np.degrees(ex['cue_ang'])).to_csv(
        paths.results('recordings', 'pca_plane.csv'), index=False)


if __name__ == '__main__':
    main()
