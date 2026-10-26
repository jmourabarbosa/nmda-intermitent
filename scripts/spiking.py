"""Spiking-network results (Fig. 4); about 2 h on 4 cores (sessions are cached in cache/spiking/).

12 sessions of 480 trials (60 per cue; 1.5 s delay with dips from cue onset), 330 recorded cells, read out like the
recordings, and one session of long trials (1 s of delay with a constant common input, then 1 s with dips).
results/spiking/
  sessions.csv          memory and On/Off statistics of each session (all-units decoder)
  example_trial.npz     Fig. 4a: E raster, common input, decoder z and states of one long trial
  state_lengths.npz     Fig. 4b: Off and On length counts of sessions 0-3
  tuning.npz            Fig. 4c: On and Off tuning of the held-out cells of session 0
results/population/net_*.csv   Fig. 4d, e: PCA plane of session 0 and Procrustes similarity of sessions 0-9

usage: python scripts/spiking.py
"""
import os

import numpy as np
import pandas as pd

from nmdaint import onoff, paths, population, spiking as SP, spiking_readout as SR

N_SESSIONS, FIG4B_SESSIONS, POP_SESSIONS = 12, range(4), range(10)
LONG = dict(seed=50, quiet=1000.0, dips=1000.0, example_trial=27)   # Fig. 4a (example: cue 180 deg, memory held,
                                                                   # On through the constant second, 2-5 Off states)


def cached(name, make):
    f = paths.cache('spiking', name + '.npz')
    if not os.path.exists(f):
        np.savez_compressed(f, **make())
    return dict(np.load(f))


def population_session(d, s):
    """Procrustes similarity (and, for session 0, the PCA plane) of session d, labelled with all cells of one half."""
    c, cue = d['counts'][d['held']], d['cue'][d['held']]
    X, tz = SR.windows(c, d['t10'])
    Xp, _ = SR.windows(c, d['t10'], -400, -50)            # pre-cue: windows that end before cue onset
    H1, H2 = population.unit_halves(X.shape[2])
    _, st = SR.states(X, cue, np.random.default_rng(s + 1), H1, tz)
    dl = (tz >= 500) & (tz <= 1400)
    half = population.trial_halves(cue, np.random.default_rng(0))
    Zd, Zpre = SR.zscore(X)[:, dl][:, :, H2], SR.zscore(Xp)[:, :, H2]
    rows = [dict(session=s, period=p, space=sp, similarity=v[0], shuffle_mean=v[1], n_trials=v[2], shuffle_p95=v[3],
                 normalised=population.normalised(*v[:2]))
            for sp, dd in population.shape_similarity(Zd, Zpre, cue, st[:, dl], half).items() for p, v in dd.items()]
    plane = population.ring_and_centroids(*population.project(Zd, cue, half), cue, st[:, dl]) if s == 0 else None
    return rows, plane


def main():
    net = SP.Network(SP.params())
    ses = [cached(f'session_{s}', lambda: SP.session(net, 100 + s, s)) for s in range(N_SESSIONS)]
    reads = [SR.read_session(d, seed=s) for s, d in enumerate(ses)]
    pd.DataFrame([dict(session=s, memory=d['held'].mean(), **{k: r[k] for k in ('n_off', 'off_median', 'frac_off', 'n_on',
                  'on_median', 'frac_on', 'mu_on')}) for s, (d, r) in enumerate(zip(ses, reads))]
                 ).to_csv(paths.results('spiking', 'sessions.csv'), index=False)
    np.savez(paths.results('spiking', 'state_lengths.npz'),
             **{f'h_{st}': np.sum([reads[s][f'h_{st}'] for s in FIG4B_SESSIONS], 0) for st in ('on', 'off')},
             **{f'n_{st}': np.mean([reads[s][f'n_{st}'] for s in FIG4B_SESSIONS]) for st in ('on', 'off')})
    X, tz = SR.windows(ses[0]['counts'][ses[0]['held']], ses[0]['t10'])
    np.savez(paths.results('spiking', 'tuning.npz'), **SR.tuning(X, tz, ses[0]['cue'][ses[0]['held']], np.random.default_rng(0)))

    rows = []
    for s in POP_SESSIONS:
        r, plane = population_session(ses[s], s)
        rows += r
        if plane:
            pd.DataFrame(plane).assign(angle_deg=45 * np.arange(8)).to_csv(paths.results('population', 'net_pca_plane.csv'), index=False)
    pd.DataFrame(rows).to_csv(paths.results('population', 'net_procrustes.csv'), index=False)

    # Fig. 4a: long trials, the all-units decoder over the whole delay
    lnet = SP.Network(SP.params(t_delay=LONG['quiet'] + LONG['dips']))
    d = cached('long_session', lambda: SP.session(lnet, LONG['seed'], LONG['seed'], dip_free=LONG['quiet']))
    X, tz = SR.windows(d['counts'][d['held']], d['t10'], -300, np.inf)
    active = np.where(X.sum((0, 1)) > 0)[0]
    z, st = SR.states(X, d['cue'][d['held']], np.random.default_rng(0), active, tz, valid=tz >= 0)
    i = LONG['example_trial']
    b0 = i // SP.BATCH * SP.BATCH
    sim = SP.simulate(lnet, 45.0 * d['cue'][b0:b0 + SP.BATCH], seed=LONG['seed'] * 1000 + i // SP.BATCH, rec_idx=d['rec'],
                      raster=True, dip_free=LONG['quiet'])
    assert np.array_equal(sim['counts'], d['counts'][b0:b0 + SP.BATCH])
    t, cell = np.nonzero(sim['raster'][i - b0])
    r = int(np.flatnonzero(np.flatnonzero(d['held']) == i)[0])
    np.savez(paths.results('spiking', 'example_trial.npz'), spike_t=sim['t1'][t], spike_cell=cell, t1=sim['t1'],
             m_eff=sim['m_eff'][i - b0], cue_deg=45.0 * d['cue'][i], tz=tz, z=z[r], state=st[r])


if __name__ == '__main__':
    main()
