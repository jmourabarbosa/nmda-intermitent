"""Read-out of the spiking network like the recordings.

Recorded cells: spike counts in 100 ms windows every 10 ms. For each cue and its opposite cue (4 pairs) and each
time point, decoder.quadratic_loo (lam = 10,000, on counts) gives leave-one-out confidences of the true cue, with
the true labels and 30 label shuffles; the true confidence is z-scored against the mean and SD of the shuffles over
trials and shuffles of the pair at each time point (BUGS.md: the recordings use each trial's own shuffles). On
threshold: 95th percentile of the cluster masses of the shuffles' z (onoff.null_threshold); Off below 0.84.
Correct trials only (memory held).
"""
import numpy as np

from . import decoder, onoff

LAM = 1e4
STEP, WIN = 10, 100
N_SHUF = 30


def windows(counts, t10, t_from=-50, t_to=1450):
    """(trials, windows, cells) counts in 100 ms windows every 10 ms, and window centres (ms from cue onset)."""
    k = WIN // STEP
    c = np.cumsum(np.concatenate([np.zeros_like(counts[:, :1], dtype=np.int32), counts.astype(np.int32)], 1), 1)
    X = (c[:, k:] - c[:, :-k]).astype(np.float64)
    tz = t10[:X.shape[1]] + WIN / 2
    keep = (tz >= t_from) & (tz <= t_to)
    return X[:, keep], tz[keep]


def zscore(X):
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 0, sd, 1)


def decode(X, cue, rng, units):
    """z of the decoder for the cued location (trials, times) and the null z (shuffles, trials, times); X: window
    counts (trials, times, cells), cue: index 0-7, units: decoded cells."""
    z = np.full(X.shape[:2], np.nan)
    zn = np.full((N_SHUF,) + X.shape[:2], np.nan)
    for c in range(4):
        idx = np.where((cue == c) | (cue == c + 4))[0]
        y = (cue[idx] == c).astype(int)
        Y = np.vstack([y[None], [rng.permutation(y) for _ in range(N_SHUF)]]).astype(float)
        p1 = decoder.quadratic_loo(X[idx][:, :, units], Y, LAM)
        o = np.where(y[None, :, None] == 1, p1, 1 - p1)                   # confidence of the true cue
        mu, sd = o[1:].mean((0, 1)), np.maximum(o[1:].std((0, 1)), 1e-300)
        z[idx], zn[:, idx] = (o[0] - mu) / sd, (o[1:] - mu) / sd
    return z, zn


def states(X, cue, rng, units, tz, valid=None, halves=None):
    """z and On/Off states (trials, times) of the decoder on `units`, or with halves = (half, units of half 0, units
    of half 1) different cells for the trials of each half."""
    valid = (tz >= 0) & (tz <= 1400) if valid is None else valid
    if halves is None:
        z, zn = decode(X, cue, rng, units)
    else:
        half, z, zns = halves[0], np.zeros(X.shape[:2]), []
        for h in (0, 1):
            zh, zn = decode(X, cue, rng, halves[1 + h])
            z[half == h] = zh[half == h]
            zns.append(zn[:, half == h])
        zn = np.concatenate(zns, 1)
    return z, onoff.label(np.nan_to_num(z), onoff.null_threshold(zn, valid), valid)


def read_session(d, seed):
    """All-units read-out of a simulated session d (spiking.session): z, states, window centres, and statistics."""
    rng = np.random.default_rng(seed)
    X, tz = windows(d['counts'][d['held']], d['t10'])
    cue = d['cue'][d['held']]
    for c in range(8):                 # trial halves (not used by the all-units decoder; they fix the random sequence)
        rng.permutation(np.where(cue == c)[0])
    z, st = states(X, cue, rng, np.where(X.sum((0, 1)) > 0)[0], tz)
    delay = (tz >= 500) & (tz <= 1400)
    return dict(z=z, state=st, tz=tz, **onoff.stats(st, delay), mu_on=onoff.mu_on(np.nan_to_num(z)[:, delay], rng))


def tuning(X, tz, cue, rng, n_rep=10):
    """Cross-validated memory tuning in On and Off states (as Panichello et al. 2024, Fig. 4d). Cells are split at
    random into halves; states are labelled by the decoder of one half; for each held-out cell, its z-scored count
    is averaged per cue over the delay time points in On, and in Off, states, aligned to the cell's preferred cue
    (largest On + Off value), minus the same with shuffled cue labels (aligned to its own maximum); halves swapped;
    n_rep random splits, averaged per cell. Returns {'on', 'off'}: (cells, 8 cue offsets from the preferred cue)."""
    Z = zscore(X)
    valid, delay = (tz >= 0) & (tz <= 1400), (tz >= 500) & (tz <= 1400)
    n_cells = X.shape[2]
    curves, ids = {'on': [], 'off': []}, []
    for _ in range(n_rep):
        perm = rng.permutation(n_cells)
        for lab, test in ((perm[:n_cells // 2], perm[n_cells // 2:]), (perm[n_cells // 2:], perm[:n_cells // 2])):
            z, zn = decode(X, cue, rng, lab)
            st = onoff.label(np.nan_to_num(z), onoff.null_threshold(zn, valid), valid)[:, delay]
            zt = Z[:, delay][:, :, test]
            aligned = []
            for cu in (cue, rng.permutation(cue)):
                tun = {nm: np.array([(zt[cu == c] * (st == v)[cu == c][:, :, None]).sum((0, 1)) /
                                     np.maximum((st == v)[cu == c].sum(), 1) for c in range(8)])
                       for nm, v in (('on', 1), ('off', -1))}
                pref = np.argmax(tun['on'] + tun['off'], 0)
                aligned.append({nm: np.array([np.roll(tun[nm][:, j], -pref[j]) for j in range(len(test))]) for nm in tun})
            for nm in curves:
                curves[nm].append(aligned[0][nm] - aligned[1][nm])
            ids.append(test)
    ids = np.concatenate(ids)
    return {nm: np.array([np.concatenate(v)[ids == j].mean(0) for j in range(n_cells)]) for nm, v in curves.items()}
