"""On and Off states (Panichello et al. 2024), shared by the recordings, the rate model and the spiking network.

A decoding z score (confidence of the decoder for the cued location, in standard deviations of a label-shuffle
null) is labelled trial by trial, at 10 ms steps:
    On   runs of z > 1.96 whose summed z (cluster mass) exceeds a threshold taken from the null;
    Off  runs of z < 0.84 lasting at least 5 time points (50 ms).
States are labelled 0-1,400 ms after cue onset; fractions of time On/Off are taken over the delay, 500-1,400 ms.
The decoding strength mu_on is the upper mean of a two-component Gaussian mixture fitted to the delay z values.
"""
import numpy as np
from sklearn.mixture import GaussianMixture

STEP = 10                  # ms between time points
ON_Z, OFF_Z = 1.96, 0.84   # OFF_Z of the recordings is 0.8416 (recordings.OFF_Z); see BUGS.md
MIN_OFF = 5                # time points
STATE_WIN, DELAY = (0, 1400), (500, 1400)


def runs(mask):
    """(start, stop) index pairs of the runs of True in a 1-d boolean array."""
    a = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(a == 1)[0], np.where(a == -1)[0]))


def label(z, thr, valid=None, off_z=OFF_Z):
    """States of each trial (row of z): +1 On, -1 Off, 0 neither. thr: On cluster-mass threshold; valid: time points
    that can be labelled (default all)."""
    valid = np.ones(z.shape[-1], bool) if valid is None else valid
    state = np.zeros(z.shape, int)
    for k, row in enumerate(z):
        for a, b in runs((row > ON_Z) & valid):
            if row[a:b].sum() > thr:
                state[k, a:b] = 1
        for a, b in runs((row < off_z) & valid):
            if b - a >= MIN_OFF:
                state[k, a:b] = -1
    return state


def null_threshold(z_null, valid=None):
    """95th percentile of the masses of all runs of z > 1.96 in null z traces (rows), within `valid`."""
    valid = np.ones(z_null.shape[-1], bool) if valid is None else valid
    masses = [row[a:b].sum() for row in z_null.reshape(-1, z_null.shape[-1]) for a, b in runs((row > ON_Z) & valid)]
    return float(np.percentile(masses, 95)) if masses else np.inf


def stats(state, delay):
    """On/Off states per trial, their lengths (mean, median, counts in 20 ms bins with a last bin for >= 600 ms) and
    the fraction of the delay (boolean mask over time points) in each state."""
    out = {}
    for name, lab in (('on', 1), ('off', -1)):
        d = [[(b - a) * STEP for a, b in runs(s == lab)] for s in state]
        dur = np.concatenate(d + [[]])
        out.update({f'n_{name}': np.mean([len(x) for x in d]), f'frac_{name}': (state[:, delay] == lab).mean(),
                    f'{name}_mean': dur.mean() if len(dur) else 0.0, f'{name}_median': np.median(dur) if len(dur) else 0.0,
                    f'h_{name}': np.bincount(np.minimum(dur.astype(int) // 20, 30), minlength=31)})
    return out


def trial_stats(state):
    """Per trial, on states restricted to the delay: fraction Off, number of Off periods, mean Off length (ms, NaN
    without Off periods) and fraction On."""
    offs = [[(b - a) * STEP for a, b in runs(s == -1)] for s in state]
    return dict(frac_off=(state == -1).mean(1), n_off=np.array([len(o) for o in offs], float),
                off_len_ms=np.array([np.mean(o) if o else np.nan for o in offs]), frac_on=(state == 1).mean(1))


def mu_on(z, rng, n_max=40000):
    """Decoding strength: upper mean of a 2-Gaussian mixture of the z values (at most n_max of them, drawn with rng)."""
    v = z.ravel()
    v = rng.choice(v, n_max, replace=False) if len(v) > n_max else v
    return float(GaussianMixture(2, random_state=0).fit(v[:, None]).means_.max())
