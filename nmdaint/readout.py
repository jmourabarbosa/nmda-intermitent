"""Read-out of the rate model like the recordings.

The decoder's z score is modelled as z(t) = c dr(t) + eta(t):
  dr    cued minus uncued pool rate in 100 ms windows every 10 ms, as the recordings' rates;
  c     read-out gain, not fitted: for each decoded population it is set so that the model's decoding strength
        (onoff.mu_on of the delay z) equals the data's mean over sessions;
  eta   unit-variance noise with the autocorrelation of overlapping 100 ms windows plus a constant per-trial offset
        holding a share RHO of the variance (the data's pre-cue z autocorrelation at 200 ms; BUGS.md).
States are labelled as in the recordings (onoff.label); the On threshold comes from pure noise eta.

The seven decoded populations: the 5, 20 and 50 most cue-selective neurons, 5, 20 and 50 random neurons, and all
neurons. The model is fitted to 'all' only (fit.py); the others are predictions.
"""
import numpy as np
import pandas as pd

from . import onoff, paths, ratemodel

STEP, WIN = 10, 100
SETS = {'top 5': ('top', 5), 'top 20': ('top', 20), 'top 50': ('top', 50), 'rand 5': ('random', 5),
        'rand 20': ('random', 20), 'rand 50': ('random', 50), 'all': ('all', None)}
RHO = {'top': 0.09, 'random': 0.11, 'all': 0.35}
GAIN_GRID = 10 ** np.linspace(-2.6, -0.3, 10)
# fit loss: (data column, model key, scale) of six statistics, and the weight of the length-distribution term
LOSS_STATS = (('off_per_trial', 'n_off', 0.3), ('off_mean_ms', 'off_mean', 20.0), ('frac_off', 'frac_off', 0.05),
              ('on_per_trial', 'n_on', 0.3), ('on_mean_ms', 'on_mean', 20.0), ('frac_on', 'frac_on', 0.05))
W_HIST = 50


def load_data():
    """Recordings, per decoded population: {set: {column: values over sessions}} (random draws averaged within each
    session) and the pooled state-length counts {(set, 'on'/'off'): 20 ms bins from 0 to 600 ms, then >= 600 ms}."""
    rows = pd.read_csv(paths.results('recordings', 'readout_per_session.csv'))
    hrows = pd.read_csv(paths.results('recordings', 'readout_histograms.csv'))
    ses, hist = {}, {}
    for lab, (s, n) in SETS.items():
        sel = rows[(rows.set == s) & ((rows.n == n) if n else True)]
        cols = [c for c in sel.columns if c not in ('session', 'set', 'n', 'draw')]
        ses[lab] = {c: np.array([np.nanmean(g[c].values) for _, g in sel.groupby('session', sort=True)]) for c in cols}
        for st in ('on', 'off'):
            h = hrows[(hrows.set == s) & ((hrows.n == n) if n else True) & (hrows.state == st)]
            hist[lab, st] = np.bincount(h.bin_lo_ms.values // 20, weights=h['count'].values, minlength=31)
            hist[lab, st][29] -= hist[lab, st][30]     # the 580 ms bin also holds the states clipped at 600 ms
    return ses, hist


def rate_diff(sim, t_from=-100):
    """Cued minus uncued rate in 100 ms windows every 10 ms, and the window centres (from t_from ms)."""
    c = np.cumsum(np.c_[np.zeros(len(sim['rc'])), sim['rc'] - sim['ru']], axis=1)
    dw = ((c[:, WIN:] - c[:, :-WIN]) / WIN)[:, ::STEP]
    tz = sim['t'][0] + np.arange(dw.shape[1]) * STEP + WIN / 2
    return dw[:, tz >= t_from], tz[tz >= t_from]


def noise(shape, rho, rng):
    """Read-out noise eta: running sums of white noise over 100 ms windows (variance 1 - rho) plus a per-trial
    offset (variance rho)."""
    k = WIN // STEP
    c = np.cumsum(np.c_[np.zeros(shape[0]), rng.standard_normal((shape[0], shape[1] + k - 1))], axis=1)
    return np.sqrt(1 - rho) * (c[:, k:] - c[:, :-k]) / np.sqrt(k) + np.sqrt(rho) * rng.standard_normal((shape[0], 1))


def null_threshold(shape, rho, rng, n_rep=3):
    """On threshold: 95th percentile of the cluster masses of pure noise eta."""
    return onoff.null_threshold(np.concatenate([noise(shape, rho, rng) for _ in range(n_rep)]))


def gain_curve(dw, eta, delay, rng, grid=GAIN_GRID):
    """mu_on of the delay z = c dw + eta for each gain c of the grid (made non-decreasing)."""
    return np.maximum.accumulate([onoff.mu_on((c * dw + eta)[:, delay], rng) for c in grid])


def gain_for(target, mus, grid=GAIN_GRID):
    """Gain at which the decoding strength equals target, interpolated in log gain (the last gain if out of range)."""
    return float(10 ** np.interp(target, mus, np.log10(grid)))


def evaluate(p, ses, n_trials=1056, seed=0, sets=tuple(SETS), sim=None, gains=None):
    """Simulate the model (or use sim) and read it out for each decoded population: memory correct, and per set the
    gain, z, states and their statistics. gains: fixed gain per set instead of calibrating to the data."""
    sim = ratemodel.simulate(p, n_trials, seed=seed * 100 + 1) if sim is None else sim
    dw, tz = rate_diff(sim)
    rng = np.random.default_rng(seed * 100 + 2)
    delay, valid = (tz >= 500) & (tz <= 1400), (tz >= 0) & (tz <= 1400)
    out = dict(mem=float(ratemodel.memory_correct(sim).mean()), sim=sim, tz=tz, sets={})
    cache = {}
    for lab in sets:
        rho = RHO[SETS[lab][0]]
        if rho not in cache:                  # populations of the same kind share the noise draw
            eta = noise(dw.shape, rho, rng)
            mus = gain_curve(dw, eta, delay, rng) if gains is None else None
            cache[rho] = eta, mus, null_threshold(dw.shape, rho, rng)
        eta, mus, thr = cache[rho]
        c = gain_for(ses[lab]['mu_on'].mean(), mus) if gains is None else gains[lab]
        state = onoff.label(c * dw + eta, thr, valid)
        out['sets'][lab] = dict(gain=c, z=c * dw + eta, state=state, **onoff.stats(state, delay))
    return out



def loss(r, d, h):
    """Fit loss of one decoded population: mean squared scaled error of the six statistics plus W_HIST x the summed
    squared Hellinger-type distance between the model's and the data's On and Off length distributions (states of
    40-580 ms). r: model statistics (onoff.stats), d: data columns over sessions, h: data length counts by state."""
    L = np.mean([((r[mk] - d[dk].mean()) / sc) ** 2 for dk, mk, sc in LOSS_STATS])
    for st in ('on', 'off'):
        p, q = r['h_' + st][2:], h[st][2:]
        L += W_HIST * np.sum((np.sqrt(p / max(p.sum(), 1)) - np.sqrt(q / q.sum())) ** 2)
    return L
