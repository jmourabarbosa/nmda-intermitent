"""Read-out of the rate model like the recordings.

The decoder's z score is modelled as z(t) = c dr(t) + eta(t):
  dr    cued minus uncued pool rate in 100 ms windows every 10 ms, as the recordings' rates;
  c     read-out gain, not fitted: for each decoded population it is set so that the model's decoding strength
        (onoff.mu_on of the delay z) equals the data's mean over sessions; reference() calibrates the gains that
        every analysis of the fitted models uses;
  eta   unit-variance noise with the autocorrelation of overlapping 100 ms windows plus a constant per-trial offset
        holding a share RHO of the variance, per decoded population (its pre-cue z autocorrelation at 200 ms).
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
GAIN_GRID = 10 ** np.linspace(-2.6, -0.3, 10)
REF_TRIALS, REF_SEED = 2112, 1                   # the reference calibration of the gains
# fit loss: (data column, model key, scale) of six statistics, and the weight of the length-distribution term
LOSS_STATS = (('off_per_trial', 'n_off', 0.3), ('off_mean_ms', 'off_mean', 20.0), ('frac_off', 'frac_off', 0.05),
              ('on_per_trial', 'n_on', 0.3), ('on_mean_ms', 'on_mean', 20.0), ('frac_on', 'frac_on', 0.05))
W_HIST = 50


def rho_from_data():
    """Share of the read-out noise that is constant within a trial, per decoded population: the recordings' pre-cue z
    autocorrelation at 200 ms (beyond the 100 ms windows; random draws averaged within each session), mean over
    sessions, to two decimals."""
    a = pd.read_csv(paths.results('recordings', 'precue_acf.csv'))
    m = a[a.lag_ms == 200].groupby('set').acf.mean()
    return {lab: float(round(m[lab], 2)) for lab in SETS}


RHO = rho_from_data()


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
    gain (and the calibration curve mus over GAIN_GRID), z, states and their statistics. gains: fixed gain per set
    instead of calibrating to the data."""
    sim = ratemodel.simulate(p, n_trials, seed=seed * 100 + 1) if sim is None else sim
    dw, tz = rate_diff(sim)
    rng = np.random.default_rng(seed * 100 + 2)
    valid, delay = onoff.windows(tz)
    out = dict(mem=float(ratemodel.memory_correct(sim).mean()), sim=sim, tz=tz, sets={})
    cache = {}
    # the 20-unit and all-units populations draw first: their noise stays the one drawn when every top and every
    # random population used the 20-unit rho
    for lab in sorted(sets, key=lambda l: l not in ('top 20', 'rand 20', 'all')):
        rho = RHO[lab]
        if rho not in cache:                  # populations with the same rho share the noise draw
            eta = noise(dw.shape, rho, rng)
            mus = gain_curve(dw, eta, delay, rng) if gains is None else None
            cache[rho] = eta, mus, null_threshold(dw.shape, rho, rng)
    for lab in sets:
        eta, mus, thr = cache[RHO[lab]]
        c = gain_for(ses[lab]['mu_on'].mean(), mus) if gains is None else gains[lab]
        state = onoff.label(c * dw + eta, thr, valid)
        out['sets'][lab] = dict(gain=c, mus=mus, z=c * dw + eta, state=state, **onoff.stats(state, delay))
    return out


def reference(p, ses):
    """The reference read-out of model p: REF_TRIALS trials, seed REF_SEED, all decoded populations. Its gains are the
    ones used for the model everywhere."""
    return evaluate(p, ses, n_trials=REF_TRIALS, seed=REF_SEED)



def gain_sweep(out, rho, factors, seed=0):
    """The simulated trials of an evaluate() output read out at the all-units gain x each factor, with noise share
    rho: one row per gain with the decoding strength (mu_on) and the On/Off statistics."""
    dw, tz = rate_diff(out['sim'])
    rng = np.random.default_rng(seed)
    valid, delay = onoff.windows(tz)
    eta, thr = noise(dw.shape, rho, rng), null_threshold(dw.shape, rho, rng)
    rows = []
    for f in factors:
        z = f * out['sets']['all']['gain'] * dw + eta
        st = onoff.stats(onoff.label(z, thr, valid), delay)
        rows.append(dict(factor=f, mu_on=onoff.mu_on(z[:, delay], rng), **{k: v for k, v in st.items() if k[:2] != 'h_'}))
    return pd.DataFrame(rows)


def summary(out, ses, hist):
    """One row per decoded population of an evaluate() output: memory, gain, fit loss and the On/Off statistics."""
    keys = ('n_off', 'off_median', 'off_mean', 'frac_off', 'n_on', 'on_median', 'on_mean', 'frac_on')
    return pd.DataFrame([dict(set=lab, memory=out['mem'], gain=r['gain'],
                              loss=loss(r, ses[lab], {s: hist[lab, s] for s in ('on', 'off')}), **{k: r[k] for k in keys})
                         for lab, r in out['sets'].items()]).set_index('set')


def loss(r, d, h):
    """Fit loss of one decoded population: mean squared scaled error of the six statistics plus W_HIST x the summed
    squared Hellinger-type distance between the model's and the data's On and Off length distributions (states of
    40-580 ms). r: model statistics (onoff.stats), d: data columns over sessions, h: data length counts by state."""
    L = np.mean([((r[mk] - d[dk].mean()) / sc) ** 2 for dk, mk, sc in LOSS_STATS])
    for st in ('on', 'off'):
        p, q = r['h_' + st][2:], h[st][2:]
        L += W_HIST * np.sum((np.sqrt(p / max(p.sum(), 1)) - np.sqrt(q / q.sum())) ** 2)
    return L
