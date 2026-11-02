"""Shared helpers of the review scripts."""
import numpy as np, pandas as pd
from nmdaint import behaviour, onoff, paths, readout

P0 = behaviour.P0


def sweep_curves(path=None):
    sweep = pd.read_csv(path or paths.results('ratemodel', 'tau_sweep.csv'))
    taus = np.sort(sweep.tau_ms.unique())
    cnt = lambda c: sweep.pivot(index='tau_ms', columns='bin', values=c).loc[taus].values
    return behaviour.held_curves(taus, cnt('held'), cnt('n'))


def counts(d):
    b, c, s = behaviour.bins(d.frac_off.values), d.correct.values, d.session.values
    names = np.unique(s)
    return (np.array([np.bincount(b[s == x], weights=c[s == x], minlength=behaviour.NB) for x in names]),
            np.array([np.bincount(b[s == x], minlength=behaviour.NB) for x in names]))


def ll_session_p0(ks, ns, Hf):
    """log-likelihood over the tau grid with one free p0 per session (profiled)."""
    q = Hf + behaviour.G * (1 - Hf)
    tot = np.zeros(len(Hf))
    for k, n in zip(ks, ns):
        p = np.clip(P0[None, :, None] * q[:, None, :], 1e-9, 1 - 1e-9)
        tot += (k * np.log(p) + (n - k) * np.log(1 - p)).sum(-1).max(1)
    return tot


def ll_flat_session(ks, ns):
    """log-likelihood of P(correct) constant within each session (no Off dependence)."""
    k, n = ks.sum(1), ns.sum(1)
    p = np.clip(k / n, 1e-9, 1 - 1e-9)
    return float((k * np.log(p) + (n - k) * np.log(1 - p)).sum())


def tau_fits(d, fine, Hf, n_boot=200, seed=0):
    """Paper's pooled fit (with bootstrap over sessions), the session-p0 fit, and the LR against a flat model."""
    ks, ns = counts(d)
    i, j, ll = behaviour.best_tau(ks.sum(0), ns.sum(0), Hf)
    rng = np.random.default_rng(seed)
    boot = [fine[behaviour.best_tau(ks[ix].sum(0), ns[ix].sum(0), Hf)[0]]
            for ix in (rng.integers(0, len(ks), len(ks)) for _ in range(n_boot))]
    lls = ll_session_p0(ks, ns, Hf)
    return dict(tau_pooled=fine[i], p0=P0[j], boot_lo=np.percentile(boot, 2.5), boot_hi=np.percentile(boot, 97.5),
                tau_session_p0=fine[np.argmax(lls)], lr_vs_flat=2 * (lls.max() - ll_flat_session(ks, ns)))


def label_trials(dw, tz, gain, rho, rng):
    """Paper's read-out of the rate model: z = c dr + eta, states, per-trial delay statistics."""
    eta = readout.noise(dw.shape, rho, rng)
    thr = readout.null_threshold(dw.shape, rho, rng)
    st = onoff.label(gain * dw + eta, thr, (tz >= 0) & (tz <= 1400))
    return gain * dw + eta, st, onoff.trial_stats(st[:, (tz >= 500) & (tz <= 1400)])
