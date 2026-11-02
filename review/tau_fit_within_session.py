"""Fig. 3c,d: is the pooled tau fit driven by between-session differences (Simpson's paradox)?

The paper fits P(correct) = p0 [P(held|Off bin; tau) + G (1 - P(held))] to all trials pooled over sessions, with one
p0. Sessions differ in both error rate (monkey H 28%, J 2%) and Off time (decoding strength), so a pooled slope can
come from between-session covariation. Here: (1) one p0 per session (only within-session variation informs tau),
(2) per monkey, (3) the between-session correlation of error rate and mean Off fraction.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from nmdaint import behaviour, paths

trials = pd.read_csv(paths.results('recordings', 'per_trial.csv'))
sweep = pd.read_csv(paths.results('ratemodel', 'tau_sweep.csv'))
taus = np.sort(sweep.tau_ms.unique())
cnt = lambda c: sweep.pivot(index='tau_ms', columns='bin', values=c).loc[taus].values
fine, Hf = behaviour.held_curves(taus, cnt('held'), cnt('n'))
P0 = behaviour.P0


def ll_session_p0(ks, ns):
    """log-lik over tau with a free p0 per session (profiled): (n_tau,)"""
    q = Hf + behaviour.G * (1 - Hf)                                   # (tau, bin)
    tot = np.zeros(len(fine))
    for k, n in zip(ks, ns):
        p = np.clip(P0[None, :, None] * q[:, None, :], 1e-9, 1 - 1e-9)
        tot += (k * np.log(p) + (n - k) * np.log(1 - p)).sum(-1).max(1)
    return tot


def counts(d):
    b, c, s = behaviour.bins(d.frac_off.values), d.correct.values, d.session.values
    names = np.unique(s)
    return (np.array([np.bincount(b[s == x], weights=c[s == x], minlength=behaviour.NB) for x in names]),
            np.array([np.bincount(b[s == x], minlength=behaviour.NB) for x in names]))


def ll_flat(ks, ns):
    k, n = ks.sum(1), ns.sum(1)
    p = np.clip(k / n, 1e-9, 1 - 1e-9)
    return float((k * np.log(p) + (n - k) * np.log(1 - p)).sum())


def ci_from_ll(ll):
    """profile-likelihood 95% interval (drop of 1.92)"""
    ok = fine[ll >= ll.max() - 1.92]
    return ok.min(), ok.max()


out = []
for name, d in [('all', trials)] + [(f'monkey {m}', g) for m, g in trials.groupby('monkey')]:
    ks, ns = counts(d)
    i, _, llp = behaviour.best_tau(ks.sum(0), ns.sum(0), Hf)
    llp = llp.max(1)
    lls = ll_session_p0(ks, ns)
    out.append(dict(subset=name, n=len(d), err=round(1 - d.correct.mean(), 3),
                    tau_pooled=round(fine[i]), pooled_ci=tuple(np.round(ci_from_ll(llp))),
                    tau_session_p0=round(fine[np.argmax(lls)]), session_p0_ci=tuple(np.round(ci_from_ll(lls))),
                    lr_vs_flat=round(2 * (lls.max() - ll_flat(ks, ns)), 1)))   # flat = P(correct) constant per session
print(pd.DataFrame(out).to_string(index=False))
print('note: the sweep is not monotonic in tau: P(held) ~ 1 for tau = 200-750 ms but the model fails (P(held) ~ 0.67 in every'
      ' bin) at tau = 1,500 ms, so the curve family contains no working long-trace model; lr_vs_flat: session-p0 fit vs constant')

g = trials.groupby('session').agg(err=('correct', lambda c: 1 - c.mean()), frac_off=('frac_off', 'mean'), monkey=('monkey', 'first'))
r = spearmanr(g.err, g.frac_off)
print(f'\nbetween sessions: error rate vs mean Off fraction, Spearman rho = {r.statistic:.2f}, p = {r.pvalue:.3f}')
print(g.groupby('monkey')[['err', 'frac_off']].mean().round(3))
# within-session centred P(correct) curve
trials['b'] = behaviour.bins(trials.frac_off.values)
trials['c_dm'] = trials.correct - trials.groupby('session').correct.transform('mean')
print('\nwithin-session (demeaned) P(correct) per Off bin:', trials.groupby('b').c_dm.mean().round(3).values)
print('pooled P(correct) per Off bin:               ', trials.groupby('b').correct.mean().round(3).values)
