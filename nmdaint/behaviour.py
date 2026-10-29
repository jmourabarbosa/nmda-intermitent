"""Off states and errors in the recordings.

Per trial (correct and error trials, all-units decoder): fraction of the delay Off, number of Off periods, mean Off
length and fraction On (onoff.trial_stats).

glmm_single: for each statistic, a logistic mixed model of errors,
    error ~ z(stat) + monkey + (1 + z(stat) || session),
with the statistic z-scored within each session (so only within-session variation is used), monkey as a fixed
effect and an uncorrelated random intercept and slope per session, fitted with the Laplace approximation (as
lme4::glmer, nAGQ = 1); trials with at least one Off period. Tests: likelihood ratio (dropping the fixed effect,
keeping its random slope) and Wald. Session weights: fixed effect + the session's random slope (conditional mode).

fit_tau: the NMDA time constant fitted to the pooled performance against the fraction of the delay Off. Model:
P(memory held | Off fraction) from simulations with each tau_N (memory.tau_sweep), in 10 bins of the Off
fraction, interpolated in log tau; P(correct) = p0 x [P(held) + G x (1 - P(held))], a lost memory being a guess
among the 8 cues (G = 1/8). Maximum likelihood over tau_N and p0; bootstrap over sessions.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, norm

STATS = ('frac_off', 'n_off', 'off_len_ms', 'frac_on')


# ------------------------------------------------------------------------------------------------ mixed model
def _session_ll(Xb, Z, y, sd, b):
    """Laplace log-likelihood of one session's trials and the mode of its random effects b (started from b)."""
    q, prec = Z.shape[1], 1.0 / sd ** 2
    b = b.copy()
    for _ in range(50):
        p = expit(Xb + Z @ b)
        step = np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + np.diag(prec), Z.T @ (y - p) - prec * b)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    eta = Xb + Z @ b
    p = expit(eta)
    H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(prec)
    ll_y = np.sum(y * eta - np.logaddexp(0, eta))
    ll_b = -0.5 * np.sum(b ** 2 * prec) - np.sum(np.log(sd)) - 0.5 * q * np.log(2 * np.pi)
    return ll_y + ll_b + 0.5 * q * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(H)[1], b, H


class GLMM:
    """Logistic GLMM, y ~ Bernoulli(sigmoid(X beta + Z b_session)), b ~ N(0, diag(sd^2)); maximum Laplace
    likelihood over (beta, log sd) by L-BFGS; standard errors of beta from a numerical Hessian."""

    def __init__(self, X, Z, y, groups):
        self.X, self.Z, self.y = np.asarray(X, float), np.asarray(Z, float), np.asarray(y, float)
        self.idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]
        self.p, self.q = self.X.shape[1], self.Z.shape[1]
        self._b = [np.zeros(self.q) for _ in self.idx]

    def loglik(self, theta):
        Xb, sd = self.X @ theta[:self.p], np.exp(theta[self.p:])
        tot = 0.0
        for k, ix in enumerate(self.idx):
            ll, self._b[k], _ = _session_ll(Xb[ix], self.Z[ix], self.y[ix], sd, self._b[k])
            tot += ll
        return tot

    def fit(self):
        beta = np.zeros(self.p)           # start from ordinary logistic regression, random-effect SDs at 0.3
        for _ in range(30):
            pr = expit(self.X @ beta)
            beta += np.linalg.solve((self.X * (pr * (1 - pr))[:, None]).T @ self.X + 1e-8 * np.eye(self.p),
                                    self.X.T @ (self.y - pr))
        f = lambda th: -self.loglik(th)
        r = minimize(f, np.r_[beta, np.log(0.3) * np.ones(self.q)], method='L-BFGS-B',
                     bounds=[(None, None)] * self.p + [(np.log(1e-4), np.log(10))] * self.q,
                     options=dict(maxiter=500, ftol=1e-12, gtol=1e-6))
        self.theta, self.llf, self.converged = r.x, -r.fun, r.success
        h, th, E = 1e-4, self.theta.copy(), np.eye(len(self.theta)) * 1e-4
        H = np.zeros((self.p, self.p))
        for i in range(self.p):
            for j in range(i, self.p):
                H[i, j] = H[j, i] = (f(th + E[i] + E[j]) - f(th + E[i] - E[j]) - f(th - E[i] + E[j])
                                     + f(th - E[i] - E[j])) / (4 * h * h)
        self.beta, self.se = self.theta[:self.p], np.sqrt(np.diag(np.linalg.inv(H)))
        self.re_sd = np.exp(self.theta[self.p:])
        return self

    def session_effects(self):
        """Conditional modes of the random effects (n_sessions, q) and their conditional SDs."""
        Xb = self.X @ self.beta
        out = [_session_ll(Xb[ix], self.Z[ix], self.y[ix], self.re_sd, np.zeros(self.q)) for ix in self.idx]
        return (np.array([b for _, b, _ in out]),
                np.array([np.sqrt(np.diag(np.linalg.inv(H))) for _, _, H in out]))


def zscore_within(d, cols):
    d = d.copy()
    for c in cols:
        g = d.groupby('session')[c]
        d['z_' + c] = ((d[c] - g.transform('mean')) / g.transform('std').replace(0, np.nan)).fillna(0)
    return d


def glmm_single(trials):
    """trials: one row per trial (session, monkey, correct and STATS). Returns the fixed effect of each statistic
    (one row per statistic) and the session weights (one row per statistic and session)."""
    d = trials[trials.n_off > 0].copy()
    d['error'] = 1 - d['correct']
    d['session'] = d['session'].astype(str)
    d = zscore_within(d, STATS)
    monkeys = np.column_stack([(d.monkey == m).values.astype(float) for m in sorted(d.monkey.unique())[1:]] or
                              [np.zeros((len(d), 0))])          # monkey as a fixed effect (none with one monkey)
    fixed, weights = [], []
    for s in STATS:
        Z = np.column_stack([np.ones(len(d)), d['z_' + s].values])
        m = GLMM(np.column_stack([Z, monkeys]), Z, d.error.values, d.session.values).fit()
        m0 = GLMM(np.column_stack([Z[:, :1], monkeys]), Z, d.error.values, d.session.values).fit()
        b, se, lr = m.beta[1], m.se[1], 2 * (m.llf - m0.llf)
        fixed.append(dict(predictor=s, n_trials=len(d), n_error=int(d.error.sum()), logodds_per_sd=b, se=se,
                          odds_ratio_per_sd=np.exp(b), or_ci_lo=np.exp(b - 1.96 * se), or_ci_hi=np.exp(b + 1.96 * se),
                          p_wald=2 * norm.sf(abs(b / se)), lr_stat=lr, p_lrt=chi2.sf(max(lr, 0), 1),
                          re_sd_slope=m.re_sd[1], converged=m.converged and m0.converged))
        be, sd = m.session_effects()
        for j, ses in enumerate(np.unique(d.session.values)):
            w = b + be[j, 1]
            weights.append(dict(predictor=s, session=ses, session_logodds=w, session_or=np.exp(w),
                                session_or_lo=np.exp(w - 1.96 * sd[j, 1]), session_or_hi=np.exp(w + 1.96 * sd[j, 1])))
    return pd.DataFrame(fixed), pd.DataFrame(weights)


# ------------------------------------------------------------------------------------------------ tau fit
EDGES = np.r_[np.arange(0, 1.0, 0.1), 1.0001]     # bins of the fraction of the delay Off
NB = len(EDGES) - 1
P0 = np.linspace(0.3, 1.0, 701)
G = 1 / 8
N_BOOT = 1000


def bins(frac):
    return np.clip(np.digitize(frac, EDGES) - 1, 0, NB - 1)


def held_curves(taus, held, n):
    """P(memory held | Off bin) of the simulations (n_tau, NB; held of n trials per bin, add-half smoothed),
    interpolated in log tau on a fine grid of 300 taus. Returns the grid and the curves (300, NB)."""
    H = (np.asarray(held) + 0.5) / (np.asarray(n) + 1)
    fine = np.exp(np.linspace(np.log(taus[0]), np.log(taus[-1]), 300))
    return fine, np.array([np.interp(np.log(fine), np.log(taus), H[:, j]) for j in range(NB)]).T


def best_tau(k, n, Hf):
    """Indices of the best (tau, p0) and the log-likelihood over (tau, p0), for k correct of n trials per bin."""
    p = np.clip(P0[None, :, None] * (Hf + G * (1 - Hf))[:, None, :], 1e-9, 1 - 1e-9)
    ll = (k * np.log(p) + (n - k) * np.log(1 - p)).sum(-1)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    return i, j, ll


def fit_tau(trials, fine, Hf, seed=0):
    """Best tau_N and p0 for all trials pooled, the bootstrap distribution of tau_N over sessions, and the data per
    Off bin (correct k, trials n)."""
    b, c, s = bins(trials.frac_off.values), trials.correct.values, trials.session.values
    names = np.unique(s)
    ks = np.array([np.bincount(b[s == x], weights=c[s == x], minlength=NB) for x in names])
    ns = np.array([np.bincount(b[s == x], minlength=NB) for x in names])
    k, n = ks.sum(0), ns.sum(0)
    i, j, _ = best_tau(k, n, Hf)
    rng = np.random.default_rng(seed)
    boot = np.array([fine[best_tau(ks[idx].sum(0), ns[idx].sum(0), Hf)[0]]
                     for idx in (rng.integers(0, len(names), len(names)) for _ in range(N_BOOT))])
    return dict(tau=fine[i], p0=P0[j], held=Hf[i], boot=boot, k=k, n=n)


def wilson(k, n, z=1.96):
    p = k / np.maximum(n, 1)
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h
