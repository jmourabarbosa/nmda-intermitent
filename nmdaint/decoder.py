"""Decoders of the cued location against the opposite location, one per time point.

ridge_logistic_loo: the recordings' decoder. L2 logistic regression, objective
    sum_i logloss_i + (lam / 2) ||w||^2,
with the intercept entering as a constant feature 1 that is penalised like a unit weight. Fitted by Newton's
method to max |gradient| < GRAD_TOL (in unit space when units + 1 <= trials, in trial space otherwise; same
objective). Leave-one-trial-out predictions by the approximate leave-one-out correction for ridge GLMs
(Rad & Maleki 2018):  eta_-i = eta_i + (p_i - y_i) h_i / (1 - v_i h_i),  h_i = x_i' (X' V X + lam I)^-1 x_i.
Several label vectors (the true labels and label shuffles) are fitted at once.

quadratic_loo: the spiking network's decoder. The same objective with the logistic loss replaced by its
second-order expansion at w = 0 (one Newton step from zero); at lam = 10,000 the fitted logits are tiny, so this
is close to the full fit, and its leave-one-out predictions are exact.
"""
import numpy as np
from scipy.special import expit

C_INT = 1.0        # intercept feature
MAX_NEWTON = 50
GRAD_TOL = 1e-9


def ridge_logistic_loo(X, Y, lam, chunk=16):
    """X: (T, n, d) features, Y: (S, n) label sets in {0, 1}. Returns leave-one-out logits (S, T, n) and the largest
    final |gradient|."""
    T, n, d = X.shape
    S = Y.shape[0]
    Xa = np.concatenate([X, np.full((T, n, 1), C_INT)], 2)
    D = d + 1
    primal = n * D * D + D ** 3 < n ** 3
    out = np.zeros((S, T, n))
    gmax = 0.0
    for t0 in range(0, T, chunk):
        Xt = Xa[t0:t0 + chunk]
        tc = Xt.shape[0]
        Yb = np.broadcast_to(Y[None], (tc, S, n))
        if primal:
            Xs = np.broadcast_to(Xt[:, None], (tc, S, n, D))
            XT = Xs.transpose(0, 1, 3, 2)
            w = np.zeros((tc, S, D, 1))
            I = lam * np.eye(D)
            for it in range(MAX_NEWTON + 1):
                eta = (Xs @ w)[..., 0]
                p = expit(eta)
                v = p * (1 - p)
                H = XT @ (Xs * v[..., None]) + I
                g = XT @ (p - Yb)[..., None] + lam * w
                gm = np.abs(g).max()
                if gm < GRAD_TOL or it == MAX_NEWTON:
                    break
                w -= np.linalg.solve(H, g)
            h = np.sum(XT * np.linalg.solve(H, XT), 2)                     # diag(X H^-1 X')
        else:
            K = Xt @ Xt.transpose(0, 2, 1)
            Kb = np.broadcast_to(K[:, None], (tc, S, n, n))
            XtT = Xt.transpose(0, 2, 1)[:, None]
            a = np.zeros((tc, S, n, 1))
            I = lam * np.eye(n)
            for it in range(MAX_NEWTON + 1):
                eta = (Kb @ a)[..., 0]
                p = expit(eta)
                v = p * (1 - p)
                M = v[..., None] * Kb + I                                  # V K + lam I
                r = (p - Yb)[..., None] + lam * a
                gm = np.abs(XtT @ r).max()                                 # gradient w.r.t. w = X' r
                if gm < GRAD_TOL or it == MAX_NEWTON:
                    break
                a -= np.linalg.solve(M, r)
            h = np.sum(Kb * np.linalg.inv(M).transpose(0, 1, 3, 2), 3)     # diag(K M^-1)
        gmax = max(gmax, gm)
        out[:, t0:t0 + chunk] = (eta + (p - Yb) * h / (1 - v * h)).transpose(1, 0, 2)
    return out, gmax


def ridge_logistic_weights(X, Y, lam, chunk=16):
    """Weights (T, S, d + 1) of the same decoder fitted on all trials (trial-space Newton), intercept last."""
    out = []
    for t0 in range(0, X.shape[0], chunk):
        Xc = X[t0:t0 + chunk]
        T, n, _ = Xc.shape
        Xa = np.concatenate([Xc, np.full((T, n, 1), C_INT)], 2)
        K = Xa @ Xa.transpose(0, 2, 1)
        Kb = np.broadcast_to(K[:, None], (T, Y.shape[0], n, n))
        a = np.zeros((T, Y.shape[0], n, 1))
        I = lam * np.eye(n)
        for it in range(MAX_NEWTON + 1):
            p = expit((Kb @ a)[..., 0])
            r = (p - Y[None])[..., None] + lam * a
            if np.abs(Xa.transpose(0, 2, 1)[:, None] @ r).max() < GRAD_TOL:
                break
            a -= np.linalg.solve((p * (1 - p))[..., None] * Kb + I, r)
        out.append((Xa.transpose(0, 2, 1)[:, None] @ a)[..., 0])
    return np.concatenate(out)


def quadratic_loo(X, Y, lam):
    """X: (n, T, d) features, Y: (S, n) label sets in {0, 1}. Returns leave-one-out probabilities of label 1,
    (S, n, T)."""
    n, n_t, _ = X.shape
    R = Y - 0.5
    o = np.empty((len(Y), n, n_t))
    for t in range(n_t):
        Z = np.c_[X[:, t], np.ones(n)]
        e, U = np.linalg.eigh(Z @ Z.T)
        g = np.maximum(e, 0) / (np.maximum(e, 0) / 4 + lam)      # hat matrix U diag(g) U'
        h = (U ** 2) @ g
        F = ((R @ U) * g) @ U.T                                  # fitted logits
        o[:, :, t] = expit(F + h / (1 - h / 4) * (F / 4 - R))
    return o
