"""Population geometry of delay activity in On and Off states, for the recordings and the spiking network alike.

Units are split at random into two halves (seed 1): On/Off states are labelled from all units of the first half;
everything below uses the activity of the other half (z-scored rates, 100 ms windows every 10 ms). Trials are
split into two halves stratified by cue (seed 0).

project: cross-validated PCA of the 8 cue-condition means of delay activity (500-1,400 ms): axes from one trial
half, projection of the other, and vice versa, the second half's axes aligned to the first's by an orthogonal
Procrustes rotation on the condition means; then each cue's mean position over its trials' On, and Off, points.

shape_similarity: Procrustes similarity (1 - disparity; translation, rotation and uniform scaling allowed) between
the ring of cue means of one trial half and the per-cue centroids of the other half's trial vectors during On, Off
and before the cue (mean activity over those time points), in all held-out units ('full') and in the 7 PCs of the
cue means ('pcs'), averaged over the two halves; chance from 50 shuffles of the cue labels within each half.
Normalised similarity = 1 - disparity / shuffle disparity (0 = chance, 1 = identical shape).
"""
import numpy as np
from scipy.spatial import procrustes as sp_procrustes

PERIODS = ('on', 'off', 'pre')


def unit_halves(n_units):
    """Labelling units (half 1) and analysed units (half 2)."""
    pu = np.random.default_rng(1).permutation(n_units)
    return np.sort(pu[:n_units // 2]), np.sort(pu[n_units // 2:])


def trial_halves(cue, rng):
    half = np.zeros(len(cue), int)
    for c in range(8):
        idx = rng.permutation(np.flatnonzero(cue == c))
        half[idx[len(idx) // 2:]] = 1
    return half


def _rotation(A, B):
    """Orthogonal R minimising ||B R - A||."""
    u, _, vt = np.linalg.svd(B.T @ A)
    return u @ vt


def project(Zd, cue, half):
    """Zd: delay activity (trials, times, units). Returns per-trial projections (trials, times, 2) on the first two
    PCs of the cue means of the other trial half, aligned across halves, and the aligned cue means (2, 8, 2)."""
    proj = np.zeros(Zd.shape[:2] + (2,))
    cms = []
    for h in (0, 1):
        C = np.stack([Zd[(half != h) & (cue == c)].mean((0, 1)) for c in range(8)])
        mu = C.mean(0)
        W = np.linalg.svd(C - mu, full_matrices=False)[2][:2].T
        cm, P = (C - mu) @ W, (Zd[half == h] - mu) @ W
        if h == 1:
            R = _rotation(cms[0], cm)
            cm, P = cm @ R, P @ R
        cms.append(cm)
        proj[half == h] = P
    return proj, np.stack(cms)


def ring_and_centroids(proj, cms, cue, state):
    """For each cue: the ring position (mean of the two halves' cue means), the mean position of its trials over
    their On points and over their Off points, and the number of trials with On / Off points."""
    rows = []
    for c in range(8):
        row = dict(cue=c, ring_pc1=cms.mean(0)[c, 0], ring_pc2=cms.mean(0)[c, 1])
        for nm, v in (('on', 1), ('off', -1)):
            pts = [proj[i][state[i] == v].mean(0) for i in np.flatnonzero(cue == c) if (state[i] == v).any()]
            m = np.mean(pts, 0) if pts else (np.nan, np.nan)
            row.update({f'{nm}_pc1': m[0], f'{nm}_pc2': m[1], f'n_{nm}': len(pts)})
        rows.append(row)
    return rows


def trial_vectors(Zd, Zpre, state):
    """Per-trial mean activity over On points, Off points and pre-cue points (NaN for a trial without any)."""
    out = {'pre': Zpre.mean(1)}
    for p, v in (('on', 1), ('off', -1)):
        m = state == v
        n = m.sum(1)
        x = np.einsum('tk,tku->tu', m.astype(float), Zd) / np.maximum(n, 1)[:, None]
        x[n == 0] = np.nan
        out[p] = x
    return out


def shape_similarity(Zd, Zpre, cue, state, half, k=7, n_shuf=50):
    """{space: {period: (similarity, shuffle mean, trials, shuffle 95th percentile)}} for space 'full' and 'pcs'
    (k PCs of the cue means)."""
    rng = np.random.default_rng(0)
    V = trial_vectors(Zd, Zpre, state)
    ref = {}
    for h in (0, 1):
        C = np.stack([Zd[(half != h) & (cue == c)].mean((0, 1)) for c in range(8)])
        mu = C.mean(0)
        ref[h] = C - mu, mu, np.linalg.svd(C - mu, full_matrices=False)[2][:k].T

    def sim(P, cu, hf, space):
        v = []
        for h in (0, 1):
            Cm, mu, W = ref[h]
            m = hf == h
            cen = np.array([P[m & (cu == c)].mean(0) - mu if (m & (cu == c)).any() else np.full(P.shape[1], np.nan)
                            for c in range(8)])
            ok = ~np.isnan(cen).any(1)
            if ok.sum() >= 4:
                A, B = (Cm[ok], cen[ok]) if space == 'full' else (Cm[ok] @ W, cen[ok] @ W)
                v.append(1 - sp_procrustes(A, B)[2])
        return np.mean(v) if v else np.nan

    res = {'full': {}, 'pcs': {}}
    for p in PERIODS:
        ok = ~np.isnan(V[p]).any(1)
        P, cu, hf = V[p][ok], cue[ok], half[ok]
        for space in res:
            sh = []
            for _ in range(n_shuf):
                c2 = cu.copy()
                for hh in (0, 1):
                    c2[hf == hh] = rng.permutation(c2[hf == hh])
                sh.append(sim(P, c2, hf, space))
            res[space][p] = (float(sim(P, cu, hf, space)), float(np.nanmean(sh)), int(ok.sum()), float(np.nanpercentile(sh, 95)))
    return res


def analyse(Zd, Zpre, cue, state, plane=False):
    """Procrustes table (one row per space and period) and, if asked, the PCA plane of one session. Zd, Zpre: delay
    and pre-cue activity of the analysed units (trials, times, units); state: On/Off states over the delay."""
    half = trial_halves(cue, np.random.default_rng(0))
    rows = [dict(space=sp, period=p, similarity=v[0], shuffle_mean=v[1], n_trials=v[2], shuffle_p95=v[3],
                 normalised=normalised(*v[:2]))
            for sp, d in shape_similarity(Zd, Zpre, cue, state, half).items() for p, v in d.items()]
    return rows, ring_and_centroids(*project(Zd, cue, half), cue, state) if plane else None


def normalised(similarity, shuffle_mean):
    return (similarity - shuffle_mean) / (1 - shuffle_mean)
