"""Decoding of the recordings and their On/Off states.

Data: 25 sessions of lateral prefrontal recordings (Panichello et al. 2024), spikes at 1 ms from -500 to 1,450 ms
after cue onset, 8 cue locations.

Per session, correct trials:
  rates     spike counts in 100 ms windows every 10 ms (window centres -400 ... 1,400 ms), in Hz, z-scored across
            trials for each unit and time point.
  decoder   for each cue and its opposite cue (4 pairs) and each time point, decoder.ridge_logistic_loo with
            lam = 10,000 (leave-one-trial-out), for the true labels and N_SHUFF label shuffles. Confidence = the
            probability given to the trial's true cue.
  z         the true confidence against the shuffles at each time point (decoder_z); the On threshold of each trial
            comes from its shuffles (null_threshold); states are labelled 0-1,400 ms (onoff.label, Off below 0.8416).
  unit sets all units; top k units (ranked by a one-way ANOVA of the delay rate across cues, computed on the other
            half of the trials, so that a trial never ranks its own units); random k units (N_RAND draws).
Error trials are decoded with the decoders fitted on all correct trials of their cue pair (same shuffles), with
rates z-scored by the correct trials' mean and SD.
"""
import os

import numpy as np
from scipy.io import loadmat
from scipy.special import expit
from scipy.stats import f_oneway

from . import decoder, onoff

LAM = 10000.0
N_SHUFF = 30
KS = (5, 10, 20, 50, 100, 200)    # all drawn, as they fix the random-number sequence; the paper shows 5, 20, 50
N_RAND = 10
OFF_Z = 0.8416
STEP, BOX, T_RANGE = 10, 100, (-400, 1400)
LEN_BINS = np.arange(0, 620, 20)


def monkey(session):
    return {'21': 'A', '22': 'H', '24': 'J'}[str(session)[:2]]


def load(path):
    """One session file: spikes (trials x 1 ms x units), cue index 0-7, cue angle (rad) of each index, the index of
    the opposite cue, correct (bool)."""
    d = loadmat(path, simplify_cells=True)
    cue = d['cueAngIdx'].astype(int) - 1
    ang = np.array([np.angle(np.mean(np.exp(1j * d['cueAng'][cue == c]))) for c in range(8)])
    opp = np.array([np.argmin(np.abs(np.angle(np.exp(1j * (ang - a - np.pi))))) for a in ang])
    return dict(name=os.path.basename(path)[:-4], spks=d['spks'], t0=d['tc'][0] - 0.5, cue=cue, cue_ang=ang, opp=opp,
                correct=d['isCorr'].astype(bool))


def rates(s, trials):
    """Times (window centres, ms) and rates (Hz, float32; trials x times x units): spikes in [t - 50, t + 50)."""
    spks = s['spks'][trials]
    n = spks.shape[1] // STEP
    counts = spks[:, :n * STEP].reshape(len(spks), n, STEP, -1).sum(2, dtype=np.float32)
    w = BOX // STEP
    cs = np.concatenate([np.zeros_like(counts[:, :1]), np.cumsum(counts, 1)], 1)
    R = (cs[:, w:] - cs[:, :-w]) / (BOX / 1000.0)
    t = s['t0'] + BOX / 2 + STEP * np.arange(R.shape[1])
    keep = (t >= T_RANGE[0]) & (t <= T_RANGE[1])
    return t[keep], R[:, keep]


def zscore(R, ref=None):
    """z-score R across trials with the mean and SD of ref (default R itself); an SD of 0 is replaced by 1."""
    ref = R if ref is None else ref
    sd = ref.std(0)
    sd[sd == 0] = 1
    return (R - ref.mean(0)) / sd


def trial_halves(cue, rng):
    """Two halves of the trials, stratified by cue."""
    half = np.zeros(len(cue), int)
    for c in range(8):
        idx = rng.permutation(np.flatnonzero(cue == c))
        half[idx[len(idx) // 2:]] = 1
    return half


def decoder_z(C):
    """C: (1 + n_shuff, T) confidences of one trial, row 0 with the true labels: z of row 0 against the shuffles."""
    C = C - 0.5                    # z is shift-invariant; centring avoids cancellation when C ~ 0.5
    return (C[0] - C[1:].mean(0)) / np.maximum(C[1:].std(0, ddof=1), 1e-300)


def null_threshold(C):
    """On threshold of one trial: 95th percentile over the shuffle rows of C of each row's largest cluster mass, with
    the row z-scored against the other rows (as the authors' clusterMassOneSampZ.m)."""
    C = C - 0.5
    S = C.shape[0]
    tot, tot2 = C.sum(0), (C ** 2).sum(0)
    masses = []
    for r in range(1, S):
        m = (tot - C[r]) / (S - 1)
        z = (C[r] - m) / np.sqrt(np.maximum(((tot2 - C[r] ** 2) - (S - 1) * m ** 2) / (S - 2), 1e-300))
        masses.append(max([z[a:b].sum() for a, b in onoff.runs(z > onoff.ON_Z)], default=0.0))
    return np.percentile(masses, 95)


class Session:
    """Rates, trial halves, unit sets and shuffled labels of one session, and decoders of any set of units.

    The random-number sequence (seed 0: trial halves, random unit sets, label shuffles, then the subsampling of the
    mixture fits in readout_row) defines the published values."""

    def __init__(self, path):
        rng = np.random.default_rng(0)
        s = load(path)
        self.name, self.opp, self.cue_ang = s['name'], s['opp'], s['cue_ang']
        self.trial_index = np.flatnonzero(s['correct'])
        self.error_index = np.flatnonzero(~s['correct'])
        self.n_trials_all = len(s['correct'])
        self.t, R = rates(s, s['correct'])
        _, Re = rates(s, ~s['correct'])
        self.cue, self.cue_err = s['cue'][s['correct']], s['cue'][~s['correct']]
        self.Z, self.Z_err = zscore(R), zscore(Re, R)
        del s, R, Re
        self.n_tr, _, self.n_units = self.Z.shape
        self.sw = (self.t >= onoff.STATE_WIN[0]) & (self.t <= onoff.STATE_WIN[1])         # labelled time points
        self.delay = (self.t >= onoff.DELAY[0]) & (self.t <= onoff.DELAY[1])
        self.half = trial_halves(self.cue, rng)
        delay_rate = self.Z[:, self.delay].mean(1)
        self.F = [np.nan_to_num(f_oneway(*[delay_rate[(self.half == h) & (self.cue == c)] for c in range(8)],
                                         axis=0).statistic) for h in (0, 1)]
        self.sets = self.unit_sets(rng)
        self.pairs = sorted({min(c, self.opp[c]) for c in range(8)})
        self.labels = {}
        for c in self.pairs:
            idx = np.flatnonzero((self.cue == c) | (self.cue == self.opp[c]))
            y = (self.cue[idx] == c).astype(float)
            self.labels[c] = idx, y, np.stack([y] + [rng.permutation(y) for _ in range(N_SHUFF)])
        self.rng = rng

    def unit_sets(self, rng):
        """{(set, k, draw): (units for trials of half 0, units for trials of half 1)}."""
        allu = np.arange(self.n_units)
        sets = {('all', self.n_units, 0): (allu, allu)}
        for k in KS:
            if k > self.n_units:
                continue
            sets['top', k, 0] = np.argsort(-self.F[1])[:k], np.argsort(-self.F[0])[:k]   # ranked on the other half
            for r in range(N_RAND):
                u = np.sort(rng.choice(self.n_units, k, replace=False))
                sets['random', k, r] = u, u
        return sets

    def confidence(self, units):
        """Leave-one-out confidences of the true cue, (1 + n_shuff, trials, times); units: (units for the trials of
        half 0, for half 1)."""
        conf = np.zeros((N_SHUFF + 1, self.n_tr, len(self.t)))
        same = np.array_equal(*units)
        for c in self.pairs:
            idx, y, Y = self.labels[c]
            for u, h in (((units[0], None),) if same else ((units[0], 0), (units[1], 1))):
                p = expit(decoder.ridge_logistic_loo(self.Z[idx][:, :, u].transpose(1, 0, 2), Y, LAM)[0])
                cf = np.where(y[None, None] == 1, p, 1 - p).transpose(0, 2, 1)
                m = np.ones(len(idx), bool) if h is None else self.half[idx] == h
                conf[:, idx[m]] = cf[:, m]
        return conf

    def error_confidence(self):
        """Confidences of the cued location on the error trials (all units; decoders fitted on all correct trials of
        the cue pair)."""
        conf = np.zeros((N_SHUFF + 1, len(self.cue_err), len(self.t)))
        for c in self.pairs:
            idx, _, Y = self.labels[c]
            e = np.flatnonzero((self.cue_err == c) | (self.cue_err == self.opp[c]))
            if len(e) == 0:
                continue
            w = decoder.ridge_logistic_weights(self.Z[idx].transpose(1, 0, 2), Y, LAM)          # T, S, units + 1
            Xe = np.concatenate([self.Z_err[e], np.full((len(e), len(self.t), 1), decoder.C_INT)], 2)
            p = expit(np.einsum('etd,tsd->set', Xe, w))
            conf[:, e] = np.where((self.cue_err[e] == c)[None, :, None], p, 1 - p)
        return conf

    def states(self, conf):
        """z (trials x times) and On/Off states (trials x labelled time points) of the confidences conf."""
        z = np.array([decoder_z(conf[:, i]) for i in range(conf.shape[1])])
        st = np.array([onoff.label(z[i, self.sw][None], null_threshold(conf[:, i][:, self.sw]), off_z=OFF_Z)[0]
                       for i in range(conf.shape[1])], np.int8)
        return z, st

    def readout_row(self, key, conf, z, st):
        """On/Off statistics of one unit set (one row of the read-out table) and its state-length counts (20 ms bins
        from 0 to 600 ms, lengths clipped at 600 ms, and > 600 ms)."""
        dsw = self.delay[self.sw]
        on = np.array([(b - a) * STEP for s in st for a, b in onoff.runs(s == 1)], int)
        off = np.array([(b - a) * STEP for s in st for a, b in onoff.runs(s == -1)], int)
        zz = z[:, self.delay].ravel()
        row = dict(session=self.name, set=key[0], n=key[1], draw=key[2], n_trials=self.n_tr,
                   accuracy=float(np.mean(conf[0][:, self.delay] > 0.5)),
                   on_per_trial=float(np.mean([len(onoff.runs(s == 1)) for s in st])),
                   off_per_trial=float(np.mean([len(onoff.runs(s == -1)) for s in st])),
                   on_mean_ms=float(on.mean()) if on.size else np.nan,
                   on_median_ms=float(np.median(on)) if on.size else np.nan,
                   off_mean_ms=float(off.mean()) if off.size else np.nan,
                   off_median_ms=float(np.median(off)) if off.size else np.nan,
                   frac_on=float(np.mean(st[:, dsw] == 1)), frac_off=float(np.mean(st[:, dsw] == -1)),
                   mu_on=onoff.mu_on(zz[np.isfinite(zz)], self.rng))
        hist = {k: np.r_[np.histogram(np.minimum(v, 600), LEN_BINS)[0], np.sum(v > 600)] for k, v in (('on', on), ('off', off))}
        return row, hist
