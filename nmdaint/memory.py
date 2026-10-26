"""Memory against time spent Off in the rate model.

single_dip: every trial has one dip of the common input (fitted depth, no other dips) of a fixed length
(0-1,000 ms in 25 ms steps) starting 500 ms after cue onset; memory held at the end of the delay (no read-out).

tau_sweep: the fitted NMDA model with its own random dips and only tau_N changed, read out like the data with the
all-units gain of the fitted model (fixed, not recalibrated): per trial, the fraction of the delay Off and whether
the memory was held. Input of behaviour.fit_tau.
"""
import numpy as np

from . import onoff, ratemodel, readout

LENGTHS = np.arange(0, 1001, 25)
ONSET, N_DIP = 500.0, 300
TAUS = (20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 750, 1000, 1500)
N_SWEEP, BATCH = 30000, 1000


def single_dip(p):
    """Dip length (ms) and memory held, per trial."""
    held = [ratemodel.memory_correct(ratemodel.simulate(p, N_DIP, seed=50000 + i, m=ratemodel.single_dip(p, N_DIP, ONSET, L)))
            for i, L in enumerate(LENGTHS)]
    return np.repeat(LENGTHS, N_DIP), np.concatenate(held)


def reference_gain(p, ses):
    """All-units read-out gain of the model (1,056 trials, seed 0)."""
    return readout.evaluate(p, ses, n_trials=1056, seed=0, sets=('all',))['sets']['all']['gain']


def tau_sweep(p, tau, gain):
    """Fraction of the delay Off and memory held of N_SWEEP trials with tau_N = tau."""
    rng = np.random.default_rng(int(tau))
    p = {**p, 'tau_N': float(tau)}
    rho = readout.RHO['all']
    frac, held, thr = [], [], None
    for b in range(N_SWEEP // BATCH):
        sim = ratemodel.simulate(p, BATCH, seed=90000 + b)
        dw, tz = readout.rate_diff(sim)
        if thr is None:
            thr = readout.null_threshold(dw.shape, rho, rng)
        state = onoff.label(gain * dw + readout.noise(dw.shape, rho, rng), thr, (tz >= 0) & (tz <= 1400))
        frac.append((state[:, (tz >= 500) & (tz <= 1400)] == -1).mean(1))
        held.append(ratemodel.memory_correct(sim))
    return np.concatenate(frac), np.concatenate(held)
