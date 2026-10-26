"""Rate model of the memory network: two memory pools (cued c, uncued u) with shared inhibition I.

    tau_r  dr_X/dt = -r_X + Phi( w_s s_X + w_a r_X - w_I r_I + I_0 m(t) - w_ff q + cue_X + sigma xi_X )
    tau_I  dr_I/dt = -r_I + w_EI (r_c + r_u)              shared inhibition (competition)
    tau_ff dq/dt   = -q + m(t)                            feed-forward inhibition, lagging the common input

Phi(x) = r_max / (1 + exp(-(x - 1) / kappa)); xi are Ornstein-Uhlenbeck noises (tau_xi). Time in ms, rates in Hz.
m(t) is the common (attentional) input: 1, except during dips (Poisson onsets, gamma-distributed lengths) when it
drops to 1 - depth, with 10 ms ramps. When a dip ends, the lagging feed-forward inhibition gives both pools a
brief kick and the pool that kept more of its slow recurrent variable s wins the competition.

The slow recurrent variable s_X:
  'nmda'  NMDA gating,  tau_N ds_X/dt = -s_X + beta r_X (1 - s_X).
  'stp'   no slow synapse; the saturating synaptic activation S_X = g / (1 + g), g = beta r_X u_X / U, weighted by
          short-term facilitation u_X (Tsodyks-Markram; Mi, Katkov & Tsodyks 2017):
          du_X/dt = (U - u_X) / tau_f + U (1 - u_X) r_X, U = 0.3, tau_f = 1.5 s (depression off).
The memory is held at the end of the delay when the slow variable (NMDA gating, or facilitation u) of the cued pool
exceeds the uncued pool's by more than 0.05 (memory_correct).
"""
import numpy as np
from numba import njit

FIXED = dict(r_max=150.0, kappa=0.06, w_a=0.0141, beta=0.02, w_I=0.04, w_EI=0.5, w_ff=0.8, tau_ff=15.0, I_0=1.496,
             tau_r=10.0, tau_I=1.0, tau_xi=10.0, cue_amp=1.0, cue_dur=50.0, t_pre=300.0, t_delay=1600.0, dt=0.1,
             U=0.3, tau_d=300.0)
# fitted to the On/Off statistics of the all-units read-out (fit.py)
FIT = dict(nmda=dict(tau_N=150.0, w_s=1.30, dip_rate=10.207, dip_mean=90.12, dip_shape=8.70, dip_depth=0.5987,
                     sigma=0.0058),
           stp=dict(tau_f=1500.0, w_s=0.6919, dip_rate=5.486, dip_mean=211.73, dip_shape=0.70, dip_depth=0.3936,
                    sigma=0.0050))
_KEYS = ('r_max', 'kappa', 'w_a', 'beta', 'w_I', 'w_EI', 'w_ff', 'tau_ff', 'I_0', 'tau_r', 'tau_I', 'cue_amp', 'dt',
         'w_s', 'sigma')


def params(model='nmda', **over):
    return {'model': model, **FIXED, **FIT[model], **over}


def common_input(n_tr, n, p, rng, ramp=10.0):
    """(n_tr, n) common input m(t) with dips, at resolution dt; the dip process is stationary at t = 0."""
    m = np.ones((n_tr, n))
    dt, T = p['dt'], n * p['dt']
    for k in range(n_tr):
        t = rng.exponential(1000 / p['dip_rate']) - 500 * rng.random()
        while t < T:
            d = rng.gamma(p['dip_shape'], p['dip_mean'] / p['dip_shape'])
            a, b = int(max(t, 0) / dt), int(min(T, t + d) / dt)
            if b > a:
                m[k, a:b] = 1 - p['dip_depth']
            t += d + rng.exponential(1000 / p['dip_rate'])
    return smooth_ramps(m, int(ramp / dt))


def smooth_ramps(m, w):
    """Linear ramps of w steps at every step of m (running mean, padded with 1)."""
    pad = np.c_[np.ones((len(m), w)), m, np.ones((len(m), w))]
    return np.array([np.convolve(r, np.ones(w) / w, mode='same') for r in pad])[:, w:-w]


def single_dip(p, n_tr, onset, length, ramp=10.0):
    """(n_tr, n) common input with one dip of the fitted depth from `onset` ms after cue onset."""
    n = int(round((p['t_pre'] + p['cue_dur'] + p['t_delay']) / p['dt']))
    t = np.arange(n) * p['dt'] - p['t_pre']
    m = np.where((t >= onset) & (t < onset + length), 1 - p['dip_depth'], 1.0)
    return np.broadcast_to(smooth_ramps(m[None], int(ramp / p['dt']))[0], (n_tr, n)).copy()


@njit(cache=True)
def _sim(m, cue, xi, p, stp, tau_s, rec):
    """Euler integration; stp = False: s is the NMDA gating (time constant tau_s = tau_N); stp = True: s = S of the
    facilitation model (tau_s = tau_f) and u is recorded. Returns rc, ru, sc, su, uc, uu every rec steps."""
    r_max, kappa, w_a, beta, w_I, w_EI, w_ff, tau_ff, I_0, tau_r, tau_I, cue_amp, dt, w_s, sigma, U = p
    n_tr, n = m.shape
    out = np.zeros((n_tr, n // rec, 6))
    for k in range(n_tr):
        rc = 1.0; ru = 1.0; rI = 1.0; sc = 0.0; su = 0.0; q = m[k, 0]
        uc = U * (1 + tau_s * 1e-3) / (1 + U * tau_s * 1e-3); uu = uc        # facilitation at rest (1 Hz)
        for i in range(n):
            if stp:
                gc = beta * rc * uc / U; gu = beta * ru * uu / U
                sc = gc / (1 + gc); su = gu / (1 + gu)
            x = -w_I * rI + I_0 * m[k, i] - w_ff * q
            fc = r_max / (1 + np.exp(-(w_s * sc + w_a * rc + x + cue_amp * cue[i] + sigma * xi[k, i, 0] - 1) / kappa))
            fu = r_max / (1 + np.exp(-(w_s * su + w_a * ru + x + sigma * xi[k, i, 1] - 1) / kappa))
            rI += dt / tau_I * (w_EI * (rc + ru) - rI)
            if stp:
                ruc = 1e-3 * rc; ruu = 1e-3 * ru                                  # Hz -> per ms
                duc = (U - uc) / tau_s + U * (1 - uc) * ruc
                duu = (U - uu) / tau_s + U * (1 - uu) * ruu
                uc += dt * duc; uu += dt * duu
            rc += dt / tau_r * (fc - rc)
            ru += dt / tau_r * (fu - ru)
            q += dt / tau_ff * (m[k, i] - q)
            if not stp:
                sc += dt / tau_s * (-sc + beta * rc * (1 - sc))
                su += dt / tau_s * (-su + beta * ru * (1 - su))
            if i % rec == rec - 1:
                j = i // rec
                out[k, j, 0] = rc; out[k, j, 1] = ru; out[k, j, 2] = sc; out[k, j, 3] = su
                out[k, j, 4] = uc; out[k, j, 5] = uu
    return out


@njit(cache=True)
def _ou(white, a):
    x = white.copy()
    for i in range(1, x.shape[1]):
        x[:, i] = a * x[:, i - 1] + np.sqrt(1 - a * a) * white[:, i]
    return x


def simulate(p, n_tr, seed=0, quiet=0.0, m=None):
    """n_tr trials with pool c cued: 1 ms traces (n_tr, n_t) of rc, ru, sc, su (and uc, uu for 'stp'), the common
    input m, and the time t from cue onset. quiet: no dips in the first `quiet` ms after cue onset. m: a given
    common input (n_tr, n steps of dt) instead of random dips."""
    rng = np.random.default_rng(seed)
    dt = p['dt']
    n = int(round((p['t_pre'] + p['cue_dur'] + p['t_delay']) / dt))
    if m is None:
        m = common_input(n_tr, n, p, rng)
        m[:, :int((p['t_pre'] + quiet) / dt)] = 1.0
    t = np.arange(n) * dt
    cue = ((t >= p['t_pre']) & (t < p['t_pre'] + p['cue_dur'])).astype(float)
    xi = _ou(rng.standard_normal((n_tr, n, 2)), np.exp(-dt / p['tau_xi']))
    stp = p['model'] == 'stp'
    rec = int(round(1 / dt))
    o = _sim(m, cue, xi, np.array([float(p[k]) for k in _KEYS + ('U',)]), stp, p['tau_f' if stp else 'tau_N'], rec)
    sim = dict(rc=o[..., 0], ru=o[..., 1], sc=o[..., 2], su=o[..., 3], m=m[:, rec - 1::rec],
               t=np.arange(n // rec) - p['t_pre'])
    if stp:
        sim.update(uc=o[..., 4], uu=o[..., 5])
    return sim


def memory_trace(sim):
    """Cued minus uncued slow variable: NMDA gating, or facilitation u in the STP model."""
    return sim['uc'] - sim['uu'] if 'uc' in sim else sim['sc'] - sim['su']


def memory_correct(sim):
    """Memory held at the end of the delay: memory_trace above 0.05 over the last 100 ms."""
    return memory_trace(sim)[:, -100:].mean(1) > 0.05
