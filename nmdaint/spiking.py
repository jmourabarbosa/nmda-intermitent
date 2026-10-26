"""Spiking ring attractor (Compte, Brunel, Goldman-Rakic & Wang 2000) with NMDA-only recurrent excitation (decay
150 ms) and a common input to the excitatory cells that briefly drops (dips) and overshoots when it recovers.

Leaky integrate-and-fire neurons, N_E pyramidal cells and N_I interneurons on a ring (angle theta):

    C dV/dt = -g_L (V - E_L) - I_ext - I_NMDA - I_GABA + I_cue

    I_ext  = g_ext s_ext (V - E_E)                      external Poisson AMPA input, rate nu_ext (x m_eff(t) on E cells)
    I_NMDA = g_N (V - E_E) B(V) sum_j W_ij s_j          recurrent NMDA from E cells, B(V) the Mg2+ block
    I_GABA = g_G (V - E_I) sum_j s_GABA_j               recurrent GABA_A from I cells (all-to-all)

    NMDA gating of E cell j:  dx/dt = -x / tau_x + sum_k delta(t - t_k),  ds/dt = -s / tau_N + alpha x (1 - s)
    AMPA / GABA_A gating:     ds/dt = -s / tau + sum_k delta(t - t_k)

E->E weights fall off with the difference in preferred angle, W = J_minus + (J_plus - J_minus) exp(-dtheta^2 / 2
sigma^2), mean 1; E->I, I->E and I->I are uniform. The E->E sum uses the Fourier series of W (K_FOURIER harmonics).
Heterogeneity, fixed per network: firing thresholds (SD sd_vth, stratified along the ring), and each E cell i gets a
fraction h_i of its E->E input through a broad kernel 1 + kappa_broad cos(dtheta) instead of W.

Common input m(t): 1, except during dips (Poisson onsets, gamma lengths, from cue onset) when it drops to 1 - depth,
with 10 ms ramps. The E cells receive m_eff = m + kick_A (m - m_bar), m_bar following m instantly into a dip and
recovering with tau_kick, so the input overshoots after each dip. Interneurons get a constant nu_ext.
Units: ms, mV, nS, pF, pA, Hz. Parameter sources: PARAMETERS.md.
"""
import numpy as np
from numba import njit, prange
from scipy import stats

P = dict(
    N_E=2048, N_I=512,                                                       # populations
    C_E=500.0, gL_E=25.0, EL=-70.0, Vth=-50.0, Vreset=-60.0, ref_E=2.0,      # pyramidal cells
    C_I=200.0, gL_I=20.0, ref_I=1.0,                                         # interneurons
    E_E=0.0, E_I=-70.0, Mg=1.0,                                              # reversal potentials, Mg2+
    tau_AMPA=2.0, tau_x=2.0, alpha=0.5, tau_N=150.0, tau_GABA=10.0,          # synapses
    g_EE=0.33, g_EI=0.292, g_IE=1.336, g_II=1.024,                           # recurrent conductances (nS)
    nu_ext=1800.0, g_ext_E=3.1, g_ext_I=2.38,                                # external input
    J_plus=3.0, sigma_deg=18.0,                                              # E->E footprint
    sd_vth=0.3, h_max=1.0, h_a=1.0, h_b=1.0, kappa_broad=0.25,               # heterogeneity
    cue_amp=500.0, cue_sigma_deg=18.0, cue_dur=50.0,                         # cue current to E cells
    t_pre=500.0, t_delay=1500.0, dt=0.1,                                     # trial (cue onset at t_pre)
    dip_rate=7.0, dip_mean=150.0, dip_shape=8.7, dip_depth=0.5, ramp=10.0, kick_A=1.0, tau_kick=150.0,
    K_FOURIER=24, net_seed=1)


def params(**over):
    assert set(over) <= set(P), f'unknown parameters {set(over) - set(P)}'
    return {**P, **over}


def _stratified(rng, n, ppf, block=16):
    """Every block of 16 neighbouring cells holds the same 16 quantiles of the distribution, in random order."""
    q = ppf((np.arange(block) + 0.5) / block)
    return np.concatenate([rng.permutation(q) for _ in range(n // block)])


class Network:
    """Fixed structure of one network: angles, thresholds, broad-input fractions, Fourier tables of W."""

    def __init__(self, p):
        self.p = p
        NE, NI = p['N_E'], p['N_I']
        self.theta_E = 2 * np.pi * np.arange(NE) / NE
        rng = np.random.default_rng(p['net_seed'])
        self.vth = np.r_[p['Vth'] + p['sd_vth'] * _stratified(rng, NE, stats.norm.ppf),
                         p['Vth'] + p['sd_vth'] * _stratified(rng, NI, stats.norm.ppf)]
        self.h = p['h_max'] * _stratified(rng, NE, lambda q: stats.beta.ppf(q, p['h_a'], p['h_b']))
        g = np.exp(-np.angle(np.exp(1j * self.theta_E)) ** 2 / (2 * np.deg2rad(p['sigma_deg']) ** 2))
        J_minus = (1 - p['J_plus'] * g.mean()) / (1 - g.mean())
        W = J_minus + (p['J_plus'] - J_minus) * g                # weight at angular distance theta_E[k]
        what = np.fft.rfft(W).real / NE
        K = p['K_FOURIER']
        # sum_j W(th_i - th_j) s_j = sum_k wk[k] (cos k th_i C_k + sin k th_i S_k), C_k = sum_j cos(k th_j) s_j, ...
        self.wk = np.r_[what[0], 2 * what[1:K + 1]]
        k = np.arange(K + 1)[:, None]
        self.cos_tab, self.sin_tab = np.cos(k * self.theta_E[None]), np.sin(k * self.theta_E[None])

    def cue_current(self, cue_deg):
        d = np.angle(np.exp(1j * (self.theta_E[None] - np.deg2rad(np.asarray(cue_deg, float))[:, None])))
        return self.p['cue_amp'] * np.exp(-d ** 2 / (2 * np.deg2rad(self.p['cue_sigma_deg']) ** 2))


def dips(n_tr, T, p, rng, t_free):
    """(n_tr, n) common input at resolution dt over T ms with random dips after t_free ms (the dip process starts
    up to 500 ms before t_free, so that it is near-stationary there)."""
    dt = p['dt']
    m = np.ones((n_tr, int(round(T / dt))))
    for k in range(n_tr):
        t = t_free - rng.random() * 500 + rng.exponential(1000 / p['dip_rate'])
        while t < T:
            d = rng.gamma(p['dip_shape'], p['dip_mean'] / p['dip_shape'])
            if t + d > t_free:
                a = max(t, t_free)
                length = t + d - a
                m[k, int(a / dt):int(min(T, a + length) / dt)] = 1 - p['dip_depth']
            t += d + rng.exponential(1000 / p['dip_rate'])
    w = int(round(p['ramp'] / dt))
    pad = np.c_[np.ones((n_tr, w)), m, np.ones((n_tr, w))]
    return np.array([np.convolve(r, np.ones(w) / w, mode='same') for r in pad])[:, w:-w]


@njit(cache=True)
def _kick(m, A, a):
    out = np.empty_like(m)
    for k in range(m.shape[0]):
        mb = m[k, 0]
        for i in range(m.shape[1]):
            mb = m[k, i] if m[k, i] < mb else m[k, i] + (mb - m[k, i]) * a
            out[k, i] = m[k, i] + A * (m[k, i] - mb)
    return out


@njit(parallel=True, cache=True)
def _run(m_eff, cue, cue_on, cue_off, pf, vth, h, kb, wk, cos_tab, sin_tab, rec_idx, seeds, raster_on, s_from, s_to):
    """Simulate trials in parallel. Returns spike counts of the recorded cells in 10 ms bins, the mean NMDA gating
    and spike count of each E cell from step s_from to s_to, and (if raster_on) the E raster in 1 ms bins."""
    (NE, NI, C_E, gL_E, EL, Vreset, ref_E, C_I, gL_I, ref_I, E_E, E_I, Mg, tau_AMPA, tau_x, alpha, tau_N, tau_GABA,
     g_EE, g_EI, g_IE, g_II, nu_ext, g_ext_E, g_ext_I, dt) = pf
    NE = int(NE); NI = int(NI)
    n_tr, n = m_eff.shape
    K1 = wk.shape[0]
    b10 = int(round(10.0 / dt)); b1 = int(round(1.0 / dt))
    counts = np.zeros((n_tr, n // b10, rec_idx.shape[0]), np.uint8)
    s_end = np.zeros((n_tr, NE))
    c_end = np.zeros((n_tr, NE), np.int32)
    raster = np.zeros((n_tr if raster_on else 0, n // b1 if raster_on else 0, NE), np.uint8)
    rec_map = -np.ones(NE + NI, np.int64)
    for r in range(rec_idx.shape[0]):
        rec_map[rec_idx[r]] = r
    dA = np.exp(-dt / tau_AMPA); dX = np.exp(-dt / tau_x); dG = np.exp(-dt / tau_GABA)
    nE_ref = int(round(ref_E / dt)); nI_ref = int(round(ref_I / dt))
    lam_I = nu_ext * dt / 1000.0
    for b in prange(n_tr):
        np.random.seed(seeds[b])
        N = NE + NI
        V = EL + (vth - EL) * np.random.random(N)
        ref = np.zeros(N, np.int64)
        sA = np.zeros(N)
        x = np.zeros(NE); s = np.zeros(NE)
        sG = np.zeros(NI)
        nxt = np.random.exponential(1.0, N)          # external Poisson spikes by time rescaling
        Ck = np.zeros(K1); Sk = np.zeros(K1)
        spk = np.zeros(N, np.uint8)
        for i in range(n):
            lam_E = nu_ext * m_eff[b, i] * dt / 1000.0
            for k in range(K1):                       # Fourier projections of the NMDA gating
                c = 0.0; sn = 0.0
                for j in range(NE):
                    c += cos_tab[k, j] * s[j]
                    sn += sin_tab[k, j] * s[j]
                Ck[k] = c * wk[k]; Sk[k] = sn * wk[k]
            sum_s = Ck[0] / wk[0]
            sum_g = 0.0
            for j in range(NI):
                sum_g += sG[j]
            cue_now = cue_on <= i < cue_off
            for j in range(N):
                spk[j] = 0
                lam = lam_E if j < NE else lam_I
                nxt[j] -= lam
                while nxt[j] <= 0:
                    sA[j] += 1.0
                    nxt[j] += np.random.exponential(1.0)
                if ref[j] > 0:
                    ref[j] -= 1
                    continue
                v = V[j]
                Bm = 1.0 / (1.0 + Mg * np.exp(-0.062 * v) / 3.57)
                if j < NE:
                    rec = 0.0
                    for k in range(K1):
                        rec += Ck[k] * cos_tab[k, j] + Sk[k] * sin_tab[k, j]
                    rec -= h[j] * (rec - Ck[0] - kb * (Ck[1] * cos_tab[1, j] + Sk[1] * sin_tab[1, j]) / wk[1])
                    I = (gL_E * (v - EL) + g_ext_E * sA[j] * (v - E_E) + g_EE * rec * (v - E_E) * Bm
                         + g_IE * sum_g * (v - E_I))
                    if cue_now:
                        I -= cue[b, j]
                    v += -dt * I / C_E
                else:
                    I = (gL_I * (v - EL) + g_ext_I * sA[j] * (v - E_E) + g_EI * sum_s * (v - E_E) * Bm
                         + g_II * sum_g * (v - E_I))
                    v += -dt * I / C_I
                if v >= vth[j]:
                    v = Vreset
                    spk[j] = 1
                    ref[j] = nE_ref if j < NE else nI_ref
                V[j] = v
            for j in range(N):
                sA[j] *= dA
            ib = i // b10
            for j in range(NE):
                x[j] = x[j] * dX + spk[j]
                s[j] += dt * (-s[j] / tau_N + alpha * x[j] * (1.0 - s[j]))
                if raster_on and spk[j]:
                    raster[b, i // b1, j] += 1
            for j in range(NI):
                sG[j] = sG[j] * dG + spk[NE + j]
            for j in range(N):
                if spk[j] and rec_map[j] >= 0:
                    counts[b, ib, rec_map[j]] += 1
            if s_from <= i < s_to:
                for j in range(NE):
                    s_end[b, j] += s[j]
                    c_end[b, j] += spk[j]
        for j in range(NE):
            s_end[b, j] /= max(s_to - s_from, 1)
    return counts, s_end, c_end, raster


PF_KEYS = ('N_E', 'N_I', 'C_E', 'gL_E', 'EL', 'Vreset', 'ref_E', 'C_I', 'gL_I', 'ref_I', 'E_E', 'E_I', 'Mg',
           'tau_AMPA', 'tau_x', 'alpha', 'tau_N', 'tau_GABA', 'g_EE', 'g_EI', 'g_IE', 'g_II', 'nu_ext', 'g_ext_E',
           'g_ext_I', 'dt')


def simulate(net, cue_deg, seed=0, rec_idx=(), raster=False, dip_free=0.0, t_after=0.0):
    """Trials with cues at cue_deg (one per trial), dips from cue onset + dip_free. Returns 10 ms spike counts of the
    recorded cells rec_idx (0..N_E-1 E cells, N_E.. I cells), the NMDA gating and spike counts of the E cells in the
    last 100 ms of the delay, the common input with and without overshoot (1 ms) and the E raster (1 ms) if asked.
    t_after: ms simulated after the delay with the common input at 1 (it only lengthens the trials)."""
    p = net.p
    cue_deg = np.atleast_1d(np.asarray(cue_deg, float))
    rng = np.random.default_rng(seed)
    T = p['t_pre'] + p['t_delay']
    m = dips(len(cue_deg), T, p, rng, p['t_pre'] + dip_free)
    m = np.c_[m, np.ones((len(cue_deg), int(round(t_after / p['dt']))))]
    m_eff = _kick(m, p['kick_A'], np.exp(-p['dt'] / p['tau_kick']))
    dt = p['dt']
    seeds = rng.integers(0, 2 ** 31 - 1, len(cue_deg))
    s_to = int(round(T / dt))
    counts, s_end, c_end, ras = _run(m_eff, net.cue_current(cue_deg), int(round(p['t_pre'] / dt)),
                                     int(round((p['t_pre'] + p['cue_dur']) / dt)), np.array([float(p[k]) for k in PF_KEYS]),
                                     net.vth, net.h, float(p['kappa_broad']), net.wk, net.cos_tab, net.sin_tab,
                                     np.asarray(rec_idx, np.int64), seeds, raster, s_to - int(round(100 / dt)), s_to)
    b1 = int(round(1 / dt))
    return dict(counts=counts, s_end=s_end, c_end=c_end, raster=ras if raster else None, m=m[:, b1 - 1::b1],
                m_eff=m_eff[:, b1 - 1::b1], cue_deg=cue_deg, t10=np.arange(counts.shape[1]) * 10.0 - p['t_pre'],
                t1=np.arange(m.shape[1] // b1) * 1.0 - p['t_pre'])


def _smooth(x, w):
    """Circular running mean over 2w + 1 cells."""
    return np.array([np.convolve(np.r_[r[-w:], r, r[:w]], np.ones(2 * w + 1) / (2 * w + 1), 'valid') for r in x])


def memory(sim, net, min_rate=15.0, min_s=0.2):
    """Memory held at the end of the delay (last 100 ms), from the whole E population: if a bump of activity is
    active (smoothed rate profile peaking > min_rate Hz above its median), whether its peak is within 22.5 deg of the
    cue; otherwise (e.g. the trial ends in a dip) the same test on the NMDA gating profile (peak > min_s)."""
    w = int(round(net.p['N_E'] * 10 / 360))
    held = {}
    for nm, prof, thr in (('rate', _smooth(sim['c_end'] / 0.1, w), min_rate), ('nmda', _smooth(sim['s_end'], w), min_s)):
        base = np.median(prof, 1, keepdims=True)
        top = np.maximum(prof - base - 0.5 * (prof.max(1, keepdims=True) - base), 0)
        ang = np.rad2deg(np.arctan2(top @ np.sin(net.theta_E), top @ np.cos(net.theta_E)))
        err = (ang - sim['cue_deg'] + 180) % 360 - 180
        held[nm] = (prof.max(1) - base[:, 0] > thr), (prof.max(1) - base[:, 0] > thr) & (np.abs(err) <= 22.5)
    return np.where(held['rate'][0], held['rate'][1], held['nmda'][1])


N_REC, N_TRIALS, BATCH = 330, 480, 64      # recorded cells (E and I in proportion), trials per session (60 per cue)


def session(net, seed, sim_seed, n_trials=N_TRIALS, **kw):
    """A simulated session: recorded cells drawn at random over the ring and cues in random order (seed), trials
    simulated in batches of BATCH, batch b seeded with sim_seed * 1000 + b. Returns the recorded cells' 10 ms counts,
    cue index, memory held, the time of each count bin and the recorded cells."""
    rng = np.random.default_rng(seed)
    NE, NI = net.p['N_E'], net.p['N_I']
    n_i = int(round(N_REC * NI / (NE + NI)))
    rec = np.sort(np.r_[rng.choice(NE, N_REC - n_i, replace=False), NE + rng.choice(NI, n_i, replace=False)])
    cue = rng.permutation(np.repeat(np.arange(8), n_trials // 8))
    parts = [simulate(net, 45.0 * cue[i:i + BATCH], seed=sim_seed * 1000 + i // BATCH, rec_idx=rec, **kw)
             for i in range(0, n_trials, BATCH)]
    return dict(counts=np.concatenate([p['counts'] for p in parts]), cue=cue, rec=rec, t10=parts[0]['t10'],
                held=np.concatenate([memory(p, net) for p in parts]))
