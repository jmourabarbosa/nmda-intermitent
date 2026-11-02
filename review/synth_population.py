"""Ground-truth test of Fig. 4e (and the 'Off states overestimate memory loss' claim).

Synthetic sessions: 330 Poisson neurons on a ring, 8 cues x 60 trials, 1 ms resolution. The memory signal s(t) is
1 between silences and, during the silences (dips of the common input drawn with the fitted rate-model dip
statistics, 1 - depth), it is
  'silent'   0: the memory is completely ABSENT from spiking (activity-silent ground truth),
  'reduced'  0.3 of its value (memory still in activity, weakly).
rate_i(t) = m_glob(t) b_i + s(t) A_i exp(kappa (cos(theta_cue - theta_i) - 1)), with m_glob = 1 - 0.3 depth(t) a
global rate drop during silences. Read-out as the paper's spiking network (spiking_readout: 100 ms windows,
quadratic L2 decoder, 30 shuffles, On/Off labels) on half of the units; population.analyse on the other half.

Reported: (i) shape similarity (normalised, 0 = chance) of On, Off and pre-cue activity, as Fig. 4e;
(ii) precision of Off labels = fraction of Off-labelled points whose 100 ms window is mostly inside a true silence;
(iii) the same similarity with Off restricted to 'core' points (>= 50 ms from the edges of their Off run).
If the 'silent' world gives Off similarity >> 0, Fig. 4e cannot show that the memory stays in activity during Off.
usage: python review/synth_population.py   (~20 min on 4 cores)
"""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from nmdaint import onoff, population, ratemodel, spiking_readout

N_U, N_PER, T0, T1 = 330, 60, -500, 1450
KAPPA, F_TUNED = 2.0, 0.6
AMP = 5.0          # Hz, tuned peak (mu_on of the all-units decoder close to the data's)


def session(world, seed, amp=AMP):
    rng = np.random.default_rng(seed)
    cue = np.repeat(np.arange(8), N_PER)
    n_tr, nt = len(cue), T1 - T0
    t = np.arange(T0, T1)
    p = ratemodel.params('nmda')
    m = ratemodel.common_input(n_tr, nt * 10, {**p, 'dt': 0.1}, rng)[:, ::10]   # 1 ms
    m[:, t < 100] = 1.0                                                         # no silences before 100 ms
    depth = (1 - m) / p['dip_depth']                                            # 0..1, 1 inside a silence
    s_in = {'silent': 0.0, 'reduced': 0.3}[world]
    s = (1 - depth * (1 - s_in)) * (t >= 0)[None]                               # memory signal
    theta = rng.uniform(0, 2 * np.pi, N_U)
    b = np.exp(rng.normal(np.log(5), 0.6, N_U))
    A = amp * rng.uniform(0.5, 1.5, N_U) * (rng.random(N_U) < F_TUNED)
    tun = np.exp(KAPPA * (np.cos(np.pi / 4 * cue[:, None] - theta[None]) - 1)) * A[None]     # trials x units
    counts = np.zeros((n_tr, nt // 10, N_U), np.int32)
    for i in range(n_tr):
        lam = ((1 - 0.3 * depth[i])[:, None] * b[None] + s[i][:, None] * tun[i][None]) / 1000.0
        counts[i] = rng.poisson(lam.reshape(nt // 10, 10, N_U).sum(1))
    t10 = np.arange(T0, T1, 10)
    X, tz = spiking_readout.windows(counts, t10, t_from=-450, t_to=1400)
    # true silence per window: mean of depth over the window's 100 ms
    cd = np.c_[np.zeros(n_tr), np.cumsum(depth, 1)]
    wc = (tz - T0).astype(int)
    true_sil = (cd[:, wc + 50] - cd[:, wc - 50]) / 100 > 0.5
    H1, H2 = population.unit_halves(N_U)
    valid = (tz >= 0) & (tz <= 1400)
    delay = (tz >= 500) & (tz <= 1400)
    z, st = spiking_readout.states(X, cue, rng, H1, tz, valid)
    Z = spiking_readout.zscore(X)
    std = st[:, delay]
    rows, _ = population.analyse(Z[:, delay][:, :, H2], Z[:, tz < 0][:, :, H2], cue, std)
    # Off core: Off points >= 5 points (50 ms) from the edges of their run
    core = np.zeros_like(std)
    for k, row in enumerate(std):
        for a, b_ in onoff.runs(row == -1):
            if b_ - a > 10:
                core[k, a + 5:b_ - 5] = -1
    core_on = np.where(std == 1, 1, core)
    rows_c, _ = population.analyse(Z[:, delay][:, :, H2], Z[:, tz < 0][:, :, H2], cue, core_on)
    sil = true_sil[:, delay]
    off = std == -1
    out = dict(world=world, seed=seed, amp=amp, mu_on=onoff.mu_on(np.nan_to_num(z)[:, delay], rng),
               frac_off=off.mean(), frac_true_silent=sil.mean(),
               off_precision=float(sil[off].mean()), silent_recall=float(off[sil].mean()),
               frac_off_points_not_silent=float(1 - sil[off].mean()))
    for r in rows:
        if r['space'] == 'pcs':
            out[f"sim_{r['period']}"] = r['normalised']
    for r in rows_c:
        if r['space'] == 'pcs' and r['period'] == 'off':
            out['sim_off_core'] = r['normalised']
    return out


if __name__ == '__main__':
    jobs = [(w, s, a) for a in (8.0, 14.0) for w in ('silent', 'reduced') for s in range(3)]
    res = pd.DataFrame(Parallel(n_jobs=4)(delayed(session)(w, s, a) for w, s, a in jobs))
    res.to_csv('review/out/synth_population.csv', index=False)
    pd.set_option('display.width', 250)
    print(res.round(3).to_string(index=False))
    print(res.groupby(['amp', 'world']).mean(numeric_only=True).drop(columns='seed').round(3).to_string())
