"""Validation of the model read-out z = c dr + eta on synthetic data.

The fitted rate model is simulated for two cue locations A and B (the pools swap roles); its pool rates drive 330
synthetic neurons: a fraction F_TUNED follows pool A or pool B (half each), the others are untuned;
    rate_i(t) = g_trial (b_i + k a_i r_pool(t) / 50 Hz),  b_i lognormal (median 5 Hz), a_i ~ U(0.5, 1.5),
g_trial ~ N(1, GAIN_SD) shared by all neurons of a trial; Poisson spikes at 1 ms. The spikes go through the
recordings' pipeline (100 ms windows, z-scored rates, decoder.ridge_logistic_loo with lam = 10,000, 30 label
shuffles, recordings.decoder_z / null_threshold, On/Off labels), and, on the same trials, through the model read-out
with its gain calibrated to the pipeline's mu_on and its noise share rho set to the pipeline's pre-cue z
autocorrelation at 200 ms. Three tuning strengths k, chosen so that the pipeline's mu_on is near the data's
random-20, all-unit and top-20 decoders. Reference: the pipeline rerun on new spikes from the same rates.
"""
import numpy as np
from scipy.special import expit

from . import decoder, onoff, ratemodel, readout, recordings

N_UNITS, F_TUNED, N_PER_CUE, GAIN_SD = 330, 0.3, 70, 0.1
LEVELS = {'weak': 3.2, 'medium': 4.6, 'strong': 7.8}          # tuning k (Hz at a pool rate of 50 Hz)
N_SESSIONS = 8
STEP, WIN = 10, 100
GRID = 10 ** np.linspace(-3, 0, 25)


def session(k, seed):
    rng = np.random.default_rng(seed)
    sim = ratemodel.simulate(ratemodel.params(), 2 * N_PER_CUE, seed=seed)
    cueA = np.r_[np.ones(N_PER_CUE, bool), np.zeros(N_PER_CUE, bool)]
    rA = np.where(cueA[:, None], sim['rc'], sim['ru'])
    rB = np.where(cueA[:, None], sim['ru'], sim['rc'])
    tuned = rng.random(N_UNITS) < F_TUNED
    pool = rng.random(N_UNITS) < 0.5                      # True: follows pool A
    b = np.exp(rng.normal(np.log(5), 0.6, N_UNITS))
    amp = rng.uniform(0.5, 1.5, N_UNITS) * k * tuned
    n_b, w, y = rA.shape[1] // STEP, WIN // STEP, cueA.astype(float)
    tt = sim['t'][0] + STEP * np.arange(n_b - w + 1) + WIN / 2          # window centres
    keep = (tt >= -250) & (tt <= 1400)
    tt = tt[keep]
    sw = (tt >= 0) & (tt <= 1400)

    def pipeline(r):
        """New trial gains and Poisson spikes from the same rates, decoded as the recordings."""
        g = np.clip(r.normal(1, GAIN_SD, 2 * N_PER_CUE), 0.3, None)
        counts = np.zeros((2 * N_PER_CUE, n_b, N_UNITS), np.float32)
        for i in range(2 * N_PER_CUE):
            drive = np.where(pool[None], rA[i][:, None], rB[i][:, None]) / 50.0
            lam = g[i] * (b[None] + amp[None] * drive) / 1000.0
            counts[i] = r.poisson(lam[:n_b * STEP].reshape(n_b, STEP, N_UNITS).sum(1))
        cs = np.concatenate([np.zeros_like(counts[:, :1]), np.cumsum(counts, 1)], 1)
        Z = recordings.zscore(((cs[:, w:] - cs[:, :-w]) / (WIN / 1000.0))[:, keep])
        Y = np.stack([y] + [r.permutation(y) for _ in range(recordings.N_SHUFF)])
        p = expit(decoder.ridge_logistic_loo(Z.transpose(1, 0, 2), Y, recordings.LAM)[0])
        conf = np.where(y[None, None] == 1, p, 1 - p).transpose(0, 2, 1)
        z = np.array([recordings.decoder_z(conf[:, i]) for i in range(len(y))])
        st = np.array([onoff.label(z[i, sw][None], recordings.null_threshold(conf[:, i][:, sw]), off_z=recordings.OFF_Z)[0]
                       for i in range(len(y))])
        return z, st

    z_p, st_p = pipeline(rng)
    z_p2, _ = pipeline(np.random.default_rng(seed + 500))
    dw, tz = readout.rate_diff(sim, t_from=-250)
    dw = dw[:, np.isin(tz, tt)]
    dsel, pre = (tt >= 500) & (tt <= 1400), tt < 0
    x = z_p[:, pre] - z_p[:, pre].mean()
    rho = float(np.clip((x[:, :-20] * x[:, 20:]).mean() / (x * x).mean(), 0, 0.9))
    mu_pipe = onoff.mu_on(z_p[:, dsel], rng)
    eta = readout.noise(dw.shape, rho, rng)
    c = readout.gain_for(mu_pipe, readout.gain_curve(dw, eta, dsel, rng, GRID), GRID)
    z_s = c * dw + eta
    st_s = onoff.label(z_s[:, sw], readout.null_threshold(dw.shape, rho, rng))
    del_sw = (tt[sw] >= 500) & (tt[sw] <= 1400)
    corr = lambda a, b_: float(np.corrcoef(a[:, dsel].ravel(), b_[:, dsel].ravel())[0, 1])
    out = dict(seed=seed, k=k, rho=rho, mu_on_pipeline=mu_pipe, gain_c=c, corr_z_pipeline_repeat=corr(z_p, z_p2),
               corr_z_pipeline_vs_shortcut=corr(z_p, z_s))
    for name, st in (('pipeline', st_p), ('shortcut', st_s)):
        S = onoff.stats(st, del_sw)
        out.update({f'{name}_{m}': float(S[m]) for m in ('n_off', 'n_on', 'frac_off')})
    mm = sim['m'][:, :n_b * STEP].reshape(len(y), n_b, STEP).mean(2)       # true dips, in the same 100 ms windows
    cm = np.concatenate([np.zeros((len(y), 1)), np.cumsum(mm, 1)], 1)
    dip = ((1 - (cm[:, w:] - cm[:, :-w]) / w)[:, keep] > 0.5 * ratemodel.FIT['nmda']['dip_depth'])[:, sw]
    example = dict(tt=tt, z_pipeline=z_p[0], z_shortcut=z_s[0], dip=dip[0])
    return out, example
