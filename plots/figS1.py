"""Supplementary Figure 1: reading the model out like the data, and the validation on synthetic data."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from scipy.stats import norm

from nmdaint import onoff, readout
from .style import C_OFF, C_ON, GREY, INK, LIGHT, letter, limits, runs, shade

C_RO, C_REP = '#2b83ba', GREY
POPC = {'all': INK, 'top20': '#636363', 'random20': '#bdbdbd'}
FS = 6.5
NAMES = {'rand 20': 'Random 20', 'all': 'All units', 'top 20': 'Top 20'}


def thresholds(a):
    a.axhline(onoff.ON_Z, color=C_ON, lw=0.6, ls=(0, (3, 2)))
    a.axhline(onoff.OFF_Z, color=C_OFF, lw=0.6, ls=(0, (3, 2)))


def labels(a, items, x, y, dy=0.12, ha='left'):
    for n, (txt, c) in enumerate(items):
        a.text(x, y - n * dy, txt, color=c, transform=a.transAxes, ha=ha, va='top', fontsize=FS)


def hlabels(a, items, loc='upper right'):
    box = HPacker(children=[TextArea(t, textprops=dict(color=c, fontsize=FS)) for t, c in items], align='baseline', pad=0, sep=8)
    a.add_artist(AnchoredOffsetbox(loc, child=box, pad=0.1, borderpad=0.2, frameon=False))


def draw(m, acf, data_z, syn, E, ses, rho, targets):
    """m: the model read-out example (t, rc, ru, m, tz, dw, eta, state, gain, rho, delay_z, mu_on_model, grid,
    mu_on_grid, mu_on_data, gains); acf: pre-cue z autocorrelation per session, set and lag; data_z: delay z of
    one session with its mixture fit (session, z, gmm_weights, gmm_means, gmm_sds, mu_on_mean); syn: one row per
    synthetic session (synthetic.session); E: a synthetic example trial; ses: readout.load_data()[0]; rho: read-out
    noise share per kind of decoded population; targets: {synthetic level: the data's decoded population it matches}."""
    LV = list(targets)
    plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5})
    C0, RHO = float(m['gain']), float(m['rho'])
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = GridSpec(4, 6, figure=fig, height_ratios=[1.35, 1.0, 1.0, 0.95], hspace=0.75, wspace=1.3,
                  left=0.105, right=0.985, top=0.97, bottom=0.06)

    # a: one model trial through the read-out
    sub = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, :], hspace=0.2)
    t1, tz = m['t'], m['tz']
    ax = [fig.add_subplot(sub[0])]
    ax += [fig.add_subplot(sub[i], sharex=ax[0]) for i in (1, 2, 3)]
    for a in ax[:3]:
        for i, j in runs(m['m'] < 0.99):
            a.axvspan(t1[i], t1[j - 1], color=LIGHT, alpha=0.7, lw=0)
    ax[0].plot(t1, m['rc'], color=INK, lw=0.8)
    ax[0].plot(t1, m['ru'], color=GREY, lw=0.8)
    ax[0].set_ylabel('Rate (Hz)')
    hlabels(ax[0], [('Cued pool', INK), ('Uncued pool', GREY), ('Grey: dips of the common input', '#9e9e9e')])
    ax[1].plot(tz, m['dw'], color=INK, lw=0.8)
    ax[1].set_ylabel(r'$\Delta r$ (Hz)')
    labels(ax[1], [(r'$\Delta r$ = cued $-$ uncued rate, 100 ms windows every 10 ms', INK)], 0.99, 0.98, ha='right')
    ax[2].plot(tz, m['eta'], color=GREY, lw=0.6)
    ax[2].plot(tz, C0 * m['dw'], color=INK, lw=0.8)
    ax[2].set_ylabel('z units')
    hlabels(ax[2], [(rf'$c\cdot\Delta r$ ($c$ = {C0:.3f} per Hz)', INK), (rf'Noise $\eta$ ($\rho$ = {RHO})', GREY)])
    shade(ax[3], tz, m['state'])
    ax[3].plot(tz, C0 * m['dw'] + m['eta'], color=INK, lw=0.8)
    thresholds(ax[3])
    ax[3].set_ylabel('z')
    ax[3].set_xlabel('Time from cue onset (ms)')
    hlabels(ax[3], [(r'$z = c\cdot\Delta r + \eta$', INK), ('On', C_ON), ('Off', C_OFF)])
    for a in ax[:3]:
        plt.setp(a.get_xticklabels(), visible=False)
    ax[0].set_xlim(-200, onoff.STATE_WIN[1])
    for a in ax:
        a.yaxis.set_label_coords(-0.045, 0.5)
    letter(ax[0], 'a', x=-0.085, y=1.0)

    # b: pre-cue z autocorrelation of three decoded populations, and of the model's noise
    cx = fig.add_subplot(gs[1, 0:2])
    za = acf
    lag = np.arange(0, acf.lag_ms.max() + 1, 10)
    for pop, kind in (('all', 'all'), ('top20', 'top'), ('random20', 'random')):
        mean = za[za.set == pop].groupby('lag_ms').acf.mean()
        cx.plot(mean.index, mean.values, color=POPC[pop], lw=1.1)
        rho_k = rho[kind]
        cx.plot(lag, (1 - rho_k) * np.clip(1 - lag / readout.WIN, 0, None) + rho_k, color=POPC[pop], lw=0.9, ls=(0, (3, 2)))
    cx.axvline(readout.WIN, color=LIGHT, lw=0.6)
    cx.set_xlim(0, lag[-1]); cx.set_ylim(0, 1.02)
    cx.set_xlabel('Lag (ms)'); cx.set_ylabel('Pre-cue z autocorrelation')
    labels(cx, [('Solid: data, dashed: model noise', INK), ('All units', POPC['all']), ('Top 20', POPC['top20']),
                ('Random 20', '#9e9e9e')], 0.98, 0.98, 0.11, 'right')
    letter(cx, 'b', x=-0.4, y=1.02)

    # c: delay z of one session with its mixture, and the model's delay z
    ex = data_z
    dx = fig.add_subplot(gs[1, 2:4])
    q = np.percentile(m['delay_z'], (0.1, 99.9))               # the paper's window, widened if the model leaves it
    lo_z, hi_z = min(-4, np.floor(q[0])), max(9, np.ceil(q[1]))
    bins = np.linspace(lo_z, hi_z, int(round((hi_z - lo_z) * 4)) + 1)
    h_data = dx.hist(ex['z'], bins, density=True, color=LIGHT, lw=0)[0]
    xs = np.linspace(lo_z, hi_z, 400)
    for q, ls in zip(np.argsort(ex['gmm_means']), ((0, (1.5, 1.5)), '-')):
        dx.plot(xs, ex['gmm_weights'][q] * norm.pdf(xs, ex['gmm_means'][q], ex['gmm_sds'][q]), color=INK, lw=0.9, ls=ls)
    h_model = dx.hist(m['delay_z'], bins, density=True, histtype='step', color=C_RO, lw=0.9)[0]
    mu_d = ex['gmm_means'].max()
    dx.axvline(mu_d, color=INK, lw=0.7, ls=(0, (3, 2)))
    dx.set_xlabel('Delay z (500-1,400 ms)'); dx.set_ylabel('Density')
    limits(dx, 'y', 0, 0.62, np.r_[h_data, h_model])
    labels(dx, [(f"Data, session {ex['session']}", '#9e9e9e'), ('Mixture: lower component', INK),
                ('Mixture: upper component', INK), ('Model', C_RO)], 0.99, 0.98, 0.1, 'right')
    dx.text(0.0, -0.36, f"Dashed: decoding strength (upper mean), session {mu_d:.2f};\nmean of sessions "
            f"{float(ex['mu_on_mean']):.2f}; model {float(m['mu_on_model']):.2f}", transform=dx.transAxes, fontsize=5.5,
            color=GREY, va='top')
    letter(dx, 'c', x=-0.36, y=1.02)

    # d: the model's decoding strength against the gain; the data's sets the gain
    ex_ = fig.add_subplot(gs[1, 4:6])
    ex_.plot(m['grid'], m['mu_on_grid'], color=C_RO, lw=1, marker='o', ms=2)
    ex_.axhline(float(m['mu_on_data']), color=INK, lw=0.7, ls=(0, (3, 2)))
    ex_.axvline(C0, color=INK, lw=0.7, ls=(0, (3, 2)))
    ex_.set_xscale('log')
    ex_.set_xlabel('Gain c (z per Hz)'); ex_.set_ylabel('Model decoding strength')
    labels(ex_, [('Data decoding strength', INK), (f'sets c = {C0:.3f}', INK)], 0.03, 0.98)
    letter(ex_, 'd', x=-0.36, y=1.02)

    # e: synthetic example trial
    fx = fig.add_subplot(gs[2, 0:2])
    tt = E['tt']
    tsw = tt[(tt >= 0) & (tt <= 1400)]
    for i, j in runs(E['dip']):
        fx.axvspan(tsw[i], tsw[j - 1], color=LIGHT, alpha=0.7, lw=0)
    fx.plot(tt, E['z_pipeline'], color=INK, lw=0.8)
    fx.plot(tt, E['z_shortcut'], color=C_RO, lw=0.8)
    thresholds(fx)
    fx.set_xlim(-200, onoff.STATE_WIN[1])
    fx.set_xlabel('Time from cue onset (ms)'); fx.set_ylabel('z')
    labels(fx, [('Decoder on synthetic spikes', INK), (r'Read-out $c\cdot\Delta r + \eta$', C_RO), ('True dips', '#9e9e9e')],
           0.99, 0.98, 0.11, 'right')
    letter(fx, 'e', x=-0.4, y=1.02)

    # f: decoding strength of the synthetic decoder vs the data's decoders
    gx = fig.add_subplot(gs[2, 2:4])
    jit = np.random.default_rng(0).uniform(-0.08, 0.08, 8)
    vals = []
    for i, (lv, lab) in enumerate(targets.items()):
        v = ses[lab]['mu_on']
        gx.errorbar(i - 0.15, v.mean(), 2 * v.std(ddof=1) / np.sqrt(len(v)), fmt='o', color=INK, ms=3, capsize=0, elinewidth=0.6)
        w = syn[syn.level == lv].mu_on_pipeline.values
        gx.plot(i + 0.15 + jit[:len(w)], w, '.', color=C_RO, ms=3, alpha=0.7, mew=0)
        gx.plot([i + 0.05, i + 0.25], [w.mean()] * 2, color=C_RO, lw=1)
        vals += [*w, v.mean() + 2 * v.std(ddof=1) / np.sqrt(len(v))]
    gx.set_xticks(range(len(targets)), [NAMES.get(lab, lab) for lab in targets.values()])
    gx.set_xlim(-0.5, len(targets) - 0.5); limits(gx, 'y', 0, 5, vals)
    gx.set_ylabel('Decoding strength')
    labels(gx, [(f"Data ({len(ses['all']['mu_on'])} sessions)", INK), (f'Synthetic decoder ({syn.level.value_counts().iloc[0]} sessions)', C_RO)], 0.03, 0.98)
    letter(gx, 'f', x=-0.36, y=1.02)

    # g: gain recovered by the calibration against the tuning strength of the synthetic neurons
    hx0 = fig.add_subplot(gs[2, 4:6])
    kk, cc = syn.k.values, syn.gain_c.values
    slope = float((kk * cc).sum() / (kk * kk).sum())
    hx0.plot(kk + np.random.default_rng(1).uniform(-0.12, 0.12, len(kk)), cc, '.', color=C_RO, ms=3, alpha=0.7, mew=0)
    kmax = limits(hx0, 'x', 0, 9, kk)[1]
    xs_ = np.linspace(0, kmax, 10)
    hx0.plot(xs_, slope * xs_, color=GREY, lw=0.7, ls=(0, (3, 2)))
    limits(hx0, 'y', 0, 0.11, cc)
    hx0.set_xlabel('Tuning strength (Hz)'); hx0.set_ylabel('Calibrated gain c')
    gm_ = syn.groupby('level').gain_c.mean()
    g = m['gains']
    labels(hx0, [('Recovered c: ' + ', '.join(f'{gm_[lv]:.3f}' for lv in LV), C_RO),
                 ('Model c: ' + ', '.join(f'{g[lab]:.3f}' for lab in targets.values()), INK), ('Dashed: proportional fit', GREY)],
           0.03, 0.98, 0.11)
    letter(hx0, 'g', x=-0.36, y=1.02)

    # h: agreement of the read-out with the decoder
    mu_lv = syn.groupby('level').mu_on_pipeline.mean()
    gx2 = fig.add_subplot(gs[3, 0:2])
    for q, (col, c_) in enumerate((('corr_z_pipeline_repeat', C_REP), ('corr_z_pipeline_vs_shortcut', C_RO))):
        s = syn.groupby('level')[col].agg(['mean', 'sem']).loc[LV]
        gx2.errorbar(np.arange(3) + (q - 0.5) * 0.2, s['mean'], 2 * s['sem'], fmt='o', color=c_, ms=3, capsize=0, elinewidth=0.6)
    gx2.set_xticks(range(3), [f'{lv.capitalize()}\n({mu_lv[lv]:.1f})' for lv in LV])
    gx2.set_xlim(-0.5, 2.5); gx2.set_ylim(0, 1)
    gx2.set_xlabel('Decoding strength'); gx2.set_ylabel('Correlation with decoder z')
    labels(gx2, [('Decoder rerun (new spikes)', C_REP), ('Read-out', C_RO)], 0.03, 0.98)
    letter(gx2, 'h', x=-0.4, y=1.02)

    # i: On/Off statistics, decoder vs read-out
    hs = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[3, 2:6], wspace=0.75)
    for q, (st, lab) in enumerate((('n_on', 'On per trial'), ('n_off', 'Off per trial'), ('frac_off', 'Fraction Off'))):
        hx = fig.add_subplot(hs[q])
        for r_, (pre, c_) in enumerate((('pipeline', INK), ('shortcut', C_RO))):
            s = syn.groupby('level')[f'{pre}_{st}'].agg(['mean', 'sem']).loc[LV]
            hx.errorbar(np.arange(3) + (r_ - 0.5) * 0.25, s['mean'], 2 * s['sem'], fmt='o', color=c_, ms=2.5, capsize=0, elinewidth=0.6)
        hx.set_xticks(range(3), ['Weak', 'Medium', 'Strong'], rotation=45, ha='right', rotation_mode='anchor')
        hx.set_xlim(-0.6, 2.6)
        hx.set_ylabel(lab, labelpad=1)
        if q == 0:
            letter(hx, 'i', x=-0.75, y=1.02)
        if q == 1:
            labels(hx, [('Decoder', INK), ('Read-out', C_RO)], 0.97, 0.98, 0.11, 'right')
    return fig
