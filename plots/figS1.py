"""Supplementary Figure 1: reading the model out like the data, and the validation on synthetic data."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from scipy.stats import norm

from nmdaint import paths, readout
from .style import C_OFF, C_ON, GREY, INK, LIGHT, letter, runs, shade

C_RO, C_REP = '#2b83ba', GREY
POPC = {'all': INK, 'top20': '#636363', 'random20': '#bdbdbd'}
FS = 6.5
LV = ['weak', 'medium', 'strong']


def thresholds(a):
    a.axhline(1.96, color=C_ON, lw=0.6, ls=(0, (3, 2)))
    a.axhline(0.84, color=C_OFF, lw=0.6, ls=(0, (3, 2)))


def labels(a, items, x, y, dy=0.12, ha='left'):
    for n, (txt, c) in enumerate(items):
        a.text(x, y - n * dy, txt, color=c, transform=a.transAxes, ha=ha, va='top', fontsize=FS)


def hlabels(a, items, loc='upper right'):
    box = HPacker(children=[TextArea(t, textprops=dict(color=c, fontsize=FS)) for t, c in items], align='baseline', pad=0, sep=8)
    a.add_artist(AnchoredOffsetbox(loc, child=box, pad=0.1, borderpad=0.2, frameon=False))


def plot():
    plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5})
    m = np.load(paths.results('ratemodel', 'readout_example.npz'), allow_pickle=True)
    syn = pd.read_csv(paths.results('synthetic', 'sessions.csv'))
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
    ax[0].set_xlim(-200, 1400)
    for a in ax:
        a.yaxis.set_label_coords(-0.045, 0.5)
    letter(ax[0], 'a', x=-0.085, y=1.0)

    # b: pre-cue z autocorrelation of three decoded populations, and of the model's noise
    cx = fig.add_subplot(gs[1, 0:2])
    za = pd.read_csv(paths.results('recordings', 'precue_acf.csv'))
    lag = np.arange(0, 310, 10)
    for pop, kind in (('all', 'all'), ('top20', 'top'), ('random20', 'random')):
        mean = za[za.set == pop].groupby('lag_ms').acf.mean()
        cx.plot(mean.index, mean.values, color=POPC[pop], lw=1.1)
        rho = readout.RHO[kind]
        cx.plot(lag, (1 - rho) * np.clip(1 - lag / 100, 0, None) + rho, color=POPC[pop], lw=0.9, ls=(0, (3, 2)))
    cx.axvline(100, color=LIGHT, lw=0.6)
    cx.set_xlim(0, 300); cx.set_ylim(0, 1.02)
    cx.set_xlabel('Lag (ms)'); cx.set_ylabel('Pre-cue z autocorrelation')
    labels(cx, [('Solid: data, dashed: model noise', INK), ('All units', POPC['all']), ('Top 20', POPC['top20']),
                ('Random 20', '#9e9e9e')], 0.98, 0.98, 0.11, 'right')
    letter(cx, 'b', x=-0.4, y=1.02)

    # c: delay z of one session with its mixture, and the model's delay z
    ex = np.load(paths.results('recordings', 'example_delay_z.npz'))
    dx = fig.add_subplot(gs[1, 2:4])
    bins = np.linspace(-4, 9, 53)
    dx.hist(ex['z'], bins, density=True, color=LIGHT, lw=0)
    xs = np.linspace(-4, 9, 400)
    for q, ls in zip(np.argsort(ex['gmm_means']), ((0, (1.5, 1.5)), '-')):
        dx.plot(xs, ex['gmm_weights'][q] * norm.pdf(xs, ex['gmm_means'][q], ex['gmm_sds'][q]), color=INK, lw=0.9, ls=ls)
    dx.hist(m['delay_z'], bins, density=True, histtype='step', color=C_RO, lw=0.9)
    mu_d = ex['gmm_means'].max()
    dx.axvline(mu_d, color=INK, lw=0.7, ls=(0, (3, 2)))
    dx.set_xlabel('Delay z (500-1,400 ms)'); dx.set_ylabel('Density'); dx.set_ylim(0, 0.62)
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
    E = np.load(paths.results('synthetic', 'example.npz'))
    fx = fig.add_subplot(gs[2, 0:2])
    tt = E['tt']
    tsw = tt[(tt >= 0) & (tt <= 1400)]
    for i, j in runs(E['dip']):
        fx.axvspan(tsw[i], tsw[j - 1], color=LIGHT, alpha=0.7, lw=0)
    fx.plot(tt, E['z_pipeline'], color=INK, lw=0.8)
    fx.plot(tt, E['z_shortcut'], color=C_RO, lw=0.8)
    thresholds(fx)
    fx.set_xlim(-200, 1400)
    fx.set_xlabel('Time from cue onset (ms)'); fx.set_ylabel('z')
    labels(fx, [('Decoder on synthetic spikes', INK), (r'Read-out $c\cdot\Delta r + \eta$', C_RO), ('True dips', '#9e9e9e')],
           0.99, 0.98, 0.11, 'right')
    letter(fx, 'e', x=-0.4, y=1.02)

    # f: decoding strength of the synthetic decoder vs the data's decoders
    ses, _ = readout.load_data()
    gx = fig.add_subplot(gs[2, 2:4])
    jit = np.random.default_rng(0).uniform(-0.08, 0.08, 8)
    for i, (lv, lab, name) in enumerate((('weak', 'rand 20', 'Random 20'), ('medium', 'all', 'All units'), ('strong', 'top 20', 'Top 20'))):
        v = ses[lab]['mu_on']
        gx.errorbar(i - 0.15, v.mean(), 2 * v.std(ddof=1) / np.sqrt(len(v)), fmt='o', color=INK, ms=3, capsize=0, elinewidth=0.6)
        w = syn[syn.level == lv].mu_on_pipeline.values
        gx.plot(i + 0.15 + jit[:len(w)], w, '.', color=C_RO, ms=3, alpha=0.7, mew=0)
        gx.plot([i + 0.05, i + 0.25], [w.mean()] * 2, color=C_RO, lw=1)
    gx.set_xticks(range(3), ['Random 20', 'All units', 'Top 20'])
    gx.set_xlim(-0.5, 2.5); gx.set_ylim(0, 5)
    gx.set_ylabel('Decoding strength')
    labels(gx, [('Data (25 sessions)', INK), (f'Synthetic decoder ({syn.level.value_counts().iloc[0]} sessions)', C_RO)], 0.03, 0.98)
    letter(gx, 'f', x=-0.36, y=1.02)

    # g: gain recovered by the calibration against the tuning strength of the synthetic neurons
    hx0 = fig.add_subplot(gs[2, 4:6])
    kk, cc = syn.k.values, syn.gain_c.values
    slope = float((kk * cc).sum() / (kk * kk).sum())
    hx0.plot(kk + np.random.default_rng(1).uniform(-0.12, 0.12, len(kk)), cc, '.', color=C_RO, ms=3, alpha=0.7, mew=0)
    xs_ = np.linspace(0, 9, 10)
    hx0.plot(xs_, slope * xs_, color=GREY, lw=0.7, ls=(0, (3, 2)))
    hx0.set_xlim(0, 9); hx0.set_ylim(0, 0.11)
    hx0.set_xlabel('Tuning strength (Hz)'); hx0.set_ylabel('Calibrated gain c')
    gm_ = syn.groupby('level').gain_c.mean()
    g = m['gains'].item()
    labels(hx0, [(f"Recovered c: {gm_['weak']:.3f}, {gm_['medium']:.3f}, {gm_['strong']:.3f}", C_RO),
                 (f"Model c: {g['rand 20']:.3f}, {g['all']:.3f}, {g['top 20']:.3f}", INK), ('Dashed: proportional fit', GREY)],
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
