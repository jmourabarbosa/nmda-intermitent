"""Figure 4: the mechanism in a spiking ring-attractor network, and the population code in On and Off states."""
import matplotlib.pyplot as plt
import numpy as np

from nmdaint import spiking
from .style import C_NET, C_OFF, C_ON, GREY, INK, LIGHT, letter, runs

QUIET, XL = 1000, (-300, 2000)


def shade(ax, t, state, step):
    for lab, c in ((1, C_ON), (-1, C_OFF)):
        for a, b in runs(state == lab):
            ax.axvspan(t[a] - step / 2, t[b - 1] + step / 2, color=c, alpha=0.15, lw=0)


def cue_colour(deg):
    return plt.cm.hsv((deg % 360) / 360 % 1)


def pca_scatter(ax, rows, title):
    """Ring of the 8 cue means (grey) and, per cue, the mean position during On (filled) and Off (open) moments."""
    deg = rows.angle_deg.values
    ring = rows[['ring_pc1', 'ring_pc2']].values
    o = np.argsort(deg)
    ax.fill(ring[o, 0], ring[o, 1], facecolor='none', edgecolor=LIGHT, lw=1.0)
    for (_, r), dg in zip(rows.iterrows(), deg):
        c = cue_colour(dg)
        on, off = (r.on_pc1, r.on_pc2), (r.off_pc1, r.off_pc2)
        ax.annotate('', xy=off, xytext=on, arrowprops=dict(arrowstyle='-|>', color=c, lw=0.7, mutation_scale=6, shrinkA=3, shrinkB=3))
        ax.plot(*on, 'o', ms=4.5, mfc=c, mec=INK, mew=0.4)
        ax.plot(*off, 'o', ms=4.5, mfc='white', mec=c, mew=1.0)
    lim = np.abs(np.r_[ring.ravel(), rows[['on_pc1', 'on_pc2']].values.ravel()]).max() * 1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.axis('off')
    x0, y0, L = -lim * 0.95, -lim * 0.95, lim * 0.3
    for dx, dy in ((L, 0), (0, L)):
        ax.annotate('', xy=(x0 + dx, y0 + dy), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=GREY, lw=0.6, mutation_scale=5))
    ax.text(x0 + L, y0 - lim * 0.05, 'PC1', ha='right', va='top', fontsize=6, color=GREY)
    ax.text(x0 - lim * 0.05, y0 + L, 'PC2', ha='right', va='top', fontsize=6, color=GREY, rotation=90)
    ax.set_title(title, fontsize=7.5, pad=3)


def pca_panels(fig, planes, procrustes):
    d1 = fig.add_axes([0.675, 0.06, 0.15, 0.36]); d2 = fig.add_axes([0.85, 0.06, 0.15, 0.36])
    pca_scatter(d1, planes['network'], 'Network')
    pca_scatter(d2, planes['data'], 'Data')
    letter(d1, 'd', x=-0.1, y=1.02)
    key = d2.inset_axes([0.0, 0.7, 0.3, 0.3]); key.set_aspect('equal'); key.axis('off')
    for k in range(8):
        a = np.radians(45 * k); key.plot(np.cos(a), np.sin(a), 'o', ms=2.5, color=cue_colour(45 * k))
    key.plot(0, 0, '+', color=GREY, ms=3, mew=0.5)
    key.set_xlim(-1.5, 1.5); key.set_ylim(-1.5, 1.5)
    key.text(0, -1.6, 'Cue location', ha='center', va='top', fontsize=6, color=GREY)
    lg = d2.inset_axes([0.72, 0.0, 0.28, 0.2]); lg.axis('off'); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
    lg.plot(0.1, 0.75, 'o', ms=4.5, mfc=GREY, mec=INK, mew=0.4); lg.text(0.3, 0.75, 'On', va='center', fontsize=6.5)
    lg.plot(0.1, 0.3, 'o', ms=4.5, mfc='white', mec=GREY, mew=1.0); lg.text(0.3, 0.3, 'Off', va='center', fontsize=6.5)
    # e: normalised Procrustes similarity in the 7 PCs of the cue means
    e = fig.add_axes([0.885, 0.6, 0.11, 0.26])
    x = np.array([0, 1, 2])
    for name, dx, col in (('data', -0.12, INK), ('network', 0.12, C_NET)):
        D = procrustes[name]
        M = D[D.space == 'pcs'].pivot(index='session', columns='period', values='normalised')[['on', 'off', 'pre']].values
        for row in M:
            e.plot(x[:2] + dx, row[:2], color=col, lw=0.4, alpha=0.3)
            e.plot(x[2] + dx, row[2], 'o', ms=1.5, color=col, alpha=0.4)
        mu, se = np.nanmean(M, 0), np.nanstd(M, 0, ddof=1) / np.sqrt((~np.isnan(M)).sum(0))
        e.plot(x[:2] + dx, mu[:2], color=col, lw=1.2)
        e.errorbar(x + dx, mu, yerr=se, fmt='o', ms=3.5, color=col, lw=1.0, capsize=0)
    e.axhline(0, color=GREY, ls=(0, (3, 2)), lw=0.6)
    e.set_xticks(x, ['On', 'Off', 'Pre']); e.set_xlim(-0.4, 2.4); e.set_ylim(-0.4, 1.05); e.set_yticks([0, 0.5, 1], ['0', '', '1'])
    e.set_ylabel('Shape similarity\nto cue ring', labelpad=1)
    e.text(0.6, 0.8, 'Data', transform=e.transAxes, color=INK, fontsize=6, va='top')
    e.text(0.6, 0.68, 'Network', transform=e.transAxes, color=C_NET, fontsize=6, va='top')
    letter(e, 'e', x=-0.6, y=1.05)


def draw(tr, lengths, tuning, planes, procrustes):
    """tr: the example trial (spike_t, spike_cell, t1, m_eff, tz, z, state); lengths: onoff.stats of the sessions'
    all-units read-out (h_on, h_off, n_on, n_off); tuning: spiking_readout.tuning; planes, procrustes: {'data',
    'network'}: population.ring_and_centroids tables and Procrustes tables (session, period, space, normalised)."""
    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 0.35, 1.6], hspace=0.22, left=0.075, right=0.37, top=0.86, bottom=0.14)
    gb = fig.add_gridspec(2, 1, hspace=0.4, left=0.48, right=0.6, top=0.86, bottom=0.14)
    gc = fig.add_gridspec(1, 1, left=0.705, right=0.825, top=0.86, bottom=0.6)
    a, m, d = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    # a, top: spikes as counts in 5 ms x 8-cell bins, cells in order of preferred location
    n_e = spiking.P['N_E']
    te, ce = np.arange(-500, 2001, 5), np.arange(0, n_e + 1, 8)
    H, _, _ = np.histogram2d(tr['spike_t'], tr['spike_cell'], bins=(te, ce))
    a.imshow(H.T, origin='lower', aspect='auto', cmap='Greys', vmin=0, vmax=np.percentile(H[H > 0], 95),
             extent=(te[0], te[-1], 0, 360), interpolation='nearest')
    a.axvspan(0, 50, color=LIGHT, alpha=0.6, lw=0, zorder=0)
    a.set_ylim(0, 360); a.set_yticks([0, 180, 360]); a.set_ylabel('Location (deg)')
    xa = a.get_xaxis_transform()
    for dx, label in ((1, 'Fluctuating attention'), (-1, 'Constant attention')):
        a.annotate('', xy=(QUIET + dx * 110, 1.06), xytext=(QUIET + dx * 15, 1.06), xycoords=xa, textcoords=xa,
                   arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.6, mutation_scale=5), annotation_clip=False)
        a.text(QUIET + dx * 125, 1.06, label, transform=xa, fontsize=6.5, va='center', ha='left' if dx > 0 else 'right')
    a.text(25, 340, 'Cue', fontsize=7, ha='center', va='top', color=GREY)
    letter(a, 'a')
    # a, middle: common input with its overshoot after each dip
    m.plot(tr['t1'], tr['m_eff'], color=INK, lw=0.7)
    m.set_ylim(0.3, max(1.6, tr['m_eff'].max() + 0.05)); m.set_yticks([])
    for s in ('left', 'bottom'):
        m.spines[s].set_visible(False)
    m.tick_params(bottom=False)
    m.text(-0.01, 0.5, 'Attention\nsignal', transform=m.transAxes, ha='right', va='center', fontsize=7.5)
    # a, bottom: all-units decoder z with On/Off states
    tz, z = tr['tz'], tr['z']
    shade(d, tz, tr['state'], 10)
    d.plot(tz, z, color=INK)
    for y, c, txt in ((0, GREY, ''), (1.96, C_ON, 'P < 0.05'), (0.84, C_OFF, 'P > 0.20')):
        d.axhline(y, color=c, ls=(0, (3, 2)), lw=0.6)
        d.text(1.005, y, txt, transform=d.get_yaxis_transform(), color=c, fontsize=7, va='bottom' if y > 1 else 'top')
    d.set_ylim(-3, max(12, np.ceil(z[tz <= XL[1]].max()) + 1)); d.set_yticks([0, 5, 10])
    d.set_ylabel('Decoding z')
    d.set_xlabel('Time from cue onset (ms)')
    for ax in (a, m, d):
        ax.axvline(QUIET, color=GREY, ls=(0, (3, 2)), lw=0.6)
        ax.set_xlim(*XL)
    for ax in (a, m):
        ax.tick_params(labelbottom=False)
    # b: Off and On lengths of the all-units decoder
    hl = lengths
    for j, (name, col) in enumerate((('off', C_OFF), ('on', C_ON))):
        h = fig.add_subplot(gb[j])
        cnt = hl[f'h_{name}'].astype(float)
        frac = (cnt[:30][0::2] + cnt[:30][1::2]) / cnt.sum()          # 40 ms bins; states >= 600 ms not shown
        h.bar(np.arange(0, 600, 40), frac, width=40, align='edge', color=col, ec='white', lw=0.4)
        h.text(0.97, 0.8, f"{float(hl[f'n_{name}']):.1f}/trial", transform=h.transAxes, ha='right', va='top', color=col, fontsize=6.5)
        h.text(0.97, 0.97, f'{name.capitalize()} states', transform=h.transAxes, ha='right', va='top', color=col, fontsize=7.5)
        h.set_xlim(0, 600); h.set_xticks([0, 300, 600])
        h.set_ylim(0, 0.4); h.set_yticks([0, 0.2, 0.4], ['0', '', '0.4'])
        h.set_ylabel('Fraction\nof states', labelpad=-14)
        if j == 0:
            letter(h, 'b', x=-0.55, y=1.0)
            h.tick_params(labelbottom=False)
        else:
            h.set_xlabel('State length (ms)')
    # c: cross-validated tuning in On and Off states (mean +- s.e.m. over cells)
    c = fig.add_subplot(gc[0])
    tun = tuning
    x = np.arange(-4, 5) * 45
    for name, col, lab in (('on', C_ON, 'On states'), ('off', C_OFF, 'Off states')):
        v = tun[name][:, [(k + 4) % 8 for k in range(9)]]          # cue offsets -180 ... 180 deg
        c.errorbar(x, v.mean(0), yerr=v.std(0) / np.sqrt(len(v)), color=col, marker='o', ms=2.5, lw=0.9, capsize=0)
        c.text(-170, 2.0 if name == 'on' else 1.74, lab, color=col, fontsize=6.5, va='top')
    c.axhline(0, color=GREY, ls=(0, (3, 2)), lw=0.6)
    c.set_xticks([-180, 0, 180]); c.set_xlim(-195, 195)
    c.set_ylim(-0.8, 2.0); c.set_yticks([-0.5, 0, 0.5, 1.0, 1.5, 2.0], ['', '0', '', '1', '', '2'])
    c.set_xlabel('Cue from pref. (deg)')
    c.set_ylabel('Firing rate (z)')
    letter(c, 'c', x=-0.5, y=1.05)
    pca_panels(fig, planes, procrustes)
    return fig
