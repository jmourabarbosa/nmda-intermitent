"""Figure 1: an example trial of the rate model read out at three gains, and the On/Off length distributions."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

from nmdaint import ratemodel
from .style import C_OFF, C_ON, C_UN, GREY, INK, LIGHT, letter, runs, shade

QUIET, DIPS, CUE = 1000, 1500, 75
SHADES = {'off': ('#16606d', C_OFF, '#86d3e0'), 'on': ('#9c1215', C_ON, '#f39a9c')}
Z_INK = ('#7f7f7f', INK, '#bdbdbd')


def draw(e, lengths, gains):
    """e: the example trial (t, rc, ru, sc, su, m at 1 ms; tz, dw, eta, state of the read-out; gain).
    gains: {name: gain relative to the calibrated one}, three of them, the calibrated one (1) in the middle;
    lengths: {name: onoff.stats of all simulated trials read out at that gain}."""
    GAINS = [(k, f, k.capitalize()) for k, f in gains.items()]
    t, tz, state = e['t'], e['tz'], e['state']
    r_max = ratemodel.FIXED['r_max']
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 0.35, 2.0], hspace=0.22, left=0.08, right=0.71, top=0.9, bottom=0.14)
    gh = fig.add_gridspec(2, 1, hspace=0.45, left=0.85, right=0.99, top=0.9, bottom=0.14)
    a, b, d = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    # a, top: pool rates and NMDA gating
    shade(a, tz, state)
    a.axvspan(0, CUE, color=LIGHT, lw=0)
    a.plot(t, e['rc'] / r_max, color=INK)
    a.plot(t, e['ru'] / r_max, color=C_UN)
    a.plot(t, e['sc'], color=INK, ls=(0, (1, 1.5)))
    a.plot(t, e['su'], color=C_UN, ls=(0, (1, 1.5)))
    a.set_ylim(0, 1); a.set_yticks([0, 0.5, 1], ['0', '', '1']); a.set_ylabel('Rate / NMDA')
    xa = a.get_xaxis_transform()
    for dx, label in ((1, 'With attention fluctuations'), (-1, 'Without attention fluctuations')):
        a.annotate('', xy=(QUIET + dx * 160, 1.06), xytext=(QUIET + dx * 15, 1.06), xycoords=xa, textcoords=xa,
                   arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.6, mutation_scale=5), annotation_clip=False)
        a.text(QUIET + dx * 180, 1.06, label, transform=xa, fontsize=7.5, va='center', ha='left' if dx > 0 else 'right')
    a.text(500, e['sc'][(t > 400) & (t < 600)].max() + 0.04, 'NMDA', fontsize=7.5, ha='center', va='bottom')
    a.text(500, e['rc'][(t > 400) & (t < 600)].min() / r_max - 0.04, 'Rate', fontsize=7.5, ha='center', va='top')
    off0 = runs(state == -1)[0]
    on1 = next((a0, b0) for a0, b0 in runs(state == 1) if tz[a0] > QUIET)
    a.text((tz[on1[0]] + tz[on1[1] - 1]) / 2, 0.96, 'On', color=C_ON, fontsize=7.5, va='top', ha='center')
    a.text((tz[off0[0]] + tz[off0[1] - 1]) / 2, 0.96, 'Off', color=C_OFF, fontsize=7.5, va='top', ha='center')
    letter(a, 'a')
    # a, middle: common input
    b.plot(t, e['m'], color=INK, lw=0.7)
    b.set_ylim(0, 1.1); b.set_yticks([])
    b.spines['left'].set_visible(False); b.spines['bottom'].set_visible(False); b.tick_params(bottom=False)
    b.text(-0.01, 0.5, 'Attention\nsignal', transform=b.transAxes, ha='right', va='center', fontsize=7.5)
    # a, bottom: decoding z at three gains, same noise
    shade(d, tz, state)
    for (_, f, _), c in zip(GAINS, Z_INK):
        d.plot(tz, f * e['gain'] * e['dw'] + e['eta'], color=c, lw=0.9 if f == 1.0 else 0.7, zorder=3 if f == 1.0 else 2)
    box = HPacker(children=[TextArea(lab + ' gain', textprops=dict(color=c, fontsize=7.5)) for (_, _, lab), c in zip(GAINS, Z_INK)],
                  align='baseline', pad=0, sep=7)
    d.add_artist(AnchoredOffsetbox('upper right', child=box, pad=0.1, borderpad=0.3, frameon=False))
    for y, c, txt in ((0, GREY, ''), (1.96, C_ON, 'P < 0.05'), (0.84, C_OFF, 'P > 0.20')):
        d.axhline(y, color=c, ls=(0, (3, 2)), lw=0.6)
        d.text(1.005, y, txt, transform=d.get_yaxis_transform(), color=c, fontsize=7.5, va='center')
    z = e['gain'] * e['dw'] + e['eta']
    d.set_ylim(z.min() - 0.3, max(gains.values()) * e['gain'] * e['dw'].max() + e['eta'].max() + 0.3)
    d.set_yticks([0, 5, 10], ['0', '', '10'])
    d.set_ylabel('Decoding z')
    d.set_title(r'$z(t) = c\,[r_\mathrm{cued}(t) - r_\mathrm{uncued}(t)] + \eta(t)$', fontsize=7.5, loc='left', pad=2)
    d.set_xlabel('Time from cue onset (ms)')
    for ax in (a, b, d):
        ax.axvline(QUIET, color=GREY, ls=(0, (3, 2)), lw=0.6)
        ax.set_xlim(-300, QUIET + DIPS)
    for ax in (a, b):
        ax.tick_params(labelbottom=False)
    # b: length distributions at each gain, fraction of states in 40 ms bins
    for j, (name, c) in enumerate((('off', C_OFF), ('on', C_ON))):
        h = fig.add_subplot(gh[j], sharex=fig.axes[-1] if j else None)
        top = 0
        for i, (key, _, lab) in enumerate(GAINS):
            cnt = lengths[key][f'h_{name}']
            hm = (cnt[:30][0::2] + cnt[:30][1::2]) / max(cnt.sum(), 1)
            w = 40 / len(GAINS)
            h.bar(np.arange(0, 600, 40) + i * w, hm, width=w, align='edge', color=SHADES[name][i], lw=0)
            top = max(top, hm.max())
            h.text(0.97, 0.66 - 0.17 * i, f"{lab}, {float(lengths[key][f'n_{name}']):.1f}/trial", transform=h.transAxes,
                   ha='right', va='top', color=SHADES[name][i], fontsize=6.5)
        h.set_xlim(0, 620); h.set_xticks([0, 200, 400, 600])
        top = np.ceil(top * 10) / 10
        h.set_ylim(0, top); h.set_yticks([0, top / 2, top], ['0', '', f'{top:g}'])
        h.set_ylabel('Fraction\nof states', labelpad=-14)
        h.text(0.97, 0.9, f'{name.capitalize()} states', transform=h.transAxes, ha='right', va='top', color=c, fontsize=7.5)
        if j == 0:
            h.tick_params(labelbottom=False)
            letter(h, 'b', x=-0.55, y=1.0)
        else:
            h.set_xlabel('State length (ms)')
    return fig
