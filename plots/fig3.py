"""Figure 3: single-dip test of the two models, Off states and errors, and the NMDA time constant fitted to the
monkeys' performance."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from nmdaint.behaviour import wilson
from .style import C_FIT, C_OFF, C_ON, C_STP, GREY, INK

DPI = 300
STATS = (('frac_off', 'Total Off time', C_OFF), ('off_len_ms', 'Off length', C_OFF), ('n_off', 'Off states', C_OFF),
         ('frac_on', 'Total On time', C_ON))


def letter(ax, txt, x=-0.3, y=1.04):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom')


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def draw(single_dip, glmm, weights, tau):
    """single_dip: {model: (dip lengths, trials, memory held)}; glmm, weights: behaviour.glmm_single; tau:
    behaviour.fit_tau with 'bin_ms' and 'p_correct_model'."""
    plt.rcParams.update({'xtick.minor.size': 1.5})
    FW, FH, SIDE, BOT = 7.2, 1.95, 1.12, 0.48
    fig = plt.figure(figsize=(FW, FH))
    sx, a, bx, dx = (fig.add_axes([x0 / FW, BOT / FH, SIDE / FW, SIDE / FH]) for x0 in (0.55, 2.72, 4.37, 6.0))
    # a: single-dip test, both models (no read-out)
    for model, col in (('nmda', INK), ('stp', C_STP)):
        L, n, held = single_dip[model]
        lo, hi = wilson(held, n)
        sx.fill_between(L, lo, hi, color=col, alpha=0.18, lw=0)
        sx.plot(L, held / n, '-', color=col, lw=1.2)
    sx.set_xlim(0, 1000); sx.set_xticks([0, 500, 1000]); sx.set_ylim(0, 1.03)
    sx.set_yticks([0, 0.5, 1], ['0', '', '1']); sx.yaxis.labelpad = 1
    sx.set_xlabel('Silence duration (ms)'); sx.set_ylabel('Fraction memory held')
    sx.set_title('Model prediction', fontsize=8, pad=4)
    sx.text(980, 0.92, 'STP model', color=C_STP, fontsize=7, ha='right', va='top')
    sx.text(450, 0.45, 'NMDA model', color=INK, fontsize=7, ha='left', va='center')
    letter(sx, 'a')
    # b: effect of each statistic on the log-odds of an error, per within-session SD
    g, sw = glmm.set_index('predictor'), weights
    jit = np.random.default_rng(0)
    for q, (stat, name, col) in enumerate(STATS):
        r, s = g.loc[stat], sw[sw.predictor == stat]
        a.plot(np.log(s.session_or), q + jit.uniform(-0.18, 0.18, len(s)), '.', ms=2.5, color=GREY, alpha=0.5, mew=0)
        a.plot([np.log(r.or_ci_lo), np.log(r.or_ci_hi)], [q, q], color=col, lw=1.4, solid_capstyle='butt')
        a.plot(np.log(r.odds_ratio_per_sd), q, 'o', ms=4, color=col)
        a.text(np.log(r.odds_ratio_per_sd), q - (0.22 if r.p_lrt < 0.05 else 0.36), stars(r.p_lrt), ha='center',
               va='center', fontsize=7.5)
    a.axvline(0, color=GREY, ls=(0, (3, 2)), lw=0.6)
    a.set_xlim(-0.7, 0.8); a.set_xticks([-0.5, 0, 0.5], ['−0.5', '0', '0.5'])
    a.set_yticks(range(len(STATS)), [r[1] for r in STATS]); a.set_ylim(len(STATS) - 0.5, -0.5)
    a.tick_params(axis='y', length=0)
    a.set_xlabel(r'$\beta$')
    letter(a, 'b', x=-0.88)
    # c: fraction correct against Off time, data and the best-fitting NMDA model
    t = tau
    k, n, cen = t['k'], t['n'], t['bin_ms']
    pk = k / n
    lo, hi = wilson(k, n)
    bx.plot(cen, t['p_correct_model'], color=C_FIT, lw=1.2)
    bx.errorbar(cen, pk, yerr=[pk - lo, hi - pk], fmt='o', ms=3, color=INK, elinewidth=0.6, capsize=0)
    bx.set_xlim(0, 900); bx.set_xticks([0, 300, 600, 900]); bx.set_ylim(0.5, 1)
    bx.set_yticks([0.5, 0.75, 1], ['0.5', '', '1']); bx.yaxis.labelpad = -6
    bx.set_xlabel('Total Off time (ms)'); bx.set_ylabel('Fraction correct')
    bx.text(30, 0.62, 'Monkeys', fontsize=6.5, color=INK)
    bx.text(30, 0.54, rf"NMDA, $\tau_N$ = {float(t['tau']):.0f} ms", fontsize=6.5, color=C_FIT)
    letter(bx, 'c')
    # d: bootstrap over sessions
    lo95, hi95 = np.percentile(t['boot'], (2.5, 97.5))
    dx.hist(t['boot'], bins=np.exp(np.linspace(np.log(20), np.log(1500), 41)), color=C_FIT, alpha=0.8, lw=0)
    dx.set_xscale('log'); dx.set_xlim(20, 1500)
    dx.set_xticks([30, 100, 300, 1000], ['30', '100', '300', '1000'])
    dx.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for x in (lo95, hi95):
        dx.axvline(x, color=C_FIT, lw=0.6, ls=(0, (1, 1.5)))
    dx.axvline(float(t['tau']), color=INK, lw=0.6, ls=(0, (3, 2)))
    dx.set_xlabel(r'Best $\tau_N$ (ms)'); dx.set_ylabel('Bootstrap samples')
    dx.text(0.62, 0.62, f"{float(t['tau']):.0f} ms\n95% CI\n{lo95:.0f}-{hi95:.0f}", transform=dx.transAxes, ha='left',
            va='center', fontsize=6.5)
    letter(dx, 'd')
    return fig
