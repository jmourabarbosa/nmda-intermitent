"""Plot of review/out/data_error_timecourse.npz: mean z, P(On), P(Off) against time, correct vs error trials
(sessions with >= 20 errors, averaged over sessions)."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
res = [r for r in np.load('review/out/data_error_timecourse.npz', allow_pickle=True)['res'] if r['n_e'] >= 20]
fig, ax = plt.subplots(1, 3, figsize=(11, 3))
for a, (k, lab) in zip(ax, (('z', 'decoder z'), ('on', 'P(On)'), ('off', 'P(Off)'))):
    t = res[0]['t'] if k == 'z' else res[0]['tsw']
    for suf, col, nm in (('_c', 'k', 'correct'), ('_e', 'tab:red', 'error')):
        m = np.mean([r[k + suf] for r in res], 0)
        a.plot(t, m, color=col, label=nm)
    a.axvspan(100, 400, color='grey', alpha=0.15); a.set_xlabel('time from cue (ms)'); a.set_title(lab)
ax[0].legend(); fig.suptitle(f'{len(res)} sessions with >= 20 errors; shaded: 100-400 ms (before a 120-150 ms trace could be lost)', fontsize=9)
fig.tight_layout(); fig.savefig('review/out/data_error_timecourse.png', dpi=120)
