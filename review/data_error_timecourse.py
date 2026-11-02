"""Real data: when do error trials depart from correct trials? (reverse-causality check of Fig. 3)

If errors are CAUSED by long Off states that outlast an NMDA trace (paper), error trials should be decoded as well
as correct trials just after the cue (encoding intact) and diverge later in the delay. If errors come from
encoding failures, wrong-item memories or lapses of engagement, decoding is already weaker right after the cue,
before any Off state could have erased a 120-150 ms trace, and the Off-error association is (partly) reverse causal.

Per session, all-units decoder exactly as scripts/recordings.py (correct: leave-one-out; error: decoders fitted on
all correct trials). Saved: mean z and P(On), P(Off) against time, for correct and error trials.
usage: python review/data_error_timecourse.py   (needs ../panichello; ~30-60 min on 4 cores)
"""
import glob, os
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from nmdaint import paths, recordings


def run(f):
    S = recordings.Session(f)
    key = [k for k in S.sets if k[0] == 'all'][0]
    z, st = S.states(S.confidence(S.sets[key]))
    ze, ste = S.states(S.error_confidence())
    tsw = S.t[S.sw]
    # control: correct trials decoded like the error trials (decoders fitted on ALL correct trials, in-sample)
    return dict(name=S.name, t=S.t, tsw=tsw, z_c=z.mean(0), z_e=ze.mean(0), n_c=len(z), n_e=len(ze),
                on_c=(st == 1).mean(0), on_e=(ste == 1).mean(0), off_c=(st == -1).mean(0), off_e=(ste == -1).mean(0),
                early_z_c=z[:, (S.t >= 100) & (S.t <= 400)].mean(1), early_z_e=ze[:, (S.t >= 100) & (S.t <= 400)].mean(1),
                pre_z_c=z[:, S.t < 0].mean(1), pre_z_e=ze[:, S.t < 0].mean(1))


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(paths.DATA, '*.mat')), key=lambda f: -os.path.getsize(f))
    res = Parallel(n_jobs=4, verbose=5)(delayed(run)(f) for f in files)
    np.savez_compressed('review/out/data_error_timecourse.npz', res=np.array(res, dtype=object))
    rows = []
    for r in res:
        if r['n_e'] < 5:
            continue
        rows.append(dict(session=r['name'], monkey=recordings.monkey(r['name']), n_err=r['n_e'],
                         pre_z_c=r['pre_z_c'].mean(), pre_z_e=r['pre_z_e'].mean(),
                         early_z_c=r['early_z_c'].mean(), early_z_e=r['early_z_e'].mean(),
                         delay_z_c=r['z_c'][(r['t'] >= 500) & (r['t'] <= 1400)].mean(),
                         delay_z_e=r['z_e'][(r['t'] >= 500) & (r['t'] <= 1400)].mean()))
    d = pd.DataFrame(rows)
    d.to_csv('review/out/data_error_timecourse.csv', index=False)
    from scipy.stats import wilcoxon
    pd.set_option('display.width', 200)
    print(d.round(3).to_string(index=False))
    for w in ('pre', 'early', 'delay'):
        print(f'{w:5s}: correct {d[w + "_z_c"].mean():.3f}  error {d[w + "_z_e"].mean():.3f}  '
              f'Wilcoxon p = {wilcoxon(d[w + "_z_c"], d[w + "_z_e"]).pvalue:.1e}  (n = {len(d)} sessions)')
    d['early_ratio'] = d.early_z_e / d.early_z_c
    d['delay_ratio'] = d.delay_z_e / d.delay_z_c
    print('error/correct z ratio: early (100-400 ms) median', round(d.early_ratio.median(), 3),
          ' delay (500-1400 ms) median', round(d.delay_ratio.median(), 3))
