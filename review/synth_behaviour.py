"""Ground-truth test of Fig. 3b-d: do the GLMM pattern and the tau fit identify an NMDA trace?

Synthetic sessions copy the data's structure (25 sessions, trial counts, monkey, session error rates). On/Off states
come from the paper's own model read-out (z = c dr + eta, all-units gain and rho) of simulated trials. Worlds:

  nmda_tau{80,150,300}  NMDA model with that tau_N; memory loss from the model's own dynamics (Off -> loss);
                        P(correct) = p0_s [held + (1 - held) / 8].  Positive controls / parameter recovery.
  stp_lapse             STP model (memory never lost); errors independent of the memory. Negative control.
  stp_encode_fail       STP model; in a fraction F of trials the cue was never encoded (dr = 0 all trial); guess.
  stp_wrong_item        STP model; in F trials a wrong location is held all trial (dr x cos(dtheta)); guess.
  stp_random_loss       STP model; in F trials the memory is lost at a uniform random time of the delay,
                        independent of the dips, after which a random wrong item is read (dr x cos(dtheta)).

In the STP/other worlds, Off states never cause errors; errors cause Off (reverse causality). If the paper's
analyses also return significant total-Off and Off-length effects, a null number-of-Off effect and a tau of
~100-200 ms there, they cannot tell an NMDA trace from these alternatives.
usage: python review/synth_behaviour.py   (~15 min on 4 cores)
"""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from nmdaint import behaviour, memory, paths, ratemodel, readout
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import common

F_ERR = 0.08            # fraction of trials with a memory failure in the alternative worlds
RHO = readout.RHO['all']
data = pd.read_csv(paths.results('recordings', 'per_trial.csv'))
sess = data.groupby('session').agg(n=('correct', 'size'), acc=('correct', 'mean'), monkey=('monkey', 'first'))
N_TOT = int(sess.n.sum())


def simulate_dw(p, n, seed):
    out = []
    for b in range(int(np.ceil(n / 1000))):
        sim = ratemodel.simulate(p, 1000, seed=seed + b)
        dw, tz = readout.rate_diff(sim)
        out.append((dw, ratemodel.memory_correct(sim)))
    return np.concatenate([o[0] for o in out])[:n], np.concatenate([o[1] for o in out])[:n], tz


def world(name, gains, seed=0):
    rng = np.random.default_rng(seed)
    model = 'nmda' if name.startswith('nmda') else 'stp'
    p = ratemodel.params(model)
    if name.startswith('nmda_tau'):
        p = {**p, 'tau_N': float(name[8:])}
    dw, held, tz = simulate_dw(p, N_TOT, 70000 + 1000 * seed)
    held = held.astype(float)
    fail = np.zeros(N_TOT, bool)
    if model == 'stp':
        held[:] = 1.0                                  # STP holds the memory (100% in the fitted model)
        if name != 'stp_lapse':
            fail = rng.random(N_TOT) < F_ERR
            wrong = np.cos(np.pi / 4 * rng.integers(1, 8, N_TOT))      # projection of a random wrong item
            if name == 'stp_encode_fail':
                dw[fail] = 0.0
            elif name == 'stp_wrong_item':
                dw[fail] *= wrong[fail, None]
            elif name == 'stp_random_loss':
                tl = rng.uniform(0, 1400, N_TOT)
                after = tz[None] > tl[:, None]
                dw = np.where(fail[:, None] & after, dw * wrong[:, None], dw)
            held[fail] = 0.0
    z, st, ts = common.label_trials(dw, tz, gains[model], RHO, rng)
    # sessions as in the data; p0 per session so that error rates resemble the data's
    s_idx = np.repeat(np.arange(len(sess)), sess.n.values)
    rng.shuffle(s_idx)
    m_hold = held.mean()
    p0 = np.clip(sess.acc.values / (m_hold + (1 - m_hold) / 8), 0.3, 0.995)[s_idx]
    pc = p0 * (held + (1 - held) / 8)
    correct = (rng.random(N_TOT) < pc).astype(int)
    df = pd.DataFrame(dict(session=sess.index.values[s_idx], monkey=sess.monkey.values[s_idx], correct=correct, **ts))
    return name, df, float(held.mean())


def analyse(name, df, held, fine, Hf):
    g, _ = behaviour.glmm_single(df)
    row = dict(world=name, memory_held=round(held, 3), err=round(1 - df.correct.mean(), 3))
    for _, r in g.iterrows():
        row[f'b_{r.predictor}'] = round(r.logodds_per_sd, 3)
        row[f'p_{r.predictor}'] = float(f'{r.p_lrt:.1g}')
    row.update({k: round(v, 3) for k, v in common.tau_fits(df, fine, Hf).items()})
    return row


if __name__ == '__main__':
    ses, _ = readout.load_data()
    gains = {'nmda': memory.reference_gain(ratemodel.params('nmda'), ses),
             'stp': readout.evaluate(ratemodel.params('stp'), ses, n_trials=1056, seed=0, sets=('all',))['sets']['all']['gain']}
    print('gains', gains, flush=True)
    names = ['nmda_tau80', 'nmda_tau150', 'nmda_tau300', 'stp_lapse', 'stp_encode_fail', 'stp_wrong_item', 'stp_random_loss']
    worlds = Parallel(n_jobs=4)(delayed(world)(n, gains) for n in names)
    fine, Hf = common.sweep_curves()
    rows = Parallel(n_jobs=4)(delayed(analyse)(n, df, h, fine, Hf) for n, df, h in worlds)
    for n, df, h in worlds:
        df.to_csv(f'review/out/synth_trials_{n}.csv.gz', index=False)
    res = pd.DataFrame(rows)
    real = analyse('DATA', data, np.nan, fine, Hf)
    res = pd.concat([pd.DataFrame([real]), res])
    res.to_csv('review/out/synth_behaviour.csv', index=False)
    pd.set_option('display.width', 250)
    print(res.to_string(index=False))
