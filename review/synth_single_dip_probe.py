"""Fig. 3a: is the memory 'held' as scored by the paper the memory the network would report?

The paper scores the memory on a hidden variable at the end of the delay (NMDA gating difference > 0.05; for the
STP model the facilitation difference u_c - u_u > 0.05), not on activity. Here, after the single dip and the delay,
a response-period input (common input raised to M_RESP for 150 ms) is applied and the report is the pool with the
higher rate 50-150 ms into it ('probe'), against chance 0.5. Also reported: the cued pool active at the end of the
delay (activity-based memory) and the paper's score. A model whose hidden trace says 'held' but whose probe
reports at chance does not hold the memory in a usable form.
usage: python review/synth_single_dip_probe.py   (~10 min)
"""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from nmdaint import ratemodel

LENGTHS = np.arange(0, 1001, 100)
N, ONSET, M_RESP, T_RESP = 100, 500.0, 1.15, 150.0


def run(model, L, m_resp=M_RESP):
    p = ratemodel.params(model)
    p_ext = {**p, 't_delay': p['t_delay'] + T_RESP + 50}
    n = int(round((p_ext['t_pre'] + p_ext['cue_dur'] + p_ext['t_delay']) / p['dt']))
    m = ratemodel.single_dip(p_ext, N, ONSET, L)
    t = np.arange(n) * p['dt'] - p['t_pre']
    t_end = p['cue_dur'] + p['t_delay']                                  # end of the normal delay (from cue onset)
    m[:, (t >= t_end) & (t < t_end + T_RESP)] = m_resp
    sim = ratemodel.simulate(p_ext, N, seed=int(L) + (0 if model == 'nmda' else 7), m=m)
    ts = sim['t']
    end = (ts >= t_end - 100) & (ts < t_end)
    pr = (ts >= t_end + 50) & (ts < t_end + T_RESP)
    tr = ratemodel.memory_trace(sim)
    return dict(model=model, dip_ms=int(L), m_resp=m_resp,
                paper_score=float((tr[:, (ts >= t_end - 100) & (ts < t_end)].mean(1) > 0.05).mean()),
                cued_active_end=float((sim['rc'][:, end].mean(1) > 20).mean()),
                uncued_active_end=float((sim['ru'][:, end].mean(1) > 20).mean()),
                probe_correct=float((sim['rc'][:, pr].mean(1) > sim['ru'][:, pr].mean(1) + 1).mean()),
                probe_wrong=float((sim['ru'][:, pr].mean(1) > sim['rc'][:, pr].mean(1) + 1).mean()))


if __name__ == '__main__':
    jobs = [(m, L, r) for m in ('nmda', 'stp') for L in LENGTHS for r in (1.0, M_RESP, 1.3)]
    res = pd.DataFrame(Parallel(n_jobs=4)(delayed(run)(*j) for j in jobs))
    res.to_csv('review/out/synth_single_dip_probe.csv', index=False)
    pd.set_option('display.width', 200)
    for r, g in res.groupby('m_resp'):
        print(f'\n== response input m = {r}')
        print(g.drop(columns='m_resp').round(2).to_string(index=False))
