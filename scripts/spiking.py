"""Simulations of the spiking network (Fig. 4), stored so that the read-out can be redone in the notebook;
about 2 h on 4 cores.

results/spiking/sessions/session_<s>.npz   12 sessions of 480 trials (60 per cue; 1.5 s delay with dips from cue
                                           onset): 10 ms spike counts of 330 recorded cells, cue, memory held
results/spiking/sessions/long_session.npz  480 long trials: 1 s of delay with a constant common input, then 1 s
                                           with dips (Fig. 4a)
results/spiking/example_raster.npz         all E spikes and the common input of the example trial of Fig. 4a

usage: python scripts/spiking.py
"""
import os

import numpy as np

from nmdaint import paths, spiking as SP

N_SESSIONS = 12
LONG = dict(seed=50, quiet=1000.0, dips=1000.0, example_trial=27)   # the example: cue 180 deg, memory held, On
                                                                   # through the constant second, 2-5 Off states


def save(name, make):
    f = paths.results('spiking', 'sessions', name + '.npz')
    if not os.path.exists(f):
        np.savez_compressed(f, **make())
    return dict(np.load(f))


def main():
    net = SP.Network(SP.params())
    for s in range(N_SESSIONS):
        save(f'session_{s}', lambda: SP.session(net, 100 + s, s))
    lnet = SP.Network(SP.params(t_delay=LONG['quiet'] + LONG['dips']))
    d = save('long_session', lambda: SP.session(lnet, LONG['seed'], LONG['seed'], dip_free=LONG['quiet']))
    i = LONG['example_trial']                      # re-simulate its batch with the raster (same seeds, same trials)
    b0 = i // SP.BATCH * SP.BATCH
    sim = SP.simulate(lnet, 45.0 * d['cue'][b0:b0 + SP.BATCH], seed=LONG['seed'] * 1000 + i // SP.BATCH, rec_idx=d['rec'],
                      raster=True, dip_free=LONG['quiet'])
    assert np.array_equal(sim['counts'], d['counts'][b0:b0 + SP.BATCH])
    t, cell = np.nonzero(sim['raster'][i - b0])
    np.savez_compressed(paths.results('spiking', 'example_raster.npz'), trial=i, spike_t=sim['t1'][t], spike_cell=cell,
                        t1=sim['t1'], m_eff=sim['m_eff'][i - b0])


if __name__ == '__main__':
    main()
