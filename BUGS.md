# Known issues

Sources of the original code: `jmourabarbosa/gamma_bursts-` (branches `paper-model`, `paper-model-mongillo`,
`claude/gallant-volta-bkjojq`, `claude/intermittent-rate-coding-paper-ced984`, `session-archive`) and
`jmourabarbosa/panichello` (`analysis-replication`).

## A. Inconsistencies kept in the code (so that the published numbers are unchanged)

A1. **Off threshold.** The recordings use z < 0.8416 (P > 0.20 exactly); the rate model and the spiking network use
   z < 0.84 (`recordings.OFF_Z`, `onoff.OFF_Z`).
A2. **State-length histograms.** The data's table puts states of exactly 600 ms in the 580-600 ms bin, the model in
   the >= 600 ms bin (enters the fit loss).

## B. Statements of the paper that this repository cannot reproduce

B1. **The pre-cue z SD of 1.06 per decoded population** (Methods, read-out) is not stored by `scripts/recordings.py`
   (only the autocorrelation is). Recomputed in the audit below: 1.060-1.064 for the seven populations, so the
   statement is correct.
B2. **"With depression, On states were too short (112 vs 172 ms in the data)"** (Methods, STP model): 172 ms is the
   data's mean On length; 112 ms comes from the STP model with depression, which is not in this repository.

B3. **Mixed-model weights of single monkeys** (Results: total Off time, H beta 0.26, A 0.13, J 0.03; Methods, last
   sentence of the behavioural analysis). The code fitting single monkeys was removed.

## C. Checks of the refactored code against the original outputs

| step | compared with | result |
|---|---|---|
| read-out table, 25 sessions x 61 decoded populations | `paper_model/data/readout_per_session_data_lam10000.csv` | identical (mu_on to 4e-15) |
| state-length histograms | `readout_histograms_data_lam10000.csv` | identical |
| rate model, NMDA and STP | `paper_model/model.py`, `model_ours_stp.py` | bit-identical traces |
| single-dip test, both models | `single_dip_readout_trials.npz` | identical |
| Procrustes and PCA plane, recordings | `procrustes_paper_handoff/`, `paper/scripts/data/pca/` | identical (PCA plane 4e-8: float64 instead of float32) |
| spiking network, 12 sessions | `spiking/results/pca_onoff/counts_s*.npz`, `sessions_dips.json` | identical spikes and statistics |
| On/Off tuning, Procrustes and PCA plane, network | `tuning_on_off.npz`, `figures/spiking/*.csv` | identical |
| synthetic validation, 24 sessions | `readout_suppl_handoff/data/synthetic_sessions.csv` | identical |

Analyses that differ from the original on purpose: the share rho of the read-out noise is measured for each decoded
population (the original used the 20-unit values for the 5- and 50-unit populations), the mixed model of errors
(Fig. 3b) has other predictors, the tau sweep uses the gain of the
other figures (0.0512 instead of its own calibration, 0.0487; tau_N 125 instead of 123 ms) and its bootstrap resamples
sessions within each monkey (10,000 resamples), and Fig. 2 plots the
statistics against decoding strength.

## D. Open problems from the audit of 16 November, with the fix tested for each

Not changed in the code or the paper. "Tested" means tried in scratch code on the stored results.

D1. **The tau sweep extends into a range where the model cannot load the memory.** With tau_N of 1,000 ms and above
   the 50 ms cue no longer charges the gating: the memory is held in 97% (1,000 ms) and 63% (1,500 ms) of trials
   whatever the time Off. The few bootstrap resamples at about 1,000 ms sit in that range. *Tested:* fitting only
   tau_N <= 750 ms gives 125 ms, 112-148, with 0.03% of resamples above 250 ms (1.4% now). Works.
D2. **Gain calibration biased low** (`readout.gain_curve`: running maximum over 10 gains, mixture fitted to 40,000
   random points). The model's decoding strength is 2.09-2.14 instead of 2.18. *Tested:* mixture on all points,
   interpolated on a fine grid: gain 0.053 instead of 0.051, decoding strength 2.18; Off states 3.10 -> 3.05,
   fraction Off 0.385 -> 0.376, fit loss 9.9 -> 10.7. Works; shifts every model figure slightly.
D3. **Fit not identified** (`fit.py`; from an independent review, not rerun here). The published w_s (NMDA) and
   dip shape and sigma (STP) lie on their search bounds, and the loss of one parameter set differs between
   simulation seeds (published NMDA 9.38 and 9.30; a refit 8.97 and 10.98) by more than the difference between
   candidate fits or between the NMDA and STP models. Fix not tested (about 2 h per fit): average the loss over
   seeds, widen the bounds.
