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
other figures (0.0512 instead of its own calibration, 0.0487; tau_N 125 instead of 123 ms), stops at 750 ms (the
model does not load the memory with slower traces) and its bootstrap resamples sessions within each monkey
(10,000 resamples), and Fig. 2 plots the
statistics against decoding strength.

## D. Stated in the Methods instead of fixed

The gain calibration leaves the model's decoding strength 1-5% below the
data's (`readout.gain_curve`: running maximum over 10 gains), and the fit loss varies between simulation seeds
(NMDA 9.9 +- 0.3, STP 9.7 +- 0.3 on all neurons; 73 +- 2 and 79 +- 2 summed over the populations; 10 seeds), with
some fitted parameters on their search bounds.
