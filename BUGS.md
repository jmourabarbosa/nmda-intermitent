# Known issues

Sources of the original code: `jmourabarbosa/gamma_bursts-` (branches `paper-model`, `paper-model-mongillo`,
`claude/gallant-volta-bkjojq`, `claude/intermittent-rate-coding-paper-ced984`, `session-archive`) and
`jmourabarbosa/panichello` (`analysis-replication`). The refactored code reproduces the original intermediate results
bit for bit (section D). Where the paper disagreed with the code, the corrections were proposed as suggestions in the
Google Doc; those items are no longer listed here.

## A. Inconsistencies kept in the code (so that the published numbers are unchanged)

A1. **Two gains for the same read-out.** Every figure uses the all-units gain of the reference calibration
   (`readout.reference`, 2,112 trials, seed 1): 0.0512. The tau sweep behind Fig. 3c, d uses its own calibration
   (`memory.sweep_gain`, 1,056 trials, seed 0): 0.0487, as in the paper. With 0.0512 the fit gives tau_N 125 ms
   instead of 123 ms and the bootstrap interval 102-932 ms instead of 101-190 ms: its upper end is fragile, as 1.8%
   (with 0.0512, 3.3%) of the resamples find no decline of performance with time Off and go to the flat end of the
   tau grid.
A2. **Off threshold.** The recordings use z < 0.8416 (P > 0.20 exactly); the rate model and the spiking network use
   z < 0.84 (`recordings.OFF_Z`, `onoff.OFF_Z`).
A3. **State-length histograms.** The data's table puts states of exactly 600 ms in the 580-600 ms bin, the model in
   the >= 600 ms bin (enters the fit loss).

## B. Statements of the paper that this repository cannot reproduce

B1. **The pre-cue z SD of 1.06 per decoded population** (Methods, read-out) comes from the original pipeline
   (`z_noise_summary.csv`). The refactored z is identical, but `scripts/recordings.py` stores only its autocorrelation.
B2. **"With depression, On states were too short (112 vs 172 ms in the data)"** (Methods, STP model): 172 ms is the
   data's mean On length; 112 ms comes from the STP model with depression, which was removed from the code (C2).

## C. Other notes

C1. `paper/main.tex` (paper branch) is older than the Google Doc (different title, captions); the figures were
    compared with the images in the Google Doc.
C2. Removed as not used by the paper: unit sets other than all/top/random and the k values they need (still drawn,
    since they fix the random sequence), the confidently-incorrect fraction, pre-cue false-On rates, CCG analysis,
    the network's response probe, overshoot and no-feed-forward model variants, the STP model with depression.
C3. **tau_N is fixed at 150 ms** (`ratemodel.FIXED`; the fit has six free parameters, as the paper says). The published
    NMDA parameters come from the original fit, in which tau_N also varied (50-150 ms) and settled at 150 ms.

## D. Checks of the refactored code against the original outputs

| step | compared with | result |
|---|---|---|
| read-out table, 25 sessions x 61 decoded populations | `paper_model/data/readout_per_session_data_lam10000.csv` | identical (mu_on to 4e-15) |
| state-length histograms | `readout_histograms_data_lam10000.csv` | identical |
| per-trial statistics, 15,468 correct and error trials | `behaviour_handoff/data/per_trial_correct_and_error.csv` | identical |
| pre-cue z autocorrelation at 200 ms | `RHO` of the read-out | all, top 20: 0.346, 0.089, identical; random 20: 0.111 instead of 0.107 (all random draws averaged instead of the first); now also computed for top/random 5 and 50, which used the 20-unit values |
| GLMM fixed effects and session weights | `joint_glmm.csv`, `joint_glmm_session_weights.csv` | equal to 1e-6 (optimiser) |
| tau sweep, 17 tau_N x 30,000 trials | `figures/off_vs_memory/nmda/tau_sweep/*.npz` | identical |
| tau_N fit and bootstrap | `fig3_behaviour_tau.py` | 123 ms, 95% interval 101-190 ms, identical |
| rate model, NMDA and STP | `paper_model/model.py`, `model_ours_stp.py` | bit-identical traces |
| model read-out, all populations, 4 conditions | `readout.evaluate`, `control_no_dips.py`, `fig_unit_sets_nmda_stp.py` | identical states and statistics |
| single-dip test, both models | `single_dip_readout_trials.npz` | identical |
| Procrustes and PCA plane, recordings | `procrustes_paper_handoff/`, `paper/scripts/data/pca/` | identical (PCA plane 4e-8: float64 instead of float32) |
| spiking network, 12 sessions | `spiking/results/pca_onoff/counts_s*.npz`, `sessions_dips.json` | identical spikes and statistics |
| On/Off tuning, Procrustes and PCA plane, network | `tuning_on_off.npz`, `figures/spiking/*.csv` | identical |
| synthetic validation, 24 sessions | `readout_suppl_handoff/data/synthetic_sessions.csv` | identical |


Figures: all the Google Doc's figures are the code's.
