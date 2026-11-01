# Known issues

Sources of the original code: `jmourabarbosa/gamma_bursts-` (branches `paper-model`, `paper-model-mongillo`,
`claude/gallant-volta-bkjojq`, `claude/intermittent-rate-coding-paper-ced984`, `session-archive`) and
`jmourabarbosa/panichello` (`analysis-replication`). The refactored code reproduces the original intermediate results
bit for bit (section D). Where the paper disagreed with the code, the corrections were proposed as suggestions in the
Google Doc; those items are no longer listed here.

## A. Inconsistencies kept in the code (so that the published numbers are unchanged)

A1. **Three gains for the same read-out.** The all-units gain of the fitted model is calibrated three times with
   different simulations: 0.0512 (Fig. 2, 2,112 trials), 0.0487 (tau sweep behind Fig. 3c, d, 1,056 trials, seed 0)
   and 0.0510 (Supplementary Fig. 1, 1,056 trials, seed 11, other grid; the Fig. S1 notebook's calibration and
   `memory.reference_gain`). With the single gain 0.0512, Supplementary Fig. 1c would read "model 2.21" instead of
   2.20 and Fig. 3c, d would change slightly.
A2. **Off threshold.** The recordings use z < 0.8416 (P > 0.20 exactly); the rate model and the spiking network use
   z < 0.84 (`recordings.OFF_Z`, `onoff.OFF_Z`).
A3. **State-length histograms.** The data's table puts states of exactly 600 ms in the 580-600 ms bin, the model in
   the >= 600 ms bin (enters the fit loss).
A4. **Read-out noise shares** `RHO` (0.09 top, 0.11 random, 0.35 all units) are typed in, not computed; they are the
   data's pre-cue z autocorrelation at 200 ms (the Fig. S1 notebook prints the recomputed values: 0.089, 0.107, 0.346).

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

## D. Checks of the refactored code against the original outputs

| step | compared with | result |
|---|---|---|
| read-out table, 25 sessions x 61 decoded populations | `paper_model/data/readout_per_session_data_lam10000.csv` | identical (mu_on to 4e-15) |
| state-length histograms | `readout_histograms_data_lam10000.csv` | identical |
| per-trial statistics, 15,468 correct and error trials | `behaviour_handoff/data/per_trial_correct_and_error.csv` | identical |
| pre-cue z autocorrelation at 200 ms | `RHO` of the read-out | 0.346, 0.089, 0.107 (all, top 20, random 20) |
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


Figures: the Google Doc's Figs. 1, 3 and 4 were replaced by the code's; its Figs. 2, S1, S2 and S3 are pixel-identical
to the code's.
