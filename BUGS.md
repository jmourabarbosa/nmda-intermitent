# Problems found while refactoring, and differences from the code behind the submitted figures

Sources: `jmourabarbosa/gamma_bursts-` (branches `paper-model`, `paper-model-mongillo`, `claude/gallant-volta-bkjojq`,
`claude/intermittent-rate-coding-paper-ced984`, `session-archive`) and `jmourabarbosa/panichello`
(`analysis-replication`). The refactored code reproduces the original intermediate results bit for bit unless
stated below (checks in the final section).

## A. Results that the original code does not reproduce

A1. **Figure 1 (example trial and "High, 3.0/trial").** The committed `paper_model/make_figures.py` (commit 92a4f33,
   the commit of the paper's PNG) selects simulated trial 56 as the example; the paper shows trial 226 of the same
   simulation (identical traces and states). Its panel b gives 3.08 On states per trial at high gain ("3.1"), the
   paper says 3.0, and the high-gain bars differ slightly. The PNG was apparently made with an uncommitted variant.
   The new code takes the example trial explicitly (226, `EXAMPLE_TRIAL` in the Fig. 1 notebook). Decision: keep the
   code's numbers (3.1); the paper's Fig. 1 image is to be replaced by `figures/fig1.png` (the text quotes no count).
A2. **The model fits.** At the published parameters, the archived best losses (`session-archive`,
   `archive/fits/fits_one/all_run0`: 8.99 for the NMDA model; `fits_fac/run0`: 9.23 for the STP model) are not
   what the current fit code computes (9.43 and 9.05). The published parameters therefore come from an earlier
   version of the loss or of the data table, and a refit with the current code will not return them.
   Also: the NMDA seed used (run0, loss 8.99) was not the best of the archived runs (run1, 8.78).
   Refits with the current code (`fit.fit`, 1,000 evaluations, seeds 0-2; best parameters not committed):
   - NMDA: losses 9.25, 9.48 and 9.10 (seeds 0-2), against 9.43 for the published parameters, so the published fit is
     close to but not at the optimum of the current loss. The best refit (seed 2) has tau_N 150 ms and w_s 1.30 (both
     at their bounds, as published), dips 9.1/s, 106 ms long, depth 0.81 (published: 10.2/s, 90 ms, 0.60).
   - STP: seed 0 reaches 8.96 (w_s 0.69, dips 4.7/s, 260 ms, depth 0.32; published, from the archived seed 0: 0.69,
     5.5/s, 212 ms, 0.39). Seeds 1 and 2 (stopped by a container restart after ~550 evaluations) had already found
     exactly the archived best parameters of their seeds, so the STP fit code reproduces those runs, but not seed 0.
   Decision: keep the published parameters (`ratemodel.FIT`), which all figures use; the Methods now call them
   "Fitted parameters" instead of "Best fit", since they are not the optimum of the loss the Methods describe.
A3. **Behavioural statistics (Fig. 3b and Results).** The paper's GLMM numbers, its Methods formula and its Fig. 3b
   PNG came from an earlier variant of the model, not the one in the committed Fig. 3b script. The committed model is
   kept (`behaviour.glmm_single`) and the paper was changed to it: Methods formula; total Off time β = 0.31 (95% CI
   0.20–0.42), p = 1×10⁻⁵ (was 0.32, 0.20–0.43); per monkey β = 0.44 (H, p = 1×10⁻⁴), 0.55 (J, p = 0.005), 0.11 (A, p
   = 0.11) (was 0.48, 0.52, 0.10 with p = 0.13); caption, Off length p = 1×10⁻⁴ (was 7×10⁻⁵), number of Off states p =
   0.78 (was 0.89). The Fig. 3 notebook prints all of them.
A4. **Figure 4b.** Its histograms came from a separate re-simulation of sessions 0-3
   (`paper/scripts/spiking_topbottom.py`), whose decoders drew different label shuffles from the main simulation of
   the same trials. The new code uses sessions 0-3 of the main simulation: 2.99 Off and 3.44 On states per trial
   (paper 2.98 and 3.42; same labels at one decimal), bars differing by a few states. Decision: keep the code; the
   paper's Fig. 4 image was replaced by `figures/fig4.png`.
A5. **Methods, "loss 9.7 for the NMDA model and 10.4 for the STP model, on 2,112 trials".** Not reproduced. The
   all-units loss at the published parameters, in the evaluation behind Figs. 2 and S3 (2,112 trials, seed 1), is
   9.87 (NMDA) and 9.83 (STP); with seeds 2 and 3, 9.97 and 9.61 (NMDA), 9.53 and 9.04 (STP). The evaluation noise
   (about ±0.4) is larger than the difference, and the STP model is not worse on this decoder. Summed over the seven
   populations (Supplementary Fig. 3h), NMDA 74 and STP 79 (seed 1). Changed in the paper to these values (9.9 and
   9.8 on the all-units decoder, 74 and 79 summed).

## B. Inconsistencies in the methods (code kept, so that the published numbers are unchanged; the paper describes them)

B1. **Three gains for the same read-out.** The all-units gain of the fitted model was calibrated three times with
   different simulations: 0.0512 (Fig. 2, 2,112 trials), 0.0487 (tau sweep behind Fig. 3c, d, 1,056 trials, seed 0)
   and 0.0510 (Supplementary Fig. 1, 1,056 trials, seed 11, other grid). The new code keeps all three so that the figures
   are unchanged (the Fig. S1 notebook's calibration; `memory.reference_gain`). With the single gain 0.0512,
   Supplementary Fig. 1c would read "model 2.21" instead of 2.20 and Fig. 3c, d would change slightly. Not in the
   paper (an implementation detail).
B2. **Off threshold.** The recordings use z < 0.8416 (P > 0.20 exactly); the rate model and the spiking network use
   z < 0.84 (`recordings.OFF_Z`, `onoff.OFF_Z`). The paper gives 0.84, which is both at two decimals.
B3. **The spiking network is not decoded exactly like the recordings**, although the Methods say so:
   (a) the decoder is fitted on raw spike counts (the data: rates z-scored across trials), with the quadratic
   approximation of the logistic loss (close at lambda = 10,000); (b) its z is taken against the shuffles pooled over
   all trials of a cue pair (the data: each trial against its own shuffles); (c) its On threshold pools the cluster
   masses of all null traces (the data: each trial's shuffles, the maximum mass per shuffle). The rate model's On
   threshold, too, pools the masses of all clusters of noise-only traces. The paper now describes all three
   differences and the three nulls (Methods, On/Off labelling and Spiking network read-out); the Results and the
   Fig. 4 caption no longer say "the decoder, null and labelling of the recordings".
B4. **State-length histograms.** The data's table puts states of exactly 600 ms in the 580-600 ms bin, the model in
   the >= 600 ms bin (enters the fit loss).
B5. **Fit conditions.** The STP model was fitted with 1 s before the cue and a penalty on spontaneous activity; the
   NMDA model with 300 ms and no such penalty. Both are evaluated with 300 ms. Now in the paper's STP Methods.
B6. **Read-out noise shares** `RHO` (0.09 top, 0.11 random, 0.35 all units) are typed in, not computed; they are the
    data's pre-cue z autocorrelation at 200 ms (the Fig. S1 notebook prints the recomputed values).
B7. **Pre-cue window.** Recordings: window centres -400 to -10 ms; network: -400 to -50 ms (the model's cue drives
    spikes within milliseconds; documented by the spiking agent). The paper said "to cue onset" for the recordings;
    corrected to -10 ms.
B8. **Supplementary Fig. 1g** had the model gains ("0.029, 0.051, 0.088") typed into the figure script; they are now
    read from the results.

## C. Other notes

C1. `paper/main.tex` (paper branch) is older than the Google Doc (different title, captions); the figures were
    compared with the images in the Google Doc.
C2. Removed as not used by the paper: unit sets other than all/top/random and the k values they need (still drawn,
    since they fix the random sequence), the confidently-incorrect fraction, pre-cue false-On rates, CCG analysis,
    the network's response probe, overshoot and no-feed-forward model variants, the STP model with depression.

## D. Caveats found by exploring the models (the notebooks' "Play" sections)

Not errors in the code: results that qualify a statement of the current paper.

D1. **Supplementary Fig. 2: noise alone gives most of the all-units Off states.** With no dips and the gain
    recalibrated, the model has 2.37 Off states per trial for the all-units decoder (fitted model 3.10, data 3.25),
    and 3.95–4.77 for the random populations (data 4.27–4.80). Only the top-unit decoders separate the two (0.08–0.27
    vs 3.15–3.50). The caption reports the top and random populations but not all units, the decoder the model is
    fitted to and whose statistics the Results quote. The case for dips rests on the top-unit decoders and the fit
    loss (313 vs 74), which is worth saying. In the Fig. S2 notebook's sweep (500 trials per depth), all-units Off
    states rise only from 2.3 without dips to 3.1 at the fitted depth. The paper's Fig. S2 caption now gives the
    all-units numbers (2.4 vs 3.3 in the data and 3.1 in the fitted model) and the random populations' (0.1–2.1 vs
    4.3–4.8, instead of "few").
D2. **Fig. 2: "quantitatively predicts the shift".** The model gets the direction and ordering, but for the top-unit
    decoders it gives too few Off states (2.49–2.68 vs 3.15–3.50 per trial), too little Off time (0.24–0.27 vs
    0.33–0.36 of the delay) and too long On states (median 150 vs 126–139 ms). The top-to-random shift in Off states
    per trial is overestimated (k = 5: 1.99 vs 1.29 in the data; k = 20: 1.67 vs 1.40; k = 50: 1.23 vs 1.12). Visible
    in Fig. 2. The paper now says the model predicts the direction of the shift and most of its size, with too few
    Off states for the most selective neurons (2.5–2.7 vs 3.2–3.5).

## E. Checks of the refactored code against the original outputs

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

Figures, against `paper/figures/` (the versions in the Google Doc): Figs. 2, S1, S2 and S3 pixel-identical; Fig. 3
identical to the original script run on the committed data (the paper's PNG, made with the earlier model variant
of A3, shows sub-pixel shifts of the session dots); Fig. 1 differs as in A1 (High-gain counts) and Fig. 4b as in A4.

## F. Audit of every statement of the paper against the code

Each number and method statement of the paper (Results, captions, Methods) was checked against the code, the
results and the notebooks' output. Besides A and B, the paper was corrected where it disagreed:

F1. **tau_N was fitted, not fixed.** The fit (`nmdaint/fit.py`, as the original `paper_model/fit.py`) has seven free
    parameters, tau_N within 50-150 ms among them, and the archived best fit has tau_N = 150 ms (upper bound) and
    w_s = 1.30 (lower bound). The paper said six parameters with tau_N fixed at 150 ms. Corrected: seven parameters,
    tau_N in bold in the equation and the fitted list, removed from the fixed list, and the bounds stated; the
    Results' "the 150 ms assumed in the fitted model" is now "of the fitted model".
F2. **Single-dip test (Fig. 3a).** The Methods said the trials were read out with the all-units gain and noise and that
    Off time was computed; the code (`memory.single_dip`) scores the memory on the model only, as the Fig. 3 caption
    says. Corrected.
F3. **"100,000 trials per model; memory held in 98.5% and 99.99%."** No code in the repository runs these; they came
    from an analysis no longer in the paper. Replaced by the values of the Figs. 2 and S3 evaluation (98.4% and 100%,
    2,112 trials each); the section is now called "Memory in the fitted models".
F4. **Spiking network trials.** Only trials with the memory held (83% of 480 per session) are analysed, as the data's
    correct trials; now in the Methods and the Fig. 4b caption.
F5. **Data decoder.** The Methods now say that it decodes each cue against the opposite one from firing rates in
    100 ms windows z-scored across trials.
F6. **Synthetic validation.** "It reproduced the number and fraction of Off states" is now "approximately reproduced
    (within 0.5 states per trial and 0.06 of the delay)": the read-out gives 0.03-0.06 more Off time than the decoder.
F7. **Fig. 1, low gain.** "Almost no On states" is now "far fewer On states" (0.58 vs 1.92 per trial).

Verified as written (examples): 25 sessions, 13,190 correct and 15,468 trials, 2,278 errors, 13,962 with an Off state,
329 units per session, 264 + 66 sampled cells; all rate-model and network parameters; Fig. 2 statistics and Wilcoxon
p < 10⁻⁶; single-dip 350/400 ms and STP to 1 s; tau_N 123 ms (101-190), p0 0.887 and 129/118/102 ms for G = 0, 1/4,
1/2; rho 0.35/0.09/0.11; decoding strengths 2.22/2.18/2.20; synthetic gains and correlations; Procrustes 10 and 25
sessions; dips recovering about 5 times per second; Fig. 1 40 ms bins and Fig. 4a 5 ms x 8-cell bins.

Not reproducible from this repository (kept in the paper):
- the pre-cue z SD of 1.06 per decoded population: from the original pipeline (`z_noise_summary.csv`; the refactored
  z is identical, but `scripts/recordings.py` stores only its autocorrelation);
- "with depression, On states too short (112 vs 172 ms)": 172 ms is the data's mean On length; 112 ms comes from the
  STP model with depression, removed from the code (C2).

