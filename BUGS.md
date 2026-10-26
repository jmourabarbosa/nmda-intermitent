# Problems found while refactoring, and differences from the code behind the submitted figures

Sources: `jmourabarbosa/gamma_bursts-` (branches `paper-model`, `paper-model-mongillo`, `claude/gallant-volta-bkjojq`,
`claude/intermittent-rate-coding-paper-ced984`, `session-archive`) and `jmourabarbosa/panichello`
(`analysis-replication`). The refactored code reproduces the original intermediate results bit for bit unless
stated below (checks in the final section).

## A. Results that the original code does not reproduce

1. **Figure 1 (example trial and "High, 3.0/trial").** The committed `paper_model/make_figures.py` (commit 92a4f33,
   the commit of the paper's PNG) selects simulated trial 56 as the example; the paper shows trial 226 of the same
   simulation (identical traces and states). Its panel b gives 3.08 On states per trial at high gain ("3.1"), the
   paper says 3.0, and the high-gain bars differ slightly. The PNG was apparently made with an uncommitted variant.
   The new code takes the example trial explicitly (226, `scripts/ratemodel.py`, `EXAMPLE`); the numbers come from
   the code (3.1).
2. **The model fits.** At the published parameters, the archived best losses (`session-archive`,
   `archive/fits/fits_one/all_run0`: 8.99 for the NMDA model; `fits_fac/run0`: 9.23 for the STP model) are not
   what the current fit code computes (9.43 and 9.05). The published parameters therefore come from an earlier
   version of the loss or of the data table, and a refit with the current code will not return them.
   Also: the NMDA seed used (run0, loss 8.99) was not the best of the archived runs (run1, 8.78).
3. **Figure 3b caption.** The p values in the caption for Off length (7×10⁻⁵) and number of Off states (0.89) are
   those of the GLMM with a time-in-session covariate (`time_covariate_check.csv`, `time_terms = t`); the plotted
   effects and stars come from the model without it, which gives p = 1.1×10⁻⁴ and 0.78 (total Off and On time agree).
4. **Figure 4b.** Its histograms came from a separate re-simulation of sessions 0-3
   (`paper/scripts/spiking_topbottom.py`), whose decoders drew different label shuffles from the main simulation of
   the same trials. The new code uses sessions 0-3 of the main simulation: 2.99 Off and 3.44 On states per trial
   (paper 2.98 and 3.42; same labels at one decimal), bars differing by a few states.

## B. Inconsistencies in the methods (kept, so that the published numbers are unchanged; your decision)

5. **Three gains for the same read-out.** The all-units gain of the fitted model was calibrated three times with
   different simulations: 0.0512 (Fig. 2, 2,112 trials), 0.0487 (tau sweep behind Fig. 3c, d, 1,056 trials, seed 0)
   and 0.0510 (Supplementary Fig. 1, 1,056 trials, seed 11, other grid). The new code uses 0.0512 for Supplementary
   Fig. 1 (printed "0.051" either way) and keeps 0.0487 for the tau sweep (`memory.reference_gain`); using 0.0512
   there would change Fig. 3c, d slightly.
6. **Off threshold.** The recordings use z < 0.8416 (P > 0.20 exactly); the rate model and the spiking network use
   z < 0.84 (`recordings.OFF_Z`, `onoff.OFF_Z`).
7. **The spiking network is not decoded exactly like the recordings**, although the Methods say so:
   (a) the decoder is fitted on raw spike counts (the data: rates z-scored across trials), with the quadratic
   approximation of the logistic loss (close at lambda = 10,000); (b) its z is taken against the shuffles pooled over
   all trials of a cue pair (the data: each trial against its own shuffles); (c) its On threshold pools the cluster
   masses of all null traces (the data: each trial's shuffles, the maximum mass per shuffle).
8. **State-length histograms.** The data's table puts states of exactly 600 ms in the 580-600 ms bin, the model in
   the >= 600 ms bin (enters the fit loss).
9. **Fit conditions.** The STP model was fitted with 1 s before the cue and a penalty on spontaneous activity; the
   NMDA model with 300 ms and no such penalty. Both are evaluated with 300 ms.
10. **Read-out noise shares** `RHO` (0.09 top, 0.11 random, 0.35 all units) are typed in, not computed; they are the
    data's pre-cue z autocorrelation at 200 ms (the Fig. S1 notebook prints the recomputed values).
11. **Pre-cue window.** Recordings: window centres -400 to -10 ms; network: -400 to -50 ms (the model's cue drives
    spikes within milliseconds; documented by the spiking agent).
12. **Supplementary Fig. 1g** had the model gains ("0.029, 0.051, 0.088") typed into the figure script; they are now
    read from the results.

## C. Other notes

13. `paper/main.tex` (paper branch) is older than the Google Doc (different title, captions); the figures were
    compared with the images in the Google Doc.
14. Removed as not used by the paper: unit sets other than all/top/random and the k values they need (still drawn,
    since they fix the random sequence), the confidently-incorrect fraction, pre-cue false-On rates, CCG analysis,
    the network's response probe, overshoot and no-feed-forward model variants, the STP model with depression.
