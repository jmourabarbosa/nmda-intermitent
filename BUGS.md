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
other figures (0.0512 instead of its own calibration, 0.0487; tau_N 125 instead of 123 ms) and its bootstrap resamples
sessions within each monkey (10,000 resamples), and Fig. 2 plots the
statistics against decoding strength.

## D. Audit (16 November): open problems, with the fix tested for each

Nothing below was changed in the code or the paper. "Tested" means tried in scratch code on the stored results.

### Results that could change

D1. **The tau_N fit pools monkeys that differ** (`behaviour.fit_tau`, Fig. 3c, d). One tau_N and one baseline accuracy
   p0 are fitted to all trials, but baseline accuracy differs (A 0.92, H 0.84, J 0.99) and so does the decline with
   time Off. Fitted separately: H 79 ms (65-91), A 165 ms (138-959), J 185 ms (174-945); no monkey's interval
   contains the pooled 125 ms. *Tested:* one tau_N with a p0 per monkey gives 158 ms (95% interval 143-174, 0.2% of
   resamples without decline). Works and is stable, but moves the headline value from 125 to 158 ms.
D2. **The tau sweep extends into a range where the model cannot load the memory.** With tau_N of 1,000 ms and above
   the 50 ms cue no longer charges the gating: the memory is held in 97% (1,000 ms) and 63% (1,500 ms) of trials
   whatever the time Off. The bootstrap resamples at about 1,000 ms sit in that range. *Tested:* fitting only
   tau_N <= 750 ms gives 125 ms, 112-148, with 0.03% of resamples above 250 ms (1.4% before). Works.
D3. **Errors go with time decoded as the opposite location, not with uninformative time** (Fig. 3b). Off (z < 0.84)
   includes time points where the decoder favours the opposite location. Error trials have more of those (15.1% vs
   10.6% of the delay with z < -0.84 for >= 50 ms) and hardly more time near zero (25.2% vs 24.5% with |z| < 0.84).
   *Tested:* the mixed model with the two parts as separate predictors: opposite-location time beta 0.21 (0.13-0.30,
   p = 2e-4), near-zero time beta 0.04 (-0.04-0.12, p = 0.36); "ends Off" and the previous error are unchanged. The
   effect of "total Off time" on errors is therefore carried by decoding of the wrong location. This is compatible
   with a memory that was lost and restarted in the other pool, but it is not evidence that silence itself precedes
   errors.
D4. **On threshold of the recordings** (`recordings.null_threshold`). Each label shuffle is z-scored against the
   other shuffles and the true labels, so strong decoding lowers the threshold. The docstring says the original
   authors' code does the same. *Tested* on two sessions, z-scoring against shuffles only: median threshold 14-17 ->
   20-24, On states per trial 8-13% fewer, fraction On 2-13% lower. Works, but changes every On statistic of the
   data and needs the 4 h pipeline.
D5. **The model sits below the data in both fractions**, also for the fitted population: fraction Off 0.39 vs 0.45,
   fraction On 0.18 vs 0.21; for the most selective neurons 2.5 vs 3.2-3.5 Off states per trial; for random neurons
   0.5-1.4 vs 1.1-1.7 On states per trial. The Results quote only the four best statistics ("matches the data
   well", "follows the data across the whole range"). *Tested:* defining the model's On threshold as in the data
   (largest cluster per noise trace) changes On states per trial from 1.92 to 1.97 only. Does not fix it.
D6. **Gain calibration biased low** (`readout.gain_curve`: running maximum over 10 gains, mixture fitted to 40,000
   random points). The model's decoding strength is 2.09-2.14 instead of 2.18. *Tested:* mixture on all points,
   interpolated on a fine grid: gain 0.053 instead of 0.051, decoding strength 2.18; Off states 3.10 -> 3.05,
   fraction Off 0.385 -> 0.376, fit loss 9.9 -> 10.7. Works; shifts every model figure slightly.
D7. **Fit not identified** (`fit.py`; from the independent review, not rerun here). The published w_s (NMDA) and
   dip shape and sigma (STP) lie on their search bounds, and the loss of one parameter set differs between
   simulation seeds (published NMDA 9.38 and 9.30; a refit 8.97 and 10.98) by more than the difference between
   candidate fits or between the NMDA and STP models. Fix not tested (about 2 h per fit): average the loss over
   seeds, widen the bounds.

### Smaller points

D8. **Fig. 4c tuning** (`spiking_readout.tuning`): the shuffle correction is also subtracted at the preferred cue of
   truly tuned cells, which flattens the peak (independent review, synthetic test: 1 : 0.71 becomes 0.56 : 0.52).
   The On-tuned versus Off-flat contrast is not explained by it. Fix (preferred cue from the other half of the
   trials) not tested on the network.
D9. **Fig. 3b** draws Wald intervals next to likelihood-ratio stars; for the previous error they disagree (Wald
   p = 0.003, likelihood ratio 0.019). The per-monkey models are singular (several random-slope SDs at zero with
   7-10 sessions); their fixed effects and tests are finite.
D10. **Off thresholds and histogram bins** (A1, A2). *Tested:* 0.8416 in the model changes the fraction Off from
   0.3851 to 0.3855; the data's bin convention moves 9 states and the loss from 9.875 to 9.888. Both harmless.
D11. **`population.off_synchrony`**: the "another trial" control leaves about one trial per cue paired with itself
   (estimated effect 0.002 on 0.52). **`synthetic.session`** seeds two generators with the same number. **Fig. 4e**
   treats 10 samples of one network as independent sessions. **Pre-cue windows** centred at -40 to -10 ms reach up
   to 40 ms past cue onset and enter rho. None retested.
D12. **Wording.** Abstract: Off states "more frequent when the decoder used smaller ... subsets" holds for random
   neurons only (all neurons are Off 45% of the delay, the 5-50 most selective 33-36%). Introduction: "the same
   decoding" as Panichello et al.; the decoder differs (Methods). Methods, synthetic validation: Off states "within
   0.5 states per trial" is 0.52 at the strongest tuning.

