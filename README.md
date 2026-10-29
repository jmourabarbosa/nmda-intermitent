# Intermittent rate coding without short-term plasticity can support working memory

Code for the paper: the analysis of the recordings of Panichello et al. (2024), the NMDA rate model (and its
short-term-plasticity variant), and the spiking ring-attractor network.

## Layout

| folder | content |
|---|---|
| `nmdaint/` | **the methods** (no plotting). Read these to check the methodology. |
| `figure_notebooks/` | **one notebook per figure: the analysis behind it**, step by step, with the parameters at the top. Change one and rerun to see what happens. |
| `model_notebooks/` | `rate_model.ipynb`, `spiking_network.ipynb`: example simulations to play with the two models (under a minute each) |
| `scripts/` | the few computations too heavy for a notebook; they write `results/` |
| `results/` | outputs of `scripts/` (decoders of the recordings, the tau sweep, spiking-network simulations) |
| `plots/` | drawing only: `draw(...)` of each figure takes the notebook's results (`style.py`: shared style) |

### Methods (`nmdaint/`)

| module | what it does |
|---|---|
| `onoff.py` | On/Off states: labelling of a decoding z score, cluster-mass threshold from a null, state statistics, decoding strength (mu_on). Shared by data, rate model and spiking network. |
| `decoder.py` | decoders of the cued vs the opposite location: L2 logistic regression with approximate leave-one-out (recordings), its quadratic version with exact leave-one-out (spiking network) |
| `recordings.py` | the recordings: rates, decoded populations (all, top k, random k units), decoder z and states, error trials |
| `behaviour.py` | Off states and errors: logistic mixed model (Laplace), NMDA time constant fitted to performance |
| `ratemodel.py` | NMDA rate model and STP variant |
| `readout.py` | reading the rate model out like the data (z = c dr + eta), fit loss |
| `fit.py` | CMA-ES fit of the rate model |
| `memory.py` | memory against time Off: single-dip test, tau_N sweep |
| `spiking.py` | LIF ring-attractor network |
| `spiking_readout.py` | reading the network out like the data; On/Off tuning of held-out cells |
| `population.py` | split-half PCA plane and Procrustes shape similarity of On, Off and pre-cue activity (data and network) |
| `synthetic.py` | validation of the model read-out on synthetic Poisson neurons |

### Figures

| figure | notebook (`figure_notebooks/`) | runs live (time) | loads from `results/` |
|---|---|---|---|
| 1 | `fig1.ipynb` | rate model, read-out (40 s) | the data's On/Off statistics |
| 2 | `fig2.ipynb` | rate model, read-out of 7 populations (30 s); optional: data decoder on one session | the data's On/Off statistics |
| 3 | `fig3.ipynb` | single-dip test, mixed model, tau fit (2 min) | per-trial statistics; tau sweep (optional rerun) |
| 4 | `fig4.ipynb` | network read-out, tuning, PCA, Procrustes (4 min) | network spike counts; the data's Procrustes and PCA plane |
| S1 | `figS1.ipynb` | read-out, calibration, synthetic validation (3 min) | the data's pre-cue z autocorrelation and one session's delay z |
| S2, S3 | `figS2.ipynb`, `figS3.ipynb` | control without dips, STP model (1-3 min) | the data's On/Off statistics |

## Running

Data: the 25 sessions (`yymmdd.mat`) of Panichello et al. (2024), Dryad doi:10.5061/dryad.kkwh70sct, mirrored at
github.com/jmourabarbosa/panichello. By default they are read from `../panichello`; set `NMDAINT_DATA` otherwise.

```
pip install -e .                    # Python 3.11-3.13; versions in requirements.txt
jupyter notebook figure_notebooks/  # the figures: about 12 min for all seven

# only to recompute results/ (not needed for the notebooks):
python scripts/recordings.py        # decoders of the recordings, ~1.5-4 h on 4 cores (needs the data)
python scripts/tau_sweep.py         # ~20 min, after recordings
python scripts/spiking.py           # network simulations, ~2 h
python scripts/fit.py               # refits of both models, ~2 h per run (the figures use the published fits)
```

The notebooks only need `results/`, which is committed. `BUGS.md` lists the differences from the code that produced
the submitted figures and the problems found while refactoring.
