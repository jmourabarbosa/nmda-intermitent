# Intermittent rate coding without short-term plasticity can support working memory

Code for the paper: the analysis of the recordings of Panichello et al. (2024), the NMDA rate model (and its
short-term-plasticity variant), and the spiking ring-attractor network.

## Layout

| folder | content |
|---|---|
| `nmdaint/` | **the methods** (no plotting). Read these to check the methodology. |
| `scripts/` | pipelines that run the methods and write `results/` |
| `results/` | precomputed results, small text and `.npz` files, read by the figures |
| `plots/` | plotting code, one module per figure (`style.py`: shared style) |
| `figure_notebooks/` | one notebook per figure: loads `results/`, draws the figure (`figures/`), prints the numbers quoted in the text |
| `model_notebooks/` | `rate_model.ipynb`, `spiking_network.ipynb`: example simulations to play with the two models (under a minute each) |

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

| figure | notebook (`figure_notebooks/`) | results from |
|---|---|---|
| 1 | `fig1.ipynb` | `scripts/ratemodel.py` |
| 2 | `fig2.ipynb` | `scripts/recordings.py`, `scripts/ratemodel.py` |
| 3 | `fig3.ipynb` | `scripts/ratemodel.py`, `scripts/behaviour.py` |
| 4 | `fig4.ipynb` | `scripts/spiking.py`, `scripts/recordings.py` |
| S1 | `figS1.ipynb` | `scripts/ratemodel.py`, `scripts/recordings.py`, `scripts/synthetic.py` |
| S2, S3 | `figS2.ipynb`, `figS3.ipynb` | `scripts/recordings.py`, `scripts/ratemodel.py` |

## Running

Data: the 25 sessions (`yymmdd.mat`) of Panichello et al. (2024), Dryad doi:10.5061/dryad.kkwh70sct, mirrored at
github.com/jmourabarbosa/panichello. By default they are read from `../panichello`; set `NMDAINT_DATA` otherwise.

```
pip install -e .                    # Python 3.13; versions in requirements.txt
python scripts/recordings.py        # ~4 h on 4 cores (sessions cached in cache/)
python scripts/ratemodel.py         # ~1 h; --fit also refits both models (~2 h per run)
python scripts/behaviour.py         # minutes
python scripts/spiking.py           # ~2 h
python scripts/synthetic.py         # ~15 min
jupyter nbconvert --to notebook --execute --inplace figure_notebooks/fig*.ipynb
```

The figures only need `results/`, which is committed. `BUGS.md` lists the differences from the code that produced
the submitted figures and the problems found while refactoring.
