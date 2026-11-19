# Neural manifolds for brain-computer interface applications

Code, results and text of my MSc thesis in Biomedical Engineering at Roma Tre University (supervisor Prof. Simone
Ranaldi). The thesis is in [`thesis/Berardocco_MSc_thesis.pdf`](thesis/Berardocco_MSc_thesis.pdf).

The thesis asks whether neural manifolds can serve as a stable basis for an intracortical BCI. Seven latent-variable
models are validated on the FALCON H1 dataset against three requirements: the structure they extract must exceed
chance, their latents must carry the task, and they must remain valid on sessions recorded on later days. The models
are PCA, factor analysis (FA), GPFA, Isomap, UMAP, CEBRA and LFADS.

The data are the FALCON H1 recordings, available on DANDI as [DANDI:000954](https://dandiarchive.org/dandiset/000954).

## Repository layout

```
thesis/          the thesis (PDF)
src/thesis_h1/   Python package used by the notebooks and scripts
  io.py            NWB loading, binning (FALCON convention), trials and task phases
  decoding.py      causal ridge decoder with 1000 ms of history
  fa_decoder.py    ridge decoder on the FA latents, streaming
  crosssession.py  few-shot cross-session protocol
  stability.py     stability of the FA subspaces across sessions
  literature.py    comparison with the literature, decision rules R1-R7
  literature2.py   rules R8-R10: Parallel Analysis robustness, intrinsic dimension, TME null
  tme.py           Python port of Tensor Maximum Entropy
  results.py       run registry (results/h1_runs.csv)
  palette.py       colours and markers of the methods
notebooks/       within-session analyses on the reference session (percent-format sources, .src_*.py)
rep/             replication on sessions S2 and S5 and with the FALCON bins; summary tables
scripts/         decoders, literature comparison, cross-session protocol, tables and figures
evalai/          streaming GPFA decoder for the FALCON evaluator, and the Docker image build for EvalAI
tests/           checks of the code against known answers, reference libraries and synthetic data
results/         run registry, tables, reports and figures
figures/         figures exported by the notebooks
docs/            cross-session protocol and decision rules
models/          trained ridge decoders
```

## Where each result comes from

| Thesis content | Code | Output |
|---|---|---|
| Dataset and task phases | `notebooks/.src_01_dataset_exploration.py`, `scripts/make_sessions_csv.py`, `scripts/fig_task_phases.py` | `figures/01_*`, `results/sessions.csv` |
| Linear models (PCA, FA, GPFA) | `notebooks/.src_02_new.py`, `rep/make_02.py` | `figures/02new_*`, `rep/tables/` |
| Isomap and UMAP | `notebooks/.src_03_geometric.py`, `rep/registry_03.py` | `figures/03_*`, `rep/tables/` |
| CEBRA and LFADS | `notebooks/.src_04_deep.py`, `rep/make_04.py`, `rep/make_04_extended.py`, `rep/registry_04_extended.py`, `rep/make_phases.py` | `figures/04_*`, `rep/tables/` |
| Comparison with the literature (R1-R10) | `scripts/literature_*.py`, `scripts/literature_tables.py`, `rep/angles.py` | `results/literature/`, `rep/tables/angles*.csv` |
| Controls | `rep/kin_check.py`, `rep/angles_diag.py`, `scripts/io_bins_check.py` | `rep/kin/`, `rep/angles/` |
| Within-session kinematic decoding | `scripts/05_*.py`, `rep/dec_rep.py`, `scripts/import_decoder_json.py`, `scripts/curve_components.py` | `results/table_B.*`, `results/curve_components/`, `models/` |
| Cross-session degradation and few-shot recalibration | `scripts/xs_run.py`, `scripts/xs_analysis.py`, `scripts/xs_checks.py`, `scripts/stats_*_per_day.py` | `results/crosssession/`, `results/conclusions_crosssession.md` |
| Stability of the subspaces | `scripts/xs_stability.py` | `results/crosssession/stability/` |
| FALCON evaluator submission | `scripts/evalai_clock_gpfa.py`, `evalai/` | `results/evalai_clock_gpfa/`, `models/gpfa25_online_ridge_h50.npz` |
| Computing time | `scripts/make_timing_table.py` | `results/table_timing*` |

Every score produced by a script is a row of the run registry `results/h1_runs.csv`, with the experiment, method,
session, binning, split, condition, fold, seed, metric, value, timestamp and code commit that produced it.
`scripts/make_tables.py` builds the tables from it. The commit hashes refer to the private development history,
which is not published here.

The cross-session protocol in `docs/crosssession_protocol.md` and the decision rules in `docs/literature/` are
documents written to be read before the computations they govern.

## Setup

```bash
conda env create -f environment.yml
conda activate thesis-manifolds
pip install falcon-challenge==1.0.2
```

The package versions are listed in `environment.yml`.

## Data

```bash
dandi download DANDI:000954 --output-dir ~/000954
```

The data are not included in this repository.

## Running the analyses

The scripts expect the following folders under the home directory:

```
~/000954            the dandiset
~/thesis_h1_git     this repository
~/thesis_h1_work    scratch folder for caches, trained models and the run registry
```

```bash
git clone https://github.com/vberardocco/tesi_h1 ~/thesis_h1_git
mkdir -p ~/thesis_h1_work/results ~/thesis_h1_git/data ~/thesis_h1_work/data
ln -s ~/000954 ~/thesis_h1_git/data/raw
ln -s ~/000954 ~/thesis_h1_work/data/raw
export PYTHONPATH=~/thesis_h1_git/src
```

`results.RESULTS` points to `~/thesis_h1_work/results`; set `THESIS_RESULTS` to change it. Each script states its
usage in its docstring. The notebooks are stored as percent-format sources and can be converted with
`python notebooks/.py2nb.py notebooks/.src_02_new.py 02_new.ipynb`. The replication notebooks are built by
`rep/make_*.py` and run by `rep/run_queue.sh`; `scripts/xs_queue.sh` runs the cross-session protocol.

Tests that do not need the data:

```bash
PYTHONPATH=src:evalai python tests/test_stability.py
```

## Limitations

The data come from one participant. The evaluation uses the minival split, not the hidden FALCON test set. The
hyperparameters were chosen on a reference session; the replication on other sessions and with the FALCON bins is
reported in `rep/tables/`.

## Citation

Valerio Berardocco, *Neural manifolds for brain-computer interface applications*, MSc thesis, Roma Tre University.

## Status of this release

- The code, notebooks and reports were translated from Italian to English. Identifiers, file names and the run registry were renamed in English as well.
- Some dictionary keys and table labels still contain Italian text. They are lookup keys shared between notebook sections and scripts; renaming them would break those lookups. Most of them appear in the figure legends and row labels of the LFADS, CEBRA and cross-session tables.
- The tables in `rep/tables/` and the test files `tests/test_results.py` and `tests/test_fa_decoder.py` depend on the run registry and on the data folder and were not re-run after the translation. Six of the eight test files were run and pass.
- Translating comments inside some functions changes the source hash used to name the cached models, so cached LFADS and CEBRA models will be retrained on the first run.
- The commit hashes in the run registry refer to a private development history and do not exist in this repository.
