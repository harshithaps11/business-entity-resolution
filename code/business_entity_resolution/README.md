# Business Entity Resolution — reproducible pipeline

Regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from
the challenge data using only the provided training data (no external lookups,
no pretrained models; every model is trained here from scratch).

## Environment

* Linux, Python 3.12, CPU only (developed on 16 cores / 16 GB RAM; peak RSS ≈ 8 GB).
* Install pinned dependencies:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

All libraries are permissively licensed (numpy/pandas/scipy/scikit-learn/joblib: BSD,
LightGBM: MIT, pyarrow: Apache-2.0, RapidFuzz: MIT, Unidecode: GPL-2.0 — used only
as a text-transliteration utility, not as a model).

## Run end-to-end

```bash
bash run_all.sh <dataset_dir> <work_dir> <output_dir>
# e.g. from the challenge root (the folder containing dataset/):
bash code/business_entity_resolution/run_all.sh "$PWD/dataset" "$PWD/work" "$PWD/output"
```

Stages (each can also be run alone, in this order, from `src/`):

| # | Script | Input → Output | Time* |
|---|--------|----------------|-------|
| 0 | `to_parquet.py DATA WORK` | raw TSVs → parquet cache | ~1 min |
| 1a | `learn_translit.py WORK WORK/translit.json` | train ground truth → native-script→Latin token map | ~0.5 min |
| 1b | `preprocess.py WORK` | all 6 tables → normalised tables (`*_n{1,2,3}.parquet`) | ~2 min |
| 2 | `run_blocking.py WORK {train,test} 50` | normalised tables → top-50 retrieval pairs per S1 | ~1–2 h per split |
| 2b | `train.py WORK` | train pairs + ground truth → candidate pruner (+ v1 baseline matcher) | ~15 min |
| 2c | `prune_pairs.py WORK {train,test}` | retrieval pairs → final candidate set (DuckDB, ≤3 GB RAM) | ~2 min per split |
| 3a | `build_features.py WORK {train,test}` | final candidates → cached pair features | ~20 min |
| 3b | `experiment.py WORK` | two-stage LightGBM, decision-rule selection on validation, `v2_config.json` | ~30 min |
| 4 | `predict_v2.py WORK OUT` | test features → both TSVs | ~6 min |

\*on the development machine. Stage 2 checkpoints per country and resumes if interrupted.

Validate the outputs with the organiser's checker (from the challenge root):

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Source layout (`src/`)

| File | Role |
|------|------|
| `normalize.py` | name/address normalisation (legal forms, DBA, domains, OCR digit→letter, abbreviations incl. French, state canonicalisation, transliteration) |
| `learn_translit.py` | learns native-script token maps from aligned training pairs |
| `preprocess.py` | parallel normalisation of all tables |
| `blocking.py`, `run_blocking.py` | hashed TF-IDF over name/address/number keys + compound keys; per-country sparse top-K retrieval |
| `features.py` | 52 pairwise features (name, address, house numbers, retrieval competition) |
| `prune.py`, `prune_pairs.py` | learned candidate pruner; memory-bounded application (defines the final candidate set) |
| `build_features.py` | cached feature extraction (train / valid / rest / test) |
| `experiment.py` | two-stage LightGBM (out-of-fold stage 1 → context stage 2), expected-F0.5 decision rule |
| `predict_v2.py` | final test inference (current submission) |
| `decide.py` | threshold + one-owner assignment rule; macro F0.5 metric |
| `train.py` | entity-level train/valid split, pruner + model comparison, threshold tuning |
| `predict.py` | test inference, writes both submission files |

Country is treated as an open set: blocking iterates over whatever country labels
appear in the data and no feature encodes the country, so France (absent from
training) is processed by exactly the same code path.
