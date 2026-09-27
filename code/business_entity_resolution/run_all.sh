#!/usr/bin/env bash
# End-to-end: raw TSVs -> normalised tables -> blocking -> pruning -> two-stage model -> output TSVs.
# Usage: bash run_all.sh <dataset_dir> <work_dir> <output_dir>
set -euo pipefail
DATA=${1:-../../dataset}; WORK=${2:-../../work}; OUT=${3:-../../output}
SRC="$(cd "$(dirname "$0")/src" && pwd)"
mkdir -p "$WORK" "$OUT"
cd "$SRC"
python3 to_parquet.py "$DATA" "$WORK"                    # 0. TSV -> parquet
python3 learn_translit.py "$WORK" "$WORK/translit.json"  # 1a. native-script maps (train only)
python3 preprocess.py "$WORK"                            # 1b. normalise all tables
python3 run_blocking.py "$WORK" train 50                 # 2a. retrieval: top-50 per S1
python3 run_blocking.py "$WORK" test 50
python3 train.py "$WORK"                                 # 2b. train candidate pruner (+ v1 baseline matcher)
python3 prune_pairs.py "$WORK" train                     # 2c. apply pruner -> final candidate sets
python3 prune_pairs.py "$WORK" test
python3 build_features.py "$WORK" train                  # 3a. cache pair features
python3 build_features.py "$WORK" test
python3 experiment.py "$WORK"                            # 3b. two-stage LightGBM + decision rule (validated)
python3 predict_v2.py "$WORK" "$OUT"                     # 4. test inference -> both submission files
