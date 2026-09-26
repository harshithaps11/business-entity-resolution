#!/usr/bin/env bash
# End-to-end: raw TSVs -> normalised tables -> blocking -> model -> output TSVs.
# Usage: bash run_all.sh <dataset_dir> <work_dir> <output_dir>
set -euo pipefail
DATA=${1:-../../dataset}; WORK=${2:-../../work}; OUT=${3:-../../output}
SRC="$(cd "$(dirname "$0")/src" && pwd)"
mkdir -p "$WORK" "$OUT"
cd "$SRC"
python3 to_parquet.py "$DATA" "$WORK"          # 0. TSV -> parquet
python3 learn_translit.py "$WORK" "$WORK/translit.json"   # 1a. native-script maps (train only)
python3 preprocess.py "$WORK"                   # 1b. normalise all tables
python3 run_blocking.py "$WORK" train 50        # 2. candidate generation
python3 run_blocking.py "$WORK" test 50
python3 train.py "$WORK"                        # 3. features + model selection
python3 predict.py "$WORK" "$OUT"               # 4. test inference + submission files
