#!/usr/bin/env bash
# Build <team>_submission.zip in the structure required by the challenge.
# Usage: bash make_submission.sh <team_name>
set -euo pipefail
TEAM=${1:?team name}
cd "$(dirname "$0")"
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
STAGE=$(mktemp -d)
mkdir -p "$STAGE/output" "$STAGE/code"
cp output/matching_results.tsv output/candidate_pairs.tsv "$STAGE/output/"
rsync -a --exclude '__pycache__' code/business_entity_resolution "$STAGE/code/"
cp Documentation_template.md "$STAGE/"
rm -f "${TEAM}_submission.zip"
(cd "$STAGE" && zip -qr "$OLDPWD/${TEAM}_submission.zip" output code Documentation_template.md)
rm -rf "$STAGE"
unzip -l "${TEAM}_submission.zip" | tail -n +1
