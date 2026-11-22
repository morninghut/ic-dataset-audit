#!/bin/bash
# Recompute the final ranking after a change to protocol.py, keeping the previous one.
#
#   nohup bash analysis/metrics/rerank.sh <tag of the previous ranking> > /dev/null 2>&1 &
#
# Moves runs/metrics/pairwise_s12-13-14.json and pairwise_tables_final.txt to
# runs/metrics/archive_rankings/<tag>/, then runs pairwise.py with three seeds and the tables.
# Logs: runs/metrics/logs/pairwise_final.log.  Ends with runs/metrics/RERANK_DONE.
set -u

TAG=$1
ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
S3=$ROOT/analysis/metrics
RUNS=$ROOT/runs/metrics
PY=$ROOT/.venv/bin/python
export PYTHONPATH=$ROOT/Polos-revised HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUTF8=1
cd "$S3"

rm -f "$RUNS/RERANK_DONE"
mkdir -p "$RUNS/archive_rankings/$TAG"
cp "$RUNS/pairwise_s12-13-14.json" "$RUNS/pairwise_tables_final.txt" "$RUNS/archive_rankings/$TAG/"
OMP_NUM_THREADS=1 $PY pairwise.py 10000 20 12,13,14 44 > "$RUNS/logs/pairwise_final.log" 2>&1 &&
  $PY pairwise_tables.py pairwise_s12-13-14.json > "$RUNS/pairwise_tables_final.txt" 2>&1
echo "exit $?" >> "$RUNS/logs/pairwise_final.log"
touch "$RUNS/RERANK_DONE"
