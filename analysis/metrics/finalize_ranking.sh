#!/bin/bash
# Final ranking of step 4: InternVL read after an empty think block, Rigel included.
#
#   nohup bash analysis/metrics/finalize_ranking.sh > /dev/null 2>&1 &
#
# Keeps the ranking computed before (InternVL without the prefill) under archive_1115_preview/,
# then runs pairwise.py with three seeds, the tables, and rigel_groups.py side by side.
# Logs: runs/metrics/logs/pairwise_final.log, pairwise_final_tables.txt, rigel_groups.log.
# Ends with runs/metrics/RANKING_FINAL_DONE.
set -u

ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
S3=$ROOT/analysis/metrics
RUNS=$ROOT/runs/metrics
PY=$ROOT/.venv/bin/python
export PYTHONPATH=$ROOT/Polos-revised HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUTF8=1
cd "$S3"

KEEP=$RUNS/archive_1115_preview
[ -f "$KEEP/pairwise_s12-13-14_internvl_conditional.json" ] || {
  cp "$RUNS/pairwise_s12-13-14.json" "$KEEP/pairwise_s12-13-14_internvl_conditional.json"
  cp "$RUNS/logs/rigel_tables.log" "$KEEP/pairwise_tables_internvl_conditional.txt"
}
OMP_NUM_THREADS=8 $PY rigel_groups.py 2000 > "$RUNS/logs/rigel_groups.log" 2>&1 &
OMP_NUM_THREADS=1 $PY pairwise.py 10000 20 12,13,14 40 > "$RUNS/logs/pairwise_final.log" 2>&1 &&
  $PY pairwise_tables.py pairwise_s12-13-14.json > "$RUNS/logs/pairwise_final_tables.txt" 2>&1
wait
touch "$RUNS/RANKING_FINAL_DONE"
