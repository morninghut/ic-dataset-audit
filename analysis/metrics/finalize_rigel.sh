#!/bin/bash
# After the step-4 finalize and the Rigel scoring: register Rigel and redo the ranking with it.
#
#   nohup bash analysis/metrics/finalize_rigel.sh > /dev/null 2>&1 &
#
# Waits for runs/metrics/STEP4_FINAL_DONE here and QUEUE_rigel_DONE on the second machine.
# The ranking without Rigel is kept as pairwise_s12-13-14_norigel.json.  Stages log to
# runs/metrics/logs/rigel_<stage>.log, progress to logs/pipeline_final.log.
# Ends with runs/metrics/RIGEL_DONE.
set -u

ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
S3=$ROOT/analysis/metrics
RUNS=$ROOT/runs/metrics
LOG=$RUNS/logs
HOST=${REMOTE_HOST:?user@host of the second machine}
SSH="ssh -n -p ${REMOTE_PORT:-22} -o BatchMode=yes -o ConnectTimeout=20"
RSH="ssh -p ${REMOTE_PORT:-22} -o BatchMode=yes -o ConnectTimeout=20"
PY=$ROOT/.venv/bin/python
export PYTHONPATH=$ROOT/Polos-revised HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONUTF8=1

say() { echo "[$(date '+%F %T')] $*" >> "$LOG/pipeline_final.log"; }
stage() {  # stage <name> <command...>
  local name=$1 code=0; shift
  say "start rigel_$name"
  echo "===== $(date '+%F %T'): $*" >> "$LOG/rigel_$name.log"
  (cd "$S3" && "$@") >> "$LOG/rigel_$name.log" 2>&1 || code=$?
  say "end   rigel_$name (exit $code)"
}

say "rigel finalize started (pid $$)"
while [ ! -f "$RUNS/STEP4_FINAL_DONE" ]; do sleep 300; done
while ! $SSH $HOST "test -f $RUNS/QUEUE_rigel_DONE"; do sleep 300; done
copy() {
  rsync -a -e "$RSH" "$HOST:$RUNS/vlm/" "$RUNS/vlm/" && rsync -a -e "$RSH" "$HOST:$RUNS/logs/" "$RUNS/logs/remote/"
}
stage copy copy
stage register $PY register.py rigel free=$RUNS/vlm/Rigel/free refs=$RUNS/vlm/Rigel/refs
stage check $PY check_rigel.py
[ -f "$RUNS/pairwise_s12-13-14_norigel.json" ] || cp "$RUNS/pairwise_s12-13-14.json" "$RUNS/pairwise_s12-13-14_norigel.json"
OMP_NUM_THREADS=1 stage pairwise $PY pairwise.py 10000 20 12,13,14 44
stage tables $PY pairwise_tables.py pairwise_s12-13-14.json
touch "$RUNS/RIGEL_DONE"
say "rigel finalize finished"
