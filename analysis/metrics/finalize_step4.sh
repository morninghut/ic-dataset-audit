#!/bin/bash
# Unattended end of step 4: wait for the last runs on both machines, then register and analyse.
#
#   nohup bash analysis/metrics/finalize_step4.sh > /dev/null 2>&1 &
#
# Waits for, here:      DENEB_S14_DONE, EPOCH_SWEEP_A_DONE
#           on B:       QUEUE_r1_DONE (Pearl matcheddrop seed 13), QUEUE_nothink_DONE, EPOCH_SWEEP_B_DONE
# Then copies B's results, registers deneb_s14 and the four-column pearl_s13, and runs
# check_trained.py, leakage_seeds.py (Pearl raw / prefixed, DENEB), compare_nothink.py,
# epoch_sweep_report.py and pairwise.py with three seeds.  Every stage logs to
# runs/metrics/logs/final_<stage>.log; progress in logs/pipeline_final.log.
# A stage that fails does not stop the later ones.  Ends with runs/metrics/STEP4_FINAL_DONE.
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
  local name=$1; shift
  say "start $name"
  echo "===== $(date '+%F %T'): $*" >> "$LOG/final_$name.log"
  (cd "$S3" && "$@") >> "$LOG/final_$name.log" 2>&1
  say "end   $name (exit $?)"
}
remote_has() { $SSH $HOST "test -f $RUNS/$1"; }

say "finalize started (pid $$)"
for flag in DENEB_S14_DONE EPOCH_SWEEP_A_DONE; do
  while [ ! -f "$RUNS/$flag" ]; do sleep 300; done
  say "have $flag"
done
for flag in QUEUE_nothink_DONE QUEUE_r1_DONE EPOCH_SWEEP_B_DONE; do
  while ! remote_has $flag; do sleep 300; done
  say "have $flag (remote)"
done

copy() {
  rsync -a -m -e "$RSH" --include='*/' --include='scores_raw/***' --include='scores_prefixed/***' \
      --include='epoch_scores/***' --include='TRAINED' --exclude='*' "$HOST:$RUNS/pearl/" "$RUNS/pearl/" &&
  rsync -a -e "$RSH" "$HOST:$RUNS/vlm/" "$RUNS/vlm/" &&
  rsync -a -e "$RSH" "$HOST:$RUNS/logs/" "$RUNS/logs/remote/"
}
stage copy copy

P=$RUNS/pearl; D=$RUNS/deneb
stage register $PY register.py pearl_s13 released=$P/released_s13/scores_raw pairclean=$P/pairclean_s13/scores_raw \
    randomdrop=$P/randomdrop_s13/scores_raw matcheddrop=$P/matcheddrop_s13/scores_raw
stage register $PY register.py pearl_s13_prefixed released=$P/released_s13/scores_prefixed pairclean=$P/pairclean_s13/scores_prefixed \
    randomdrop=$P/randomdrop_s13/scores_prefixed matcheddrop=$P/matcheddrop_s13/scores_prefixed
stage register $PY register.py deneb_s14 released=$D/released_s14/scores pairclean=$D/pairclean_s14/scores randomdrop=$D/randomdrop_s14/scores
stage check_trained $PY check_trained.py pearl pearl_s13
stage check_trained $PY check_trained.py deneb deneb_s14

stage nothink $PY compare_nothink.py
stage epoch_sweep $PY epoch_sweep_report.py
stage leakage_pearl $PY leakage_seeds.py pearl spica_train 12 13 14
stage leakage_pearl_prefixed $PY leakage_seeds.py pearl spica_train 12 13 14 --suffix _prefixed
stage leakage_deneb $PY leakage_seeds.py deneb nebula_train 12 13 14
OMP_NUM_THREADS=1 stage pairwise $PY pairwise.py 10000 20 12,13,14 44
stage tables $PY pairwise_tables.py pairwise_s12-13-14.json

touch "$RUNS/STEP4_FINAL_DONE"
say "finalize finished"
