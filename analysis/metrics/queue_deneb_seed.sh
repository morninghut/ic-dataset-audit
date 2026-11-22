#!/bin/bash
# One more DENEB seed on one GPU: the three variants train side by side, then are scored side by side.
#
#   nohup bash analysis/metrics/queue_deneb_seed.sh <seed> > /dev/null 2>&1 &
#
# Same commands as queue.sh (run_deneb.sh, then score_deneb.py on the correlation and PASCAL
# splits).  Progress in runs/metrics/logs/pipeline_deneb_s<seed>.log, stages in
# logs/<stage>.log.  Ends with runs/metrics/DENEB_S<seed>_DONE.
set -u

SEED=$1
ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
S3=$ROOT/analysis/metrics
RUNS=$ROOT/runs/metrics
LOG=$RUNS/logs
VARIANTS="released pairclean randomdrop"
SPLITS="spica_test polaris_test nebula_test flickr8k_expert composite flickr8k_cf pascal50s polaris_val nebula_valid spica_val"

say() { echo "[$(date '+%F %T')] $*" >> "$LOG/pipeline_deneb_s$SEED.log"; }

stage() {  # stage <name> <command...>: one retry unless it timed out
  local name=$1 code=1 attempt; shift
  for attempt in 1 2; do
    say "start $name (attempt $attempt)"
    echo "===== $(date '+%F %T') attempt $attempt: $*" >> "$LOG/$name.log"
    "$@" >> "$LOG/$name.log" 2>&1
    code=$?
    say "end   $name (exit $code)"
    [ $code -eq 0 ] && return 0
    [ $code -eq 124 ] && return 124
    sleep 60
  done
  return $code
}

score() {  # score <run dir>
  local run=$1 checkpoint
  checkpoint=$(cd "$run" && "$ROOT/external/envs/deneb/bin/python" "$S3/best_checkpoint.py" "$run") || return 1
  checkpoint=$(cd "$run" && realpath "$checkpoint")
  [ -f "$checkpoint" ] || { echo "no checkpoint in $run: $checkpoint"; return 1; }
  echo "checkpoint $checkpoint"
  (cd "$run" && PYTHONPATH=$ROOT/external/DENEB timeout -k 60 6h "$ROOT/external/envs/deneb/bin/python" \
      "$S3/score_deneb.py" "$checkpoint" "$run/scores" $SPLITS)
}

say "deneb seed $SEED queue started (pid $$)"
JOBS=
for V in $VARIANTS; do stage "deneb_${V}_s$SEED" timeout -k 60 16h bash "$S3/run_deneb.sh" "$V" "$SEED" & JOBS="$JOBS $!"; sleep 120; done
wait $JOBS
JOBS=
for V in $VARIANTS; do stage "score_deneb_${V}_s$SEED" score "$RUNS/deneb/${V}_s$SEED" & JOBS="$JOBS $!"; done
wait $JOBS
touch "$RUNS/DENEB_S${SEED}_DONE"
say "deneb seed $SEED queue finished"
