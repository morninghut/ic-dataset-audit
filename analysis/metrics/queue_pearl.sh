#!/bin/bash
# Train and score a list of Pearl runs one after the other on one GPU.
#
#   CUDA_VISIBLE_DEVICES=0 nohup bash analysis/metrics/queue_pearl.sh <slot> <variant:seed>... > /dev/null 2>&1 &
#
# Same commands as queue.sh (run_pearl.sh, then score_pearl.py with both text lookups on the
# correlation, PASCAL and FOIL splits), so that several GPUs or machines can each take a
# list.  <slot> only names the progress file, runs/metrics/logs/pipeline_<slot>.log;
# each stage writes logs/<stage>.log.  A run is skipped when its directory holds a file
# named SKIP or its scores are complete, and trained again unless it holds TRAINED, which
# is written after a training that ran to its end.  Ends with runs/metrics/QUEUE_<slot>_DONE.
set -u

SLOT=$1; shift
ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
S3=$ROOT/analysis/metrics
RUNS=$ROOT/runs/metrics
LOG=$RUNS/logs
SPLITS="spica_test polaris_test nebula_test flickr8k_expert composite flickr8k_cf pascal50s polaris_val nebula_valid spica_val foil_1ref foil_4ref"
mkdir -p "$LOG"

say() { echo "[$(date '+%F %T')] $*" >> "$LOG/pipeline_$SLOT.log"; }

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
  local run=$1 checkpoint code=0
  checkpoint=$(cd "$run" && "$ROOT/external/envs/pearl/bin/python" "$S3/best_checkpoint.py" "$run") || return 1
  checkpoint=$(cd "$run" && realpath "$checkpoint")
  [ -f "$checkpoint" ] || { echo "no checkpoint in $run: $checkpoint"; return 1; }
  echo "checkpoint $checkpoint"
  (cd "$run" && PYTHONPATH=$ROOT/external/Pearl timeout -k 60 6h "$ROOT/external/envs/pearl/bin/python" \
      "$S3/score_pearl.py" "$checkpoint" "$run/scores_raw" $SPLITS) || code=$?
  (cd "$run" && PYTHONPATH=$ROOT/external/Pearl timeout -k 60 6h "$ROOT/external/envs/pearl/bin/python" \
      "$S3/score_pearl.py" "$checkpoint" "$run/scores_prefixed" --prefixed $SPLITS) || code=$?
  return $code
}

say "queue $SLOT started on GPU ${CUDA_VISIBLE_DEVICES:-all}: $* (pid $$)"
for JOB in "$@"; do
  V=${JOB%%:*}; SEED=${JOB##*:}
  RUN=$RUNS/pearl/${V}_s$SEED
  [ -f "$RUN/SKIP" ] && { say "pearl_${V}_s$SEED skipped (SKIP file)"; continue; }
  [ -f "$RUN/scores_prefixed/foil_4ref.npz" ] && { say "pearl_${V}_s$SEED already scored"; continue; }
  if [ ! -f "$RUN/TRAINED" ]; then
    stage "pearl_${V}_s$SEED" timeout -k 60 24h bash "$S3/run_pearl.sh" "$V" "$SEED" || continue
    touch "$RUN/TRAINED"
  fi
  stage "score_pearl_${V}_s$SEED" score "$RUN"
done
touch "$RUNS/QUEUE_${SLOT}_DONE"
say "queue $SLOT finished"
