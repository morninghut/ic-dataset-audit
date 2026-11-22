#!/bin/bash
# One more Pearl seed on one GPU: train and score the variants one after the other.
#
#   nohup bash analysis/metrics/queue_pearl_seed.sh <seed> [variants...] > /dev/null 2>&1 &
#
# Same commands as queue.sh (run_pearl.sh, then score_pearl.py with both text lookups on
# the correlation, PASCAL and FOIL splits).  A variant is skipped when its directory holds
# a file named SKIP (so it can be handed to another machine while the queue runs) or when
# its scores are complete.  Progress goes to runs/metrics/logs/pipeline.log, each
# stage to logs/<stage>.log.  Ends with runs/metrics/PEARL_S<seed>_DONE.
set -u

SEED=$1; shift
VARIANTS=${*:-released pairclean randomdrop}
ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
S3=$ROOT/analysis/metrics
RUNS=$ROOT/runs/metrics
LOG=$RUNS/logs
SPLITS="spica_test polaris_test nebula_test flickr8k_expert composite flickr8k_cf pascal50s polaris_val nebula_valid spica_val foil_1ref foil_4ref"

say() { echo "[$(date '+%F %T')] $*" >> "$LOG/pipeline.log"; }

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
  (cd "$run" && PYTHONPATH=$ROOT/external/Pearl timeout -k 60 4h "$ROOT/external/envs/pearl/bin/python" \
      "$S3/score_pearl.py" "$checkpoint" "$run/scores_raw" $SPLITS) || code=$?
  (cd "$run" && PYTHONPATH=$ROOT/external/Pearl timeout -k 60 4h "$ROOT/external/envs/pearl/bin/python" \
      "$S3/score_pearl.py" "$checkpoint" "$run/scores_prefixed" --prefixed $SPLITS) || code=$?
  return $code
}

say "pearl seed $SEED queue started: $VARIANTS (pid $$)"
for V in $VARIANTS; do
  RUN=$RUNS/pearl/${V}_s$SEED
  [ -f "$RUN/SKIP" ] && { say "pearl_${V}_s$SEED skipped (SKIP file)"; continue; }
  [ -f "$RUN/scores_prefixed/foil_4ref.npz" ] && { say "pearl_${V}_s$SEED already scored"; continue; }
  if [ ! -f "$RUN/TRAINED" ]; then  # written only after a training that ran to its end
    stage "pearl_${V}_s$SEED" timeout -k 60 16h bash "$S3/run_pearl.sh" "$V" "$SEED" || continue
    touch "$RUN/TRAINED"
  fi
  [ -f "$RUN/SKIP" ] && { say "pearl_${V}_s$SEED scoring skipped (SKIP file)"; continue; }
  stage "score_pearl_${V}_s$SEED" score "$RUN"
done
touch "$RUNS/PEARL_S${SEED}_DONE"
say "pearl seed $SEED queue finished"
