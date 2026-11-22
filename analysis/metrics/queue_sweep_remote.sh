#!/bin/bash
# On the second machine: score every epoch of the Pearl trainings whose checkpoints live there.
#
#   nohup bash analysis/metrics/queue_sweep_remote.sh > /dev/null 2>&1 &
#
# Waits for the no-think judge check to free GPU 0 (QUEUE_nothink_DONE) and for the last training
# to end (matcheddrop_s13/TRAINED), then runs epoch_sweep.py on the four trainings.
# Log: runs/metrics/logs/epoch_sweep_B.log.  Ends with runs/metrics/EPOCH_SWEEP_B_DONE.
set -u

ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
S3=$ROOT/analysis/metrics
RUNS=$ROOT/runs/metrics
LOG=$RUNS/logs/epoch_sweep_B.log
export CUDA_VISIBLE_DEVICES=0
ulimit -n 524288

sweep() {
  echo "[$(date '+%F %T')] sweep $*" >> "$LOG"
  local code=0
  (cd "$RUNS/pearl" && PYTHONPATH=$ROOT/external/Pearl:$S3 "$ROOT/external/envs/pearl/bin/python" \
      "$S3/epoch_sweep.py" "$@") >> "$LOG" 2>&1 || code=$?
  echo "[$(date '+%F %T')] exit $code" >> "$LOG"
}

while [ ! -f "$RUNS/QUEUE_nothink_DONE" ]; do sleep 120; done
while [ ! -f "$RUNS/pearl/matcheddrop_s13/TRAINED" ]; do sleep 300; done
sweep "$RUNS/pearl/pairclean_s14" "$RUNS/pearl/randomdrop_s14" "$RUNS/pearl/matcheddrop_s12" "$RUNS/pearl/matcheddrop_s13"
touch "$RUNS/EPOCH_SWEEP_B_DONE"
