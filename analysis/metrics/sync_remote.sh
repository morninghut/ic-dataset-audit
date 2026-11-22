#!/bin/bash
# Bring the results of the second machine back, every 20 minutes until its queues end.
#
#   nohup bash analysis/metrics/sync_remote.sh <queue slots...> > /dev/null 2>&1 &
#
# The second machine (${REMOTE_HOST:?user@host of the second machine}, port ${REMOTE_PORT:-22}) holds a copy of this tree at the same
# path and runs queue_pearl.sh / queue_vlm.sh there.  Copied back: the score files of
# every Pearl run, everything under vlm/, and the logs (into logs/remote/).  Checkpoints
# stay there.  When runs/metrics/QUEUE_<slot>_DONE exists for every slot, one last
# copy is made, the validation score of every checkpoint is written to
# logs/remote/checkpoints.txt, and runs/metrics/REMOTE_DONE is created here.
set -u

ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
RUNS=$ROOT/runs/metrics
HOST=${REMOTE_HOST:?user@host of the second machine}
SSH="ssh -n -p ${REMOTE_PORT:-22} -o BatchMode=yes -o ConnectTimeout=20"
RSH="ssh -p ${REMOTE_PORT:-22} -o BatchMode=yes -o ConnectTimeout=20"
mkdir -p "$RUNS/logs/remote"

copy() {
  rsync -a -m -e "$RSH" --include='*/' --include='scores_raw/***' --include='scores_prefixed/***' \
      --include='TRAINED' --exclude='*' "$HOST:$RUNS/pearl/" "$RUNS/pearl/"
  rsync -a -e "$RSH" "$HOST:$RUNS/vlm/" "$RUNS/vlm/"
  rsync -a -e "$RSH" "$HOST:$RUNS/logs/" "$RUNS/logs/remote/"
  echo "[$(date '+%F %T')] copied" >> "$RUNS/logs/remote/sync.log"
}

finished() {
  local slot
  for slot in "$@"; do
    $SSH $HOST "test -f $RUNS/QUEUE_${slot}_DONE" || return 1
  done
}

while ! finished "$@"; do
  copy
  sleep 1200
done
copy
$SSH $HOST "cd $RUNS/pearl && for run in */; do [ -d \$run/experiments/lightning ] || continue; echo \"== \$run\"; \
    (cd \$run && $ROOT/external/envs/pearl/bin/python $ROOT/analysis/metrics/best_checkpoint.py $RUNS/pearl/\$run 2>&1 | grep -a 'monitored\|ckpt'); done" \
    > "$RUNS/logs/remote/checkpoints.txt" 2>&1
touch "$RUNS/REMOTE_DONE"
echo "[$(date '+%F %T')] remote queues finished" >> "$RUNS/logs/remote/sync.log"
