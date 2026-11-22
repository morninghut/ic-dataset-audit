#!/bin/bash
# Score the evaluation splits with further vision-language judges, one model after the other.
#
#   CUDA_VISIBLE_DEVICES=1 nohup bash analysis/metrics/queue_vlm.sh <slot> <hub id>... > /dev/null 2>&1 &
#
# Per model: wait for its download to be complete, then score_vlm.py with references and
# without on the correlation and PASCAL splits, then without references on FOIL (as for
# the first judge).  Finished triples are kept, so a smaller batch or a restart resumes.
# VLM_BATCHES (default "16 8 4") lists the batch sizes to try; start lower when the GPU is shared.
# Progress in runs/metrics/logs/pipeline_<slot>.log, stages in logs/<stage>.log.
set -u

SLOT=$1; shift
ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
S3=$ROOT/analysis/metrics
LOG=$ROOT/runs/metrics/logs
PY=$ROOT/external/envs/vlm/bin/python
CORE="spica_test polaris_test nebula_test flickr8k_expert composite flickr8k_cf pascal50s polaris_val nebula_valid spica_val"
FOIL="foil_1ref foil_4ref"
mkdir -p "$LOG"

say() { echo "[$(date '+%F %T')] $*" >> "$LOG/pipeline_$SLOT.log"; }

stage() {  # stage <name> <command...>
  local name=$1 code; shift
  say "start $name"
  echo "===== $(date '+%F %T'): $*" >> "$LOG/$name.log"
  "$@" >> "$LOG/$name.log" 2>&1
  code=$?
  say "end   $name (exit $code)"
  return $code
}

download() {  # blocks until every file of the model is in the local cache
  HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} "$PY" -c "
import sys, time
from huggingface_hub import snapshot_download
for attempt in range(60):
    try:
        print(snapshot_download(sys.argv[1], max_workers=4), flush=True); sys.exit(0)
    except Exception as error:
        print('attempt', attempt, repr(error)[:300], flush=True); time.sleep(30)
sys.exit(1)" "$1"
}

judge() {  # judge <model> <mode> <splits...>: smaller batches after a failure
  local model=$1 mode=$2 batch; shift 2
  for batch in ${VLM_BATCHES:-16 8 4}; do
    echo "== batch size $batch"
    VLM_MODEL=$model HF_HUB_OFFLINE=1 timeout -k 60 24h "$PY" "$S3/score_vlm.py" "$mode" "$batch" "$@" && return 0
  done
  return 1
}

say "queue $SLOT started on GPU ${CUDA_VISIBLE_DEVICES:-all}: $* (pid $$)"
for MODEL in "$@"; do
  NAME=${MODEL##*/}
  stage "download_$NAME" download "$MODEL" || continue
  stage "vlm_${NAME}_refs" judge "$MODEL" refs $CORE
  stage "vlm_${NAME}_free" judge "$MODEL" free $CORE
done
for MODEL in "$@"; do
  stage "vlm_${MODEL##*/}_free_foil" judge "$MODEL" free $FOIL
done
touch "$ROOT/runs/metrics/QUEUE_${SLOT}_DONE"
say "queue $SLOT finished"
