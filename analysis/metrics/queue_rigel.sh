#!/bin/bash
# On the second machine: score the six test sets with the released Rigel checkpoint.
#
#   CUDA_VISIBLE_DEVICES=1 nohup bash analysis/metrics/queue_rigel.sh > /dev/null 2>&1 &
#
# Needs external/Rigel (SHUonkei/Rigel at 72394e2), external/envs/rigel_extra (peft, omegaconf,
# installed with uv pip install --no-deps --target) and the two models in the Hugging Face cache
# (logs/download_rigel.log).  With references, then without.  Progress in
# runs/metrics/logs/pipeline_rigel.log; ends with runs/metrics/QUEUE_rigel_DONE.
set -u

ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
S3=$ROOT/analysis/metrics
RUNS=$ROOT/runs/metrics
LOG=$RUNS/logs
PY=$ROOT/external/envs/vlm/bin/python
SPLITS="composite flickr8k_expert nebula_test flickr8k_cf spica_test polaris_test"
export PYTHONPATH=$ROOT/external/envs/rigel_extra HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

say() { echo "[$(date '+%F %T')] $*" >> "$LOG/pipeline_rigel.log"; }

score() {  # score <mode>: smaller batches after a failure; finished triples are kept
  local mode=$1 batch code=1
  say "start rigel_$mode"
  for batch in 8 4 2; do
    echo "===== $(date '+%F %T') batch size $batch" >> "$LOG/rigel_$mode.log"
    timeout -k 60 24h "$PY" "$S3/score_rigel.py" "$mode" "$batch" $SPLITS >> "$LOG/rigel_$mode.log" 2>&1
    code=$?
    [ $code -eq 0 ] && break
  done
  say "end   rigel_$mode (exit $code)"
}

say "rigel queue started on GPU ${CUDA_VISIBLE_DEVICES:-all} (pid $$)"
while pgrep -f 'snapshot_downloa[d]' > /dev/null; do sleep 30; done
score refs
score free
touch "$RUNS/QUEUE_rigel_DONE"
say "rigel queue finished"
