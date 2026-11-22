#!/bin/bash
# Score every saved epoch of the nine DENEB trainings on the Polaris test set (the DENEB
# counterpart of metrics/epoch_sweep.py).  Four workers share the GPU; a checkpoint whose
# score file exists is skipped, so the script can be restarted.
#
#   nohup setsid bash analysis/summary/deneb_epoch_sweep.sh > /dev/null 2>&1 &
#
# Writes <run>/epoch_scores/<checkpoint stem>/polaris_test.npz and, at the end,
# runs/summary/DENEB_SWEEP_DONE.  Progress: runs/summary/logs/deneb_sweep_w*.log
ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
OUT=$ROOT/runs/summary
WORKERS=4
mkdir -p "$OUT/logs"
rm -f "$OUT/DENEB_SWEEP_DONE"
ls "$ROOT"/runs/metrics/deneb/*_s*/experiments/lightning/version_*/epoch=*.ckpt > "$OUT/deneb_checkpoints.txt"

worker() {
  local w=$1 n=0
  while read -r checkpoint; do
    n=$((n + 1))
    [ $(( (n - 1) % WORKERS )) -eq "$w" ] || continue
    run=$(dirname "$(dirname "$(dirname "$(dirname "$checkpoint")")")")
    out=$run/epoch_scores/$(basename "$checkpoint" .ckpt)
    [ -f "$out/polaris_test.npz" ] && continue
    echo "$(date +%H:%M:%S) $checkpoint"
    (cd "$run" && PYTHONPATH=$ROOT/external/DENEB TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        timeout -k 60 2h "$ROOT/external/envs/deneb/bin/python" "$ROOT/analysis/metrics/score_deneb.py" \
        "$checkpoint" "$out" polaris_test) || echo "FAILED $checkpoint"
  done < "$OUT/deneb_checkpoints.txt"
}

for w in $(seq 0 $((WORKERS - 1))); do worker "$w" > "$OUT/logs/deneb_sweep_w$w.log" 2>&1 & done
wait
date > "$OUT/DENEB_SWEEP_DONE"
