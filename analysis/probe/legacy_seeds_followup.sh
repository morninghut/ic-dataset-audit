#!/bin/bash
# Score and analyze seeds 43 and 44 of the authors' configuration (clip + legacy_cached).
set -e
cd ${CAPAUDIT_ROOT:?set CAPAUDIT_ROOT to the repository root}/analysis/probe
export PYTHONPATH=${CAPAUDIT_ROOT:?set CAPAUDIT_ROOT to the repository root}/Polos-revised HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python=../../.venv/bin/python
for seed in 43 44; do
  while ! grep -q "ALL DONE" ../../runs/probe/logs/legacy_polaris_s$seed.log; do sleep 30; done
  $python predict_mlp.py polaris_train $seed clip legacy_cached
  $python analyze_effects.py mlp_legacy_cached_clip_s$seed polaris_train 1000
  $python memorization.py mlp_legacy_cached_clip_s$seed polaris_train 1000
done
echo "ALL DONE legacy seeds"
