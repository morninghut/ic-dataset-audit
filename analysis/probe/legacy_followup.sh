#!/bin/bash
# After the authors' configuration (clip + legacy_cached) finishes its fold models: score them
# together with the released Polos checkpoint, then the cross-fit effect and the pair-level
# memorization, and finally its own decontamination pair.
set -e
cd ${CAPAUDIT_ROOT:?set CAPAUDIT_ROOT to the repository root}/analysis/probe
export PYTHONPATH=${CAPAUDIT_ROOT:?set CAPAUDIT_ROOT to the repository root}/Polos-revised HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python=../../.venv/bin/python
while ! grep -q "ALL DONE" ../../runs/probe/logs/legacy_polaris_s42.log; do sleep 60; done
$python predict_mlp.py polaris_train 42 clip legacy_cached
$python analyze_effects.py mlp_legacy_cached_clip_s42 polaris_train 1000
$python memorization.py mlp_legacy_cached_clip_s42 polaris_train 1000
./train_variants.sh polaris_train clip legacy_cached 42
$python predict_mlp.py polaris_train 42 clip legacy_cached full,pairclean,randomdrop
echo "ALL DONE legacy followup"
