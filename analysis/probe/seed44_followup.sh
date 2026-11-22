#!/bin/bash
# Score and analyze the third seed of the MLP head on both pools.
set -e
cd ${CAPAUDIT_ROOT:?set CAPAUDIT_ROOT to the repository root}/analysis/probe
export PYTHONPATH=${CAPAUDIT_ROOT:?set CAPAUDIT_ROOT to the repository root}/Polos-revised HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python=../../.venv/bin/python
for pool in polaris_train spica_train; do
  $python predict_mlp.py "$pool" 44
  $python analyze_effects.py mlp_siglip_s44 "$pool" 1000
done
echo "ALL DONE seed44"
