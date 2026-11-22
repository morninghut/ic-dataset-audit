#!/bin/bash
# Cross-fit the small network (MLP on frozen SigLIP features) on the Nebula training set, three
# seeds, as was done for the Polaris and Spica training sets in step 2.  Nebula stores one
# averaged score per image-candidate pair, so this tests whether a learner that does gain from
# duplicated samples elsewhere also gains from Nebula's averaged copies of the Polaris test set.
#
#   nohup setsid bash analysis/summary/mlp_nebula.sh > runs/summary/logs/mlp_nebula.log 2>&1 &
set -e
ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
cd $ROOT/analysis/probe
export PYTHONPATH=$ROOT/Polos-revised HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUTF8=1
python=$ROOT/.venv/bin/python
$python make_subsets.py siglip nebula_train
for seed in 42 43 44; do
  bash train_folds.sh nebula_train $seed
  $python predict_mlp.py nebula_train $seed
  $python analyze_effects.py mlp_siglip_s$seed nebula_train 1000
done
date > $ROOT/runs/summary/MLP_NEBULA_DONE
