#!/bin/bash
# Train the Polos MLP head on one pool: the released rows and the five held-out-fold subsets.
set -e
pool=$1
seed=${2:-42}
cd ${CAPAUDIT_ROOT:?set CAPAUDIT_ROOT to the repository root}/Polos-revised
for name in full heldout0 heldout1 heldout2 heldout3 heldout4; do
  ../.venv/bin/python -m revision.train --backbone siglip --variant full --seed "$seed" \
    --cache ../feature_cache_v2 \
    --train-name "${pool}_${name}" --val-name "${pool}_cleanval" \
    --stats "../feature_cache_v2/siglip_stats_${pool}_${name}.pt" \
    --output "../runs/probe/mlp/${pool}_s${seed}/${name}"
  echo "DONE $pool $name seed $seed"
done
echo "ALL DONE $pool seed $seed"
