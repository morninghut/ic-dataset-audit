#!/bin/bash
# Fold models of one pool for any backbone and training variant of revision.train.
# legacy_cached with clip is the authors' Polos configuration (batch layer normalization,
# no feature standardization), so the statistics file is not used by it.
set -e
pool=$1; backbone=$2; variant=$3; seed=${4:-42}
cd ${CAPAUDIT_ROOT:?set CAPAUDIT_ROOT to the repository root}/Polos-revised
for name in full heldout0 heldout1 heldout2 heldout3 heldout4; do
  stats="../feature_cache_v2/${backbone}_stats_${pool}_${name}.pt"
  [ -f "$stats" ] || stats="../feature_cache_v2/${backbone}_stats.pt"
  ../.venv/bin/python -m revision.train --backbone "$backbone" --variant "$variant" --seed "$seed" \
    --cache ../feature_cache_v2 --train-name "${pool}_${name}" --val-name "${pool}_cleanval" \
    --stats "$stats" --output "../runs/probe/mlp_${variant}_${backbone}/${pool}_s${seed}/${name}"
  echo "DONE $pool $name $backbone $variant seed $seed"
done
echo "ALL DONE $pool $backbone $variant seed $seed"
