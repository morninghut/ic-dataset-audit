#!/bin/bash
# Pair-decontamination experiment for the MLP head: the pool without duplicated pairs and the
# size-matched control. The "full" model of the same seed is trained by train_folds*.sh.
set -e
pool=$1; backbone=${2:-siglip}; variant=${3:-full}; seed=${4:-42}
cd ${CAPAUDIT_ROOT:?set CAPAUDIT_ROOT to the repository root}/Polos-revised
for name in pairclean randomdrop; do
  stats="../feature_cache_v2/${backbone}_stats_${pool}_${name}.pt"
  [ -f "$stats" ] || stats="../feature_cache_v2/${backbone}_stats.pt"
  if [ "$backbone" = siglip ] && [ "$variant" = full ]; then
    output="../runs/probe/mlp/${pool}_s${seed}/${name}"
  else
    output="../runs/probe/mlp_${variant}_${backbone}/${pool}_s${seed}/${name}"
  fi
  ../.venv/bin/python -m revision.train --backbone "$backbone" --variant "$variant" --seed "$seed" \
    --cache ../feature_cache_v2 --train-name "${pool}_${name}" --val-name "${pool}_cleanval" \
    --stats "$stats" --output "$output"
  echo "DONE $pool $name $backbone $variant seed $seed"
done
echo "ALL DONE variants $pool $backbone $variant seed $seed"
