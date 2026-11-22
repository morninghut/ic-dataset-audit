#!/bin/bash
# Train one Pearl variant with the released code and hyper-parameters.
#
#   run_pearl.sh <released|pairclean|randomdrop> <seed> [extra cli args]
#
# Pearl resolves data, features and experiments/hadamard.pth relative to the
# working directory, so every run gets its own directory of symlinks and its own
# experiments/ for checkpoints.  Weights & Biases runs offline.
set -eu

ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
PEARL=$ROOT/external/Pearl
VARIANT=$1
SEED=$2
shift 2

RUN=$ROOT/runs/metrics/pearl/${VARIANT}_s${SEED}
mkdir -p "$RUN/experiments" "$RUN/configs"
ln -sfn "$PEARL/data" "$RUN/data"
ln -sfn "$PEARL/features" "$RUN/features"
ln -sfn "$PEARL/pacscore" "$RUN/pacscore"
ln -sfn "$PEARL/experiments/hadamard.pth" "$RUN/experiments/hadamard.pth"
sed "s/^seed: .*/seed: $SEED/" "$PEARL/configs/variants/$VARIANT.yaml" > "$RUN/configs/config.yaml"

cd "$RUN"
export PYTHONPATH=$PEARL
export PEARL_SKIP_IMAGES=${PEARL_SKIP_IMAGES:-1}
export WANDB_MODE=offline
export WANDB_SILENT=true
export TOKENIZERS_PARALLELISM=false
exec "$ROOT/external/envs/pearl/bin/python" "$ROOT/analysis/metrics/released_cli.py" pearl train -f configs/config.yaml "$@"
