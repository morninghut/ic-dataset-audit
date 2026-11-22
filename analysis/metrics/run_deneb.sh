#!/bin/bash
# Train one DENEB variant with the released code and hyper-parameters.
#
#   run_deneb.sh <released|pairclean|randomdrop> <seed> [extra cli args]
#
# DENEB resolves data_en/, experiments/openClip_ViT-L-14.pth and its checkpoints
# relative to the working directory, so every run gets its own directory.
set -eu

ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
DENEB=$ROOT/external/DENEB
VARIANT=$1
SEED=$2
shift 2

case $VARIANT in
  released) TRAIN=train.csv ;;
  pairclean) TRAIN=train_pairclean.csv ;;
  randomdrop) TRAIN=train_randomdrop.csv ;;
  *) echo "unknown variant $VARIANT" >&2; exit 2 ;;
esac

RUN=$ROOT/runs/metrics/deneb/${VARIANT}_s${SEED}
mkdir -p "$RUN/experiments" "$RUN/configs"
ln -sfn "$DENEB/data_en" "$RUN/data_en"
ln -sfn "$ROOT/external/weights/pacs_drive/openClip_ViT-L-14.pth" "$RUN/experiments/openClip_ViT-L-14.pth"
sed -e "s/^seed: .*/seed: $SEED/" \
    -e "s#^train_path: .*#train_path: data_en/nebula/$TRAIN#" \
    "$DENEB/configs/xlmr/base/da-estimator-cvpr.yaml" > "$RUN/configs/config.yaml"

cd "$RUN"
export PYTHONPATH=$DENEB
export WANDB_MODE=offline
export WANDB_SILENT=true
export TOKENIZERS_PARALLELISM=false
exec "$ROOT/external/envs/deneb/bin/python" "$ROOT/analysis/metrics/released_cli.py" deneb train -f configs/config.yaml "$@"
