#!/bin/bash
# Is PEARL_SKIP_IMAGES numerically inert?  Train one epoch on the 8,773-row
# validation CSV three times with the same seed: once decoding every image as
# released (run 0) and twice on the fast path (runs 1 and 2), then compare the
# checkpoints tensor by tensor.  Runs 1 and 2 are the same configuration, so
# their difference is what GPU non-determinism alone produces; the fast path is
# inert when run 0 differs from run 1 by no more than that.
set -eu
ROOT=${CAPAUDIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CAPAUDIT_ROOT=$ROOT
PEARL=$ROOT/external/Pearl
for MODE in 0 1 2; do
  SKIP=1; [ "$MODE" = 0 ] && SKIP=0
  RUN=$ROOT/runs/metrics/pearl/fastpath_check_$MODE
  rm -rf "$RUN"; mkdir -p "$RUN/experiments" "$RUN/configs"
  ln -sfn "$PEARL/data" "$RUN/data"; ln -sfn "$PEARL/features" "$RUN/features"; ln -sfn "$PEARL/pacscore" "$RUN/pacscore"
  ln -sfn "$PEARL/experiments/hadamard.pth" "$RUN/experiments/hadamard.pth"
  sed -e 's#^train_path: .*#train_path: data/spica_val.csv#' -e 's#^max_epochs: .*#max_epochs: 1#' \
      -e 's#^test_path: .*#test_path: data/spica_val.csv#' \
      "$PEARL/configs/variants/released.yaml" > "$RUN/configs/config.yaml"
  (cd "$RUN" && PYTHONPATH=$PEARL PEARL_SKIP_IMAGES=$SKIP WANDB_MODE=offline WANDB_SILENT=true TOKENIZERS_PARALLELISM=false \
     "$ROOT/external/envs/pearl/bin/python" "$ROOT/analysis/metrics/released_cli.py" pearl train -f configs/config.yaml) \
     > "$ROOT/runs/metrics/logs/fastpath_check_$MODE.log" 2>&1
done

"$ROOT/external/envs/pearl/bin/python" - <<'PY'
import glob, torch
paths = [sorted(glob.glob(f'${CAPAUDIT_ROOT:?set CAPAUDIT_ROOT to the repository root}/runs/metrics/pearl/fastpath_check_{m}/experiments/lightning/*/*.ckpt'))[-1] for m in (0, 1, 2)]
states = [torch.load(p, map_location='cpu', weights_only=False)['state_dict'] for p in paths]
assert states[0].keys() == states[1].keys() == states[2].keys()
worst = lambda a, b: max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
print('tensors', len(states[0]))
print('released path vs fast path  : max abs difference', worst(states[0], states[1]))
print('fast path vs fast path again: max abs difference', worst(states[1], states[2]))
PY
