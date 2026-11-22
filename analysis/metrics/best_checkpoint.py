"""Print the checkpoint the released training loop considers best.

    best_checkpoint.py <run dir>

Both released trainers keep every epoch (save_top_k 10) and stop early on the
validation Kendall tau.  The newest checkpoint of the newest version records
which of them scored best; that path is printed, followed on stderr by the score
of every epoch so the choice can be audited.
"""
import sys
from pathlib import Path

import torch


def main(run):
    versions = sorted((Path(run) / 'experiments/lightning').glob('version_*'), key=lambda p: p.stat().st_mtime)
    checkpoints = sorted(versions[-1].glob('*.ckpt'), key=lambda p: p.stat().st_mtime)
    best, score = None, None
    for path in checkpoints:
        state = torch.load(path, map_location='cpu')['callbacks']
        for key, value in state.items():
            if 'ModelCheckpoint' in str(key):
                print(f"{path.name}: monitored {float(value['current_score']):.4f}, "
                      f"best so far {float(value['best_model_score']):.4f}", file=sys.stderr)
                best, score = value['best_model_path'], float(value['best_model_score'])
    print(best)


if __name__ == '__main__':
    main(sys.argv[1])
