"""Score every saved epoch of Pearl trainings, not only the one validation selected.

    epoch_sweep.py <run dir> [<run dir> ...] [-- split ...]

The released trainer keeps one checkpoint per epoch and stops early on the validation
Kendall tau; the per-epoch validation scores of one training differ by less than 0.005,
so which epoch gets reported is close to arbitrary.  This scores all of them on a few
evaluation sets, so the report can separate what a training variant does from what the
choice of epoch does.

Features are loaded once.  Each checkpoint is loaded from inside its run directory, because the
model constructor reads experiments/hadamard.pth relative to the working directory (the file is
the same in every run and is overwritten by the checkpoint's weights).  Training-side text lookup only (scores_raw).  Writes
<run dir>/epoch_scores/<checkpoint stem>/<split>.npz; an existing file is skipped.
Run with the Pearl environment and PYTHONPATH=external/Pearl.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import score_pearl

SPLITS = ['composite', 'flickr8k_expert', 'polaris_test', 'spica_test', 'nebula_test']


def main(arguments):
    runs, splits = arguments, SPLITS
    if '--' in arguments:
        cut = arguments.index('--')
        runs, splits = arguments[:cut], arguments[cut + 1:]
    from pearl.models import load_checkpoint

    jobs = []
    for run in (Path(r).resolve() for r in runs):
        versions = sorted((run / 'experiments/lightning').glob('version_*'), key=lambda p: p.stat().st_mtime)
        if not versions:
            print('no training under', run, flush=True)
            continue
        for checkpoint in sorted(versions[-1].glob('epoch=*.ckpt')):
            out = run / 'epoch_scores' / checkpoint.stem
            if not all((out / f'{split}.npz').exists() for split in splits):
                jobs.append((checkpoint, out))
    print(len(jobs), 'checkpoints to score', flush=True)
    if not jobs:
        return 0

    features = score_pearl.load_features()
    rows = {split: score_pearl.read_csv(score_pearl.DATA / 'csv' / f'{split}.csv') for split in splits}
    samples = {split: score_pearl.build(rows[split], features, False) for split in splits}
    print('features loaded', flush=True)
    for checkpoint, out in jobs:
        started = time.time()
        out.mkdir(parents=True, exist_ok=True)
        os.chdir(checkpoint.parents[3])   # the model constructor reads experiments/hadamard.pth of its run
        model = load_checkpoint(str(checkpoint)).cuda().eval()
        for split in splits:
            if (out / f'{split}.npz').exists():
                continue
            with torch.no_grad():
                _, score, img_score, ref_score = model.predict(samples[split], cuda=True, batch_size=256,
                                                               show_progress=False)
            np.savez(out / f'{split}.npz', score=np.array(score, dtype=np.float32),
                     img_score=np.array(img_score, dtype=np.float32),
                     ref_score=np.array(ref_score, dtype=np.float32))
        del model
        torch.cuda.empty_cache()
        print(checkpoint.parents[3].name, checkpoint.stem, f'{time.time() - started:.0f}s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
