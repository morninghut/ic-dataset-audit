"""Kendall tau of every saved epoch of the DENEB trainings on the Polaris test set.

The DENEB counterpart of metrics/epoch_sweep_report.py, for the checkpoints scored by
deneb_epoch_sweep.sh.  Samples are grouped by what the released Nebula training set holds of
them: "pair" = duplicated sample (image and candidate both in it), "not pair" = the others.

    deneb_epoch_report.py        (PYTHONPATH=Polos-revised, from analysis/summary)

Writes runs/summary/deneb_epoch_sweep.json in the layout of runs/metrics/epoch_sweep.json.
A run or checkpoint that has not been scored yet is left out, so it can be run on a partial sweep.
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import sys
from pathlib import Path

import numpy as np

ROOT = Path(_ROOT)
sys.path.insert(0, str(ROOT / 'analysis/probe'))
from analyze_effects import item_classes, pool_keys  # noqa: E402
from effects import Benchmark  # noqa: E402
from lib import write_json  # noqa: E402

SPLIT = 'polaris_test'


def main():
    pictures, keys = pool_keys('nebula_train')
    benchmark = Benchmark(SPLIT)
    classes = item_classes(benchmark, pictures, keys)
    above = benchmark.target > benchmark.target.min()
    groups = {'all': classes >= 0, 'pair': classes == 2, 'not pair': classes != 2,
              'pair above': (classes == 2) & above, 'not pair above': (classes != 2) & above}
    groups = {name: np.flatnonzero(mask) for name, mask in groups.items()}
    out = {}
    for directory in sorted((ROOT / 'runs/metrics/deneb').glob('*_s*/epoch_scores')):
        epochs = {}
        for checkpoint in sorted(directory.glob('epoch=*'), key=lambda p: int(p.name.split('=')[1].split('-')[0])):
            path = checkpoint / f'{SPLIT}.npz'
            if not path.exists():
                continue
            with np.load(path) as f:
                prediction = f['score'].astype(np.float64)
            epoch = int(checkpoint.name.split('=')[1].split('-')[0])
            epochs[epoch] = dict(tau={f'{SPLIT}/{group}': benchmark.measure(prediction, items) for group, items in groups.items()})
        if epochs:
            out[directory.parent.name] = dict(epochs=epochs)
            for epoch, row in epochs.items():
                print(f'{directory.parent.name:16s} {epoch:2d} ' + ' '.join(f'{v:7.2f}' for v in row['tau'].values()), flush=True)
    write_json(ROOT / 'runs/summary/deneb_epoch_sweep.json',
               dict(columns=[f'{SPLIT}/{group}' for group in groups], sizes={g: int(len(i)) for g, i in groups.items()}, runs=out))


if __name__ == '__main__':
    main()
