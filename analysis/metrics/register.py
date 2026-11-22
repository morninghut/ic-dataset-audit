"""Put externally scored metrics where the analysis tools look for predictions.

    register.py <name> <label>=<dir>[:<npz key>] ...

writes runs/probe/models/external/<name>/{report.json,predictions/<split>.npy}, one
column per label, taken from <dir>/<split>.npz (key defaults to "score").  The rows of the
npz files are in pearl_data/csv order, which is the order of the feature cache, so the
only check needed is the row count.  A split missing for any label is skipped.
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, (_ROOT + '/analysis/probe'))
from lib import CACHE, EVAL_SETS, RUNS, load_rows, write_json  # noqa: E402


def main(name, *columns):
    labels, sources = [], []
    for column in columns:
        label, _, source = column.partition('=')
        directory, _, key = source.partition(':')
        labels.append(label)
        sources.append((Path(directory), key or 'score'))
    out = RUNS / 'models/external' / name
    (out / 'predictions').mkdir(parents=True, exist_ok=True)
    written = []
    for split in EVAL_SETS:
        if not all((directory / f'{split}.npz').exists() for directory, _ in sources):
            continue
        rows = len(load_rows(split, CACHE)['target'])
        table = np.empty((rows, len(labels)), np.float32)
        for index, (directory, key) in enumerate(sources):
            with np.load(directory / f'{split}.npz') as f:
                values = f[key]
            if len(values) != rows:
                raise ValueError(f'{split}: {directory} has {len(values)} rows, the cache has {rows}')
            if not np.isfinite(values).all():
                raise ValueError(f'{split}: {directory}:{key} has non-finite scores')
            table[:, index] = values
        np.save(out / 'predictions' / f'{split}.npy', table)
        written.append(split)
    write_json(out / 'report.json', dict(labels=labels, sources=[f'{d}:{k}' for d, k in sources],
                                         splits=written))
    print(name, labels, written)


if __name__ == '__main__':
    main(*sys.argv[1:])
