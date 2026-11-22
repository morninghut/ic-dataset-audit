"""The released Rigel checkpoint under each benchmark's published protocol, next to the paper.

    check_rigel.py

Reads the registered external/rigel (columns free, refs) and prints 100 x Kendall tau-c
(tau-b on Flickr8k-CF) per evaluation set, with the numbers of Table 1 of the paper where it
reports the set.  Run with PYTHONPATH=Polos-revised.
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import sys

import numpy as np

sys.path.insert(0, (_ROOT + '/analysis/probe'))
from effects import Benchmark  # noqa: E402
from lib import EVAL_SETS, RUNS, read_json  # noqa: E402

PUBLISHED = {'refs': dict(flickr8k_expert=59.0, flickr8k_cf=40.4, nebula_test=57.5, composite=66.1),
             'free': dict(flickr8k_expert=59.9, flickr8k_cf=40.4, nebula_test=55.1, composite=64.3)}

path = RUNS / 'models/external/rigel'
report = read_json(path / 'report.json')
print(f"{'set':16s} " + ' '.join(f'{label:>8s} {"paper":>6s}' for label in report['labels']))
for split in EVAL_SETS:
    if split not in report['splits']:
        continue
    bench = Benchmark(split)
    predictions = np.load(path / 'predictions' / f'{split}.npy').astype(np.float64)
    items = np.arange(bench.item_count)
    cells = []
    for index, label in enumerate(report['labels']):
        paper = PUBLISHED.get(label, {}).get(split)
        cells.append(f'{bench.measure(predictions[:, index], items):8.2f} ' + (f'{paper:6.1f}' if paper else f'{"-":>6s}'))
    print(f'{split:16s} ' + ' '.join(cells))
