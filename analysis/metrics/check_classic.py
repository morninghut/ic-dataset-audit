"""The classic metrics under each benchmark's published protocol, next to the published numbers.

Published: Wada et al., "Polos" (CVPR 2024), Tables 1-3, whose classic rows repeat the CLIPScore
paper for Composite, Flickr8K and PASCAL-50S and add Polaris and FOIL.  "BLEU" there is BLEU-4.

    check_classic.py [group ...]      groups default to ngram meteor spice
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import sys

import numpy as np

sys.path.insert(0, (_ROOT + '/analysis/probe'))
from effects import Benchmark  # noqa: E402
from lib import EVAL_SETS, ROOT  # noqa: E402

COLUMNS = ['composite', 'flickr8k_expert', 'flickr8k_cf', 'polaris_test', 'pascal50s', 'foil_1ref', 'foil_4ref']
PUBLISHED = {'bleu4': (30.6, 30.8, 16.4, 46.3, 72.7, 66.5, 82.6),
             'rougel': (32.4, 32.3, 19.9, 46.3, 78.1, 71.7, 79.3),
             'cider': (37.7, 43.9, 24.6, 52.1, 79.6, 82.5, 90.6),
             'meteor': (38.9, 41.8, 22.2, 51.2, 80.2, 78.8, 82.6),
             'spice': (40.3, 44.9, 24.4, 51.0, 78.7, 75.5, 86.1)}

groups = sys.argv[1:] or ['ngram', 'meteor', 'spice']
for name in EVAL_SETS:
    bench = Benchmark(name)
    items = np.arange(bench.item_count)
    cells = []
    for group in groups:
        path = ROOT / f'runs/metrics/classic/{group}/{name}.npz'
        if not path.exists():
            continue
        with np.load(path) as f:
            for key in f.files:
                paper = PUBLISHED.get(key, ())
                paper = f' ({paper[COLUMNS.index(name)]:.1f})' if paper and name in COLUMNS else ''
                cells.append(f'{key} {bench.measure(f[key].astype(np.float64), items):6.2f}{paper}')
    print(f'{name:16s} ' + '  '.join(cells))
