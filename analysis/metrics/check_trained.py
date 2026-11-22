"""A retrained DENEB or Pearl under each benchmark's published protocol, next to the paper.

    check_trained.py <deneb|pearl> <name under runs/probe/models/external>

Prints one line per evaluation set with every registered column (released, pairclean,
randomdrop), measured as effects.Benchmark does (100 x Kendall tau-c, tau-b on Flickr8k-CF,
accuracy on PASCAL-50S and FOIL).  The published numbers are the ones each paper prints for
its released model; neither paper reports Polaris test, and Pearl's Nebula number here is
its tau-c (it also prints tau-b 58.4).
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import sys

import numpy as np

sys.path.insert(0, (_ROOT + '/analysis/probe'))
from effects import Benchmark  # noqa: E402
from lib import EVAL_SETS, RUNS, read_json  # noqa: E402

PUBLISHED = {'deneb': dict(composite=58.2, flickr8k_expert=56.8, flickr8k_cf=38.3, nebula_test=54.3,
                           pascal50s=87.8, foil_1ref=95.4, foil_4ref=96.5),
             'pearl': dict(composite=60.4, flickr8k_expert=58.6, flickr8k_cf=38.6, nebula_test=55.4)}

family, name = sys.argv[1:3]
path = RUNS / 'models/external' / name
report = read_json(path / 'report.json')
print(f"{'set':16s} " + ' '.join(f'{label:>11s}' for label in report['labels']) + '   paper (released pool)')
for split in EVAL_SETS:
    if split not in report['splits']:
        continue
    bench = Benchmark(split)
    predictions = np.load(path / 'predictions' / f'{split}.npy').astype(np.float64)
    items = np.arange(bench.item_count)
    values = [bench.measure(predictions[:, i], items) for i in range(predictions.shape[1])]
    paper = PUBLISHED[family].get(split)
    print(f'{split:16s} ' + ' '.join(f'{v:11.2f}' for v in values) + (f'   {paper:.1f}' if paper else ''))
