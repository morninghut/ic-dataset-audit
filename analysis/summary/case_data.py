"""Per image-caption pair of an evaluation set: texts, individual ratings and the score of every
metric as a percentile of that set, for the case studies.

    from case_data import load;  table = load('polaris_test')

A learned metric is the mean of its three seeds.  "Pearl +dup" style names are the models whose
training set holds duplicated samples of the set (metrics/protocol.py).
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import sys
from pathlib import Path

import numpy as np

ROOT = Path(_ROOT)
sys.path.insert(0, str(ROOT / 'analysis/probe'))
sys.path.insert(0, str(ROOT / 'analysis/metrics'))
import validity  # noqa: E402
from analyze_effects import item_classes, pool_keys  # noqa: E402
from effects import Benchmark  # noqa: E402
from lib import CACHE, read_json, row_pictures  # noqa: E402
from pairwise import gather  # noqa: E402

SEEDS = [12, 13, 14]
_catalog = None


def catalog():
    global _catalog
    if _catalog is None:
        _catalog = read_json(CACHE / 'catalog.json')
    return _catalog


def load(name, pool=None):
    names, clean, specs, members, _ = gather(name, SEEDS)
    benchmark = Benchmark(name)
    rows = benchmark.rows
    columns = np.stack([column for _, column in validity.columns(specs, name)], 1)
    scores = np.stack([columns[:, member].mean(1) for member in members], 1)
    pictures = row_pictures(rows)
    keys = pictures.astype(np.int64) * 1000000 + rows['candidate'].astype(np.int64)
    unique, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    ratings = [[] for _ in unique]
    for position, value in zip(inverse, rows['target']):
        ratings[position].append(float(value))
    mean = np.array([np.mean(r) for r in ratings])
    above = mean > rows['target'].min()
    # Percentile of a score among the pairs rated above the lowest score: the lowest-score pairs
    # are most of some sets and would compress everything else into the top of the scale.
    reference = np.sort(scores[first][above], axis=0)
    percentile = np.stack([100 * np.searchsorted(reference[:, m], scores[first][:, m], side='right') / above.sum()
                           for m in range(scores.shape[1])], 1)
    cat = catalog()
    table = dict(name=name, metrics=names, ranked=clean, pair_rows=first, picture=pictures[first],
                 image=[cat['images'][i] for i in rows['image'][first]],
                 candidate=[cat['texts'][i] for i in rows['candidate'][first]],
                 references=[[cat['texts'][j] for j in rows['references'][i][:rows['counts'][i]]] for i in first],
                 ratings=ratings, mean=np.array([np.mean(r) for r in ratings]), count=np.array([len(r) for r in ratings]),
                 score=scores[first], percentile=percentile, low=float(rows['target'].min()), high=float(rows['target'].max()))
    if pool:
        pool_pictures, pool_pairs = pool_keys(pool, every_row=True)
        table['group'] = item_classes(benchmark, pool_pictures, pool_pairs)[first]   # 2 duplicated, 1 image only, 0 unseen
    return table
