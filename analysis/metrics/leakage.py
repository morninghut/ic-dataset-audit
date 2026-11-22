"""What the duplicated pairs do to a metric trained with its authors' code.

A registered model directory holds three columns that differ only in the training rows:

  released    the pool as released
  pairclean   the pool without the rows whose (picture, candidate) pair is in an evaluation set
  randomdrop  the pool without the same number of rows drawn at random

randomdrop still holds almost all duplicated pairs, so randomdrop - pairclean is the effect
of the duplicated pairs at equal training size, released - randomdrop is the effect of the
extra rows alone, and released - pairclean is what a reader of the released numbers gets.
On the rows that were never duplicated all three contrasts are the placebo.

Every evaluation set is split by what the released pool holds of each row (the very pair,
only the picture, nothing), and all columns are resampled together on the same pictures, so
the differences are paired.  On the duplicated pairs the script also reports how closely each
column reproduces the label the pool carries for that pair.

    leakage.py <model dir under runs/probe/models> <released pool> [resamples]

Run with PYTHONPATH=Polos-revised.  Writes <model dir>/leakage.json and prints the tables.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probe'))
from analyze_effects import finite, item_classes, pool_keys  # noqa: E402
from effects import Benchmark, bootstrap, interval  # noqa: E402
from lib import CACHE, EVAL_SETS, RUNS, load_rows, read_json, row_pictures, write_json  # noqa: E402
from memorization import correlation, leaked_pairs  # noqa: E402

LABELS = ['released', 'pairclean', 'randomdrop']
CONTRASTS = [('released', 'pairclean'), ('released', 'randomdrop'), ('randomdrop', 'pairclean')]
GROUPS = ['all', 'pair', 'picture_only', 'absent']
MINIMUM_ITEMS = 200


def cell(point, draws=None):
    out = dict(value=float(point))
    if draws is not None:
        span = interval(draws)
        out.update(low=span['low'], high=span['high'])
    return out


def scores(benchmark, columns, mask, resamples):
    """Score of every column and every contrast on the items of the mask."""
    selection = benchmark.select(mask)
    point = {label: benchmark.measure(columns[label], selection.items) for label in LABELS}
    draws = None
    if resamples and len(selection) > 1:
        table = bootstrap(benchmark, [columns[label] for label in LABELS], selection, resamples)
        draws = {label: table[:, index] for index, label in enumerate(LABELS)}
    out = dict(items=int(len(selection.items)), pictures=len(selection))
    for label in LABELS:
        out[label] = cell(point[label], draws and draws[label])
    for first, second in CONTRASTS:
        out[f'{first}-{second}'] = cell(point[first] - point[second],
                                        draws and draws[first] - draws[second])
    return out


def memorization(name, pool, columns, resamples):
    """Pearson r between each column and the pool's label of the same pair, on duplicated pairs."""
    index, _, pool_label, counts = leaked_pairs(name, pool)
    if len(index) < MINIMUM_ITEMS:
        return None
    pictures = row_pictures(load_rows(name, CACHE))[index]
    units, inverse = np.unique(pictures, return_inverse=True)
    values = {label: columns[label][index] for label in LABELS}

    def statistics(weights):
        return np.array([correlation(values[label], pool_label, weights) for label in LABELS])

    point = statistics(np.ones(len(index)))
    draws = None
    if resamples:
        rng = np.random.default_rng(0)
        draws = np.empty((resamples, len(LABELS)))
        for r in range(resamples):
            weight = np.bincount(rng.integers(len(units), size=len(units)), minlength=len(units))
            draws[r] = statistics(weight[inverse].astype(np.float64))
    out = dict(pairs=int(len(index)), rows=int(counts.sum()))
    for position, label in enumerate(LABELS):
        out[label] = cell(point[position], None if draws is None else draws[:, position])
    for first, second in CONTRASTS:
        a, b = LABELS.index(first), LABELS.index(second)
        out[f'{first}-{second}'] = cell(point[a] - point[b],
                                        None if draws is None else draws[:, a] - draws[:, b])
    return out


def show(entry, digits=2):
    if entry is None:
        return '-'
    text = f"{entry['value']:+.{digits}f}" if digits == 2 and abs(entry['value']) < 20 else f"{entry['value']:.{digits}f}"
    if 'low' in entry:
        text += f" [{entry['low']:+.{digits}f},{entry['high']:+.{digits}f}]"
    return text


def main(directory, pool, resamples=1000):
    resamples = int(resamples)
    path = RUNS / 'models' / directory
    labels = read_json(path / 'report.json')['labels']
    missing = [label for label in LABELS if label not in labels]
    if missing:
        raise ValueError(f'{directory} has no column {missing}')
    pictures, keys = pool_keys(pool, every_row=True)     # the authors' code trains on every released row
    out = {}
    for name in EVAL_SETS:
        file = path / 'predictions' / f'{name}.npy'
        if not file.exists() or name == pool:
            continue
        predictions = np.load(file)
        columns = {label: predictions[:, labels.index(label)].astype(np.float64) for label in LABELS}
        benchmark = Benchmark(name)
        classes = item_classes(benchmark, pictures, keys)
        masks = dict(all=classes >= 0, pair=classes == 2, picture_only=classes == 1, absent=classes == 0)
        entry = dict(items=benchmark.item_count,
                     fraction={group: float(mask.mean()) for group, mask in masks.items()},
                     groups={group: scores(benchmark, columns, masks[group], resamples)
                             for group in GROUPS if masks[group].sum() >= MINIMUM_ITEMS})
        if not benchmark.pairwise:
            entry['memorization'] = memorization(name, pool, columns, resamples)
        out[name] = entry
        print(f"\n# {name}: {benchmark.item_count} items; pair in pool {entry['fraction']['pair']:.1%}, "
              f"picture only {entry['fraction']['picture_only']:.1%}, absent {entry['fraction']['absent']:.1%}")
        header = LABELS + [f'{a}-{b}' for a, b in CONTRASTS]
        print(f"{'rows':13s} {'items':>6s} " + ' '.join(f'{h:>22s}' for h in header))
        for group, cells in entry['groups'].items():
            print(f"{group:13s} {cells['items']:6d} " + ' '.join(f'{show(cells[h]):>22s}' for h in header))
        if entry.get('memorization'):
            cells = entry['memorization']
            print(f"{'r(pool label)':13s} {cells['pairs']:6d} "
                  + ' '.join(f'{show(cells[h], 3):>22s}' for h in header))
    write_json(path / 'leakage.json', finite(dict(model=directory, pool=pool, resamples=resamples,
                                                  labels=LABELS, benchmarks=out)))


if __name__ == '__main__':
    main(*sys.argv[1:4])
