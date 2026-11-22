"""The causal leakage effect by cross-fitting, plus the decontaminated-training comparison.

Every picture has a fold. For a row whose picture is in fold f the "unseen" prediction comes
from the model that held fold f out, and the "seen" predictions from the four models that held
another fold out and therefore trained on this picture. All five models are exchangeable (same
number of folds, pictures drawn the same way), so the difference on the same rows isolates
having trained on the picture itself. Rows whose picture is absent from the training pool are
the placebo: all models are equally ignorant of them, so their difference is the noise floor.

Rows are split further by what exactly the pool contains: the picture only, or the very same
picture-and-candidate pair, which also leaks the human label.

Usage: analyze_effects.py <backbone> <pool> [resamples]
"""
import sys

import numpy as np

from effects import Benchmark, bootstrap, interval
from lib import CACHE, EVAL_SETS, FOLDS, RUNS, load_rows, picture_fold, read_json, row_pictures, write_json

RESAMPLES = 1000


# The feature cache keeps the rows of a released training set that carry no reference captions in
# a second row file.  Metrics trained with their authors' code (Pearl, Rigel) see those rows; the
# ridge and MLP models of this directory are trained on the rows with references only.
REFERENCE_LESS = {'spica_train': 'spica_train_noref'}


def pool_keys(pool, every_row=False):
    """Pictures and (picture, candidate) pairs of a training set.

    every_row adds the reference-less rows of the released file, for a model trained on all of it.
    """
    names = [pool] + ([REFERENCE_LESS[pool]] if every_row and pool in REFERENCE_LESS else [])
    pictures, keys = [], []
    for name in names:
        rows = load_rows(name, CACHE)
        pictures.append(row_pictures(rows))
        keys.append(pictures[-1].astype(np.int64) * 1000000 + rows['candidate'].astype(np.int64))
    return np.unique(np.concatenate(pictures)), np.unique(np.concatenate(keys))


def item_classes(benchmark, pictures, keys):
    """For every item: 0 = picture absent from the pool, 1 = picture present, 2 = pair present."""
    rows = benchmark.rows
    row_keys = row_pictures(rows).astype(np.int64) * 1000000 + rows['candidate'].astype(np.int64)
    exact = np.isin(row_keys, keys)
    if benchmark.pairwise:
        exact = exact[::2] | exact[1::2]
    seen = benchmark.unit_mask(pictures)[np.searchsorted(benchmark.units, benchmark.pictures)]
    return np.where(exact, 2, seen.astype(int))


def crossfit(benchmark, predictions, labels):
    """Per-row unseen prediction and the four seen predictions from the held-out-fold models.

    The model that held out fold f never saw the picture; each of the other four did. Using
    all four as separate "seen" arms (never averaged into an ensemble) spends every model
    equally and removes most of the model-to-model noise from the seen arm.
    """
    folds = picture_fold(row_pictures(benchmark.rows))
    columns = [labels.index(f'heldout{f}') for f in range(FOLDS)]
    index = np.arange(len(folds))
    unseen = predictions[index, [columns[f] for f in folds]].astype(np.float64)
    seen = [predictions[index, [columns[(f + shift) % FOLDS] for f in folds]].astype(np.float64)
            for shift in range(1, FOLDS)]
    return seen, unseen


def effect(benchmark, seen, unseen, selection, resamples, seed=0):
    """Mean score of the seen arms minus the score of the unseen arm, on the same items."""
    arms = [benchmark.measure(column, selection.items) for column in seen]
    point = dict(units=len(selection), items=int(len(selection.items)), seen_arms=arms,
                 seen=float(np.mean(arms)), unseen=benchmark.measure(unseen, selection.items))
    point['difference'] = point['seen'] - point['unseen']
    if len(selection) > 1 and resamples:
        draws = bootstrap(benchmark, seen + [unseen], selection, resamples, seed)
        point['bootstrap_difference'] = interval(draws[:, :-1].mean(1) - draws[:, -1])
        point['bootstrap_seen'] = interval(draws[:, :-1].mean(1))
        point['bootstrap_unseen'] = interval(draws[:, -1])
    return point


def analyze(arguments):
    name, directory, labels, pool, resamples = arguments
    pictures, keys = pool_keys(pool)
    benchmark = Benchmark(name)
    predictions = np.load(directory / 'predictions' / f'{name}.npy')
    classes = item_classes(benchmark, pictures, keys)
    groups = {'clean': classes == 0, 'picture_leaked': classes == 1, 'pair_leaked': classes == 2,
              'any_leaked': classes > 0, 'all': classes >= 0}
    entry = dict(items=benchmark.item_count, units=len(benchmark.units),
                 item_fraction={key: float(mask.mean()) for key, mask in groups.items()})
    if all(f'heldout{f}' in labels for f in range(FOLDS)):
        seen, unseen = crossfit(benchmark, predictions, labels)
        entry['crossfit'] = {key: effect(benchmark, seen, unseen, benchmark.select(mask), resamples)
                             for key, mask in groups.items() if mask.any()}
        leaked, clean = groups['any_leaked'], groups['clean']
        if leaked.any() and clean.any():
            # Difference in differences: the leaked rows' effect beyond the placebo rows' drift.
            first = bootstrap(benchmark, seen + [unseen], benchmark.select(leaked), resamples, 1)
            second = bootstrap(benchmark, seen + [unseen], benchmark.select(clean), resamples, 2)
            entry['leaked_minus_placebo'] = dict(
                difference=entry['crossfit']['any_leaked']['difference'] - entry['crossfit']['clean']['difference'],
                bootstrap=interval((first[:, :-1].mean(1) - first[:, -1]) - (second[:, :-1].mean(1) - second[:, -1])))
    entry['models'] = {}
    for label in labels:
        if label.startswith('heldout'):
            continue
        column = predictions[:, labels.index(label)].astype(np.float64)
        entry['models'][label] = {key: benchmark.measure(column, benchmark.select(mask).items)
                                  for key, mask in groups.items() if mask.any()}
    return name, entry


def finite(value):
    """Undefined statistics (a resample with one distinct score) are JSON null."""
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(backbone='siglip', pool='spica_train', resamples=RESAMPLES):
    from concurrent.futures import ProcessPoolExecutor
    resamples = int(resamples)
    directory = RUNS / 'models' / backbone / pool
    labels = read_json(directory / 'report.json')['labels']
    jobs = [(name, directory, labels, pool, resamples) for name in EVAL_SETS]
    with ProcessPoolExecutor(len(jobs)) as pool_executor:
        out = dict(pool_executor.map(analyze, jobs))
    for name, entry in out.items():
        if 'crossfit' in entry:
            print(name, {k: round(v['difference'], 2) for k, v in entry['crossfit'].items()}, flush=True)
    write_json(directory / 'effects.json', finite(dict(backbone=backbone, pool=pool, resamples=resamples,
                                                       folds=FOLDS, seen_arms=FOLDS - 1, benchmarks=out)))


if __name__ == '__main__':
    main(*sys.argv[1:4])
