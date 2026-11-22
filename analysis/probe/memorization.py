"""How closely does a model reproduce the human label of a pair it was trained on?

On the rows of an evaluation set whose (picture, candidate) pair also sits in the training
pool, the pool carries a label for the same pair. For Polaris, Nebula and Spica these labels
are the same annotations (pair-level correlation 1.000 on three of four sets), so the question
reduces to how much better a model fits the label of a pair when that pair was in its training
data.

Every pair is scored by the fold model that held its picture out (unseen), by the four fold
models that trained on it (seen) and by every other column of the run, which is the model
trained on the whole pool (released) and, for the authors' configuration, the released Polos
checkpoint (official). The fold models are exchangeable, so seen minus unseen on the same
pairs is the effect of having trained on the pair; for a column that is not one of the fold
models the same difference mixes that effect with the difference between the models, which
the clean rows of analyze_effects.py measure separately.

Usage: memorization.py <model dir> <pool> [resamples]   (model dir is <name>/<pool>)
"""
import sys

import numpy as np

from analyze_effects import finite, pool_keys
from lib import CACHE, EVAL_SETS, FOLDS, RUNS, load_rows, picture_fold, read_json, row_pictures, write_json

RESAMPLES = 1000
MINIMUM_ROWS = 200


def pool_labels(pool):
    """Mean human score of every (picture, candidate) pair of the training pool."""
    rows = load_rows(pool, CACHE)
    keys = row_pictures(rows).astype(np.int64) * 1000000 + rows['candidate'].astype(np.int64)
    order = np.argsort(keys, kind='stable')
    unique, starts = np.unique(keys[order], return_index=True)
    sums = np.add.reduceat(rows['target'].astype(np.float64)[order], starts)
    counts = np.diff(np.append(starts, len(order)))
    return unique, sums / counts


def leaked_pairs(name, pool):
    """One record per duplicated pair: a representative row, the evaluation label, the pool label.

    The evaluation label is the mean over that pair's rows, because a model scores the pair
    once and the per-annotator rows differ only by annotator noise.
    """
    rows = load_rows(name, CACHE)
    keys = row_pictures(rows).astype(np.int64) * 1000000 + rows['candidate'].astype(np.int64)
    unique, mean = pool_labels(pool)
    position = np.searchsorted(unique, keys)
    hit = (position < len(unique)) & (unique[np.minimum(position, len(unique) - 1)] == keys)
    index = np.flatnonzero(hit)
    pairs, inverse = np.unique(keys[index], return_inverse=True)
    first = np.zeros(len(pairs), np.int64)
    first[inverse[::-1]] = index[::-1]
    counts = np.bincount(inverse)
    label = np.bincount(inverse, weights=rows['target'][index].astype(np.float64)) / counts
    return first, label, mean[position[first]], counts


def correlation(prediction, label, weights):
    mean_p = np.average(prediction, weights=weights)
    mean_l = np.average(label, weights=weights)
    cov = np.average((prediction - mean_p) * (label - mean_l), weights=weights)
    return cov / np.sqrt(np.average((prediction - mean_p) ** 2, weights=weights)
                         * np.average((label - mean_l) ** 2, weights=weights))


def arms(predictions, labels, index, pictures):
    """Per-pair unseen prediction, the four seen predictions and every other column."""
    folds = picture_fold(pictures)
    columns = [labels.index(f'heldout{f}') for f in range(FOLDS)]
    unseen = predictions[index, [columns[f] for f in folds]].astype(np.float64)
    seen = [predictions[index, [columns[(f + shift) % FOLDS] for f in folds]].astype(np.float64)
            for shift in range(1, FOLDS)]
    other = {label: predictions[index, position].astype(np.float64)
             for position, label in enumerate(labels) if not label.startswith('heldout')}
    return unseen, seen, other


def analyze(unseen, seen, other, pool_label, pictures, resamples):
    units, inverse = np.unique(pictures, return_inverse=True)
    extra = list(other)

    def statistics(weights):
        u = correlation(unseen, pool_label, weights)
        s = float(np.mean([correlation(arm, pool_label, weights) for arm in seen]))
        rest = [correlation(other[label], pool_label, weights) for label in extra]
        return np.array([u, s] + rest + [s - u] + [value - u for value in rest])

    names = (['unseen', 'seen'] + extra + ['seen_minus_unseen']
             + [f'{label}_minus_unseen' for label in extra])
    point = statistics(np.ones(len(pool_label)))
    out = {name: dict(value=float(value)) for name, value in zip(names, point)}
    if resamples:
        rng = np.random.default_rng(0)
        draws = np.empty((resamples, len(names)))
        for r in range(resamples):
            counts = np.bincount(rng.integers(len(units), size=len(units)), minlength=len(units))
            draws[r] = statistics(counts[inverse].astype(np.float64))
        for position, name in enumerate(names):
            out[name]['low'] = float(np.percentile(draws[:, position], 2.5))
            out[name]['high'] = float(np.percentile(draws[:, position], 97.5))
    return out


def main(directory='mlp_legacy_cached_clip_s42', pool='polaris_train', resamples=RESAMPLES):
    resamples = int(resamples)
    path = RUNS / 'models' / directory / pool
    labels = read_json(path / 'report.json')['labels']
    out = {}
    for name in EVAL_SETS:
        if name in ('pascal50s', 'foil_1ref', 'foil_4ref') or name == pool:
            continue
        index, y, m, counts = leaked_pairs(name, pool)
        if len(index) < MINIMUM_ROWS:
            continue
        rows = load_rows(name, CACHE)
        pictures = row_pictures(rows)[index]
        predictions = np.load(path / 'predictions' / f'{name}.npy')
        unseen, seen, other = arms(predictions, labels, index, pictures)
        entry = analyze(unseen, seen, other, m, pictures, resamples)
        entry['pairs'] = int(len(index))
        entry['rows'] = int(counts.sum())
        entry['correlation_pool_label'] = float(np.corrcoef(y, m)[0, 1])
        out[name] = entry
        print(f"\n# {name}: {len(index)} duplicated pairs ({entry['rows']} rows), "
              f"corr(evaluation label, pool label) {entry['correlation_pool_label']:.3f}")
        for key in [k for k in entry if isinstance(entry[k], dict)]:
            cell = entry[key]
            span = f" [{cell['low']:+.3f},{cell['high']:+.3f}]" if 'low' in cell else ''
            print(f"  r(score, pool label) {key:22s} {cell['value']:+.3f}{span}")
    write_json(path / 'memorization.json', finite(dict(model=directory, pool=pool, resamples=resamples,
                                                       benchmarks=out)))


if __name__ == '__main__':
    main(*sys.argv[1:4])
