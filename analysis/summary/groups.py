"""Sample groups by what the training set holds of them: descriptive statistics, label agreement
of the duplicated pairs, and the retraining and cross-fit contrasts with the lowest-score
samples removed.

For an evaluation set and a training set every evaluation row is a duplicated sample (image and
candidate both in the training set), an image-only overlap (image in the training set with other
candidates) or an unseen image.  The three groups differ in composition, and Kendall's tau of a
group that consists mostly of lowest-score rows barely reacts to anything, so every contrast is
also computed on the rows above the lowest score.

    groups.py [resamples]     (PYTHONPATH=Polos-revised, from analysis/summary)

Writes runs/summary/groups.json.
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(_ROOT)
sys.path.insert(0, str(ROOT / 'analysis/probe'))
from analyze_effects import REFERENCE_LESS, crossfit, finite, item_classes, pool_keys  # noqa: E402
from effects import Benchmark, bootstrap, interval  # noqa: E402
from lib import CACHE, RUNS, load_rows, read_json, row_pictures, sample_structure  # noqa: E402

OUT = ROOT / 'runs/summary'
GROUPS = {'duplicated': 2, 'image_only': 1, 'unseen': 0}
PAIRS = [('spica_train', 'polaris_test'), ('nebula_train', 'polaris_test'),
         ('polaris_train', 'spica_test'), ('polaris_train', 'nebula_test')]
RETRAIN = {'pearl': ('spica_train', ['released', 'pairclean', 'randomdrop', 'matcheddrop'],
                     [('released', 'pairclean'), ('matcheddrop', 'pairclean'), ('randomdrop', 'pairclean')]),
           'deneb': ('nebula_train', ['released', 'pairclean', 'randomdrop'],
                     [('released', 'pairclean'), ('randomdrop', 'pairclean')])}
SEEDS = [12, 13, 14]
CROSSFIT = {'ridge_siglip': ['siglip'], 'ridge_clip': ['clip'],
            'mlp_siglip': [f'mlp_siglip_s{s}' for s in (42, 43, 44)],
            'polos_arch': [f'mlp_legacy_cached_clip_s{s}' for s in (42, 43, 44)]}
MINIMUM = 300


def masks_for(benchmark, pool, every_row=True):
    """Groups relative to the released training set; every_row=False for the models fitted on its rows with references."""
    pictures, keys = pool_keys(pool, every_row=every_row)
    classes = item_classes(benchmark, pictures, keys)
    above = benchmark.target > benchmark.target.min()
    out = {}
    for group, code in GROUPS.items():
        out[group, 'all'] = classes == code
        out[group, 'above'] = (classes == code) & above
    out['all', 'all'] = classes >= 0
    out['all', 'above'] = above
    return out


def describe(benchmark, mask):
    target = benchmark.target[mask]
    size = sample_structure(benchmark.rows)['size'][mask]
    _, counts = np.unique(target, return_counts=True)
    n = len(target)
    return dict(rows=int(n), images=len(benchmark.select(mask)),
                lowest_share=float((target == benchmark.target.min()).mean()),
                single_rating_share=float((size == 1).mean()),
                score_mean=float(target.mean()), score_variance=float(target.var()),
                tied_pair_share=float((counts * (counts - 1)).sum() / (n * (n - 1))))


def label_agreement(pool, name):
    """Scores of the duplicated pairs in the training set against those in the evaluation set.

    Both sides are averaged per image-caption pair first: one side may store every annotator's
    rating and the other their mean.  The training set is taken with every released row, also
    those without references.
    """
    def table(splits):
        parts = [load_rows(split, CACHE) for split in splits]
        rows = {key: np.concatenate([part[key] for part in parts]) for key in ('image', 'candidate', 'target')}
        keys = row_pictures(rows).astype(np.int64) * 1000000 + rows['candidate'].astype(np.int64)
        unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        mean = np.bincount(inverse.ravel(), weights=rows['target'].astype(np.float64)) / counts
        return rows, keys, unique, mean, counts
    rows, keys, unique, mean, counts = table([name])
    _, _, pool_unique, pool_mean, pool_counts = table([pool] + ([REFERENCE_LESS[pool]] if pool in REFERENCE_LESS else []))
    _, a, b = np.intersect1d(unique, pool_unique, return_indices=True)
    difference = np.abs(mean[a] - pool_mean[b])
    hit = np.isin(keys, pool_unique)
    row_same = np.abs(rows['target'][hit] - pool_mean[np.searchsorted(pool_unique, keys[hit])]) < 1e-6
    return dict(pairs=int(len(unique)), duplicated_pairs=int(len(a)), duplicated_rows=int(hit.sum()), rows=int(len(keys)),
                same_mean_share=float((difference < 1e-6).mean()), correlation=float(np.corrcoef(mean[a], pool_mean[b])[0, 1]),
                mean_absolute_difference=float(difference.mean()),
                ratings_per_pair_evaluation=float(counts[a].mean()), ratings_per_pair_training=float(pool_counts[b].mean()),
                single_rating_equals_training_mean=float(row_same.mean()))


def retrain(family, name, resamples):
    pool, labels, pairs = RETRAIN[family]
    paths = [RUNS / 'models/external' / f'{family}_s{seed}' for seed in SEEDS]
    reports = [read_json(path / 'report.json') for path in paths]
    columns = []
    for path, report in zip(paths, reports):
        predictions = np.load(path / 'predictions' / f'{name}.npy').astype(np.float64)
        columns += [predictions[:, report['labels'].index(label)] for label in labels]
    benchmark = Benchmark(name)
    masks = masks_for(benchmark, pool)
    out, kept = {}, {}
    for position, (key, mask) in enumerate(masks.items()):
        if mask.sum() < MINIMUM:
            continue
        selection = benchmark.select(mask)
        point = np.array([benchmark.measure(c, selection.items) for c in columns]).reshape(len(SEEDS), len(labels))
        draws = bootstrap(benchmark, columns, selection, resamples, seed=position).reshape(resamples, len(SEEDS), len(labels))
        entry = dict(rows=int(mask.sum()), level={label: float(point[:, i].mean()) for i, label in enumerate(labels)})
        for a, b in pairs:
            ia, ib = labels.index(a), labels.index(b)
            span = interval((draws[:, :, ia] - draws[:, :, ib]).mean(1))
            entry[f'{a}-{b}'] = dict(mean=float((point[:, ia] - point[:, ib]).mean()), low=span['low'], high=span['high'],
                                     seeds=(point[:, ia] - point[:, ib]).tolist())
        kept[key] = (point, draws)
        out['/'.join(key)] = entry
    for subset in ('all', 'above'):
        for other in ('image_only', 'unseen'):
            if ('duplicated', subset) in kept and (other, subset) in kept:
                (p1, d1), (p2, d2) = kept['duplicated', subset], kept[other, subset]
                entry = {}
                for a, b in pairs:
                    ia, ib = labels.index(a), labels.index(b)
                    value = (p1[:, ia] - p1[:, ib]).mean() - (p2[:, ia] - p2[:, ib]).mean()
                    span = interval((d1[:, :, ia] - d1[:, :, ib]).mean(1) - (d2[:, :, ia] - d2[:, :, ib]).mean(1))
                    entry[f'{a}-{b}'] = dict(mean=float(value), low=span['low'], high=span['high'])
                out[f'duplicated-{other}/{subset}'] = entry
    return out


def crossfitted(directories, pool, name, resamples):
    """Seen minus unseen, each arm one model per row; mean over the four seen arms and over seeds."""
    benchmark = Benchmark(name)
    masks = masks_for(benchmark, pool, every_row=False)
    out = {}
    for position, (key, mask) in enumerate(masks.items()):
        if mask.sum() < MINIMUM:
            continue
        selection = benchmark.select(mask)
        cells = []
        for directory in directories:
            path = RUNS / 'models' / directory / pool
            if not (path / 'predictions' / f'{name}.npy').exists():
                continue
            labels = read_json(path / 'report.json')['labels']
            seen, unseen = crossfit(benchmark, np.load(path / 'predictions' / f'{name}.npy'), labels)
            arms = [benchmark.measure(column, selection.items) for column in seen]
            base = benchmark.measure(unseen, selection.items)
            draws = bootstrap(benchmark, seen + [unseen], selection, resamples, seed=position)
            cells.append(dict(arms=[a - base for a in arms], unseen=base, seen=float(np.mean(arms)),
                              draws=draws[:, :-1].mean(1) - draws[:, -1]))
        if not cells:
            return None
        span = interval(np.mean([c['draws'] for c in cells], 0))
        arms = np.array([c['arms'] for c in cells])
        out['/'.join(key)] = dict(rows=int(mask.sum()), unseen=float(np.mean([c['unseen'] for c in cells])),
                                  seen=float(np.mean([c['seen'] for c in cells])), effect=float(arms.mean()),
                                  low=span['low'], high=span['high'], arm_min=float(arms.min()), arm_max=float(arms.max()),
                                  arms_by_model=arms.tolist(), models=len(cells))
    return out


def labels():
    out = {}
    for pool, name in PAIRS + [('spica_train', 'polaris_val'), ('nebula_train', 'polaris_val'), ('spica_train', 'nebula_valid'),
                               ('spica_train', 'spica_test'), ('polaris_train', 'spica_val'), ('nebula_train', 'spica_val')]:
        out[f'{pool}>{name}'] = label_agreement(pool, name)
        print(pool, name, out[f'{pool}>{name}'], flush=True)
    return out


def main(resamples=1000):
    resamples = int(resamples)
    out = dict(resamples=resamples, description={}, labels={}, retrain={}, crossfit={})
    for pool, name in PAIRS:
        benchmark = Benchmark(name)
        masks = masks_for(benchmark, pool)
        out['description'][f'{pool}>{name}'] = {'/'.join(key): describe(benchmark, mask) for key, mask in masks.items() if mask.sum() > 1}
    out['labels'] = labels()
    for family in RETRAIN:
        out['retrain'][family] = retrain(family, 'polaris_test', resamples)
        for key, entry in out['retrain'][family].items():
            print(family, key, {k: (round(v['mean'], 2), round(v['low'], 2), round(v['high'], 2)) for k, v in entry.items() if isinstance(v, dict) and 'mean' in v}, flush=True)
    for title, directories in CROSSFIT.items():
        for pool, name in PAIRS:
            entry = crossfitted(directories, pool, name, resamples)
            if entry:
                out['crossfit'][f'{title}|{pool}>{name}'] = entry
                print(title, pool, name, {k: (v['rows'], round(v['effect'], 2), round(v['low'], 2), round(v['high'], 2), round(v['arm_min'], 2), round(v['arm_max'], 2)) for k, v in entry.items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'groups.json').write_text(json.dumps(finite(out), indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:2])
