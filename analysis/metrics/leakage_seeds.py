"""The duplicated-pair effect of a retrained metric across seeds, against rows it cannot have memorized.

    leakage_seeds.py <family> <released pool> <seed> [seed ...] [--resamples N] [--suffix S] [--tag T]

reads the registered models external/<family>_s<seed><suffix> (columns released, pairclean,
randomdrop; see leakage.py for what the three contrasts mean) and prints, per evaluation set
and per row group (all, pair, picture_only, absent), every contrast for each seed and its mean
over the seeds.  The interval of the mean resamples pictures and measures all seeds on the
same resample; it covers the evaluation sample, not the variation between trainings, which is
what the per-seed columns are for.

Where a set has both duplicated pairs and pictures the pool never held, the last line is the
difference between the two groups.  The groups share no picture, so they are resampled
independently.  This difference is the part of a contrast that only the rows whose pair was in
the pool show; whatever makes one training better than another everywhere cancels in it.

When every seed also has a matcheddrop column (prep_matcheddrop.py: the same number of rows
removed, of the kind pairclean removes, none of them in an evaluation set), two contrasts are
added: matcheddrop-pairclean, what the duplicated pairs themselves add over losing equally
many rows of their kind, and released-matcheddrop, what losing such rows costs.

Run with PYTHONPATH=Polos-revised.  Writes runs/metrics/leakage_<family><suffix><tag>.json.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probe'))
from analyze_effects import finite, item_classes, pool_keys  # noqa: E402
from effects import Benchmark, bootstrap, interval  # noqa: E402
from lib import EVAL_SETS, ROOT, RUNS, read_json, write_json  # noqa: E402

LABELS = ['released', 'pairclean', 'randomdrop']
CONTRASTS = [('released', 'pairclean'), ('randomdrop', 'pairclean'), ('released', 'randomdrop')]
MATCHED = 'matcheddrop'
MATCHED_CONTRASTS = [('matcheddrop', 'pairclean'), ('released', 'matcheddrop')]
GROUPS = ['all', 'pair', 'picture_only', 'absent']
MINIMUM_ITEMS = 200


def contrasts(table, seeds, labels, pairs):
    """(resamples or 1, seeds x labels) -> {contrast: (resamples or 1, seeds)}."""
    table = table.reshape(len(table), seeds, len(labels))
    return {f'{a}-{b}': table[:, :, labels.index(a)] - table[:, :, labels.index(b)] for a, b in pairs}


def summary(point, draws):
    """Per-seed values, their mean, and the interval of the mean."""
    span = interval(draws.mean(1))
    return dict(seeds=point[0].tolist(), mean=float(point[0].mean()), low=span['low'], high=span['high'])


def show(cell):
    return (' '.join(f'{v:+.2f}' for v in cell['seeds'])
            + f" | {cell['mean']:+.2f} [{cell['low']:+.2f},{cell['high']:+.2f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('family')
    parser.add_argument('pool')
    parser.add_argument('seeds', nargs='+')
    parser.add_argument('--resamples', type=int, default=1000)
    parser.add_argument('--suffix', default='')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()

    paths = [RUNS / 'models/external' / f'{args.family}_s{seed}{args.suffix}' for seed in args.seeds]
    reports = [read_json(path / 'report.json') for path in paths]
    pictures, keys = pool_keys(args.pool, every_row=True)   # Pearl and DENEB train on every released row
    matched = all(MATCHED in report['labels'] for report in reports)
    labels = LABELS + [MATCHED] * matched
    pairs = CONTRASTS + MATCHED_CONTRASTS * matched
    names = [f'{a}-{b}' for a, b in pairs]
    out = {}
    for name in EVAL_SETS:
        if name == args.pool or not all((path / 'predictions' / f'{name}.npy').exists() for path in paths):
            continue
        columns = []
        for path, report in zip(paths, reports):
            predictions = np.load(path / 'predictions' / f'{name}.npy').astype(np.float64)
            columns += [predictions[:, report['labels'].index(label)] for label in labels]
        benchmark = Benchmark(name)
        classes = item_classes(benchmark, pictures, keys)
        masks = dict(all=classes >= 0, pair=classes == 2, picture_only=classes == 1, absent=classes == 0)
        print(f"\n# {name}: {benchmark.item_count} items; pair in pool {masks['pair'].mean():.1%}, "
              f"picture only {masks['picture_only'].mean():.1%}, absent {masks['absent'].mean():.1%}")
        print(f"{'rows':13s} {'items':>6s} " + ' '.join(f'{n + " (seeds | mean)":>42s}' for n in names))
        entry, kept = {}, {}
        for position, group in enumerate(GROUPS):
            if masks[group].sum() < MINIMUM_ITEMS:
                continue
            selection = benchmark.select(masks[group])
            point = contrasts(np.array([[benchmark.measure(c, selection.items) for c in columns]]), len(paths), labels, pairs)
            draws = contrasts(bootstrap(benchmark, columns, selection, args.resamples, seed=position),
                              len(paths), labels, pairs)
            kept[group] = (point, draws)
            entry[group] = dict(items=int(len(selection.items)),
                                **{n: summary(point[n], draws[n]) for n in names})
            print(f"{group:13s} {entry[group]['items']:6d} " + ' '.join(f'{show(entry[group][n]):>42s}' for n in names))
        if 'pair' in kept and 'absent' in kept:
            entry['pair-absent'] = {n: summary(kept['pair'][0][n] - kept['absent'][0][n],
                                               kept['pair'][1][n] - kept['absent'][1][n]) for n in names}
            print(f"{'pair - absent':13s} {'':6s} " + ' '.join(f"{show(entry['pair-absent'][n]):>42s}" for n in names))
        out[name] = entry
    write_json(ROOT / f'runs/metrics/leakage_{args.family}{args.suffix}{args.tag}.json',
               finite(dict(family=args.family, pool=args.pool, seeds=args.seeds, suffix=args.suffix,
                           resamples=args.resamples, contrasts=names, benchmarks=out)))


if __name__ == '__main__':
    main()
