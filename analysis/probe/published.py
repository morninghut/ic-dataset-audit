"""What the duplicated pairs are worth to the released checkpoint itself.

The released Polos checkpoint trained on the whole Polaris training pool, so every pair that
pool shares with Nebula or Spica is a pair it has seen with its human label. It cannot be
cross-fitted, so it is compared with the fold model that held the picture out: on the rows
whose pair is duplicated and, as a control, on the rows of the same benchmark that are clean.
The difference of those two differences is what the duplicated pairs add to the published
checkpoint beyond its general advantage over our fold models.

Usage: published.py <model dir> <pool> [resamples] [column]
"""
import sys

import numpy as np

from analyze_effects import crossfit, finite, item_classes, pool_keys
from effects import Benchmark, bootstrap, interval
from lib import EVAL_SETS, RUNS, read_json, write_json

RESAMPLES = 1000
MINIMUM = 100


def main(directory='mlp_legacy_cached_clip_s42', pool='polaris_train', resamples=RESAMPLES,
         column='official'):
    resamples = int(resamples)
    path = RUNS / 'models' / directory / pool
    labels = read_json(path / 'report.json')['labels']
    pictures, keys = pool_keys(pool)
    out = {}
    for name in EVAL_SETS:
        benchmark = Benchmark(name)
        predictions = np.load(path / 'predictions' / f'{name}.npy')
        classes = item_classes(benchmark, pictures, keys)
        leaked, clean = classes == 2, classes == 0
        if leaked.sum() < MINIMUM or clean.sum() < MINIMUM:
            continue
        _, unseen = crossfit(benchmark, predictions, labels)
        published = predictions[:, labels.index(column)].astype(np.float64)
        entry = {}
        draws = {}
        for key, mask in (('pair_leaked', leaked), ('clean', clean)):
            selection = benchmark.select(mask)
            cell = dict(items=int(len(selection.items)),
                        published=benchmark.measure(published, selection.items),
                        unseen=benchmark.measure(unseen, selection.items))
            cell['difference'] = cell['published'] - cell['unseen']
            if resamples:
                sample = bootstrap(benchmark, [published, unseen], selection, resamples,
                                   1 if key == 'pair_leaked' else 2)
                draws[key] = sample[:, 0] - sample[:, 1]
                cell['bootstrap_difference'] = interval(draws[key])
            entry[key] = cell
        entry['leaked_minus_clean'] = dict(
            difference=entry['pair_leaked']['difference'] - entry['clean']['difference'],
            bootstrap=interval(draws['pair_leaked'] - draws['clean']) if draws else None)
        out[name] = entry
        print(f"\n# {name}: {entry['pair_leaked']['items']} duplicated items, "
              f"{entry['clean']['items']} clean items")
        for key in ('pair_leaked', 'clean'):
            cell = entry[key]
            span = cell.get('bootstrap_difference')
            print(f"  {key:12s} {column} {cell['published']:6.2f}  unseen fold model {cell['unseen']:6.2f}"
                  f"  difference {cell['difference']:+6.2f}"
                  + (f" [{span['low']:+.2f},{span['high']:+.2f}]" if span else ''))
        span = entry['leaked_minus_clean']['bootstrap']
        print(f"  {'difference in differences':38s} {entry['leaked_minus_clean']['difference']:+6.2f}"
              + (f" [{span['low']:+.2f},{span['high']:+.2f}]" if span else ''))
    write_json(path / f'published_{column}.json',
               finite(dict(model=directory, pool=pool, column=column, resamples=resamples,
                           benchmarks=out)))


if __name__ == '__main__':
    main(*sys.argv[1:5])
