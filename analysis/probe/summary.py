"""Collect every effects.json into the tables the report uses.

Usage: summary.py [effect|variants|all]
effect   : the cross-fit contamination effect per training pool, model and benchmark,
           averaged over the seeds of the same configuration.
variants : the decontamination experiment, released vs pairclean vs the size-matched control.
"""
import re
import sys
from collections import defaultdict

import numpy as np

from lib import RUNS, read_json

CLASSES = ['clean', 'picture_leaked', 'pair_leaked', 'all']
SEED = re.compile(r'_s(\d+)(_variants)?$')


def model_dirs(suffix='effects.json'):
    for path in sorted((RUNS / 'models').glob(f'*/*/{suffix}')):
        yield path.parent.parent.name, path.parent.name, path


def configuration(name):
    """Directory name without its seed, and the seed."""
    match = SEED.search(name)
    return (name[:match.start()] + (match.group(2) or ''), int(match.group(1))) if match else (name, None)


def effect_table():
    runs = defaultdict(list)
    for name, pool, path in model_dirs():
        data = read_json(path)
        if not any('crossfit' in entry for entry in data['benchmarks'].values()):
            continue
        config, seed = configuration(name)
        runs[(config, pool)].append((seed, data))
    for (config, pool), group in sorted(runs.items()):
        seeds = [seed for seed, _ in group if seed is not None]
        print(f"\n# cross-fit effect  model {config}  pool {pool}  "
              f"{len(group)} run(s){' seeds ' + ','.join(map(str, seeds)) if seeds else ''}")
        print(f"{'benchmark':16s} {'class':15s} {'items':>7s} {'share':>6s} {'unseen':>7s} "
              f"{'seen':>7s} {'effect':>7s} {'95% CI':>16s} {'spread':>7s}")
        for benchmark in group[0][1]['benchmarks']:
            for label in CLASSES:
                cells = [data['benchmarks'][benchmark].get('crossfit', {}).get(label) for _, data in group]
                cells = [cell for cell in cells if cell and cell['difference'] is not None]
                if not cells:
                    continue
                differences = np.array([cell['difference'] for cell in cells])
                interval = [cell.get('bootstrap_difference') for cell in cells]
                span = ''
                if all(interval):
                    low = float(np.mean([x['low'] for x in interval]))
                    high = float(np.mean([x['high'] for x in interval]))
                    span = f'[{low:+.2f},{high:+.2f}]'
                share = group[0][1]['benchmarks'][benchmark]['item_fraction'][label]
                print(f"{benchmark:16s} {label:15s} {cells[0]['items']:7d} {share:6.1%} "
                      f"{np.mean([c['unseen'] for c in cells]):7.2f} {np.mean([c['seen'] for c in cells]):7.2f} "
                      f"{differences.mean():+7.2f} {span:>16s} "
                      f"{(differences.max() - differences.min()) if len(differences) > 1 else 0:7.2f}")


def variant_table():
    runs = defaultdict(list)
    for name, pool, path in model_dirs('report.json'):
        data = read_json(path)
        if 'pairclean' not in data['labels']:
            continue
        config, seed = configuration(name)
        runs[(config, pool)].append((seed, data))
    for (config, pool), group in sorted(runs.items()):
        labels = group[0][1]['labels']
        seeds = [seed for seed, _ in group if seed is not None]
        print(f"\n# decontamination  model {config}  pool {pool}  {len(group)} run(s)"
              f"{' seeds ' + ','.join(map(str, seeds)) if seeds else ''}")
        print(f"{'benchmark':16s} " + ' '.join(f'{label:>12s}' for label in labels) +
              f" {'clean-rel':>10s} {'control-rel':>12s}")
        for benchmark in group[0][1]['evaluation']:
            scores = {label: np.mean([data['evaluation'][benchmark][label]['score'] for _, data in group])
                      for label in labels}
            print(f'{benchmark:16s} ' + ' '.join(f'{scores[label]:12.2f}' for label in labels) +
                  f" {scores['pairclean'] - scores['released']:+10.2f}"
                  f" {scores['randomdrop'] - scores['released']:+12.2f}")


def main(which='all'):
    if which in ('all', 'effect'):
        effect_table()
    if which in ('all', 'variants'):
        variant_table()


if __name__ == '__main__':
    main(*sys.argv[1:2])
