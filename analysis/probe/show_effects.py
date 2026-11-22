"""Readable tables of effects.json. Usage: show_effects.py <pool> [classes...]"""
import sys

from lib import RUNS, read_json

CLASSES = ['clean', 'picture_leaked', 'pair_leaked', 'all']


def main(pool='spica_train', *classes):
    data = read_json(RUNS / 'models/siglip' / pool / 'effects.json')
    print(f"# cross-fit leakage effect, training pool {pool}")
    print(f"{'benchmark':16s} {'class':15s} {'units':>6s} {'items':>7s} {'unseen':>7s} {'seen':>7s} {'diff':>6s} {'95% CI':>16s}")
    for name, entry in data['benchmarks'].items():
        for label in (classes or CLASSES):
            value = entry.get('crossfit', {}).get(label)
            if not value or value['difference'] is None:
                continue
            ci = value.get('bootstrap_difference')
            span = f"[{ci['low']:+.2f},{ci['high']:+.2f}]" if ci else ''
            print(f"{name:16s} {label:15s} {value['units']:6d} {value['items']:7d} "
                  f"{value['unseen']:7.2f} {value['seen']:7.2f} {value['difference']:+6.2f} {span:>16s}")
    labels = [k for k in next(iter(data['benchmarks'].values()))['models']]
    print(f"\n# benchmark score of each training variant (all items)")
    print(f"{'benchmark':16s} " + ' '.join(f'{l[:14]:>14s}' for l in labels))
    for name, entry in data['benchmarks'].items():
        print(f"{name:16s} " + ' '.join(f"{entry['models'][l]['all']:14.2f}" for l in labels))


if __name__ == '__main__':
    main(*sys.argv[1:])
