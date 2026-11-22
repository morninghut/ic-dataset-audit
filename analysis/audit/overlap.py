"""Match images across datasets by exact decoded-pixel hash and report overlap.

Answers: how much of each evaluation benchmark is physically present in the
Spica / Polaris training images, and how the Spica, Polaris and Nebula splits
overlap with each other.
"""
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, SPICA, read_json, write_json
from csv_audit import load as load_spica


def inventory(group):
    return read_json(OUT / f'inventory_{group}.json')


def hashes(rows, names=None):
    """name -> pixel hash, optionally restricted to a set of names."""
    return {r['name']: r['pixel_sha1'] for r in rows
            if 'pixel_sha1' in r and (names is None or r['name'] in names)}


def report_pair(name_a, a, name_b, b):
    shared = set(a.values()) & set(b.values())
    return dict(
        images_a=len(set(a.values())), images_b=len(set(b.values())),
        shared_images=len(shared),
        share_of_a=round(len(shared) / max(len(set(a.values())), 1), 4),
        share_of_b=round(len(shared) / max(len(set(b.values())), 1), 4))


def main():
    spica_rows = inventory('spica')
    spica_hash = hashes(spica_rows)
    frames = {split: load_spica(split) for split in ['train', 'val', 'test']}
    spica = {split: {i: spica_hash[i] for i in set(frame.imgid) if i in spica_hash}
             for split, frame in frames.items()}

    polaris_hash = hashes(inventory('polaris'))
    polaris = {}
    for split in ['train', 'val', 'test']:
        ids = set(pd.read_csv(DATA / f'polaris/polaris_{split}.csv', dtype=str).imgid)
        polaris[split] = {i: polaris_hash[i] for i in ids if i in polaris_hash}

    nebula_rows = inventory('nebula')
    nebula = defaultdict(dict)
    for row in nebula_rows:
        if 'pixel_sha1' in row:
            nebula[row['split']][row['name']] = row['pixel_sha1']

    others = {name: hashes(inventory(name)) for name in
              ['flickr8k', 'composite', 'pascal', 'coco_val2014']}

    sources = {f'spica_{k}': v for k, v in spica.items()}
    sources.update({f'polaris_{k}': v for k, v in polaris.items()})
    sources.update({f'nebula_{k}': v for k, v in nebula.items()})
    sources.update(others)

    report = dict(dataset_sizes={k: dict(names=len(v), unique_images=len(set(v.values())))
                                 for k, v in sources.items()})
    train_pools = ['spica_train', 'polaris_train', 'nebula_train']
    targets = [k for k in sources if k not in train_pools]
    report['benchmark_in_training_pool'] = {
        pool: {target: report_pair(pool, sources[pool], target, sources[target]) for target in targets}
        for pool in train_pools}
    report['cross_dataset'] = {
        f'{a}_vs_{b}': report_pair(a, sources[a], b, sources[b])
        for a, b in [('spica_test', 'nebula_test'), ('spica_val', 'nebula_valid'),
                     ('spica_train', 'nebula_train'), ('spica_train', 'spica_test'),
                     ('spica_train', 'spica_val'), ('spica_val', 'spica_test'),
                     ('polaris_train', 'polaris_test'), ('nebula_train', 'nebula_test'),
                     ('composite', 'coco_val2014'), ('flickr8k', 'spica_train')]}

    # Which benchmark images sit in the Spica training pool: save the name lists so
    # later experiments can split each benchmark into leaked and clean subsets.
    leaked = {}
    for pool in train_pools:
        pool_hashes = set(sources[pool].values())
        for target in targets:
            names = sorted(n for n, h in sources[target].items() if h in pool_hashes)
            if names:
                leaked[f'{target}_in_{pool}'] = names
    write_json(OUT / 'leaked_image_names.json', leaked)
    write_json(OUT / 'overlap.json', report)

    for pool in train_pools:
        print(pool)
        for target, stats in report['benchmark_in_training_pool'][pool].items():
            print(f"  {target:16s} {stats['shared_images']:6d}/{stats['images_b']:6d}"
                  f"  {stats['share_of_b']:.1%}")


if __name__ == '__main__':
    main()
