"""Per-benchmark leakage masks: which evaluation rows use an image that is
physically present in a training pool (Spica train, Polaris train, Nebula train).

Works on the evaluation sets exactly as the reproduction scores them, i.e. the
rows in feature_cache_v1/<dataset>.npz, so the masks can be reused to split any
cached prediction file into leaked and clean subsets.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DATA, OUT, SPICA, read_json, write_json
from csv_audit import load as load_spica

ROOTS = {  # inventory group -> directory the catalog paths live in
    'polaris': DATA / 'polaris/images',
    'flickr8k': DATA / 'flickr8k/images',
    'composite': DATA / 'composite/images',
    'pascal': DATA / 'pascal/VOCdevkit',
    'coco_val2014': DATA / 'coco/val2014',
}


def path_to_hash():
    mapping = {}
    for group, root in ROOTS.items():
        by_name = {r['name']: r.get('pixel_sha1') for r in read_json(OUT / f'inventory_{group}.json')}
        for path in root.rglob('*'):
            if path.is_file() and path.name in by_name:
                mapping[str(path.resolve())] = by_name[path.name]
    return mapping


def training_pools():
    spica_hash = {r['name']: r.get('pixel_sha1') for r in read_json(OUT / 'inventory_spica.json')}
    train_ids = set(load_spica('train').imgid)
    pools = {'spica_train': {spica_hash[i] for i in train_ids if i in spica_hash}}
    polaris_hash = {r['name']: r.get('pixel_sha1') for r in read_json(OUT / 'inventory_polaris.json')}
    ids = set(pd.read_csv(DATA / 'polaris/polaris_train.csv', dtype=str).imgid)
    pools['polaris_train'] = {polaris_hash[i] for i in ids if i in polaris_hash}
    nebula = read_json(OUT / 'inventory_nebula.json')
    pools['nebula_train'] = {r['pixel_sha1'] for r in nebula
                             if r['split'] == 'train' and 'pixel_sha1' in r}
    # Spica train without the rows that have no references (those are the CC-style
    # pseudo-labelled half of the release).
    frame = load_spica('train')
    with_ref = set(frame.loc[frame.reference_count > 0, 'imgid'])
    pools['spica_train_with_refs'] = {spica_hash[i] for i in with_ref if i in spica_hash}
    return pools


def main():
    catalog = read_json(CACHE / 'catalog.json')
    images = [str(Path(p).resolve()) for p in catalog['images']]
    mapping = path_to_hash()
    missing = [p for p in images if p not in mapping]
    image_hash = np.array([mapping.get(p) or '' for p in images], dtype=object)

    pools = training_pools()
    report = dict(catalog_images=len(images), catalog_images_without_hash=len(missing),
                  pool_sizes={k: len(v) for k, v in pools.items()}, datasets={})
    masks = {}
    for path in sorted(CACHE.glob('*.npz')):
        name = path.stem
        data = np.load(path, allow_pickle=True)
        idx = data['image']
        row_hash = image_hash[idx]
        unique = set(row_hash) - {''}
        entry = dict(rows=int(len(idx)), images=len(unique))
        for pool, members in pools.items():
            mask = np.array([h in members for h in row_hash])
            masks[f'{name}__{pool}'] = mask
            entry[pool] = dict(
                leaked_rows=int(mask.sum()), row_share=round(float(mask.mean()), 4),
                leaked_images=len(unique & members),
                image_share=round(len(unique & members) / max(len(unique), 1), 4))
        report['datasets'][name] = entry
    np.savez_compressed(OUT / 'leakage_masks.npz', **masks)
    write_json(OUT / 'benchmark_leakage.json', report)

    header = f"{'dataset':16s} {'rows':>8s} {'imgs':>6s}" + ''.join(f' {p:>22s}' for p in pools)
    print(header)
    for name, entry in report['datasets'].items():
        line = f"{name:16s} {entry['rows']:8d} {entry['images']:6d}"
        for pool in pools:
            line += f" {entry[pool]['image_share']:>10.1%}img{entry[pool]['row_share']:>8.1%}row"
        print(line)


if __name__ == '__main__':
    main()
