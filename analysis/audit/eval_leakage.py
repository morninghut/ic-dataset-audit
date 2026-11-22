"""Sample-level leakage of every benchmark the paper reports into each training pool.

For each evaluation row (as scored from feature_cache_v1) it checks whether the
training pool contains the same picture, the same (picture, candidate) pair, the
same (picture, references, candidate) triple, and the same triple with an equal
human score. Pictures are matched by decoded-pixel hash.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DATA, OUT, read_json, write_json
from benchmark_leakage import path_to_hash
from csv_audit import load as load_spica
from label_leakage import nebula_rows
from provenance import polaris_rows

KEY = '\u241f'


def pool_tables():
    pools = {}
    spica_pixel = {r['name']: r.get('pixel_sha1') for r in read_json(OUT / 'inventory_spica.json')}
    train = load_spica('train')
    train['pixel'] = train.imgid.map(spica_pixel)
    pools['spica_train'] = train.rename(columns={'score': 'human'})
    polaris = polaris_rows()
    pools['polaris_train'] = polaris[polaris.split == 'train'].rename(columns={'score': 'human'})
    nebula = nebula_rows()
    pools['nebula_train'] = nebula[nebula.split == 'train'].rename(columns={'human_score': 'human'})
    return pools


def index(frame):
    images = set(frame.pixel)
    pairs = set(zip(frame.pixel, frame.mt))
    triples = set(zip(frame.pixel, frame.reference_key, frame.mt))
    scored = {}
    for p, k, mt, s in zip(frame.pixel, frame.reference_key, frame.mt, frame.human):
        scored.setdefault((p, k, mt), set()).add(round(float(s), 6))
    return images, pairs, triples, scored


def main():
    catalog = read_json(CACHE / 'catalog.json')
    texts = catalog['texts']
    mapping = path_to_hash()
    image_hash = np.array([mapping.get(str(Path(p).resolve()), '') for p in catalog['images']], dtype=object)

    pools = {name: index(frame) for name, frame in pool_tables().items()}
    report = {}
    for path in sorted(CACHE.glob('*.npz')):
        name = path.stem
        data = np.load(path, allow_pickle=True)
        pixels = image_hash[data['image']]
        candidates = [texts[i] for i in data['candidate']]
        refs = data['references']
        counts = data['counts']
        reference_keys = [KEY.join(texts[j] for j in row[:n]) for row, n in zip(refs, counts)]
        targets = data['target'].astype(float)
        entry = dict(rows=int(len(pixels)), pictures=int(len(set(pixels))))
        for pool, (images, pairs, triples, scored) in pools.items():
            image_hit = np.fromiter((p in images for p in pixels), bool, len(pixels))
            pair_hit = np.fromiter(((p, c) in pairs for p, c in zip(pixels, candidates)), bool, len(pixels))
            triple = list(zip(pixels, reference_keys, candidates))
            triple_hit = np.fromiter((t in triples for t in triple), bool, len(triple))
            score_hit = np.fromiter((t in scored and round(float(y), 6) in scored[t]
                                     for t, y in zip(triple, targets)), bool, len(triple))
            entry[pool] = dict(
                image_rows=int(image_hit.sum()), image_row_share=round(float(image_hit.mean()), 4),
                pair_rows=int(pair_hit.sum()), pair_row_share=round(float(pair_hit.mean()), 4),
                triple_rows=int(triple_hit.sum()), triple_row_share=round(float(triple_hit.mean()), 4),
                same_score_rows=int(score_hit.sum()), same_score_row_share=round(float(score_hit.mean()), 4))
        report[name] = entry
        print(f'{name:16s} ' + ' | '.join(
            f"{pool.split('_')[0]:7s} img {entry[pool]['image_row_share']:5.1%} "
            f"pair {entry[pool]['pair_row_share']:5.1%} triple {entry[pool]['triple_row_share']:5.1%} "
            f"+score {entry[pool]['same_score_row_share']:5.1%}" for pool in pools))
    write_json(OUT / 'eval_leakage.json', report)


if __name__ == '__main__':
    main()
