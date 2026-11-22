"""Pearl-format data for every split, keyed by picture identity.

Pearl addresses an image by file name and a text by its exact string, so one
shared image directory whose file names are picture identities lets the four
feature extractors run once per distinct picture and text, and every split then
reuses the same feature dictionaries.

Written under pearl_data/:
    images/p<picture>.img   symlink to one file of that picture
    csv/<split>.csv         imgid, refs, mt, score  (Pearl's columns)
    csv/<split>.ids.json    row ids of the cache, in CSV order
    variants.json           pairclean / randomdrop masks of spica_train

The Spica training pool is the union of the cache's spica_train (rows with
references) and spica_train_noref, restored to the order of the released CSV,
so the pool is exactly the released one (309,386 rows).
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(_ROOT)
CACHE = ROOT / 'feature_cache_v2'
OUT = ROOT / 'pearl_data'
DROP_SEED = 7

# Splits used for training/validation and for evaluation.
TRAIN_SPLITS = ['spica_train', 'spica_val']
EVAL_SPLITS = ['spica_test', 'polaris_test', 'polaris_val', 'nebula_test', 'nebula_valid',
               'flickr8k_expert', 'flickr8k_cf', 'composite', 'pascal50s', 'foil_1ref', 'foil_4ref']


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, value):
    with open(path, 'w') as f:
        json.dump(value, f)


def load_rows(name):
    with np.load(CACHE / f'{name}.npz', allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def concatenate(parts):
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def spica_train_rows():
    """The released training pool: referenced and reference-less rows, in CSV order."""
    rows = concatenate([load_rows('spica_train'), load_rows('spica_train_noref')])
    order = np.argsort([int(i.split(':')[1]) for i in rows['ids']], kind='stable')
    return {key: value[order] for key, value in rows.items()}


def rows_of(name):
    return spica_train_rows() if name == 'spica_train' else load_rows(name)


def main():
    catalog = read_json(CACHE / 'catalog.json')
    texts, images = catalog['texts'], catalog['images']
    picture_of_image = np.load(CACHE / 'picture_ids.npy')

    splits = {name: rows_of(name) for name in TRAIN_SPLITS + EVAL_SPLITS}

    # One symlink per distinct picture, named by picture identity.
    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    (OUT / 'csv').mkdir(parents=True, exist_ok=True)
    used = sorted({int(p) for rows in splits.values() for p in picture_of_image[rows['image']]})
    first_image = {}
    for index, picture in enumerate(picture_of_image):
        first_image.setdefault(int(picture), index)
    for picture in used:
        link = OUT / 'images' / f'p{picture}.img'
        target = Path(images[first_image[picture]])
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    print('pictures', len(used), flush=True)

    # Pearl CSVs.  refs is a repr of the reference list, as in the released files.
    import csv as csv_module
    for name, rows in splits.items():
        pictures = picture_of_image[rows['image']]
        with open(OUT / 'csv' / f'{name}.csv', 'w', newline='') as f:
            writer = csv_module.writer(f)
            writer.writerow(['imgid', 'refs', 'mt', 'score'])
            for index in range(len(rows['target'])):
                refs = [texts[r] for r in rows['references'][index][:rows['counts'][index]]]
                writer.writerow([f'p{int(pictures[index])}.img', repr(refs),
                                 texts[rows['candidate'][index]], float(rows['target'][index])])
        write_json(OUT / 'csv' / f'{name}.ids.json', list(map(str, rows['ids'])))
        print(name, len(rows['target']), flush=True)

    # Training-pool variants: drop rows whose (picture, candidate) pair occurs in an
    # evaluation set, and a same-size random control.
    def pair_keys(rows):
        return picture_of_image[rows['image']].astype(np.int64) * 1000000 + rows['candidate'].astype(np.int64)

    evaluation_pairs = np.unique(np.concatenate([pair_keys(splits[name]) for name in EVAL_SPLITS]))
    report = {}
    for pool in ['spica_train', 'spica_val']:
        keys = pair_keys(splits[pool])
        leaked = np.isin(keys, evaluation_pairs)
        rng = np.random.default_rng(DROP_SEED)
        control = np.zeros(len(keys), bool)
        control[rng.choice(len(keys), int(leaked.sum()), replace=False)] = True
        counts = {name: int(np.isin(keys, np.unique(pair_keys(splits[name]))).sum()) for name in EVAL_SPLITS}
        np.save(OUT / f'{pool}_pairclean.npy', ~leaked)
        np.save(OUT / f'{pool}_randomdrop.npy', ~control)
        report[pool] = dict(rows=len(keys), leaked=int(leaked.sum()), per_evaluation_set=counts)
        print(pool, report[pool]['rows'], 'leaked', report[pool]['leaked'], flush=True)
    write_json(OUT / 'variants.json', dict(drop_seed=DROP_SEED, pools=report))


if __name__ == '__main__':
    sys.exit(main())
