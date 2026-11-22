"""Nebula data for DENEB, in the same picture identities as pearl_data.

DENEB reads data_en/nebula/{train,validation,test}.csv (imgid, refs, mt, score)
and opens data_en/nebula/images/<imgid> with PIL.  The released Nebula parquet
files carry the image bytes inline, but every Nebula picture is already present
in the feature cache (interned with the COCO/Polaris/Spica pictures), so
the CSVs are written from the cache and the images are the same
pearl_data/images/p<picture>.img symlinks Pearl uses.  One picture identity
across both models is what lets the leakage masks be the same objects.

Written:
    pearl_data/csv/nebula_train.csv (+ .ids.json)
    pearl_data/nebula_train_{pairclean,randomdrop}.npy
    pearl_data/nebula_variants.json
    external/DENEB/data_en/nebula/{train,train_pairclean,train_randomdrop,
                                   validation,test}.csv
    external/DENEB/data_en/nebula/images -> pearl_data/images
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(_ROOT)
CACHE = ROOT / 'feature_cache_v2'
PEARL_DATA = ROOT / 'pearl_data'
DENEB_DATA = ROOT / 'external/DENEB/data_en/nebula'
DROP_SEED = 7

# The same evaluation sets prep_pearl_data.py cleans the Spica pool against.
EVAL_SPLITS = ['spica_test', 'polaris_test', 'polaris_val', 'nebula_test', 'nebula_valid',
               'flickr8k_expert', 'flickr8k_cf', 'composite', 'pascal50s', 'foil_1ref', 'foil_4ref']


def load_rows(name):
    with np.load(CACHE / f'{name}.npz', allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def write_csv(path, rows, texts, pictures):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['imgid', 'refs', 'mt', 'score'])
        for index in range(len(rows['target'])):
            refs = [texts[r] for r in rows['references'][index][:rows['counts'][index]]]
            writer.writerow([f'p{int(pictures[index])}.img', repr(refs),
                             texts[rows['candidate'][index]], float(rows['target'][index])])


def subset(rows, keep):
    return {key: value[keep] for key, value in rows.items()}


def main():
    catalog = json.load(open(CACHE / 'catalog.json'))
    texts, images = catalog['texts'], catalog['images']
    picture_of_image = np.load(CACHE / 'picture_ids.npy')

    splits = {name: load_rows(name) for name in ['nebula_train', 'nebula_valid', 'nebula_test'] + EVAL_SPLITS}

    # Symlinks for any Nebula picture pearl_data does not have yet.
    first_image = {}
    for index, picture in enumerate(picture_of_image):
        first_image.setdefault(int(picture), index)
    added = 0
    for name in ['nebula_train', 'nebula_valid', 'nebula_test']:
        for picture in np.unique(picture_of_image[splits[name]['image']]):
            link = PEARL_DATA / 'images' / f'p{int(picture)}.img'
            if not link.is_symlink() and not link.exists():
                link.symlink_to(Path(images[first_image[int(picture)]]))
                added += 1
    print('new picture links', added, flush=True)

    pictures = {name: picture_of_image[rows['image']] for name, rows in splits.items()}
    write_csv(PEARL_DATA / 'csv' / 'nebula_train.csv', splits['nebula_train'], texts, pictures['nebula_train'])
    with open(PEARL_DATA / 'csv' / 'nebula_train.ids.json', 'w') as f:
        json.dump(list(map(str, splits['nebula_train']['ids'])), f)

    def pair_keys(name):
        return pictures[name].astype(np.int64) * 1000000 + splits[name]['candidate'].astype(np.int64)

    evaluation_pairs = np.unique(np.concatenate([pair_keys(name) for name in EVAL_SPLITS]))
    keys = pair_keys('nebula_train')
    leaked = np.isin(keys, evaluation_pairs)
    rng = np.random.default_rng(DROP_SEED)
    control = np.zeros(len(keys), bool)
    control[rng.choice(len(keys), int(leaked.sum()), replace=False)] = True
    np.save(PEARL_DATA / 'nebula_train_pairclean.npy', ~leaked)
    np.save(PEARL_DATA / 'nebula_train_randomdrop.npy', ~control)
    report = dict(drop_seed=DROP_SEED, rows=len(keys), leaked=int(leaked.sum()),
                  per_evaluation_set={name: int(np.isin(keys, np.unique(pair_keys(name))).sum())
                                      for name in EVAL_SPLITS})
    with open(PEARL_DATA / 'nebula_variants.json', 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report), flush=True)

    # DENEB's own data directory.
    DENEB_DATA.mkdir(parents=True, exist_ok=True)
    link = DENEB_DATA / 'images'
    if not link.is_symlink() and not link.exists():
        link.symlink_to(PEARL_DATA / 'images')
    write_csv(DENEB_DATA / 'train.csv', splits['nebula_train'], texts, pictures['nebula_train'])
    write_csv(DENEB_DATA / 'train_pairclean.csv', subset(splits['nebula_train'], ~leaked), texts,
              pictures['nebula_train'][~leaked])
    write_csv(DENEB_DATA / 'train_randomdrop.csv', subset(splits['nebula_train'], ~control), texts,
              pictures['nebula_train'][~control])
    write_csv(DENEB_DATA / 'validation.csv', splits['nebula_valid'], texts, pictures['nebula_valid'])
    write_csv(DENEB_DATA / 'test.csv', splits['nebula_test'], texts, pictures['nebula_test'])
    for name in ['train', 'train_pairclean', 'train_randomdrop', 'validation', 'test']:
        print(name, sum(1 for _ in open(DENEB_DATA / f'{name}.csv')) - 1, flush=True)


if __name__ == '__main__':
    sys.exit(main())
