"""Pearl's data directory and the three training configurations.

Pearl is run from external/Pearl with the released code and the released
hyper-parameters; only the paths change.  Three training pools:

    released    the authors' spica_train / spica_val
    pairclean   both with every row whose (picture, candidate) pair also occurs
                in an evaluation set removed
    randomdrop  a same-size random control, so the pairclean effect is not just
                the smaller pool

The feature directories hold one dictionary over all distinct pictures and
texts (features/<model>/all), so train, val and test point at the same files.

Written under external/Pearl/:
    data/images -> pearl_data/images
    data/<split>.csv for every Pearl CSV, and spica_{train,val}_{variant}.csv
    features/<model> -> ../../features/<model>
    configs/variants/<variant>.yaml
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(_ROOT)
DATA = ROOT / 'pearl_data'
PEARL = ROOT / 'external/Pearl'
BASE_CONFIG = PEARL / 'configs/xlmr/base/da-estimator-base.yaml'
VARIANTS = ['released', 'pairclean', 'randomdrop']

csv.field_size_limit(10 ** 9)


def link(path, target):
    if path.is_symlink() or path.exists():
        if path.is_symlink() and Path(path).resolve() == target.resolve():
            return
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)


def filtered_csv(source, destination, keep):
    with open(source, newline='') as f, open(destination, 'w', newline='') as g:
        reader, writer = csv.reader(f), csv.writer(g)
        writer.writerow(next(reader))
        written = 0
        for index, row in enumerate(reader):
            if keep[index]:
                writer.writerow(row)
                written += 1
    return written


def main():
    link(PEARL / 'data' / 'images', DATA / 'images')
    for path in sorted((DATA / 'csv').glob('*.csv')):
        link(PEARL / 'data' / path.name, path)
    for model in ['clippp', 'blip2', 'beit3', 'stella']:
        link(PEARL / 'features' / model, ROOT / 'features' / model)

    for variant in VARIANTS[1:]:
        for pool in ['spica_train', 'spica_val']:
            keep = np.load(DATA / f'{pool}_{variant}.npy')
            rows = filtered_csv(DATA / 'csv' / f'{pool}.csv', PEARL / 'data' / f'{pool}_{variant}.csv', keep)
            print(variant, pool, 'rows', rows, 'of', len(keep), flush=True)

    base = BASE_CONFIG.read_text()
    (PEARL / 'configs/variants').mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        text = base
        suffix = '' if variant == 'released' else f'_{variant}'
        text = text.replace('train_path: data/spica_train.csv', f'train_path: data/spica_train{suffix}.csv')
        text = text.replace('val_path: data/spica_val.csv', f'val_path: data/spica_val{suffix}.csv')
        for model, key in [('clippp', ''), ('blip', '_blip'), ('beit3', '_beit3'), ('stella', '_stella')]:
            for split in ['train', 'val', 'test']:
                directory = {'clippp': 'clippp', 'blip': 'blip2', 'beit3': 'beit3', 'stella': 'stella'}[model]
                text = text.replace(f'{split}{key}_features_path: features/{directory}/{split}',
                                    f'{split}{key}_features_path: features/{directory}/all')
        path = PEARL / 'configs/variants' / f'{variant}.yaml'
        path.write_text(text)
        print('wrote', path, flush=True)


if __name__ == '__main__':
    sys.exit(main())
