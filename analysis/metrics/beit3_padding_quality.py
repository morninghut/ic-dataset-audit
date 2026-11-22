"""Does the padding change BEiT-3's features in a way that matters?

Uses the plain BEiT-3 image-text cosine as a reference-free caption metric on
the labelled splits, once with the padding-free text features and once with the
released pipeline's padded ones, and reports Kendall tau against the human
scores.  A drop means the padding does not just move the vectors, it degrades
them.
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import ast
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(_ROOT)
FEATURES = ROOT / 'features/beit3'
DATA = ROOT / 'pearl_data/csv'
SPLITS = ['spica_test', 'polaris_test', 'nebula_test', 'flickr8k_expert', 'composite']

csv.field_size_limit(10 ** 9)


def load(directory, kind):
    merged = {}
    for chunk in range(5):
        with np.load(directory / f'clip_{kind}_features_{chunk + 1}.npz', allow_pickle=True) as f:
            merged.update(f['features'].item())
    return merged


def main():
    images = load(FEATURES / 'all', 'img')
    texts = {'padding-free': load(FEATURES / 'all', 'mt'),
             'released (padded)': load(FEATURES / 'all_batchpadded', 'mt')}
    for split in SPLITS:
        with open(DATA / f'{split}.csv', newline='') as f:
            rows = list(csv.DictReader(f))
        human = np.array([float(r['score']) for r in rows])
        line = [f'{split:16s} n={len(rows):6d}']
        for name, table in texts.items():
            image = np.stack([images[r['imgid']] for r in rows])
            text = np.stack([table[r['mt']] for r in rows])
            cosine = (image * text).sum(1)
            line.append(f'{name}: tau_c={kendalltau(cosine, human, variant="c")[0]:+.4f}')
        # reference-based variant: max cosine to a reference, same text features
        print('  '.join(line), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
