"""How much BEiT-3 text features move when the batch's padding changes.

The released save_features pipeline tokenizes a batch with padding=True and
passes only input_ids to BEiT-3, with no padding mask, so a caption's feature
depends on the longest caption batched with it.  features/beit3/all holds the
padding-free features (texts batched by equal token length) and
features/beit3/all_batchpadded the released behaviour on our CSV order.
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import sys
from pathlib import Path

import numpy as np

FEATURES = Path((_ROOT + '/features/beit3'))


def load(directory):
    merged = {}
    for chunk in range(5):
        with np.load(directory / f'clip_mt_features_{chunk + 1}.npz', allow_pickle=True) as f:
            merged.update(f['features'].item())
    return merged


def main():
    free, padded = load(FEATURES / 'all'), load(FEATURES / 'all_batchpadded')
    keys = sorted(set(free) & set(padded))
    print('texts', len(keys))
    a = np.stack([free[k] for k in keys])
    b = np.stack([padded[k] for k in keys])
    cosine = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    print('cosine(padding-free, released)  min %.4f  1%% %.4f  median %.4f  mean %.4f'
          % (cosine.min(), np.percentile(cosine, 1), np.median(cosine), cosine.mean()))
    print('share below 0.99', float((cosine < 0.99).mean()), 'below 0.95', float((cosine < 0.95).mean()))

    # How far apart is this compared with the distance between different texts?
    rng = np.random.default_rng(0)
    other = rng.permutation(len(keys))
    between = (a * a[other]).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(a[other], axis=1))
    print('cosine between unrelated texts: median %.4f' % np.median(between))
    return 0


if __name__ == '__main__':
    sys.exit(main())
