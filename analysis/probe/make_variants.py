"""Training subsets that remove duplicated (picture, candidate) pairs, and a size control.

pairclean  : every row whose picture-and-candidate pair also occurs in some evaluation set
             is dropped, so no evaluation item's human label can be read off the pool.
randomdrop : the same number of rows, chosen uniformly at random with a fixed seed, is
             dropped instead. It holds the training-set size fixed, so the difference
             between the two is the duplicated labels and not the amount of data.

Both are written as row files for the MLP head and as Gram pieces (partition 'variant')
for the ridge head, together with the standardization statistics of exactly those rows.
Usage: make_variants.py <backbone> <pool> [drop seed]
"""
import sys
import time

import numpy as np
import torch

from lib import CACHE, EVAL_SETS, Features, accumulate, load_rows, row_pictures, sample_structure, \
    subset, write_json
from gram_pass import trivial_rows  # noqa: F401  (same trivial definition, kept importable)
from models import pieces_dir

DROP_SEED = 7


def pair_keys(rows):
    return row_pictures(rows).astype(np.int64) * 1000000 + rows['candidate'].astype(np.int64)


def leaked_mask(rows, names=EVAL_SETS):
    """Rows whose (picture, candidate) pair appears in an evaluation set, and the per-set counts."""
    keys = pair_keys(rows)
    mask = np.zeros(len(keys), bool)
    counts = {}
    for name in names:
        hit = np.isin(keys, np.unique(pair_keys(load_rows(name, CACHE))))
        counts[name] = int(hit.sum())
        mask |= hit
    return mask, counts


def variants(rows, seed=DROP_SEED):
    mask, counts = leaked_mask(rows)
    rng = np.random.default_rng(seed)
    control = np.zeros(len(mask), bool)
    control[rng.choice(len(mask), int(mask.sum()), replace=False)] = True
    return {'pairclean': ~mask, 'randomdrop': ~control}, counts


def save(backbone, pool, name, rows, index, targets, features):
    """Row file, Gram piece and standardization statistics of one subset."""
    part = subset(rows, index)
    np.savez(CACHE / f'{pool}_{name}.npz', **part)
    gram, rhs = accumulate(features, part, targets[index])
    out = pieces_dir(backbone, pool, 'variant')
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(gram=gram.cpu(), rhs=rhs.cpu()), out / f'{name}.pt')
    d = gram.shape[0] - 1
    n = gram[d, d]
    mean = gram[:d, d] / n
    scale = (gram[:d, :d].diagonal() / n - mean.square()).clamp_min(0).sqrt()
    torch.save(dict(mean=mean.float().cpu(), scale=scale.clamp_min(1e-6).float().cpu(),
                    raw_scale=scale.float().cpu(), count=int(n), description=f'{pool}_{name}'),
               CACHE / f'{backbone}_stats_{pool}_{name}.pt')
    report = dict(rows=len(index), pictures=int(len(np.unique(row_pictures(part)))), weight=float(n))
    del gram, rhs
    torch.cuda.empty_cache()
    return report


def main(backbone='siglip', pool='polaris_train', seed=DROP_SEED):
    rows = load_rows(pool, CACHE)
    structure = sample_structure(rows)
    targets = np.stack([rows['target'], structure['mean']], 1)
    keep, counts = variants(rows, int(seed))
    features = Features(backbone)
    report, started = {}, time.time()
    for name, mask in keep.items():
        report[name] = save(backbone, pool, name, rows, np.flatnonzero(mask), targets, features)
        print(pool, name, report[name], round(time.time() - started), 's', flush=True)
    write_json(pieces_dir(backbone, pool, 'variant') / 'pieces.json',
               dict(pool=pool, partition='variant', backbone=backbone, drop_seed=int(seed),
                    target_columns=['score', 'sample_mean'], rows=len(rows['target']),
                    leaked_rows_per_evaluation_set=counts, pieces=report,
                    seconds=time.time() - started))


if __name__ == '__main__':
    main(*sys.argv[1:4])
