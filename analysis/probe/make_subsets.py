"""Row files and standardization statistics for the fold models of the MLP head.

The statistics come from the already accumulated Gram of exactly those rows, so each model
standardizes with its own training subset and nothing is recomputed from the features.
Usage: make_subsets.py <backbone> <pool>
"""
import sys

import numpy as np
import torch

from lib import CACHE, FOLDS, load_rows, picture_fold, row_pictures, subset, write_json
from models import clean_validation, fold_names, pieces_dir, sum_pieces
from lib import read_json


def save_rows(name, rows):
    np.savez(CACHE / f'{name}.npz', **rows)
    return dict(name=name, rows=len(rows['target']), pictures=int(len(np.unique(row_pictures(rows)))))


def save_stats(backbone, pool, names, path):
    gram, _ = sum_pieces(backbone, pool, 'fold', names, device='cuda')
    d = gram.shape[0] - 1
    n = gram[d, d]
    mean = gram[:d, d] / n
    scale = (gram[:d, :d].diagonal() / n - mean.square()).clamp_min(0).sqrt()
    torch.save(dict(mean=mean.float().cpu(), scale=scale.clamp_min(1e-6).float().cpu(),
                    raw_scale=scale.float().cpu(), count=int(n),
                    description=f'from the Gram of {names} of {pool}'), path)
    del gram
    torch.cuda.empty_cache()
    return float(n)


def main(backbone='siglip', pool='spica_train'):
    rows = load_rows(pool, CACHE)
    folds = picture_fold(row_pictures(rows))
    pieces = read_json(pieces_dir(backbone, pool, 'fold') / 'pieces.json')['pieces']
    report = {'validation': save_rows(f'{pool}_cleanval', clean_validation(pool))}
    report['full'] = save_rows(f'{pool}_full', rows)
    report['full']['weight'] = save_stats(backbone, pool, fold_names(pieces), CACHE / f'{backbone}_stats_{pool}_full.pt')
    for f in range(FOLDS):
        index = np.flatnonzero(folds != f)
        name = f'{pool}_heldout{f}'
        report[name] = save_rows(name, subset(rows, index))
        report[name]['weight'] = save_stats(backbone, pool, fold_names(pieces, folds=[k for k in range(FOLDS) if k != f]),
                                            CACHE / f'{backbone}_stats_{name}.pt')
        print(name, report[name], flush=True)
    write_json(CACHE / f'subsets_{pool}.json', report)
    print(report['validation'], report['full'], flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])
