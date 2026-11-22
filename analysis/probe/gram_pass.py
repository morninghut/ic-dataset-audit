"""Accumulate ridge sufficient statistics of a training pool as reusable additive pieces.

fold : a partition of the rows into 2*fold(picture) + trivial(row); any union of folds and
       any trivial-row ablation is a sum of these cells.
mask : the rows that contain each benchmark's pictures (overlapping, so a decontaminated
       model's statistics are full minus that piece) plus the rows that contain none.

Targets: column 0 = the row's own human score, column 1 = the mean score of its sample.
Usage: gram_pass.py <backbone> <pool> <partition>
"""
import sys
import time

import numpy as np
import torch

from lib import CACHE, RUNS, Features, accumulate, load_rows, picture_fold, row_pictures, \
    sample_structure, subset, write_json

BITS = ['foil_1ref', 'composite', 'polaris_test', 'nebula_test', 'spica_test', 'spica_val',
        'polaris_val', 'nebula_valid', 'flickr8k_expert', 'pascal50s']


def trivial_rows(rows):
    """A row is trivial if its sample was judged once and scored zero (audit definition)."""
    structure = sample_structure(rows)
    return ((structure['size'] == 1) & (rows['target'] == 0)).astype(np.int32), structure


def benchmark_mask(pictures):
    mask = np.zeros(len(pictures), np.int64)
    for bit, name in enumerate(BITS):
        benchmark = np.unique(row_pictures(load_rows(name, CACHE)))
        mask |= np.isin(pictures, benchmark).astype(np.int64) << bit
    return mask


def pieces(rows, partition):
    """{name: row index}, plus the sample structure used for the targets."""
    pictures = row_pictures(rows)
    trivial, structure = trivial_rows(rows)
    if partition == 'fold':
        cells = 2 * picture_fold(pictures).astype(np.int32) + trivial
        return {f'fold{cell // 2}_trivial{cell % 2}': np.flatnonzero(cells == cell)
                for cell in np.unique(cells)}, structure
    if partition == 'mask':
        mask = benchmark_mask(pictures)
        out = {'clean': np.flatnonzero(mask == 0)}
        for bit, name in enumerate(BITS):
            index = np.flatnonzero((mask >> bit) & 1)
            if len(index):
                out['in_' + name] = index
        return out, structure
    raise ValueError(partition)


def main(backbone, pool, partition):
    rows = load_rows(pool, CACHE)
    parts, structure = pieces(rows, partition)
    targets = np.stack([rows['target'], structure['mean']], 1)
    features = Features(backbone)
    out = RUNS / 'stats' / backbone / pool / partition
    out.mkdir(parents=True, exist_ok=True)
    report, started = {}, time.time()
    for name, index in parts.items():
        gram, rhs = accumulate(features, subset(rows, index), targets[index])
        torch.save(dict(gram=gram.cpu(), rhs=rhs.cpu()), out / f'{name}.pt')
        report[name] = dict(rows=len(index), pictures=int(len(np.unique(row_pictures(subset(rows, index))))),
                            weight=float(gram[-1, -1]))
        print(pool, partition, name, report[name], round(time.time() - started), 's', flush=True)
        del gram, rhs
        torch.cuda.empty_cache()
    write_json(out / 'pieces.json', dict(pool=pool, partition=partition, backbone=backbone, bits=BITS,
                                         partition_of_rows=partition == 'fold',
                                         target_columns=['score', 'sample_mean'], pieces=report,
                                         rows=len(rows['target']), seconds=time.time() - started))


if __name__ == '__main__':
    main(*sys.argv[1:4])
