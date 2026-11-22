"""Publishable identity lists: which (picture, candidate) pairs of a training pool are
duplicated in which evaluation set, and which rows of each evaluation set are duplicated in
which pool.

Spica and Nebula grant no redistribution licence, so nothing of their content is written
here. A pair is identified by the sha1 of its decoded picture, which the audit already uses
as picture identity, and the sha1 of the candidate caption; both are reproducible from a
local copy of the data without us shipping it. Reference sentences are not part of the key,
because a model scores a (picture, candidate) pair and a duplicated pair carries its human
label whatever the reference set is.

Usage: export_filters.py [output directory]  (default analysis/probe/filters)
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

from lib import CACHE, EVAL_SETS, load_rows, read_json, row_pictures, write_json

POOLS = ['polaris_train', 'nebula_train', 'spica_train']
DEFAULT = Path(__file__).resolve().parent / 'filters'


def text_hashes():
    texts = read_json(CACHE / 'catalog.json')['texts']
    return np.array([hashlib.sha1(text.encode()).hexdigest() for text in texts])


def keys(rows):
    return row_pictures(rows).astype(np.int64) * 1000000 + rows['candidate'].astype(np.int64)


def main(output=DEFAULT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pictures = read_json(CACHE / 'pictures.json')
    texts = text_hashes()
    pool_keys = {pool: np.unique(keys(load_rows(pool, CACHE))) for pool in POOLS}
    index = {}
    for name in EVAL_SETS + POOLS:
        rows = load_rows(name, CACHE)
        own = keys(rows)
        picture = row_pictures(rows)
        others = [pool for pool in POOLS if pool != name]
        duplicated = {pool: np.isin(own, pool_keys[pool]) for pool in others}
        any_hit = np.logical_or.reduce(list(duplicated.values()))
        unique, first = np.unique(own[any_hit], return_index=True)
        position = np.flatnonzero(any_hit)[first]
        records = [dict(picture_sha1=pictures[picture[row]], candidate_sha1=texts[rows['candidate'][row]],
                        pools=[pool for pool in others if duplicated[pool][row]])
                   for row in position]
        write_json(output / f'{name}_duplicated_pairs.json',
                   dict(dataset=name, rows=len(own), duplicated_rows=int(any_hit.sum()),
                        duplicated_pairs=len(records),
                        rows_per_pool={pool: int(hit.sum()) for pool, hit in duplicated.items()},
                        key='sha1 of the decoded picture and sha1 of the candidate caption',
                        pairs=records))
        index[name] = dict(rows=len(own), duplicated_rows=int(any_hit.sum()),
                           duplicated_pairs=len(records),
                           rows_per_pool={pool: int(hit.sum()) for pool, hit in duplicated.items()})
        print(name, index[name], flush=True)
    write_json(output / 'index.json', dict(pools=POOLS, evaluation_sets=EVAL_SETS, datasets=index))


if __name__ == '__main__':
    main(*sys.argv[1:2])
