"""A fourth Pearl training pool: a control matched to what pairclean removes.

pairclean removes the rows whose (picture, candidate) pair occurs in an evaluation set.
Those rows are not a random part of the Spica pool: all of them carry references, almost
all belong to a pair judged by several annotators, and their scores avoid the floor.
randomdrop, which removes the same number of rows uniformly, therefore does not control
for losing that kind of data.  matcheddrop does: it removes whole pairs that occur in no
evaluation set, as many per stratum as pairclean removes, where a stratum is

    (has references, number of rows of the pair, mean score of the pair in five bins)

and then pads or trims to exactly the number of rows pairclean removes.

Written:
    pearl_data/<pool>_matcheddrop.npy              keep mask, like the other variants
    pearl_data/matcheddrop.json                    how well the removed rows match
    external/Pearl/data/<pool>_matcheddrop.csv
    external/Pearl/configs/variants/matcheddrop.yaml
"""
import json
import sys

import numpy as np

import prep_pearl_data as data
import prep_pearl_run as run

SEED = 7
POOLS = ['spica_train', 'spica_val']
SIZE_CAP = 10
SCORE_EDGES = [0.2, 0.4, 0.6, 0.8]


def profile(rows, pictures, pair_size, mask):
    target = rows['target'][mask].astype(np.float64)
    return dict(rows=int(mask.sum()), with_references=float(np.mean(rows['counts'][mask] > 0)),
                references=float(rows['counts'][mask].mean()), score_mean=float(target.mean()),
                score_sd=float(target.std()), at_floor=float(np.mean(target <= rows['target'].min())),
                score_histogram=(np.histogram(target, [0, .01, .2, .4, .6, .8, 1.01])[0] / mask.sum()).round(4).tolist(),
                pictures=int(len(np.unique(pictures[mask]))),
                in_repeated_pair=float(np.mean(pair_size[mask] > 1)))


def matched(rows, pictures, leaked, rng):
    """Boolean mask of the rows to remove instead of the leaked ones."""
    keys = pictures.astype(np.int64) * 1000000 + rows['candidate'].astype(np.int64)
    _, first, pair, size = np.unique(keys, return_index=True, return_inverse=True, return_counts=True)
    pair = pair.ravel()
    mean = np.bincount(pair, weights=rows['target'].astype(np.float64)) / size
    pair_leaked = leaked[first]
    if not np.array_equal(pair_leaked[pair], leaked):
        raise ValueError('a pair is only partly leaked')
    stratum = ((rows['counts'][first] > 0).astype(np.int64) * 1000 + np.minimum(size, SIZE_CAP) * 10
               + np.digitize(mean, SCORE_EDGES))
    chosen = np.zeros(len(size), bool)
    short = 0
    for value in np.unique(stratum[pair_leaked]):
        wanted = int((pair_leaked & (stratum == value)).sum())
        available = np.flatnonzero(~pair_leaked & (stratum == value))
        take = min(wanted, len(available))
        short += wanted - take
        chosen[rng.choice(available, take, replace=False)] = True
    goal = int(leaked.sum())
    # Pad with other referenced pairs when a stratum ran out; trim single rows when over.
    spare = rng.permutation(np.flatnonzero(~pair_leaked & ~chosen & (rows['counts'][first] > 0)))
    position = 0
    while size[chosen].sum() < goal:
        chosen[spare[position]] = True
        position += 1
    remove = chosen[pair]
    extra = int(remove.sum()) - goal
    if extra:
        remove[rng.choice(np.flatnonzero(remove), extra, replace=False)] = False
    return remove, dict(pairs_removed_by_pairclean=int(pair_leaked.sum()), pairs_removed=int(chosen.sum()),
                        pairs_missing_in_their_stratum=short, pairs_padded=position, rows_trimmed=extra,
                        size=size[pair])


def main():
    picture_of_image = np.load(data.CACHE / 'picture_ids.npy')
    report = {}
    for pool in POOLS:
        rows = data.rows_of(pool)
        pictures = picture_of_image[rows['image']]
        leaked = ~np.load(data.OUT / f'{pool}_pairclean.npy')
        random = ~np.load(data.OUT / f'{pool}_randomdrop.npy')
        remove, counts = matched(rows, pictures, leaked, np.random.default_rng(SEED))
        size = counts.pop('size')
        if (remove & leaked).any() or remove.sum() != leaked.sum():
            raise ValueError('the control overlaps the leaked rows or has another size')
        np.save(data.OUT / f'{pool}_matcheddrop.npy', ~remove)
        written = run.filtered_csv(data.OUT / 'csv' / f'{pool}.csv', run.PEARL / 'data' / f'{pool}_matcheddrop.csv',
                                   ~remove)
        report[pool] = dict(counts, written=written,
                            removed_by={'pairclean': profile(rows, pictures, size, leaked),
                                        'matcheddrop': profile(rows, pictures, size, remove),
                                        'randomdrop': profile(rows, pictures, size, random)},
                            pool=profile(rows, pictures, size, np.ones(len(leaked), bool)))
        print(pool, json.dumps(report[pool], indent=1), flush=True)
    data.write_json(data.OUT / 'matcheddrop.json', dict(seed=SEED, pools=report))

    text = (run.PEARL / 'configs/variants/randomdrop.yaml').read_text()
    if text.count('_randomdrop.csv') != 2:
        raise ValueError('unexpected randomdrop configuration')
    path = run.PEARL / 'configs/variants/matcheddrop.yaml'
    path.write_text(text.replace('_randomdrop.csv', '_matcheddrop.csv'))
    print('wrote', path)


if __name__ == '__main__':
    sys.exit(main())
