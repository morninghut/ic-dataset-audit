"""How many duplicated judgments does exact matching miss?

Everywhere else a picture is the hash of its decoded pixels and a candidate is its exact
text, so the duplication counts are lower bounds.  This script bounds what is missed, per
training pool and evaluation set, in two steps.

Text.  Among the rows whose picture is in the pool with other candidates only
("picture only"), how many have a pool candidate on the same picture that is the same after
normalization (case, punctuation, whitespace), or nearly the same (character similarity of
the normalized strings at least 0.9)?

Pictures.  Among the rows whose picture is not in the pool ("absent"), how many pictures
are the same photograph stored as different pixels?  The test is the cosine of the SigLIP
image features to the pool pictures.  Looking at pairs (Spica train against Spica test):
at 0.99 and above the two files are the same photograph with another compression, between
0.97 and 0.99 they are that or another shot of the same scene, below 0.97 other
photographs.  So 0.99 is reported as "re-encoded copy" and 0.97 as the generous bound, and
the rows on those pictures are classified against the matching pool pictures like any other
row.  (Sharing a reference caption is not a usable test: short references such as "a man
riding a wave on a surfboard" recur across photographs.)

    near_duplicates.py

Run with PYTHONPATH=Polos-revised.  Writes runs/metrics/near_duplicates.json.
"""
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probe'))
from lib import CACHE, ROOT, load_rows, picture_ids, read_json, write_json  # noqa: E402
import prep_pearl_data  # noqa: E402

POOLS = ['polaris_train', 'nebula_train', 'spica_train']
SETS = ['polaris_test', 'polaris_val', 'nebula_test', 'nebula_valid', 'spica_test', 'spica_val',
        'composite', 'flickr8k_expert', 'flickr8k_cf', 'pascal50s', 'foil_1ref']
NEAR = 0.9
COSINES = [0.99, 0.97]      # re-encoded copy; generous bound
_strip = re.compile(r'[^a-z0-9 ]+')


def normalize(text):
    return ' '.join(_strip.sub(' ', text.lower()).split())


def rows_of(name):
    """The released Spica pool includes its reference-less rows; the cache keeps them apart."""
    return prep_pearl_data.spica_train_rows() if name == 'spica_train' else load_rows(name, CACHE)


def main():
    catalog = read_json(CACHE / 'catalog.json')
    texts = catalog['texts']
    pictures_of_image = picture_ids()
    # normalized text id: equal ids mean equal text after normalization
    table, normalized = {}, np.empty(len(texts), np.int64)
    strings = []
    for index, text in enumerate(texts):
        key = normalize(text)
        if key not in table:
            table[key] = len(strings)
            strings.append(key)
        normalized[index] = table[key]
    print(f'{len(texts)} texts, {len(strings)} after normalization', flush=True)

    image_features = np.load(CACHE / 'siglip_image.npy', mmap_mode='r')
    first_image = np.full(pictures_of_image.max() + 1, -1)
    first_image[pictures_of_image[::-1]] = np.arange(len(pictures_of_image))[::-1]

    def features(pictures):
        block = np.asarray(image_features[first_image[pictures]], np.float32)
        return block / np.linalg.norm(block, axis=1, keepdims=True)

    out = {}
    for pool in POOLS:
        pool_rows = rows_of(pool)
        pool_pictures = pictures_of_image[pool_rows['image']]
        exact, by_picture = set(), defaultdict(set)
        for picture, candidate in zip(pool_pictures.tolist(), pool_rows['candidate'].tolist()):
            exact.add((picture, candidate))
            by_picture[picture].add(int(normalized[candidate]))
        pool_unique = np.array(sorted(by_picture))
        pool_block = features(pool_unique)
        print(f'\n## pool {pool}: {len(pool_pictures)} rows, {len(pool_unique)} pictures', flush=True)
        print(f"{'set':16s} {'rows':>7s} {'pair':>7s} {'pic only':>8s} {'absent':>7s} | "
              f"{'+norm':>6s} {'+near':>6s} | {'absent pict.':>12s} | "
              + ' | '.join(f"{'>=%.2f: pict.' % c:>13s} {'rows':>6s} {'+pair':>6s} {'+norm':>6s}" for c in COSINES)
              + f" | {'pair, bound':>13s}")
        out[pool] = {}
        for name in SETS:
            if name == pool:
                continue
            rows = load_rows(name, CACHE)
            pictures = pictures_of_image[rows['image']]
            candidates = rows['candidate']
            in_pool = np.isin(pictures, pool_unique)
            pair = np.array([(p, c) in exact for p, c in zip(pictures.tolist(), candidates.tolist())])
            only = in_pool & ~pair
            # text: picture-only rows against the pool's candidates on the same picture
            same, near = 0, 0
            for index in np.flatnonzero(only):
                mine = int(normalized[candidates[index]])
                others = by_picture[int(pictures[index])]
                if mine in others:
                    same += 1
                elif any(abs(len(strings[mine]) - len(strings[o])) <= 0.25 * len(strings[mine])
                         and SequenceMatcher(None, strings[mine], strings[o], autojunk=False).ratio() >= NEAR
                         for o in others):
                    near += 1
            # pictures: absent ones whose features match a pool picture
            absent_pictures = np.unique(pictures[~in_pool])
            position = {picture: index for index, picture in enumerate(absent_pictures.tolist())}
            twins = {c: {} for c in COSINES}      # threshold -> absent picture -> matching pool pictures
            if len(absent_pictures):
                block = features(absent_pictures)
                for start in range(0, len(block), 2048):
                    similarity = block[start:start + 2048] @ pool_block.T
                    for c in COSINES:
                        for row, column in zip(*np.nonzero(similarity >= c)):
                            twins[c].setdefault(int(absent_pictures[start + row]), set()).add(int(pool_unique[column]))
            copies = {}
            for c in COSINES:
                on_twin = hidden_pair = hidden_norm = 0
                for index in np.flatnonzero(~in_pool):
                    hits = twins[c].get(int(pictures[index]))
                    if not hits:
                        continue
                    on_twin += 1
                    if any((hit, int(candidates[index])) in exact for hit in hits):
                        hidden_pair += 1
                    elif any(int(normalized[candidates[index]]) in by_picture[hit] for hit in hits):
                        hidden_norm += 1
                copies[f'{c:.2f}'] = dict(pictures=len(twins[c]), rows=on_twin, pair=hidden_pair,
                                          pair_after_normalization=hidden_norm)
            loose = copies[f'{COSINES[-1]:.2f}']
            bound = int(pair.sum()) + same + near + loose['pair'] + loose['pair_after_normalization']
            entry = dict(rows=int(len(pictures)), pair=int(pair.sum()), picture_only=int(only.sum()),
                         absent=int((~in_pool).sum()), picture_only_same_after_normalization=same,
                         picture_only_near=near, absent_pictures=int(len(absent_pictures)),
                         absent_pictures_matching_a_pool_picture=copies, pair_upper=bound)
            out[pool][name] = entry
            print(f"{name:16s} {entry['rows']:7d} {entry['pair']:7d} {entry['picture_only']:8d} {entry['absent']:7d} | "
                  f"{same:6d} {near:6d} | {len(absent_pictures):12d} | "
                  + ' | '.join(f"{v['pictures']:13d} {v['rows']:6d} {v['pair']:6d} {v['pair_after_normalization']:6d}"
                               for v in copies.values())
                  + f" | {bound:6d} {100 * bound / len(pictures):5.1f}%", flush=True)
    write_json(ROOT / 'runs/metrics/near_duplicates.json', dict(near=NEAR, cosines=COSINES, pools=out))


if __name__ == '__main__':
    main()
