"""How each collection is built: images, image-caption pairs and individual ratings.

A "pair" is one image with one candidate caption; a "rating" is one stored human score (one CSV
row).  Polaris and Spica store every annotator's rating as its own row; Nebula stores one
averaged score per pair.  Images are counted by pixel identity (pearl_data/csv uses one file
name per distinct picture).

    structure.py        (run with PYTHONPATH=Polos-revised from analysis/summary)

Writes runs/summary/structure.json.
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(_ROOT)
sys.path.insert(0, str(ROOT / 'analysis/probe'))
from lib import CACHE, load_rows, picture_ids  # noqa: E402

DATA = ROOT / 'pearl_data'
OUT = ROOT / 'runs/summary'
csv.field_size_limit(10 ** 9)
SPLITS = ['polaris_train', 'polaris_val', 'polaris_test', 'nebula_train', 'nebula_valid', 'nebula_test',
          'spica_train', 'spica_val', 'spica_test', 'flickr8k_expert', 'flickr8k_cf', 'composite']


def read(split):
    """[(image, candidate, score, number of references)] of a split."""
    path = DATA / 'csv' / f'{split}.csv'
    if path.exists():
        import ast
        with open(path, newline='') as f:
            return [(r['imgid'], r['mt'], float(r['score']), len(ast.literal_eval(r['refs']))) for r in csv.DictReader(f)]
    rows = load_rows(split, CACHE)                      # polaris_train is not in Pearl's format
    pictures = picture_ids()[rows['image']]
    return [(int(p), int(c), float(t), int(n)) for p, c, t, n in zip(pictures, rows['candidate'], rows['target'], rows['counts'])]


def describe(rows):
    pairs = defaultdict(list)
    for image, candidate, score, _ in rows:
        pairs[image, candidate].append(score)
    sizes = np.array([len(v) for v in pairs.values()])
    scores = np.array([r[2] for r in rows])
    means = np.array([np.mean(v) for v in pairs.values()])
    per_image = Counter(image for image, _ in pairs)
    refs = np.array([r[3] for r in rows])
    return dict(ratings=len(rows), pairs=len(pairs), images=len(per_image),
                ratings_per_pair=float(sizes.mean()), pairs_with_one_rating=float((sizes == 1).mean()),
                max_ratings_per_pair=int(sizes.max()), pairs_per_image=float(np.mean(list(per_image.values()))),
                score_levels=int(len(np.unique(scores))), score_min=float(scores.min()), score_max=float(scores.max()),
                lowest_share_ratings=float((scores == scores.min()).mean()),
                lowest_share_pairs=float((means == scores.min()).mean()),
                references_mean=float(refs.mean()), without_references=float((refs == 0).mean()))


def main():
    out = {}
    data = {split: read(split) for split in SPLITS}
    for split in SPLITS:
        out[split] = describe(data[split])
    # The training sets after the duplicated pairs of every evaluation set are removed.
    keep = {'spica_train': np.load(DATA / 'spica_train_pairclean.npy'),
            'nebula_train': np.load(DATA / 'nebula_train_pairclean.npy')}
    for split, mask in keep.items():
        assert len(mask) == len(data[split])
        out[split]['deduplicated'] = describe([row for row, k in zip(data[split], mask) if k])
    clean = load_rows('polaris_train_pairclean', CACHE)
    pictures = picture_ids()[clean['image']]
    out['polaris_train']['deduplicated'] = describe(
        [(int(p), int(c), float(t), int(n)) for p, c, t, n in zip(pictures, clean['candidate'], clean['target'], clean['counts'])])
    # Spica test against Nebula test: the same images, Spica keeps the individual ratings.
    def table(split):
        pairs = defaultdict(list)
        for image, candidate, score, _ in data[split]:
            pairs[image, candidate].append(score)
        return pairs
    spica, nebula = table('spica_test'), table('nebula_test')
    common = set(spica) & set(nebula)
    same = sum(abs(np.mean(spica[k]) - nebula[k][0]) < 1e-6 for k in common)
    spica_images, nebula_images = {k[0] for k in spica}, {k[0] for k in nebula}
    out['spica_test_vs_nebula_test'] = dict(
        spica_pairs=len(spica), nebula_pairs=len(nebula), common_pairs=len(common), common_same_mean=int(same),
        spica_images=len(spica_images), nebula_images=len(nebula_images), common_images=len(spica_images & nebula_images),
        spica_ratings_on_common_pairs=int(sum(len(spica[k]) for k in common)),
        spica_pairs_only=len(set(spica) - set(nebula)),
        spica_only_single_rating=float(np.mean([len(spica[k]) == 1 for k in set(spica) - set(nebula)])),
        spica_only_lowest=float(np.mean([np.mean(spica[k]) == 0 for k in set(spica) - set(nebula)])))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'structure.json').write_text(json.dumps(out, indent=1))
    for split in SPLITS:
        e = out[split]
        print(f"{split:16s} ratings {e['ratings']:7d} pairs {e['pairs']:6d} images {e['images']:6d} ratings/pair {e['ratings_per_pair']:.2f} "
              f"single {e['pairs_with_one_rating']:.2f} levels {e['score_levels']:3d} lowest {e['lowest_share_ratings']:.3f} refs {e['references_mean']:.2f} noref {e['without_references']:.2f}"
              + (f" | dedup ratings {e['deduplicated']['ratings']} pairs {e['deduplicated']['pairs']} images {e['deduplicated']['images']}" if 'deduplicated' in e else ''))
    print(out['spica_test_vs_nebula_test'])


if __name__ == '__main__':
    main()
