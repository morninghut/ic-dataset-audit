"""Does the Spica training set contain the evaluation samples themselves?

Image overlap alone only changes the domain a model has seen. The decisive
question is whether the same (picture, candidate caption) pair -- and its human
score -- is also in training. Pictures are matched by decoded-pixel hash, so the
two imgids the release uses for one picture collapse into one key.
"""
import ast
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NEBULA, OUT, read_json, write_json
from csv_audit import load as load_spica, parse_refs


def spica_pixel_hash():
    return {r['name']: r.get('pixel_sha1') for r in read_json(OUT / 'inventory_spica.json')}


def nebula_rows():
    inventory = {(r['split'], r['name']): r.get('pixel_sha1')
                 for r in read_json(OUT / 'inventory_nebula.json')}
    frames = []
    for path in sorted((NEBULA / 'data').glob('*.parquet')):
        split = path.name.split('-')[0]
        table = pq.read_table(path, columns=['file_name', 'refs', 'mt', 'human_score'])
        frame = table.to_pandas()
        frame['split'] = split
        frame['pixel'] = [inventory.get((split, n)) for n in frame.file_name]
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame['reference_list'] = frame['refs'].map(
        lambda v: list(v) if isinstance(v, (list, np.ndarray)) else parse_refs(v))
    frame['reference_key'] = frame['reference_list'].map(lambda r: '\u241f'.join(r))
    return frame


def main():
    pixel = spica_pixel_hash()
    spica = {split: load_spica(split) for split in ['train', 'val', 'test']}
    for frame in spica.values():
        frame['pixel'] = frame.imgid.map(pixel)

    train = spica['train']
    train_images = set(train.pixel)
    train_pairs = defaultdict(list)  # (picture, candidate) -> scores seen in training
    for p, mt, score in zip(train.pixel, train.mt, train.score):
        train_pairs[(p, mt)].append(float(score))
    train_triples = {(p, k, mt) for p, k, mt in zip(train.pixel, train.reference_key, train.mt)}

    report = dict(train_rows=int(len(train)), train_pictures=len(train_images),
                  train_pairs=len(train_pairs), evaluation_sets={})

    def score_set(name, pixels, candidates, reference_keys, targets):
        pairs = list(zip(pixels, candidates))
        image_hit = np.array([p in train_images for p in pixels])
        pair_hit = np.array([pair in train_pairs for pair in pairs])
        triple_hit = np.array([(p, k, c) in train_triples
                               for p, k, c in zip(pixels, reference_keys, candidates)])
        targets = np.asarray(targets, dtype=float)
        copy = np.array([np.mean(train_pairs[pair]) if hit else np.nan
                         for pair, hit in zip(pairs, pair_hit)])
        agree = np.isclose(copy[pair_hit], targets[pair_hit])
        entry = dict(
            rows=len(pairs), pictures=len(set(pixels)),
            image_in_train_rows=int(image_hit.sum()), image_in_train_row_share=round(float(image_hit.mean()), 4),
            image_in_train_pictures=len(set(pixels) & train_images),
            pair_in_train_rows=int(pair_hit.sum()), pair_in_train_row_share=round(float(pair_hit.mean()), 4),
            triple_in_train_rows=int(triple_hit.sum()), triple_in_train_row_share=round(float(triple_hit.mean()), 4),
            identical_score_share_of_matched_pairs=round(float(agree.mean()), 4) if agree.size else None,
            mean_abs_score_difference_of_matched_pairs=round(float(np.nanmean(np.abs(copy[pair_hit] - targets[pair_hit]))), 4)
            if pair_hit.any() else None)
        report['evaluation_sets'][name] = entry
        print(f"{name:14s} rows {entry['rows']:6d} image {entry['image_in_train_row_share']:6.1%} "
              f"pair {entry['pair_in_train_row_share']:6.1%} pair+refs {entry['triple_in_train_row_share']:6.1%} "
              f"same score {entry['identical_score_share_of_matched_pairs']}")

    for split in ['val', 'test']:
        frame = spica[split]
        score_set(f'spica_{split}', list(frame.pixel), list(frame.mt),
                  list(frame.reference_key), frame.score)

    nebula = nebula_rows()
    for split, frame in nebula.groupby('split'):
        score_set(f'nebula_{split}', list(frame.pixel), list(frame.mt),
                  list(frame.reference_key), frame.human_score)

    # Is the Nebula score simply the mean of the Spica rows for the same sample?
    spica_all = pd.concat(spica.values(), ignore_index=True)
    group = spica_all.groupby(['pixel', 'mt'])['score'].agg(['mean', 'size'])
    merged = nebula.join(group, on=['pixel', 'mt'])
    matched = merged.dropna(subset=['mean'])
    report['nebula_vs_spica_scores'] = dict(
        nebula_rows=int(len(nebula)), matched_rows=int(len(matched)),
        identical_share=round(float(np.isclose(matched['mean'], matched.human_score).mean()), 4) if len(matched) else None,
        mean_abs_difference=round(float((matched['mean'] - matched.human_score).abs().mean()), 4) if len(matched) else None,
        annotations_per_matched_sample=round(float(matched['size'].mean()), 2) if len(matched) else None)
    print('nebula vs spica group mean', report['nebula_vs_spica_scores'])
    write_json(OUT / 'label_leakage.json', report)


if __name__ == '__main__':
    main()
