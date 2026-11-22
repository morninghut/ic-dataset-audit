"""Row types inside Spica: pseudo-labelled rows without references, and
"wrong-image" rows whose candidate caption belongs to a different picture.

The wrong-image rule is the one used earlier on Polaris: the candidate string
occurs as a reference sentence of another picture in the same release. Such rows
are trivially separable without looking at the image, so they inflate any
correlation reported on the split.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, read_json, write_json
from csv_audit import load as load_spica


def main():
    pixel = {r['name']: r.get('pixel_sha1') for r in read_json(OUT / 'inventory_spica.json')}
    frames = {split: load_spica(split) for split in ['train', 'val', 'test']}
    for frame in frames.values():
        frame['pixel'] = frame.imgid.map(pixel)

    # Reference sentence -> set of pictures it describes, over the whole release.
    owner = defaultdict(set)
    for frame in frames.values():
        for p, refs in zip(frame.pixel, frame.reference_list):
            for sentence in refs:
                owner[sentence.strip()].add(p)

    report = dict(reference_sentences=len(owner), splits={})
    for split, frame in frames.items():
        with_ref = frame[frame.reference_count > 0]
        pictures = with_ref.pixel.to_numpy()
        candidates = [c.strip() if isinstance(c, str) else '' for c in with_ref.mt]
        wrong = np.array([bool(owner.get(c)) and p not in owner[c] for c, p in zip(candidates, pictures)])
        own = np.array([p in owner.get(c, ()) for c, p in zip(candidates, pictures)])
        y = with_ref.score.to_numpy(float)
        entry = dict(
            rows_with_references=int(len(with_ref)),
            wrong_image_rows=int(wrong.sum()), wrong_image_share=round(float(wrong.mean()), 4),
            candidate_is_own_reference_rows=int(own.sum()),
            candidate_is_own_reference_share=round(float(own.mean()), 4),
            mean_score_wrong_image=round(float(y[wrong].mean()), 4) if wrong.any() else None,
            mean_score_own_reference=round(float(y[own].mean()), 4) if own.any() else None,
            mean_score_other=round(float(y[~wrong & ~own].mean()), 4),
            zero_share_wrong_image=round(float((y[wrong] == 0).mean()), 4) if wrong.any() else None,
            detector_kendall_tau_c=round(float(kendalltau((~wrong).astype(float) + own, y, variant='c')[0]), 4),
            kendall_tau_c_on_remaining_rows=None)
        rest = ~wrong & ~own
        if rest.sum() > 100:
            entry['remaining_rows'] = int(rest.sum())
            entry['remaining_mean_score'] = round(float(y[rest].mean()), 4)
        report['splits'][split] = entry
        print(split, {k: entry[k] for k in ['wrong_image_share', 'candidate_is_own_reference_share',
                                            'mean_score_wrong_image', 'mean_score_own_reference',
                                            'mean_score_other', 'detector_kendall_tau_c']})

    # The half of the training split that carries no references at all.
    train = frames['train']
    no_ref = train[train.reference_count == 0]
    no_ref_pictures = set(no_ref.pixel)
    with_ref_pictures = set(train.loc[train.reference_count > 0, 'pixel'])
    coco = {r.get('pixel_sha1') for r in read_json(OUT / 'inventory_coco_val2014.json')}
    report['no_reference_rows'] = dict(
        rows=int(len(no_ref)), pictures=len(no_ref_pictures),
        mean_score=round(float(no_ref.score.mean()), 4),
        score_histogram={str(k): int(v) for k, v in sorted(no_ref.score.value_counts().items())},
        pictures_shared_with_reference_rows=len(no_ref_pictures & with_ref_pictures),
        pictures_in_coco_val2014=len(no_ref_pictures & coco),
        example_candidates=[str(c) for c in no_ref.mt.head(5)])
    print('no-reference rows', report['no_reference_rows'])
    write_json(OUT / 'row_types.json', report)


if __name__ == '__main__':
    main()
