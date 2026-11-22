"""Audit the released Spica CSVs: size, composition, split leakage and label structure.

Everything here is text-level only (imgid strings and reference sentences);
image-content matching lives in image_inventory.py / overlap.py.
"""
import ast
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, SPICA, write_json

SPLITS = ['train', 'val', 'test']


def parse_refs(value):
    """Spica stores reference lists as repr strings; 38,670 train rows are double quoted."""
    if not isinstance(value, str):
        return []
    text = value.strip()
    for _ in range(2):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return [str(s) for s in parsed]
        text = str(parsed)
    return []


def load(split):
    frame = pd.read_csv(SPICA / f'spica_{split}.csv', dtype={'imgid': str})
    frame['split'] = split
    frame['reference_list'] = frame['refs'].map(parse_refs)
    frame['reference_count'] = frame['reference_list'].map(len)
    frame['reference_key'] = frame['reference_list'].map(lambda r: '\u241f'.join(r))
    frame['malformed_refs'] = frame['refs'].astype(str).str.startswith('"')
    return frame


def label_structure(frame):
    """Rows sharing one (image, candidate) pair are repeated annotations of one sample."""
    with_ref = frame[frame.reference_count > 0]
    groups = with_ref.groupby(['imgid', 'mt'])['score']
    sizes = groups.transform('size')
    singleton = sizes == 1
    zero = with_ref.score == 0
    out = dict(
        rows_with_references=int(len(with_ref)),
        samples=int(groups.ngroups),
        score_histogram={str(k): int(v) for k, v in sorted(Counter(frame.score).items())},
        reference_count_histogram={str(k): int(v) for k, v in sorted(Counter(frame.reference_count).items())},
        annotations_per_sample_histogram={str(k): int(v) for k, v in sorted(Counter(groups.size()).items())},
        singleton_row_share=float(singleton.mean()),
        singleton_zero_share=float((singleton & zero).mean()),
        zero_share_in_singletons=float(zero[singleton].mean()),
        zero_share_in_multi=float(zero[~singleton].mean()),
        mean_score=float(frame.score.mean()),
    )
    # Oracle: how far does "is this sample annotated once?" alone get on this split.
    y = with_ref.score.to_numpy(float)
    oracle = (~singleton).to_numpy(float)
    out['singleton_oracle_kendall_tau_c'] = float(kendalltau(oracle, y, variant='c')[0])
    out['singleton_oracle_pearson'] = float(pearsonr(oracle, y)[0])
    # Human ceiling: leave-one-out mean of the other annotators of the same sample.
    totals = groups.transform('sum').to_numpy(float)
    counts = sizes.to_numpy(float)
    multi = counts > 1
    loo = (totals[multi] - y[multi]) / (counts[multi] - 1)
    out['human_leave_one_out'] = dict(
        rows=int(multi.sum()),
        kendall_tau_c=float(kendalltau(loo, y[multi], variant='c')[0]),
        pearson=float(pearsonr(loo, y[multi])[0]))
    group_mean = (totals / counts)
    out['group_mean_oracle'] = dict(
        kendall_tau_c_all=float(kendalltau(group_mean, y, variant='c')[0]),
        kendall_tau_c_multi=float(kendalltau(group_mean[multi], y[multi], variant='c')[0]))
    return out


def main():
    frames = {split: load(split) for split in SPLITS}
    report = dict(splits={}, composition={}, leakage={}, provenance={})

    for split, frame in frames.items():
        report['splits'][split] = dict(
            rows=int(len(frame)), images=int(frame.imgid.nunique()),
            rows_without_references=int((frame.reference_count == 0).sum()),
            rows_with_malformed_refs=int(frame.malformed_refs.sum()),
            images_without_references=int(frame.loc[frame.reference_count == 0, 'imgid'].nunique()),
            mean_score_without_references=float(frame.loc[frame.reference_count == 0, 'score'].mean())
            if (frame.reference_count == 0).any() else None,
            **label_structure(frame))

    # Provenance: which Polaris split does each Spica image come from, by imgid string.
    polaris = {split: set(pd.read_csv(DATA / f'polaris/polaris_{split}.csv', dtype=str).imgid)
               for split in SPLITS}
    for split, frame in frames.items():
        images = set(frame.imgid)
        report['provenance'][f'spica_{split}'] = dict(
            images=len(images),
            **{f'in_polaris_{name}': len(images & ids) for name, ids in polaris.items()},
            in_no_polaris_split=len(images - set().union(*polaris.values())))
    for name, ids in polaris.items():
        report['provenance'][f'polaris_{name}'] = dict(
            images=len(ids),
            **{f'in_spica_{split}': len(ids & set(frames[split].imgid)) for split in SPLITS},
            in_no_spica_split=len(ids - set().union(*[set(frames[s].imgid) for s in SPLITS])))

    # Split leakage by imgid and by reference set.
    for a in SPLITS:
        for b in SPLITS:
            if a >= b:
                continue
            report['leakage'][f'{a}_vs_{b}'] = dict(
                shared_imgids=len(set(frames[a].imgid) & set(frames[b].imgid)),
                shared_reference_keys=len({k for k in frames[a].reference_key if k} &
                                          {k for k in frames[b].reference_key if k}))

    # Reference sets that appear under more than one imgid (candidate duplicate pictures).
    everything = pd.concat(frames.values(), ignore_index=True)
    with_ref = everything[everything.reference_count > 0]
    by_key = defaultdict(set)
    for key, imgid in zip(with_ref.reference_key, with_ref.imgid):
        by_key[key].add(imgid)
    multi = {k: sorted(v) for k, v in by_key.items() if len(v) > 1}
    split_of = defaultdict(set)
    for split, frame in frames.items():
        for imgid in set(frame.imgid):
            split_of[imgid].add(split)
    pair_counts = Counter()
    for imgids in multi.values():
        splits = sorted(set().union(*[split_of[i] for i in imgids]))
        pair_counts['+'.join(splits)] += 1
    report['reference_key_collisions'] = dict(
        reference_keys_with_multiple_imgids=len(multi),
        imgid_count_histogram={str(k): int(v) for k, v in sorted(Counter(len(v) for v in multi.values()).items())},
        split_pattern=dict(sorted(pair_counts.items())),
        examples=[dict(imgids=v, references=k.split('\u241f')[:2]) for k, v in list(multi.items())[:5]])
    write_json(OUT / 'collision_imgids.json', {k[:200]: v for k, v in multi.items()})

    report['composition'] = dict(
        total_rows=int(len(everything)), total_images=int(everything.imgid.nunique()),
        rows_without_references=int((everything.reference_count == 0).sum()))
    write_json(OUT / 'csv_audit.json', report)
    print(f"rows {report['composition']['total_rows']} images {report['composition']['total_images']}")
    print('reference-set collisions', report['reference_key_collisions']['reference_keys_with_multiple_imgids'],
          report['reference_key_collisions']['split_pattern'])


if __name__ == '__main__':
    main()
