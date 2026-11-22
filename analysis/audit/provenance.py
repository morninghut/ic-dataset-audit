"""Where does every Spica row come from: Polaris, Nebula, or neither?

Rows are matched on (picture, reference set, candidate caption), pictures by
decoded-pixel hash, so the match survives re-encoded image copies and the two
imgids the Spica release assigns to one picture.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, read_json, write_json
from csv_audit import load as load_spica, parse_refs
from label_leakage import nebula_rows


def polaris_rows():
    pixel = {r['name']: r.get('pixel_sha1') for r in read_json(OUT / 'inventory_polaris.json')}
    frames = []
    for split in ['train', 'val', 'test']:
        frame = pd.read_csv(DATA / f'polaris/polaris_{split}.csv', dtype={'imgid': str})
        frame['split'] = split
        frame['pixel'] = frame.imgid.map(pixel)
        frame['reference_key'] = frame['refs'].map(lambda v: '\u241f'.join(parse_refs(v)))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def keys(frame, score='score'):
    return list(zip(frame.pixel, frame.reference_key, frame.mt, frame[score].astype(float).round(6)))


def main():
    spica_pixel = {r['name']: r.get('pixel_sha1') for r in read_json(OUT / 'inventory_spica.json')}
    spica = {split: load_spica(split) for split in ['train', 'val', 'test']}
    for frame in spica.values():
        frame['pixel'] = frame.imgid.map(spica_pixel)

    polaris = polaris_rows()
    nebula = nebula_rows()

    # Multisets: a sample annotated k times must match k source rows, not one.
    polaris_counts = Counter(keys(polaris))
    polaris_split = {k[:3]: s for k, s in zip(keys(polaris), polaris.split)}
    nebula_counts = Counter(keys(nebula, 'human_score'))
    nebula_split = {k[:3]: s for k, s in zip(keys(nebula, 'human_score'), nebula.split)}

    report = dict(sources=dict(polaris_rows=int(len(polaris)), nebula_rows=int(len(nebula))), spica={})
    for split, frame in spica.items():
        remaining_polaris = Counter(polaris_counts)
        remaining_nebula = Counter(nebula_counts)
        labels = []
        for key in keys(frame):
            if remaining_polaris.get(key):
                remaining_polaris[key] -= 1
                labels.append('polaris_' + polaris_split[key[:3]])
            elif remaining_nebula.get(key):
                remaining_nebula[key] -= 1
                labels.append('nebula_' + nebula_split[key[:3]])
            elif frame.reference_count.iloc[len(labels)] == 0:
                labels.append('no_references')
            else:
                labels.append('unmatched')
        counts = Counter(labels)
        report['spica'][split] = dict(rows=int(len(frame)),
                                      **{k: int(v) for k, v in sorted(counts.items())},
                                      shares={k: round(v / len(frame), 4) for k, v in sorted(counts.items())})
        frame['provenance'] = labels
        print(split, report['spica'][split]['shares'])

    # Do Polaris test rows end up in the Spica training split?
    train = spica['train']
    for name in ['polaris_test', 'polaris_val', 'polaris_train', 'nebula_test']:
        report.setdefault('rows_of_source_inside_spica_train', {})[name] = int((train.provenance == name).sum())
    write_json(OUT / 'provenance.json', report)
    pd.DataFrame(dict(split='train', provenance=train.provenance)).to_csv(
        OUT / 'spica_train_provenance.csv.gz', index=False, compression='gzip')
    print(report['rows_of_source_inside_spica_train'])


if __name__ == '__main__':
    main()
