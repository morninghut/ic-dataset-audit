"""Check the Nebula CSVs written for DENEB against the released parquet files.

Compares candidate, references and score row by row in released order, and
checks that rows sharing a picture identity in the CSV also share the released
file_name (and the other way round), so the interned pictures are the released
pictures.
"""
import ast
import csv
import glob
import sys
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq

csv.field_size_limit(10 ** 9)
SPLITS = {'train': 'train', 'validation': 'valid', 'test': 'test'}
DENEB = 'external/DENEB/data_en/nebula'


def released(split):
    frames = [pq.read_table(path, columns=['file_name', 'refs', 'mt', 'human_score']).to_pandas()
              for path in sorted(glob.glob(f'datasets_nebula/data/{split}-*.parquet'))]
    return pd.concat(frames, ignore_index=True)


def ours(name):
    with open(f'{DENEB}/{name}.csv', newline='') as f:
        return list(csv.DictReader(f))


def main():
    ok = True
    for name, split in SPLITS.items():
        them, us = released(split), ours(name)
        if len(them) != len(us):
            print(name, 'LENGTH', len(them), len(us))
            ok = False
            continue
        bad_text = bad_refs = bad_score = 0
        pictures, files = defaultdict(set), defaultdict(set)
        for i in range(len(us)):
            if them['mt'][i] != us[i]['mt']:
                bad_text += 1
            if list(them['refs'][i]) != ast.literal_eval(us[i]['refs']):
                bad_refs += 1
            # The cache stores scores as float32, so compare at float32 precision.
            if abs(float(them['human_score'][i]) - float(us[i]['score'])) > 1e-6:
                bad_score += 1
            pictures[us[i]['imgid']].add(them['file_name'][i])
            files[them['file_name'][i]].add(us[i]['imgid'])
        split_ok = not (bad_text or bad_refs or bad_score)
        many = sum(1 for v in pictures.values() if len(v) > 1), sum(1 for v in files.values() if len(v) > 1)
        print(name, len(us), 'mt/refs/score mismatches', bad_text, bad_refs, bad_score,
              'pictures', len(pictures), 'file_names', len(files),
              'picture->several file_names', many[0], 'file_name->several pictures', many[1])
        ok = ok and split_ok and many[1] == 0
    print('OK' if ok else 'MISMATCH')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
