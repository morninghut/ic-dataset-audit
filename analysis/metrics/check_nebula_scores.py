"""Look at the rows where the cache's Nebula score differs from the parquet."""
import ast
import csv
import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

csv.field_size_limit(10 ** 9)


def released(split):
    return pd.concat([pq.read_table(p, columns=['file_name', 'refs', 'mt', 'human_score']).to_pandas()
                      for p in sorted(glob.glob(f'datasets_nebula/data/{split}-*.parquet'))], ignore_index=True)


def ours(name):
    with open(f'external/DENEB/data_en/nebula/{name}.csv', newline='') as f:
        return list(csv.DictReader(f))


them, us = released('train'), ours('train')
a = them['human_score'].to_numpy(float)
b = np.array([float(r['score']) for r in us])
d = np.abs(a - b)
bad = np.flatnonzero(d > 1e-9)
print('rows', len(a), 'differing', len(bad), 'max |diff|', d.max(), 'mean |diff| on differing', d[bad].mean())
print('released distinct values', len(np.unique(a)), 'ours', len(np.unique(b)))
print('released head', np.unique(a)[:10])
print('ours head', np.unique(b)[:10])
for i in bad[:10]:
    print(i, 'released', a[i], 'ours', b[i], '|', us[i]['mt'][:60])
# Is ours a rounding of theirs, or theirs of ours?
print('ours == round(released,2)?', np.allclose(b, np.round(a, 2)))
print('ours == released rounded to 1/4?', np.allclose(b, np.round(a * 4) / 4))
