"""Score every evaluation split with a trained DENEB checkpoint.

DENEB reads the image itself (its frozen PAC-S CLIP encodes it) and tokenizes
the candidate and every reference, so scoring needs only the shared picture
symlinks in pearl_data/images.  Its collate transposes the reference lists
(torchnlp.collate_tensors), which silently truncates a batch to the smallest
reference count in it, so rows are grouped by reference count before batching:
every row is then scored with all of its own references.  Rows that repeat a
(picture, candidate, references) triple, as the per-annotator rows of Flickr8k-Expert,
Polaris and Spica do, are scored once.  Pictures are decoded by a thread pool; the
model's own preprocessing is untouched.

    score_deneb.py <checkpoint> <out dir> [splits...]

Saved per split: <split>.npz with score, in CSV order.  A split whose file exists is skipped.
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import ast
import csv
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(_ROOT)
DENEB = ROOT / 'external/DENEB'
DATA = ROOT / 'pearl_data'
SPLITS = ['spica_test', 'polaris_test', 'polaris_val', 'nebula_test', 'nebula_valid',
          'flickr8k_expert', 'flickr8k_cf', 'composite', 'pascal50s', 'foil_1ref', 'foil_4ref',
          'spica_val']

AHEAD = 4

csv.field_size_limit(10 ** 9)
sys.path.insert(0, str(DENEB))


def read_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return [dict(imgid=r['imgid'], mt=str(r['mt']), refs=ast.literal_eval(r['refs'])) for r in rows]


def main(checkpoint, out_dir, *splits):
    from deneb.models import load_checkpoint

    splits = list(splits) or SPLITS
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(checkpoint).cuda().eval()

    for split in splits:
        if (out / f'{split}.npz').exists():   # a rerun after a failure continues where it stopped
            continue
        started = time.time()
        rows = read_csv(DATA / 'csv' / f'{split}.csv')
        scores = np.full(len(rows), np.nan, np.float32)
        first, owner = {}, np.empty(len(rows), np.int64)
        for index, row in enumerate(rows):
            owner[index] = first.setdefault((row['imgid'], row['mt'], tuple(row['refs'])), index)
        groups = {}
        for index in sorted(first.values()):
            groups.setdefault(len(rows[index]['refs']), []).append(index)
        batches = []
        for count, indices in sorted(groups.items()):
            indices.sort(key=lambda i: rows[i]['imgid'])
            batch_size = max(1, 256 // max(count, 1))
            batches += [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]

        def load(chunk):
            return [Image.open(DATA / 'images' / rows[i]['imgid']).convert('RGB') for i in chunk]

        with ThreadPoolExecutor(8) as pool:
            ahead = deque()                      # at most AHEAD decoded batches in memory
            for position in range(len(batches) + AHEAD):
                if position < len(batches):
                    ahead.append((batches[position], pool.submit(load, batches[position])))
                if position < AHEAD - 1 or not ahead:
                    continue
                chunk, future = ahead.popleft()
                images = future.result()
                samples = [dict(mt=rows[i]['mt'], refs=rows[i]['refs'] or ['NO REFS!!!'], img=image)
                           for i, image in zip(chunk, images)]
                # The released predict() returns a bare float for a batch of one and then fails
                # on len(); a lone sample is scored twice and the first value kept.
                alone = len(samples) == 1
                with torch.no_grad():
                    _, values = model.predict(samples * 2 if alone else samples, cuda=True,
                                              batch_size=max(2, len(samples)), show_progress=False)
                scores[chunk] = np.asarray(values, np.float32).reshape(-1)[:len(samples)]
        scores = scores[owner]
        assert np.isfinite(scores).all()
        np.savez(out / f'{split}.npz', score=scores)
        print(split, len(rows), 'rows', len(first), 'scored', f'{time.time() - started:.0f}s', flush=True)

    with open(out / f"scoring_{'_'.join(splits)[:80]}.json", 'w') as f:
        json.dump(dict(checkpoint=str(checkpoint), splits=splits), f)


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
