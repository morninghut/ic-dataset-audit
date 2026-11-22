"""Score every evaluation split with the released Rigel checkpoint.

Rigel (Koyama et al., EMNLP 2026, arXiv:2606.29997) is Qwen3-VL-2B with LoRA adapters and a
five-class scoring head, fine-tuned on the Spica training set (and Vid-Lepus for video).  Only
inference code and one checkpoint are public, so it enters the study as released: clean on the
evaluation sets the Spica pool shares no judged pair with, "+dup" on the others.

    score_rigel.py <free|refs> [batch size] [splits...]

Everything that decides a score is the authors': the model class, prompt, picture budget
(config.yaml of the checkpoint) and the forward pass come from external/Rigel (commit 72394e2)
through rigel.from_pretrained(), pinned to the checkpoint revision below.  As in their
evaluation scripts, refs passes every reference caption of the row and free passes none.

A (picture, candidate, references) triple is scored once and shared by the rows and splits
that contain it; finished triples are kept in runs/metrics/vlm/Rigel/<mode>/table.npz, so
an interrupted run resumes.  Per split the script writes <split>.npz with score, in CSV order.

Run with the vlm environment plus peft and omegaconf (queue_rigel.sh sets this up).
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import ast
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(_ROOT)
DATA = ROOT / 'pearl_data'
OUT = ROOT / 'runs/metrics/vlm/Rigel'
CODE = ROOT / 'external/Rigel'
REVISION = '303d20866908a9dcb67db83fc4c156e886e809f4'    # ShuitsuKoyama/Rigel on the Hugging Face Hub
ORDER = ['composite', 'flickr8k_expert', 'nebula_test', 'flickr8k_cf', 'spica_test', 'polaris_test']

csv.field_size_limit(10 ** 9)
sys.path.insert(0, str(CODE / 'src'))


def read_split(name, mode):
    """Per row: (key, picture, candidate, references shown to the model)."""
    with open(DATA / 'csv' / f'{name}.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        references = tuple(ast.literal_eval(row['refs'])) if mode == 'refs' else ()
        candidate = str(row['mt'])
        key = hashlib.sha1('\x1f'.join([row['imgid'], candidate, *references]).encode()).hexdigest()
        out.append((key, row['imgid'], candidate, references))
    return out


class Table:
    """Finished triples: key -> score."""

    def __init__(self, path):
        self.path, self.values = path, {}
        if path.exists():
            with np.load(path) as f:
                self.values = dict(zip(f['keys'].tolist(), f['values'].tolist()))

    def save(self):
        keys = np.array(list(self.values))
        temporary = self.path.with_suffix('.tmp.npz')
        np.savez(temporary, keys=keys, values=np.array([self.values[k] for k in keys], np.float32))
        temporary.replace(self.path)


def main(mode='free', batch_size=8, *splits):
    batch_size = int(batch_size)
    splits = list(splits) or ORDER
    out = OUT / mode
    out.mkdir(parents=True, exist_ok=True)
    table = Table(out / 'table.npz')

    from rigel import from_pretrained
    model = from_pretrained(revision=REVISION)

    for name in splits:
        rows = read_split(name, mode)
        pending, seen = [], set()
        for item in rows:
            if item[0] not in table.values and item[0] not in seen:
                seen.add(item[0])
                pending.append(item)
        pending.sort(key=lambda item: (item[1], len(item[2])))
        print(f'{name}: {len(rows)} rows, {len(pending)} triples to score', flush=True)
        started = time.time()
        for step, start in enumerate(range(0, len(pending), batch_size)):
            batch = pending[start:start + batch_size]
            scores = model(media_path=[str(DATA / 'images' / picture) for _, picture, _, _ in batch],
                           cand=[candidate for _, _, candidate, _ in batch],
                           refs=[list(references) for _, _, _, references in batch],
                           modality=['image'] * len(batch)).float().cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError(f'{name}: non-finite score in batch {step}')
            for (key, _, _, _), score in zip(batch, scores):
                table.values[key] = float(score)
            if step % 100 == 0:
                done = start + len(batch)
                rate = done / (time.time() - started)
                print(f'[{name}] {done}/{len(pending)} {rate:.1f}/s eta '
                      f'{(len(pending) - done) / rate / 60:.1f} min', flush=True)
            if step % 1000 == 999:
                table.save()
        table.save()
        values = np.array([table.values[item[0]] for item in rows], np.float32)
        np.savez(out / f'{name}.npz', score=values)
        print(f'{name}: score mean {values.mean():.4f}, sd {values.std():.4f}', flush=True)

    with open(out / 'scoring.json', 'w') as f:
        json.dump(dict(model='ShuitsuKoyama/Rigel', revision=REVISION, code='SHUonkei/Rigel@72394e2', mode=mode,
                       references='all' if mode == 'refs' else 'none', batch_size=batch_size, splits=splits), f, indent=1)


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
