"""Score every evaluation split with a trained Pearl checkpoint.

The released validate/ scripts print aggregate correlations per benchmark with
two different text lookups (training uses the raw caption, evaluation prefixes
it with "A photo depicts ").  The analysis needs one prediction per row
instead, so this runs the model's own predict() over each split's CSV and saves
the scores.

    score_pearl.py <checkpoint> <output dir> [--prefixed] [splits...]

--prefixed uses the evaluation-side CLIP++ lookup; the default uses the
training-side one.  Both key sets are in the feature files.

Saved per split: <split>.npz with score, img_score, ref_score, in CSV order.  A split whose
file exists is skipped.
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import ast
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(_ROOT)
PEARL = ROOT / 'external/Pearl'
DATA = ROOT / 'pearl_data'
FEATURES = ROOT / 'features'
PREFIX = 'A photo depicts '
SPLITS = ['spica_test', 'polaris_test', 'polaris_val', 'nebula_test', 'nebula_valid',
          'flickr8k_expert', 'flickr8k_cf', 'composite', 'pascal50s', 'foil_1ref', 'foil_4ref',
          'spica_val', 'spica_train']

csv.field_size_limit(10 ** 9)
sys.path.insert(0, str(PEARL))


def read_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return [dict(imgid=r['imgid'], mt=str(r['mt']), refs=ast.literal_eval(r['refs']),
                 score=float(r['score']) if r['score'] else 0.0) for r in rows]


def load_features():
    from pearl.models.utils import load_clip_features
    clippp = load_clip_features(FEATURES / 'clippp/all')
    blip2 = load_clip_features(FEATURES / 'blip2/all')
    beit3 = load_clip_features(FEATURES / 'beit3/all')
    stella = load_clip_features(FEATURES / 'stella/all', img=False)
    return clippp, blip2, beit3, stella


def build(rows, features, prefixed):
    clippp, blip2, beit3, stella = features
    samples = []
    for row in rows:
        refs = row['refs'] if row['refs'] else ['NO REFS!!!']
        key = (lambda t: PREFIX + t) if prefixed else (lambda t: t)
        samples.append({
            'mt': row['mt'], 'refs': refs, 'img': row['imgid'], 'score': row['score'],
            'img_features': clippp['img_features'][row['imgid']],
            'mt_features': clippp['mt_features'][key(row['mt'])],
            'refs_features': [clippp['ref_features'].get(key(r)) for r in refs],
            'blip_img_features': blip2['img_features'][row['imgid']],
            'blip_mt_features': blip2['mt_features'][row['mt']],
            'blip_refs_features': [blip2['ref_features'].get(r) for r in refs],
            'beit3_img_features': beit3['img_features'][row['imgid']],
            'beit3_mt_features': beit3['mt_features'][row['mt']],
            'beit3_refs_features': [beit3['ref_features'].get(r) for r in refs],
            'stella_mt_features': stella['mt_features'][row['mt']],
            'stella_refs_features': [stella['ref_features'].get(r) for r in refs],
        })
    return samples


def main(checkpoint, out_dir, *arguments):
    from pearl.models import load_checkpoint

    prefixed = '--prefixed' in arguments
    splits = [a for a in arguments if not a.startswith('--')] or SPLITS
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    features = load_features()
    print('features loaded', flush=True)
    model = load_checkpoint(checkpoint)
    model = model.cuda().eval()

    for split in splits:
        if (out / f'{split}.npz').exists():   # a rerun after a failure continues where it stopped
            continue
        started = time.time()
        rows = read_csv(DATA / 'csv' / f'{split}.csv')
        samples = build(rows, features, prefixed)
        with torch.no_grad():
            _, score, img_score, ref_score = model.predict(samples, cuda=True, batch_size=256,
                                                           show_progress=False)
        np.savez(out / f'{split}.npz', score=np.array(score, dtype=np.float32),
                 img_score=np.array(img_score, dtype=np.float32),
                 ref_score=np.array(ref_score, dtype=np.float32))
        print(split, len(rows), f'{time.time() - started:.0f}s', flush=True)

    with open(out / 'scoring.json', 'w') as f:
        json.dump(dict(checkpoint=str(checkpoint), prefixed=prefixed, splits=splits), f)


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
