"""Control for image domain when reading the Composite leakage gap.

Composite rows carry their source in the id (Flickr8k/30k vs COCO val2014).
Training pools are COCO-heavy, so "leaked" and "COCO" are confounded; this
script reports the leaked/clean gap inside each source separately.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, ROOT, write_json

MODELS = ['cached_reprod', 'ridge_siglip', 'ridge_extended_clip']


def sources():
    rows = [json.loads(line) for line in open(DATA / 'composite/composite.jsonl')]
    return np.array(['coco' if 'COCO_val2014' in r['id'] else 'flickr' for r in rows]), \
        np.array([r['id'] for r in rows])


def tau(prediction, target):
    if len(prediction) < 30 or np.std(target) == 0:
        return None
    return round(float(kendalltau(prediction, target, variant='c')[0]), 4)


def main():
    source, ids = sources()
    masks = np.load(OUT / 'leakage_masks.npz')
    report = {}
    for model in MODELS:
        path = ROOT / f'runs/revision/{model}/evaluation/composite.npz'
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        assert list(data['ids']) == list(ids), 'row order differs from composite.jsonl'
        prediction, target = data['prediction'], data['target']
        entry = {}
        for pool in ['polaris_train', 'spica_train', 'nebula_train']:
            mask = masks[f'composite__{pool}']
            entry[pool] = {}
            for name in ['coco', 'flickr', 'all']:
                keep = np.ones(len(source), bool) if name == 'all' else source == name
                entry[pool][name] = dict(
                    leaked_rows=int((keep & mask).sum()), clean_rows=int((keep & ~mask).sum()),
                    leaked=tau(prediction[keep & mask], target[keep & mask]),
                    clean=tau(prediction[keep & ~mask], target[keep & ~mask]))
        report[model] = entry
    report['row_counts'] = dict(coco=int((source == 'coco').sum()), flickr=int((source == 'flickr').sum()))
    write_json(OUT / 'composite_control.json', report)
    for model, entry in report.items():
        if model == 'row_counts':
            continue
        print('==', model)
        for pool, stats in entry.items():
            parts = ' '.join(f"{n}: {s['leaked']}({s['leaked_rows']}) vs {s['clean']}({s['clean_rows']})"
                             for n, s in stats.items())
            print(f'  {pool:14s} {parts}')


if __name__ == '__main__':
    main()
