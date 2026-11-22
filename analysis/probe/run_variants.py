"""Ridge fits of the pair-decontamination experiment: released vs pairclean vs randomdrop.

The three models differ only in which training rows they see, and the last two see the same
number of rows, so the released-minus-pairclean difference that the control does not show is
caused by the duplicated (picture, candidate) labels.

Usage: run_variants.py <backbone> <pool>
Writes runs/probe/models/ridge_variant_<backbone>/<pool>/{weights.pt,report.json,predictions}.
"""
import sys
import time

import numpy as np
import torch

from evaluation import measure
from lib import CACHE, EVAL_SETS, RUNS, Features, load_rows, read_json, score, write_json
from models import clean_validation, definitions, fit, pieces_dir

NAMES = ['pairclean', 'randomdrop']


def main(backbone='siglip', pool='polaris_train'):
    out = RUNS / 'models' / f'ridge_variant_{backbone}' / pool
    (out / 'predictions').mkdir(parents=True, exist_ok=True)
    pieces = read_json(pieces_dir(backbone, pool, 'variant') / 'pieces.json')['pieces']
    chosen = {'released': definitions(backbone, pool)['released']}
    chosen.update({name: ('variant', [name], None) for name in NAMES if name in pieces})
    features = Features(backbone)
    validation = clean_validation(pool)
    models, reports, started, fixed = {}, {}, time.time(), None
    for label, definition in chosen.items():
        weight, bias, report = fit(features, label, definition, validation, backbone, pool,
                                   fixed=fixed if label != 'released' else None)
        models[label], reports[label] = (weight, bias), report
        if label == 'released':
            fixed = report['regularization']
        print(label, 'rows', report['rows'], 'lambda', report['regularization'], 'val tau_c',
              round(report['best_validation_tau_c'], 4), round(time.time() - started), 's', flush=True)
    torch.save({label: dict(weight=w.cpu(), bias=b.cpu()) for label, (w, b) in models.items()}, out / 'weights.pt')
    labels = list(models)
    results = {}
    for name in EVAL_SETS:
        rows = load_rows(name, CACHE)
        predictions = score(features, rows, [models[k][0] for k in labels], [models[k][1] for k in labels])
        np.save(out / 'predictions' / f'{name}.npy', predictions)
        results[name] = {label: measure(name, predictions[:, i].astype(np.float64), rows)
                         for i, label in enumerate(labels)}
        print(name, {k: round(v['score'], 2) for k, v in results[name].items()}, flush=True)
    write_json(out / 'report.json', dict(backbone=backbone, pool=pool, head='ridge', labels=labels,
                                         fits=reports, evaluation=results, seconds=time.time() - started,
                                         prediction_columns=labels))


if __name__ == '__main__':
    main(*sys.argv[1:3])
