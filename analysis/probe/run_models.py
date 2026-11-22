"""Fit every ridge model of the design and score it on all evaluation sets.

Usage: run_models.py <backbone> <pool> [label ...]
Writes runs/probe/models/<backbone>/<pool>/{weights.pt,report.json,predictions/<set>.npy}.
"""
import sys
import time

import numpy as np
import torch

from evaluation import measure
from lib import CACHE, EVAL_SETS, RUNS, Features, load_rows, score, write_json
from models import clean_validation, definitions, fit


def main(backbone='siglip', pool='spica_train', only=()):
    out = RUNS / 'models' / backbone / pool
    (out / 'predictions').mkdir(parents=True, exist_ok=True)
    features = Features(backbone)
    validation = clean_validation(pool)
    print('clean validation rows', len(validation['target']), flush=True)
    models, reports = {}, {}
    chosen = definitions(backbone, pool)
    if only:
        chosen = {k: v for k, v in chosen.items() if k in only}
    started = time.time()
    fixed = None
    for label, definition in chosen.items():
        weight, bias, report = fit(features, label, definition, validation, backbone, pool,
                                   fixed=fixed if label != 'released' else None)
        models[label] = (weight, bias)
        reports[label] = report
        if label == 'released':
            fixed = report['regularization']  # one regularization for every variant of this pool
        print(report['label'], 'lambda', report['regularization'], 'val tau_c',
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
    write_json(out / 'report.json', dict(backbone=backbone, pool=pool, labels=labels, fits=reports,
                                         evaluation=results, seconds=time.time() - started,
                                         prediction_columns=labels))


if __name__ == '__main__':
    main(*(sys.argv[1:3] or ['siglip', 'spica_train']), only=tuple(sys.argv[3:]))
