"""Score the trained MLP fold models on every evaluation set, in the ridge report format.

Usage: predict_mlp.py <pool> <seed> [backbone] [variant] [names]
`names` is a comma separated list of trained subsets; the default is the released pool and its
five held-out-fold models. Any other list is written to a separate "<label>_variants" directory.
With clip and legacy_cached (the authors' configuration) the released Polos checkpoint is
scored as one more column, "official".
"""
import sys
import time

import numpy as np
import torch

from evaluation import measure
from lib import CACHE, EVAL_SETS, RUNS, load_rows, write_json
from revision.common import FeatureStore, checkpoint_original
from revision.model import CachedPolos, make_model
from revision.train import score

NAMES = ['full', 'heldout0', 'heldout1', 'heldout2', 'heldout3', 'heldout4']


def run_directory(pool, seed, backbone, variant):
    if (backbone, variant) == ('siglip', 'full'):
        return RUNS / 'mlp' / f'{pool}_s{seed}', f'mlp_siglip_s{seed}'
    return RUNS / f'mlp_{variant}_{backbone}' / f'{pool}_s{seed}', f'mlp_{variant}_{backbone}_s{seed}'


def load(directory, pool, name, backbone):
    saved = torch.load(directory / name / 'best.pt', map_location='cpu', weights_only=True)
    stats = None
    if saved['config']['standardize']:
        stats = torch.load(CACHE / f'{backbone}_stats_{pool}_{name}.pt', map_location='cpu', weights_only=True)
    model = make_model(saved['config'], stats)
    model.load_state_dict(saved['model'])
    return model.cuda().eval(), saved['validation']['kendall_tau_c'], saved['epoch']


def official():
    state = torch.load(checkpoint_original('reprod'), map_location='cpu', weights_only=False)['state_dict']
    return CachedPolos(6144, 'batch', False).load_original(state).cuda().eval()


def main(pool='spica_train', seed=42, backbone='siglip', variant='full', names=None):
    directory, label = run_directory(pool, seed, backbone, variant)
    chosen = NAMES if names is None else names.split(',')
    if names is not None:
        label += '_variants'
    out = RUNS / 'models' / label / pool
    (out / 'predictions').mkdir(parents=True, exist_ok=True)
    store = FeatureStore(backbone, CACHE, scope='all')
    models, fits = {}, {}
    for name in chosen:
        model, validation, epoch = load(directory, pool, name, backbone)
        key = 'released' if name == 'full' else name
        models[key] = model
        fits[key] = dict(label=key, train_name=f'{pool}_{name}', best_validation_tau_c=validation, epoch=epoch)
        print(key, round(validation, 4), 'epoch', epoch, flush=True)
    if (backbone, variant) == ('clip', 'legacy_cached'):
        models['official'] = official()
        fits['official'] = dict(label='official', checkpoint=str(checkpoint_original('reprod')))
    labels = list(models)
    results, started = {}, time.time()
    for name in EVAL_SETS:
        rows = load_rows(name, CACHE)
        predictions = np.stack([score(models[key], store, rows, 'cuda', 32) for key in labels], 1)
        np.save(out / 'predictions' / f'{name}.npy', predictions)
        results[name] = {key: measure(name, predictions[:, i].astype(np.float64), rows)
                         for i, key in enumerate(labels)}
        print(name, {k: round(v['score'], 2) for k, v in results[name].items()}, flush=True)
    write_json(out / 'report.json', dict(backbone=backbone, head='mlp', variant=variant, seed=int(seed),
                                         pool=pool, labels=labels, fits=fits, evaluation=results,
                                         seconds=time.time() - started, prediction_columns=labels))


if __name__ == '__main__':
    main(*sys.argv[1:6])
