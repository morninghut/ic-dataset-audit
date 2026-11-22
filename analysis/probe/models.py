"""Build ridge models from the saved statistics pieces and choose the regularization.

A model is a named set of statistics pieces (summed) and one regularization strength.
Selection uses a validation set that shares no picture with the training pool or with any
evaluation set, so neither the fit nor the model choice can see an evaluation picture.
"""
import numpy as np
import torch

from lib import CACHE, FOLDS, RUNS, load_rows, read_json, row_pictures, score, solve, subset, tau

LAMBDAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def pieces_dir(backbone, pool, partition):
    return RUNS / 'stats' / backbone / pool / partition


def load_piece(backbone, pool, partition, name, device='cuda'):
    saved = torch.load(pieces_dir(backbone, pool, partition) / f'{name}.pt', map_location=device, weights_only=True)
    return saved['gram'], saved['rhs']


def sum_pieces(backbone, pool, partition, names, device='cuda'):
    gram = rhs = None
    for name in names:
        g, r = load_piece(backbone, pool, partition, name, device)
        gram, rhs = (g, r) if gram is None else (gram + g, rhs + r)
    return gram, rhs


def fold_names(partition_pieces, folds=range(FOLDS), trivial=(0, 1)):
    return [f'fold{f}_trivial{t}' for f in folds for t in trivial
            if f'fold{f}_trivial{t}' in partition_pieces]


def definitions(backbone='siglip', pool='spica_train'):
    """{label: (partition, [piece names], subtract piece or None)} for every model we fit."""
    fold = read_json(pieces_dir(backbone, pool, 'fold') / 'pieces.json')['pieces']
    out = {'released': ('fold', fold_names(fold), None),
           'nontrivial': ('fold', fold_names(fold, trivial=(0,)), None)}
    for f in range(FOLDS):
        out[f'heldout{f}'] = ('fold', fold_names(fold, folds=[k for k in range(FOLDS) if k != f]), None)
    mask_path = pieces_dir(backbone, pool, 'mask') / 'pieces.json'
    if mask_path.exists():
        for name in read_json(mask_path)['pieces']:
            if name == 'clean':
                out['clean_all'] = ('mask', ['clean'], None)
            else:
                out['without_' + name[3:]] = ('fold', fold_names(fold), name)
    return out


VALIDATION = {'spica_train': ('spica_val', ('spica_test', 'nebula_test')),
              'polaris_train': ('polaris_val', ('polaris_test',)),
              'nebula_train': ('nebula_valid', ('nebula_test', 'spica_test'))}


def clean_validation(pool='spica_train', val=None, avoid=None):
    """Validation rows whose picture is in neither the training pool nor the matching test set."""
    default_val, default_avoid = VALIDATION[pool]
    val, avoid = val or default_val, default_avoid if avoid is None else avoid
    rows = load_rows(val, CACHE)
    pictures = row_pictures(rows)
    used = np.unique(np.concatenate([row_pictures(load_rows(name, CACHE)) for name in (pool,) + tuple(avoid)]))
    return subset(rows, np.flatnonzero(~np.isin(pictures, used)))


def fit(features, label, definition, validation, backbone='siglip', pool='spica_train',
        lambdas=LAMBDAS, column=0, fixed=None):
    """Returns (weight, bias, report); the regularization is selected on validation tau_c."""
    partition, names, subtract = definition
    gram, rhs = sum_pieces(backbone, pool, partition, names)
    if subtract is not None:
        g, r = load_piece(backbone, pool, 'mask', subtract)
        gram, rhs = gram - g, rhs - r
    grid = [fixed] if fixed is not None else lambdas
    solutions = solve(gram, rhs, grid, column=column)
    predictions = score(features, validation, [w for w, _ in solutions], [b for _, b in solutions])
    taus = [tau(predictions[:, i], validation['target']) for i in range(len(grid))]
    best = int(np.argmax(taus))
    report = dict(label=label, pieces=names, subtract=subtract, target_column=column,
                  rows=float(gram[-1, -1]), validation_rows=len(validation['target']),
                  grid=list(map(float, grid)), validation_tau_c=list(map(float, taus)),
                  regularization=float(grid[best]), best_validation_tau_c=float(taus[best]))
    del gram, rhs
    torch.cuda.empty_cache()
    return solutions[best][0], solutions[best][1], report
