"""Check lib.py against the released Polaris ridge run: same lambda grid, same validation tau_c."""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import sys
import time

import numpy as np
import torch

from lib import CACHE, Features, accumulate, load_rows, read_json, score, tau

LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1., 3., 10.]


def main(backbone='siglip'):
    reference = read_json((_ROOT + '/runs/revision/ridge_siglip/training_summary.json'))
    expected = {c['regularization']: c['validation']['kendall_tau_c'] for c in reference['candidates']}
    features = Features(backbone, allow_partial=True)
    train, val = load_rows('polaris_train', CACHE), load_rows('polaris_val', CACHE)
    started = time.time()
    gram, rhs = accumulate(features, train, train['target'])
    print('gram', round(time.time() - started), 's', flush=True)
    models = solve_all(gram, rhs)
    predictions = score(features, val, [w for w, _ in models], [b for _, b in models])
    for lam, column in zip(LAMBDAS, predictions.T):
        got = tau(column, val['target'])
        print(f'lambda={lam:<8} tau_c={got:.6f} released={expected.get(lam, float("nan")):.6f} '
              f'diff={got - expected.get(lam, float("nan")):+.2e}', flush=True)


def solve_all(gram, rhs):
    from lib import solve
    return solve(gram, rhs, LAMBDAS)


if __name__ == '__main__':
    main(*sys.argv[1:])
