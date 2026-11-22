"""Closed-form ridge on the frozen Polos feature groups, with additive sufficient statistics.

A model trained on any union of row groups ("cells") is obtained by summing the
cells' statistics, so leave-pictures-out models cost one linear solve each.
Everything is accumulated on raw (unstandardized) features in float64; the
per-coordinate standardization of the training rows is applied analytically,
which makes it exactly the standardization of *that* training subset.

Run with PYTHONPATH=Polos-revised.
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import hashlib
import json
import os
from pathlib import Path

import numpy as np

try:
    import torch
    from revision.common import FeatureStore, batches, load_rows, read_json, write_json  # noqa: F401
    from revision.model import CachedPolos
except ImportError:
    # Analysing stored predictions needs neither torch nor the Polos code; fitting and scoring do.
    class torch:                                         # stands in for the decorators below
        @staticmethod
        def no_grad():
            return lambda function: function

    def write_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temp.replace(path)

    def read_json(path):
        return json.loads(Path(path).read_text())

    def load_rows(name, cache):
        with np.load(Path(cache) / f'{name}.npz', allow_pickle=False) as f:
            return {key: f[key] for key in f.files}

ROOT = Path(_ROOT)
CACHE = ROOT / 'feature_cache_v2'
RUNS = ROOT / 'runs/probe'
FOLDS = 5
EVAL_SETS = ['composite', 'flickr8k_expert', 'flickr8k_cf', 'pascal50s', 'foil_1ref', 'foil_4ref',
             'polaris_test', 'polaris_val', 'nebula_test', 'nebula_valid', 'spica_test', 'spica_val']


class Features:
    """Raw ten-group features: fixed uniform layer mix, per-sentence normalization."""

    def __init__(self, backbone='siglip', device='cuda', allow_partial=False):
        self.store = FeatureStore(backbone, CACHE, scope='all', allow_partial=allow_partial)
        self.dimension, self.device = self.store.dimension, device
        self.model = CachedPolos(self.dimension, 'sample', standardize=False, mix_trainable=False,
                                 head='ridge').to(device).eval()

    @torch.no_grad()
    def __call__(self, rows, indices):
        """[rows, references, dimension] float32."""
        return self.model.features(self.store.batch(rows, indices, self.device), standardize=False)


def subset(rows, index):
    return {key: value[index] for key, value in rows.items()}


@torch.no_grad()
def accumulate(features, rows, targets, batch_size=128):
    """Sum over rows of mean over references of [x,1][x,1]^T and [x,1] y^T (y may have columns)."""
    d, device = features.dimension, features.device
    targets = np.asarray(targets, np.float64).reshape(len(rows['candidate']), -1)
    gram = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=device)
    rhs = torch.zeros(d + 1, targets.shape[1], dtype=torch.float64, device=device)
    for ids in batches(rows, batch_size):
        x = features(rows, ids)
        refs = x.shape[1]
        x = x.flatten(0, 1).double()
        x = torch.cat((x, torch.ones(len(x), 1, dtype=torch.float64, device=device)), dim=1)
        y = torch.from_numpy(targets[ids]).to(device)[:, None, :].expand(-1, refs, -1).reshape(len(x), -1)
        gram.addmm_(x.T, x, alpha=1 / refs)
        rhs.addmm_(x.T, y, alpha=1 / refs)
    return gram, rhs


@torch.no_grad()
def solve(gram, rhs, lambdas, column=0):
    """Ridge on features standardized with this training set; intercept unpenalized.

    Returns per lambda (weight, bias) acting on raw features, as float64 tensors.
    """
    d = gram.shape[0] - 1
    n = gram[d, d]
    mean = gram[:d, d] / n
    cov = gram[:d, :d] / n - torch.outer(mean, mean)
    scale = cov.diagonal().clamp_min(0).sqrt().clamp_min(1e-6)
    cov = cov / torch.outer(scale, scale)
    ybar = rhs[d, column] / n
    cxy = ((rhs[:d, column] / n - mean * ybar) / scale)[:, None]
    out = []
    for lam in lambdas:
        system = cov.clone()
        system.diagonal().add_(lam)
        w = torch.cholesky_solve(cxy, torch.linalg.cholesky(system))[:, 0] / scale
        out.append((w, ybar - (mean * w).sum()))
    return out


@torch.no_grad()
def score(features, rows, weights, biases, batch_size=256):
    """Max over references of the linear score, for several models at once -> [rows, models]."""
    w = torch.stack([x.float() for x in weights], 1).to(features.device)
    b = torch.stack([x.float() for x in biases]).to(features.device)
    out = np.empty((len(rows['candidate']), w.shape[1]), np.float32)
    for ids in batches(rows, batch_size):
        out[ids] = (features(rows, ids) @ w + b).max(1).values.cpu().numpy()
    return out


def picture_ids():
    return np.load(CACHE / 'picture_ids.npy')


def picture_fold(pictures=None, salt='fold'):
    """Deterministic pseudo-random fold of every picture, from its pixel hash."""
    hashes = read_json(CACHE / 'pictures.json')
    fold = np.array([int(hashlib.sha1(f'{salt}:{h}'.encode()).hexdigest(), 16) % FOLDS for h in hashes], np.int8)
    return fold if pictures is None else fold[pictures]


def row_pictures(rows):
    return picture_ids()[rows['image']]


def sample_structure(rows):
    """Group rows that judge the same (picture, candidate, references): size and mean score."""
    pictures = row_pictures(rows)
    keys = np.stack([pictures.astype(np.int64), rows['candidate'].astype(np.int64),
                     rows['references'][:, 0].astype(np.int64)], 1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    sums = np.bincount(inverse, weights=rows['target'].astype(np.float64))
    return dict(sample=inverse, size=counts[inverse], mean=(sums / counts)[inverse])


def tau(pred, target, variant='c'):
    from scipy.stats import kendalltau
    return float(kendalltau(pred, target, variant=variant)[0])
