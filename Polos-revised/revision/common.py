import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'Polos-revised'
CACHE = ROOT / 'feature_cache_v1'
RUNS = ROOT / 'runs/revision'
DATA = Path((_POLOS_HOME + '/data_en'))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text())

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def setup(seed=42, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def metrics(pred, target):
    from evaluate import compute_metrics
    result = compute_metrics(pred, target)
    result.update(mse=float(np.mean((np.asarray(pred) - target) ** 2)),
                  prediction_mean=float(np.mean(pred)), prediction_std=float(np.std(pred)))
    return result

def load_rows(name, cache=CACHE):
    with np.load(Path(cache) / f'{name}.npz', allow_pickle=False) as f:
        return {key: f[key] for key in f.files}

def batches(rows, size, seed=None):
    """Shuffle rows, keeping all references; group only for efficient rectangular tensors."""
    rng = np.random.default_rng(seed)
    groups = []
    for count in np.unique(rows['counts']):
        indices = np.flatnonzero(rows['counts'] == count)
        if seed is not None:
            rng.shuffle(indices)
        groups.extend(indices[i:i + size] for i in range(0, len(indices), size))
    if seed is not None:
        rng.shuffle(groups)
    return groups

class FeatureStore:
    def __init__(self, backbone, cache=CACHE, allow_partial=False, scope='polaris'):
        self.cache = Path(cache)
        self.backbone = backbone
        self.catalog = read_json(self.cache / 'catalog.json')
        self.arrays = {}
        for key in ['simcse_cls', 'simcse_mean', 'simcse_var', 'simcse_count',
                    f'{backbone}_text', f'{backbone}_image']:
            self.arrays[key] = np.load(self.cache / f'{key}.npy', mmap_mode='r')
        if not allow_partial:
            for part in ['simcse', backbone]:
                meta = read_json(self.cache / f'{part}_status.json')
                if meta['status'] == 'complete':
                    continue
                required = self.catalog['datasets']['polaris_test']
                polaris_ready = (meta.get('completed_texts',0) >= required['text_prefix_count']
                                 and (part == 'simcse' or meta.get('completed_images',0) >= required['image_prefix_count']))
                if meta['status'] != 'complete' and not (scope == 'polaris' and polaris_ready):
                    raise RuntimeError(f'{part} cache incomplete')
        self.dimension = 6144 if backbone == 'clip' else 9984

    def take(self, key, index, device):
        return torch.from_numpy(np.array(self.arrays[key][index], copy=True)).to(device)

    def text(self, ids, device):
        return {name: self.take('simcse_' + name, ids, device)
                for name in ['cls', 'mean', 'var', 'count']}

    def batch(self, rows, indices, device='cuda'):
        count = int(rows['counts'][indices[0]])
        if not np.all(rows['counts'][indices] == count):
            raise ValueError('Batch contains different reference counts')
        cand = rows['candidate'][indices]
        refs = rows['references'][indices, :count]
        return dict(candidate=self.text(cand, device), references=self.text(refs, device),
                    candidate_vl=self.take(f'{self.backbone}_text', cand, device),
                    reference_vl=self.take(f'{self.backbone}_text', refs, device),
                    image=self.take(f'{self.backbone}_image', rows['image'][indices], device),
                    target=torch.from_numpy(rows['target'][indices]).to(device))

def checkpoint_original(name):
    if name == 'reprod':
        return Path((_POLOS_HOME + '/checkpoints/reprod/reprod.ckpt'))
    return Path(read_json(ROOT / f'runs/{name}/training_summary.json')['best_checkpoint'])
