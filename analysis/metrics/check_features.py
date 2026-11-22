"""Spot-check the extracted Pearl features: counts, shapes, norms, NaNs, and a
zero-shot image-text cosine tau_c on spica_test per backbone."""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import ast, csv, sys
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau

ROOT = Path(_ROOT)
csv.field_size_limit(10 ** 9)
rows = list(csv.DictReader(open(ROOT / 'pearl_data/csv/spica_test.csv', newline='')))
human = np.array([float(r['score']) for r in rows])

def load(model, kind):
    out = {}
    for i in range(1, 6):
        out.update(np.load(ROOT / f'features/{model}/all/clip_{kind}_features_{i}.npz', allow_pickle=True)['features'].item())
    return out

for model in sys.argv[1:]:
    mt = load(model, 'mt')
    sample = np.stack([mt[r['mt']] for r in rows]).astype(np.float64)
    line = [model, f'texts {len(mt)}', f'text dim {sample.shape[1:]}', f'dtype {next(iter(mt.values())).dtype}',
            f'text norm {np.linalg.norm(sample, axis=-1).mean():.3f}', f'nan {int(np.isnan(sample).sum())}']
    if model != 'stella':
        img = load(model, 'img')
        im = np.stack([img[r['imgid']] for r in rows]).astype(np.float64)
        line += [f'images {len(img)}', f'img dim {im.shape[1:]}', f'img norm {np.linalg.norm(im, axis=-1).mean():.3f}',
                 f'nan {int(np.isnan(im).sum())}']
        unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
        if im.ndim == 3:
            cos = (unit(im) * unit(sample)[:, None]).sum(-1).max(1)
        else:
            cos = (unit(im) * unit(sample)).sum(-1)
        line.append(f'spica_test tau_c {100 * kendalltau(cos, human, variant="c")[0]:.2f}')
        if model == 'clippp':
            pre = np.stack([mt['A photo depicts ' + r['mt']] for r in rows]).astype(np.float64)
            cos = (unit(im) * unit(pre)).sum(-1)
            line.append(f'prefixed tau_c {100 * kendalltau(cos, human, variant="c")[0]:.2f}')
    else:
        refs = [ast.literal_eval(r['refs']) for r in rows]
        unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
        cos = np.array([max(float(unit(mt[x]) @ unit(s)) for x in rf) if rf else 0.0 for rf, s in zip(refs, sample)])
        line.append(f'spica_test max-ref cosine tau_c {100 * kendalltau(cos, human, variant="c")[0]:.2f}')
    print(' | '.join(line), flush=True)
