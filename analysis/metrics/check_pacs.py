"""PAC-S / RefPAC-S under each benchmark's published protocol, next to the published numbers."""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import sys

import numpy as np

sys.path.insert(0, (_ROOT + '/analysis/probe'))
from effects import Benchmark  # noqa: E402
from lib import EVAL_SETS, ROOT  # noqa: E402

# PAC-S++: Sarto et al., arXiv 2410.07336v2 (Tables 1-4 and the backbone table).
# ViT-B/32: Sarto et al., CVPR 2023.  ViT-L/14: the DENEB paper's Table 1 rows "PAC-S (ViT-L/14)"
# and "RefPAC-S (ViT-L/14)" (which also gives Nebula for both backbones) and, for PASCAL-50S,
# the "CLIP ViT-L/14 / PAC-S" rows of the PAC-S++ paper.
PUBLISHED = {'pacs': {'flickr8k_expert': (54.3, 55.9), 'flickr8k_cf': (36.0, 37.6), 'composite': (55.7, 57.3),
                      'pascal50s': (82.4, 84.7), 'foil_1ref': (89.9, 93.7), 'foil_4ref': (89.9, 94.9),
                      'nebula_test': (47.2, 50.6)},
             'pacs_l14': {'flickr8k_expert': (55.7, 56.4), 'flickr8k_cf': (37.0, 37.8), 'composite': (54.0, 56.0),
                          'pascal50s': (82.2, 85.0), 'foil_1ref': (93.1, 94.4), 'foil_4ref': (93.1, 94.9),
                          'nebula_test': (47.9, 50.4)},
             'pacspp': {'flickr8k_expert': (54.5, 55.7), 'flickr8k_cf': (37.0, 37.9), 'composite': (58.3, 59.1),
                        'pascal50s': (82.3, 84.5), 'foil_1ref': (90.2, 93.5), 'foil_4ref': (90.2, 94.1)},
             'pacspp_l14': {'flickr8k_expert': (57.4, 57.9), 'flickr8k_cf': (38.5, 38.8), 'composite': (62.0, 61.6),
                            'pascal50s': (82.4, 85.4), 'foil_1ref': (93.6, 95.1)}}

variant = sys.argv[1] if len(sys.argv) > 1 else 'pacs'
for name in EVAL_SETS:
    bench = Benchmark(name)
    with np.load(ROOT / f'runs/metrics/pacs/{variant}/{name}.npz') as f:
        items = np.arange(bench.item_count)
        got = [bench.measure(f[key].astype(np.float64), items) for key in ('pacs', 'refpacs')]
        released = (f"   RefPAC-S with the released w=2: {bench.measure(f['refpacs_w2'].astype(np.float64), items):6.2f}"
                    if 'refpacs_w2' in f else '')
    paper = PUBLISHED[variant].get(name, (float('nan'),) * 2)
    print(f'{name:18s} PAC-S {got[0]:6.2f} (paper {paper[0]:5.1f})   RefPAC-S {got[1]:6.2f} (paper {paper[1]:5.1f})' + released)
