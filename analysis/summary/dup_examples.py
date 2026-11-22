"""Concrete examples of duplicated samples for the paper.

  exact     an image-caption pair of the Polaris test set with its ratings there, in the Spica
            training set and (averaged) in the Nebula training set
  image     two stored images with different pixels that are the same photograph (SigLIP cosine
            >= 0.99), one in the Spica test set and one in the Spica training set
  caption   a Polaris test caption that differs from a Spica training caption of the same image
            only in case, punctuation or a few characters

    dup_examples.py     (PYTHONPATH=Polos-revised, from analysis/summary)

Writes runs/summary/dup_examples.json (a few candidates of each kind; the figure script picks).
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import json
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(_ROOT)
sys.path.insert(0, str(ROOT / 'analysis/probe'))
sys.path.insert(0, str(ROOT / 'analysis/metrics'))
from lib import CACHE, load_rows, picture_ids, read_json  # noqa: E402
from near_duplicates import normalize  # noqa: E402

catalog = read_json(CACHE / 'catalog.json')
texts, images = catalog['texts'], catalog['images']
picture = picture_ids()


def pairs(name):
    rows = load_rows(name, CACHE)
    table = defaultdict(list)
    for index, (image, candidate, target) in enumerate(zip(rows['image'], rows['candidate'], rows['target'])):
        table[int(picture[image]), int(candidate)].append((float(target), index))
    return rows, table


out = {'exact': [], 'image': [], 'caption': []}
polaris_rows, polaris = pairs('polaris_test')
spica_rows, spica = pairs('spica_train')
nebula_rows, nebula = pairs('nebula_train')
for key in sorted(set(polaris) & set(spica) & set(nebula)):
    a = sorted(t for t, _ in polaris[key])
    b = sorted(t for t, _ in spica[key])
    if a == b and 6 <= len(a) <= 8 and len(set(a)) >= 3:
        index = polaris[key][0][1]
        out['exact'].append(dict(image=images[polaris_rows['image'][index]], candidate=texts[key[1]],
                                 references=[texts[j] for j in polaris_rows['references'][index][:polaris_rows['counts'][index]]],
                                 polaris_test=[t for t, _ in polaris[key]], spica_train=[t for t, _ in spica[key]],
                                 nebula_train=[t for t, _ in nebula[key]],
                                 spica_file=images[spica_rows['image'][spica[key][0][1]]],
                                 nebula_file=images[nebula_rows['image'][nebula[key][0][1]]]))
print('exact candidates', len(out['exact']))
out['exact'] = out['exact'][:: max(1, len(out['exact']) // 24)][:24]

# Caption near-duplicates: same image, different stored caption.
by_picture = defaultdict(set)
for (p, c) in spica:
    by_picture[p].add(c)
seen = set()
for (p, c), values in polaris.items():
    if (p, c) in spica or p not in by_picture or len(values) < 5:
        continue
    mine = normalize(texts[c])
    for other in by_picture[p]:
        theirs = normalize(texts[other])
        ratio = SequenceMatcher(None, mine, theirs, autojunk=False).ratio()
        if (mine == theirs or ratio >= 0.9) and texts[c] != texts[other] and (p, c) not in seen:
            seen.add((p, c))
            index = values[0][1]
            out['caption'].append(dict(image=images[polaris_rows['image'][index]], polaris_test_caption=texts[c],
                                       spica_train_caption=texts[other], same_after_normalization=mine == theirs, similarity=ratio,
                                       polaris_test=[t for t, _ in values], spica_train=[t for t, _ in spica[p, other]]))
print('caption candidates', len(out['caption']))
out['caption'] = sorted(out['caption'], key=lambda e: (not e['same_after_normalization'], -e['similarity']))[:30]

# Image near-duplicates between the Spica test and training sets.
features = np.load(CACHE / 'siglip_image.npy', mmap_mode='r')
test_rows = load_rows('spica_test', CACHE)
import prep_pearl_data  # noqa: E402
first = {}
for image in prep_pearl_data.spica_train_rows()['image']:      # the released set, with its reference-less rows
    first.setdefault(int(picture[image]), int(image))
train_pictures = np.array(sorted(first))
train = np.asarray(features[[first[p] for p in train_pictures]], np.float32)
train /= np.linalg.norm(train, axis=1, keepdims=True)
test_first = {}
for image in test_rows['image']:
    test_first.setdefault(int(picture[image]), int(image))
absent = np.array([p for p in sorted(test_first) if p not in first])
block = np.asarray(features[[test_first[p] for p in absent]], np.float32)
block /= np.linalg.norm(block, axis=1, keepdims=True)
similarity = block @ train.T
best = similarity.argmax(1)
from PIL import Image  # noqa: E402
for row in np.argsort(-similarity.max(1))[:400]:
    value = float(similarity[row, best[row]])
    if value < 0.99:
        break
    a, b = images[test_first[int(absent[row])]], images[first[int(train_pictures[best[row]])]]
    with Image.open(a) as ia, Image.open(b) as ib:
        entry = dict(spica_test_file=a, spica_train_file=b, cosine=value, test_size=ia.size, train_size=ib.size,
                     test_bytes=Path(a).stat().st_size, train_bytes=Path(b).stat().st_size, test_format=ia.format, train_format=ib.format)
    out['image'].append(entry)
print('image candidates', len(out['image']))
out['image'] = out['image'][:: max(1, len(out['image']) // 16)][:16]
(ROOT / 'runs/summary/dup_examples.json').write_text(json.dumps(out, indent=1))
for e in out['caption'][:12]:
    print(repr(e['polaris_test_caption']), '|', repr(e['spica_train_caption']), e['same_after_normalization'], round(e['similarity'], 3), e['polaris_test'], e['spica_train'])
for e in out['image'][:8]:
    print(e['cosine'], e['test_size'], e['train_size'], e['test_bytes'], e['train_bytes'], e['test_format'], e['train_format'])
for e in out['exact'][:6]:
    print(e['candidate'], e['polaris_test'], e['spica_train'], e['nebula_train'])
