"""PAC-S / RefPAC-S and PAC-S++ / RefPAC-S++ with the released checkpoints, for every split.

The released implementation (pacscore/evaluation/pac_score/pac_score.py) walks
one benchmark at a time and re-encodes a picture or a caption once per row.  The
arithmetic here is the released one, unchanged:

    PAC-S    = 2 * max(0, cos(image, "A photo depicts " + candidate))
    RefPAC-S = harmonic mean of PAC-S and max_r cos(candidate, reference),
               both captions encoded with the same "A photo depicts " prefix

only the iteration differs: every distinct picture and text of pearl_data is
encoded once with the authors' own loaders and checkpoints, and the splits then
index into those tables.

    score_pacs.py <pacs|pacs_l14|pacspp|pacspp_l14> [batch size]

  pacs        CLIP ViT-B/32 (clip_ViT-B-32.pth), the configuration of the paper's main table
  pacs_l14    OpenCLIP ViT-L/14 (openClip_ViT-L-14.pth); this is PAC-S, not PAC-S++
  pacspp      PAC-S++, CLIP ViT-B/32 with rank-4 LoRA (PAC++_clip_ViT-B-32.pth)
  pacspp_l14  PAC-S++, CLIP ViT-L/14 with rank-4 LoRA (PAC++_clip_ViT-L-14.pth), the backbone
              compute_correlations_pac++.py hard-codes

The output keys are pacs and refpacs for all four.  The released scripts call PACScore with
w=2.0 for every checkpoint; the PAC-S++ paper states w=2.5 (ViT-B/32) and w=3 (ViT-L/14).  w
does not change the ranking of PAC-S but it does change the harmonic mean, so for the two
PAC-S++ variants pacs/refpacs use the paper's w and pacs_w2/refpacs_w2 the released w=2.0.

Writes runs/metrics/pacs/<variant>/<split>.npz with pacs and refpacs, in
CSV order, plus features/<variant>/ holding the encodings.
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import ast
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(_ROOT)
DATA = ROOT / 'pearl_data'
PACSCORE = ROOT / 'external/DENEB/pacscore'
WEIGHTS = ROOT / 'external/weights/pacs_drive'
OUT = ROOT / 'runs/metrics/pacs'
PREFIX = 'A photo depicts '
PAPER_W = {'pacs': 2.0, 'pacs_l14': 2.0, 'pacspp': 2.5, 'pacspp_l14': 3.0}

csv.field_size_limit(10 ** 9)
sys.path.insert(0, str(PACSCORE))


def items():
    images, texts = set(), set()
    for path in sorted((DATA / 'csv').glob('*.csv')):
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                images.add(row['imgid'])
                texts.add(str(row['mt']))
                texts.update(ast.literal_eval(row['refs']))
    return sorted(images), sorted(texts)


def load_model(variant, device):
    """The authors' own loading path for each of the two released checkpoints."""
    if variant == 'pacs':
        from models.clip import clip
        model, preprocess = clip.load('ViT-B/32', device=device)
        state = torch.load(WEIGHTS / 'clip_ViT-B-32.pth', map_location=device)['state_dict']
        model.load_state_dict(state)
        tokenize = lambda batch: clip.tokenize(batch, truncate=True)
    elif variant in ('pacspp', 'pacspp_l14'):
        # compute_correlations_pac++.py: OpenAI CLIP with rank-4 LoRA layers, then the state dict
        from models.clip import clip
        from models.clip_lora import clip_lora
        name = 'ViT-B/32' if variant == 'pacspp' else 'ViT-L/14'
        model, preprocess = clip_lora.load(name, device=device, lora=4)
        state = torch.load(ROOT / 'external/weights/pacspp' / f"PAC++_clip_{name.replace('/', '-')}.pth",
                           map_location=device)['state_dict']
        model.load_state_dict(state)
        tokenize = lambda batch: clip.tokenize(batch, truncate=True)
    elif variant == 'pacs_l14':
        from models import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14',
                                                                     pretrained='laion2b_s32b_b82k')
        state = torch.load(WEIGHTS / 'openClip_ViT-L-14.pth', map_location='cpu')['state_dict']
        model.load_state_dict(state)
        model = model.to(device)
        # the released scorer tokenizes with OpenAI CLIP's tokenizer for both checkpoints
        from models.clip import clip
        tokenize = lambda batch: clip.tokenize(batch, truncate=True)
    else:
        raise ValueError(variant)
    return model.eval().float(), preprocess, tokenize


def progress(tag, done, total, started):
    rate = done / max(time.time() - started, 1e-9)
    print(f'[{tag}] {done}/{total} {rate:.0f}/s eta {(total - done) / max(rate, 1e-9) / 60:.1f} min',
          flush=True)


def encode_texts(model, tokenize, texts, device, batch_size, tag):
    out, started = [], time.time()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = [PREFIX + t for t in texts[start:start + batch_size]]
            out.append(model.encode_text(tokenize(batch).to(device)).float().cpu().numpy())
            if (start // batch_size) % 100 == 0:
                progress(tag, start, len(texts), started)
    return np.vstack(out)


def encode_images(model, preprocess, names, device, batch_size, tag):
    from PIL import Image

    class Pictures(torch.utils.data.Dataset):
        def __len__(self):
            return len(names)

        def __getitem__(self, index):
            return preprocess(Image.open(DATA / 'images' / names[index]).convert('RGB'))

    loader = torch.utils.data.DataLoader(Pictures(), batch_size=batch_size, num_workers=12)
    out, started, done = [], time.time(), 0
    with torch.no_grad():
        for batch in loader:
            out.append(model.encode_image(batch.to(device)).float().cpu().numpy())
            done += len(batch)
            if done % (batch_size * 50) == 0:
                progress(tag, done, len(names), started)
    return np.vstack(out)


def main(variant='pacs', batch_size=256):
    batch_size = int(batch_size)
    device = 'cuda'
    images, texts = items()
    model, preprocess, tokenize = load_model(variant, device)

    cache = ROOT / 'features' / variant
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / 'encodings.npz').exists():
        with np.load(cache / 'encodings.npz') as f:
            image_features, text_features = f['images'], f['texts']
    else:
        image_features = encode_images(model, preprocess, images, device, 64, f'{variant} img')
        text_features = encode_texts(model, tokenize, texts, device, batch_size, f'{variant} txt')
        np.savez(cache / 'encodings.npz', images=image_features, texts=text_features)

    unit = lambda x: x / np.sqrt((x ** 2).sum(-1, keepdims=True))
    image_features = unit(image_features.astype(np.float64))
    text_features = unit(text_features.astype(np.float64))
    image_index = {name: i for i, name in enumerate(images)}
    text_index = {text: i for i, text in enumerate(texts)}

    out = OUT / variant
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted((DATA / 'csv').glob('*.csv')):
        if path.stem.endswith('_train'):
            continue
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        picture = image_features[[image_index[r['imgid']] for r in rows]]
        candidate = text_features[[text_index[str(r['mt'])] for r in rows]]
        cosine = np.clip((picture * candidate).sum(-1), 0, None)
        text_text = np.full(len(rows), np.nan)
        for i, row in enumerate(rows):
            refs = ast.literal_eval(row['refs'])
            if refs:
                text_text[i] = max(float(candidate[i] @ text_features[text_index[r]]) for r in refs)

        def harmonic(pacs):
            total = pacs + text_text
            return np.where(total > 0, 2 * pacs * text_text / np.where(total > 0, total, 1), 0.0)

        saved = {}
        for suffix, w in [('', PAPER_W[variant])] + ([('_w2', 2.0)] if PAPER_W[variant] != 2.0 else []):
            saved['pacs' + suffix] = (w * cosine).astype(np.float32)
            saved['refpacs' + suffix] = np.where(np.isnan(text_text), np.nan,
                                                 harmonic(w * cosine)).astype(np.float32)
        np.savez(out / f'{path.stem}.npz', **saved)
        pacs = saved['pacs']
        print(path.stem, len(rows), f'pacs mean {pacs.mean():.4f}', flush=True)

    with open(out / 'scoring.json', 'w') as f:
        json.dump(dict(variant=variant, w=PAPER_W[variant], images=len(images), texts=len(texts)), f)


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
