"""How much of a benchmark's correlation comes from trivially separable rows?

Encodes Spica/Polaris test rows with the same frozen CLIP ViT-B/32 recipe the
reproduction uses and reports plain CLIPScore and RefCLIPScore on the whole
split, on the rows that carry more than one human annotation, and the AUC with
which CLIPScore alone separates the singleton zero-score rows from the rest.

Run from the Polos-revised directory so `import polos.clip` resolves:
    cd Polos-revised && python ../analysis/audit/clip_probe.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, SPICA, write_json
from csv_audit import load as load_spica, parse_refs

PREFIX = 'A photo depicts '


def encode(texts, images, device='cuda', text_batch=256, image_batch=64):
    import polos.clip as clip
    model, preprocess = clip.load('ViT-B/32', device='cpu')  # float32 path, as in cache_features.py
    model = model.to(device).eval().requires_grad_(False)
    text_features = np.empty((len(texts), 512), dtype=np.float32)
    image_features = np.empty((len(images), 512), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(texts), text_batch):
            chunk = texts[start:start + text_batch]
            tokens = clip.tokenize([PREFIX + s for s in chunk], truncate=True).to(device)
            text_features[start:start + len(chunk)] = model.encode_text(tokens).float().cpu().numpy()
        for start in range(0, len(images), image_batch):
            chunk = images[start:start + image_batch]
            tensors = []
            for path in chunk:
                with Image.open(path) as source:
                    tensors.append(preprocess(source.convert('RGB')))
            image_features[start:start + len(chunk)] = \
                model.encode_image(torch.stack(tensors).to(device)).float().cpu().numpy()
    return text_features, image_features


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def spica_test():
    frame = load_spica('test')
    paths = {i: str(SPICA / 'images' / i) for i in frame.imgid.unique()}
    return frame.assign(path=frame.imgid.map(paths))


def polaris_test():
    frame = pd.read_csv(DATA / 'polaris/polaris_test.csv', dtype={'imgid': str})
    frame['reference_list'] = frame['refs'].map(parse_refs)
    frame['path'] = [str(DATA / 'polaris/images' / i) for i in frame.imgid]
    return frame


def analyse(name, frame, report):
    texts = sorted({s for refs in frame.reference_list for s in refs} | set(frame.mt.astype(str)))
    text_index = {s: i for i, s in enumerate(texts)}
    images = sorted(frame.path.unique())
    image_index = {p: i for i, p in enumerate(images)}
    text_features, image_features = encode(texts, images)
    np.savez_compressed(OUT / f'clip_{name}.npz', texts=np.array(texts, dtype=object),
                        images=np.array(images, dtype=object),
                        text_features=text_features, image_features=image_features)
    t, v = unit(text_features), unit(image_features)

    candidate = np.array([text_index[str(c)] for c in frame.mt])
    picture = np.array([image_index[p] for p in frame.path])
    clip_score = np.sum(t[candidate] * v[picture], axis=1)
    reference_score = np.array([
        max(float(np.dot(t[text_index[str(c)]], t[text_index[s]])) for s in refs) if refs else np.nan
        for c, refs in zip(frame.mt, frame.reference_list)])
    harmonic = 2 * np.clip(clip_score, 0, None) * np.clip(reference_score, 0, None) / \
        np.clip(np.clip(clip_score, 0, None) + np.clip(reference_score, 0, None), 1e-6, None)

    y = frame.score.to_numpy(float)
    size = frame.groupby(['imgid', 'mt'])['score'].transform('size').to_numpy()
    singleton_zero = (size == 1) & (y == 0)
    multi = size > 1

    def tau(mask, scores):
        return round(float(kendalltau(scores[mask], y[mask], variant='c')[0]), 4)

    entry = dict(rows=int(len(frame)), singleton_zero_rows=int(singleton_zero.sum()),
                 singleton_zero_share=round(float(singleton_zero.mean()), 4),
                 multi_rows=int(multi.sum()))
    for label, scores in [('clipscore', clip_score), ('refclipscore', harmonic),
                          ('reference_only', reference_score)]:
        entry[label] = dict(
            all=tau(np.ones(len(y), bool), scores),
            without_singleton_zero=tau(~singleton_zero, scores),
            multi_annotated_only=tau(multi, scores),
            auc_singleton_zero=round(float(roc_auc_score(singleton_zero, -scores)), 4))
    report[name] = entry
    print(name, entry)


def main():
    report = {}
    analyse('spica_test', spica_test(), report)
    analyse('polaris_test', polaris_test(), report)
    write_json(OUT / 'clip_probe.json', report)


if __name__ == '__main__':
    main()
