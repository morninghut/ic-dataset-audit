"""Training-free reference points scored on every evaluation set, in the model report format.

clipscore     : 2.5 max(0, cos(image, candidate)) with the authors' cached CLIP ViT-B/32.
refclipscore  : harmonic mean of clipscore and the best candidate-reference cosine.
textonly      : the best candidate-reference cosine alone, so the image is never read.
textonly_simcse : the same, with the SimCSE sentence embedding the metric itself uses.

Usage: baselines.py  ->  runs/probe/models/baselines/none/{report.json,predictions/<set>.npy}
"""
import time

import numpy as np

from evaluation import measure
from lib import CACHE, EVAL_SETS, RUNS, load_rows, write_json

LABELS = ['clipscore', 'refclipscore', 'textonly', 'textonly_simcse']


def unit(array):
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12)


def main():
    out = RUNS / 'models' / 'baselines' / 'none'
    (out / 'predictions').mkdir(parents=True, exist_ok=True)
    text = unit(np.load(CACHE / 'clip_text.npy', mmap_mode='r').astype(np.float64))
    image = unit(np.load(CACHE / 'clip_image.npy', mmap_mode='r').astype(np.float64))
    cls = np.load(CACHE / 'simcse_cls.npy', mmap_mode='r')  # [text, layer, dimension]

    def simcse(ids):
        """Last-layer CLS embedding of these texts, unit length, shape ids.shape + [dimension]."""
        flat = np.asarray(ids).reshape(-1)
        order = np.argsort(flat, kind='stable')
        out = np.empty((len(flat), cls.shape[2]), np.float64)
        out[order] = cls[flat[order], -1, :].astype(np.float64)
        return unit(out).reshape(np.shape(ids) + (cls.shape[2],))

    results, started = {}, time.time()
    for name in EVAL_SETS:
        rows = load_rows(name, CACHE)
        candidate, references = rows['candidate'], rows['references']
        valid = references >= 0
        picture = 2.5 * np.maximum(0.0, np.sum(image[rows['image']] * text[candidate], 1))
        similarity = np.einsum('nd,nrd->nr', text[candidate], text[np.maximum(references, 0)])
        reference = np.where(valid, similarity, -np.inf).max(1)
        sentence = np.einsum('nd,nrd->nr', simcse(candidate), simcse(np.maximum(references, 0)))
        columns = dict(clipscore=picture, refclipscore=2 * picture * reference / np.maximum(picture + reference, 1e-12),
                       textonly=reference, textonly_simcse=np.where(valid, sentence, -np.inf).max(1))
        predictions = np.stack([columns[label] for label in LABELS], 1)
        np.save(out / 'predictions' / f'{name}.npy', predictions.astype(np.float32))
        results[name] = {label: measure(name, predictions[:, i], rows) for i, label in enumerate(LABELS)}
        print(name, {k: round(v['score'], 2) for k, v in results[name].items()}, flush=True)
    write_json(out / 'report.json', dict(backbone='clip', head='none', pool='none', labels=LABELS,
                                         evaluation=results, seconds=time.time() - started,
                                         prediction_columns=LABELS))


if __name__ == '__main__':
    main()
