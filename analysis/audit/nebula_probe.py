"""The same trivial-row probe on the Nebula test set, the benchmark Pearl reports.

Nebula gives one averaged human score per sample, so there is no annotation
count to key on; instead we use the score itself to show how much of the
correlation comes from the zero-score rows, and we report how a text-only
similarity (candidate vs references, no image at all) scores on it.

Run with PYTHONPATH pointing at Polos-revised so `import polos.clip` resolves.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NEBULA, OUT, write_json
from clip_probe import encode, unit


def load(split='test'):
    frames = []
    for path in sorted((NEBULA / 'data').glob(f'{split}-*.parquet')):
        frames.append(pq.read_table(path).to_pandas())
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


def main():
    frame = load('test')
    with tempfile.TemporaryDirectory() as folder:
        paths = []
        for name, image in zip(frame.file_name, frame.image):
            path = Path(folder) / name
            path.write_bytes(image['bytes'] if isinstance(image, dict) else image)
            paths.append(str(path))
        references = [list(r) for r in frame.refs]
        texts = sorted({s for refs in references for s in refs} | set(frame.mt.astype(str)))
        text_index = {s: i for i, s in enumerate(texts)}
        text_features, image_features = encode(texts, paths)
    t, v = unit(text_features), unit(image_features)
    candidate = np.array([text_index[str(c)] for c in frame.mt])
    clip_score = np.sum(t[candidate] * v[np.arange(len(frame))], axis=1)
    reference_score = np.array([max(float(np.dot(t[text_index[str(c)]], t[text_index[s]])) for s in refs)
                                for c, refs in zip(frame.mt, references)])
    harmonic = 2 * np.clip(clip_score, 0, None) * np.clip(reference_score, 0, None) / \
        np.clip(np.clip(clip_score, 0, None) + np.clip(reference_score, 0, None), 1e-6, None)
    y = frame.human_score.to_numpy(float)
    zero = y == 0

    def tau(scores, mask=None):
        mask = np.ones(len(y), bool) if mask is None else mask
        return dict(kendall_tau_c=round(float(kendalltau(scores[mask], y[mask], variant='c')[0]), 4),
                    kendall_tau_b=round(float(kendalltau(scores[mask], y[mask], variant='b')[0]), 4))

    report = dict(rows=int(len(frame)), zero_score_rows=int(zero.sum()),
                  zero_score_share=round(float(zero.mean()), 4),
                  distinct_scores=int(len(set(np.round(y, 6)))))
    for name, scores in [('clipscore', clip_score), ('refclipscore', harmonic),
                         ('reference_only_no_image', reference_score)]:
        report[name] = dict(all=tau(scores), without_zero_rows=tau(scores, ~zero),
                            auc_zero_rows=round(float(roc_auc_score(zero, -scores)), 4))
    write_json(OUT / 'nebula_probe.json', report)
    print(report)


if __name__ == '__main__':
    main()
