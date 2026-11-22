"""Split cached predictions of the Polaris-trained models into rows whose image
is in the Polaris training pool and rows whose image is not.

This is the cheap preview of how much a benchmark score moves when the training
images overlap the benchmark images; the same masks will be applied to the
Spica-trained models in step 2.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, read_json, write_json

RUNS = ROOT / 'runs/revision'
MODELS = ['cached_reprod', 'ridge_siglip', 'ridge_extended_clip']


def correlations(prediction, target):
    if len(prediction) < 10 or np.std(target) == 0:
        return None
    return dict(rows=int(len(prediction)),
                kendall_tau_c=float(kendalltau(prediction, target, variant='c')[0]),
                kendall_tau_b=float(kendalltau(prediction, target, variant='b')[0]),
                pearson=float(pearsonr(prediction, target)[0]))


def pairwise_accuracy(data, prediction, pairs, strict):
    """FOIL / PASCAL-50S style: rows come in a/b pairs keyed by id.

    `strict` reproduces evaluate.py: FOIL counts a tie as wrong, PASCAL-50S
    resolves a tie in favour of side b.
    """
    index = {str(i): k for k, i in enumerate(data['ids'])}
    correct, used = [], []
    for pair in pairs:
        a = index.get(f"{pair.get('trial', 0)}:{pair['id']}:a")
        b = index.get(f"{pair.get('trial', 0)}:{pair['id']}:b")
        if a is None or b is None:
            continue
        correct.append(bool(prediction[a] > prediction[b]) if strict
                       else (0 if prediction[a] > prediction[b] else 1) == pair['preferred'])
        used.append((a, b))
    return np.array(correct), used


def main():
    masks = np.load(OUT / 'leakage_masks.npz')
    report = {}
    for model in MODELS:
        folder = RUNS / model / 'evaluation'
        if not folder.exists():
            continue
        entry = {}
        for path in sorted(folder.glob('*.npz')):
            name = path.stem
            key = f'{name}__polaris_train'
            if key not in masks:
                continue
            data = np.load(path, allow_pickle=True)
            prediction, target = data['prediction'], data['target']
            mask = masks[key]
            if len(mask) != len(prediction):
                continue
            pairs_file = ROOT / f'feature_cache_v1/{name}_pairs.json'
            if pairs_file.exists():
                pairs = read_json(pairs_file)
                correct, used = pairwise_accuracy(data, prediction, pairs, name.startswith('foil_'))
                leaked = np.array([mask[a] or mask[b] for a, b in used])
                entry[name] = dict(
                    kind='pairwise',
                    all=dict(pairs=int(len(correct)), accuracy=float(correct.mean())),
                    leaked=dict(pairs=int(leaked.sum()), accuracy=float(correct[leaked].mean()) if leaked.any() else None),
                    clean=dict(pairs=int((~leaked).sum()), accuracy=float(correct[~leaked].mean()) if (~leaked).any() else None))
            else:
                entry[name] = dict(
                    kind='correlation', all=correlations(prediction, target),
                    leaked=correlations(prediction[mask], target[mask]),
                    clean=correlations(prediction[~mask], target[~mask]))
        report[model] = entry
    write_json(OUT / 'leakage_sensitivity.json', report)

    for model, entry in report.items():
        print('==', model)
        for name, stats in entry.items():
            if stats['kind'] == 'pairwise':
                print(f"  {name:16s} all {stats['all']['accuracy']:.3f} "
                      f"leaked {stats['leaked']['accuracy'] if stats['leaked']['accuracy'] is None else round(stats['leaked']['accuracy'],3)} "
                      f"({stats['leaked']['pairs']}) clean {round(stats['clean']['accuracy'],3) if stats['clean']['accuracy'] else None} ({stats['clean']['pairs']})")
            else:
                def tc(part):
                    return 'n/a' if not stats[part] else f"{stats[part]['kendall_tau_c']:.3f}({stats[part]['rows']})"
                print(f"  {name:16s} all {tc('all')} leaked {tc('leaked')} clean {tc('clean')}")


if __name__ == '__main__':
    main()
