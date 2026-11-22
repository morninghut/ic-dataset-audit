"""Metrics of one or more models on the benchmark row files, with the paper's protocols."""
import numpy as np

from lib import CACHE, EVAL_SETS, Features, load_rows, read_json, score, tau

PAIRWISE = {'pascal50s': False, 'foil_1ref': True, 'foil_4ref': True}  # value = strict (A must win)
PAPER_METRIC = {'flickr8k_cf': 'kendall_tau_b'}


def measure(name, predictions, rows):
    """Metric of one prediction column, in the protocol the papers report for this benchmark."""
    if name in PAIRWISE:
        from polos.evaluation_data import pairwise_metrics
        result = pairwise_metrics(read_json(CACHE / f'{name}_pairs.json'), predictions, strict=PAIRWISE[name])
        return dict(score=100 * result['accuracy'], by_category=result['by_category'], ties=result['ties'])
    variant = 'b' if PAPER_METRIC.get(name) == 'kendall_tau_b' else 'c'
    return dict(score=100 * tau(predictions, rows['target'], variant),
                kendall_tau_b=tau(predictions, rows['target'], 'b'),
                kendall_tau_c=tau(predictions, rows['target'], 'c'))


def evaluate(features, models, names=EVAL_SETS, keep=None):
    """models: {label: (weight, bias)}. Returns metrics and, if keep is given, per-row predictions."""
    labels = list(models)
    results, saved = {}, {}
    for name in names:
        rows = load_rows(name, CACHE)
        predictions = score(features, rows, [models[k][0] for k in labels], [models[k][1] for k in labels])
        results[name] = {label: measure(name, np.ascontiguousarray(predictions[:, i]).astype(np.float64), rows)
                         for i, label in enumerate(labels)}
        if keep is not None:
            saved[name] = predictions
        print(name, {label: round(results[name][label]['score'], 2) for label in labels}, flush=True)
    if keep is not None:
        np.savez(keep, **{f'{name}': saved[name] for name in saved}, labels=np.array(labels))
    return results
