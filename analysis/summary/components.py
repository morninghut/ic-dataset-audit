"""Split Kendall's tau over all samples into its two parts.

Let L be the samples that carry the lowest human score and A the others.  Pairs of two L samples
are tied in the human score and contribute nothing to tau.  Every pair of one L and one A sample
is ordered by the humans, and the metric orders it correctly with probability AUC, the area under
the ROC curve of the metric for telling A from L.  So, with S = concordant - discordant pairs,

    S(all) = |L| |A| (2 AUC - 1)  +  S(A)

and tau over all samples is a fixed mixture of a detection term (AUC) and of the ranking among the
samples above the lowest score, with weights set by the share of lowest-score samples alone.

    components.py [resamples=1000]     (PYTHONPATH=Polos-revised, from analysis/summary)

Uses the metrics, seeds and ranking rule of metrics/pairwise.py.  Writes
runs/summary/components.json.
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata

ROOT = Path(_ROOT)
sys.path.insert(0, str(ROOT / 'analysis/probe'))
sys.path.insert(0, str(ROOT / 'analysis/metrics'))
import validity  # noqa: E402
from effects import Benchmark, TAU_B  # noqa: E402
from lib import sample_structure  # noqa: E402
from pairwise import gather, reduce  # noqa: E402
from protocol import SETS  # noqa: E402

SEEDS = [12, 13, 14]


def concordance(x, y):
    """S = concordant - discordant pairs, from tau-b and its tie-corrected denominator."""
    n = len(x)
    def tied(v):
        counts = np.unique(v, return_counts=True)[1]
        return (counts * (counts - 1) // 2).sum()
    n0 = n * (n - 1) // 2
    return float(kendalltau(x, y, variant='b')[0] * np.sqrt(float(n0 - tied(x)) * float(n0 - tied(y))))


def auc(score, positive):
    ranks = rankdata(score)
    n1 = positive.sum()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(score) - n1)))


def main(resamples=1000):
    resamples = int(resamples)
    out = {}
    for name in SETS:
        names, clean, specs, members, _ = gather(name, SEEDS)
        benchmark = Benchmark(name)
        columns = [column for _, column in validity.columns(specs, name)]
        target = benchmark.target
        above = target > target.min()
        n_low, n_above = int((~above).sum()), int(above.sum())
        variant = 'b' if name in TAU_B else 'c'
        rows = np.empty((len(specs), 5))
        for index, column in enumerate(columns):
            a = auc(column, above)
            cross = n_low * n_above * (2 * a - 1)
            within = concordance(column[above], target[above])
            total = concordance(column, target)
            assert abs(cross + within - total) < 1e-6 * max(1.0, abs(total)), (name, specs[index], cross + within, total)
            rows[index] = [a, cross / total, 100 * kendalltau(column, target, variant=variant)[0],
                           100 * kendalltau(column[above], target[above], variant=variant)[0], cross]
        # Interval of the AUC over resampled images.
        rng = np.random.default_rng(0)
        selection = benchmark.select(np.ones(len(target), bool))
        draws = np.empty((resamples, len(specs)))
        for r in range(resamples):
            items = selection.resample(rng)
            draws[r] = [auc(column[items], above[items]) for column in columns]
        rows, draws = reduce(rows.T, members).T, reduce(draws, members)
        size = sample_structure(benchmark.rows)['size']
        levels, levels_above = len(np.unique(target)), len(np.unique(target[above]))
        out[name] = dict(
            samples=len(target), lowest=n_low, above=n_above, lowest_share=n_low / len(target),
            lowest_with_single_rating=float((size[~above] == 1).mean()), variant=variant,
            cross_pairs=n_low * n_above, within_pairs=n_above * (n_above - 1) // 2,
            # tau_c(all) = weight_detection * (2 AUC - 1) + weight_ranking * tau_c(above)
            weight_detection=float(2 * levels / (levels - 1) * n_low * n_above / len(target) ** 2),
            weight_ranking=float((levels_above - 1) / levels_above * levels / (levels - 1) * (n_above / len(target)) ** 2),
            metrics=[dict(metric=m, ranked=bool(c), auc=float(r[0]), auc_low=float(np.percentile(d, 2.5)),
                          auc_high=float(np.percentile(d, 97.5)), cross_share=float(r[1]), tau_all=float(r[2]), tau_above=float(r[3]))
                     for m, c, r, d in zip(names, clean, rows, draws.T)])
        print(f"\n# {name}: lowest {n_low / len(target):.3f}, weights detection {out[name]['weight_detection']:.3f} ranking {out[name]['weight_ranking']:.3f}")
        for cell in sorted(out[name]['metrics'], key=lambda c: -c['tau_all']):
            print(f"{cell['metric']:36s} {'' if cell['ranked'] else '(unranked)':10s} tau {cell['tau_all']:6.2f} above {cell['tau_above']:6.2f} AUC {cell['auc']:.3f} [{cell['auc_low']:.3f},{cell['auc_high']:.3f}] from detection {100 * cell['cross_share']:5.1f}%")
    (ROOT / 'runs/summary/components.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:2])
