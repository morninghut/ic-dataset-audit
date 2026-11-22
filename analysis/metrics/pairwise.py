"""Which metrics can each benchmark actually tell apart?

protocol.py prints one interval per metric, which says nothing about whether two metrics
differ: both are scored on the same pictures, so their errors move together.  Here every
duplication-free metric of protocol.py is resampled on the same pictures and the difference
of every two of them is tested, for the same three quantities:

  all           100 x Kendall tau over every row
  above_floor   the same over the rows scored above the floor of the scale
  order         accuracy of ordering two captions of the same picture

For each quantity it reports, per metric, the interval of its rank over the resamples, how
often it is first, and how many other metrics it is separated from; per benchmark, the
share of metric pairs that are separated and the group that cannot be told from the best
one.  The p-value of a difference is the two-sided normal one for the point estimate over
its bootstrap standard error (a percentile p-value cannot get below 2 / resamples, which
is too coarse to adjust over ~300 pairs).  A pair counts as separated when its
Holm-adjusted p-value over all pairs of that table is below 0.05; the unadjusted 95%
percentile interval is kept next to it.

Two paired contrasts are added: a model trained on a pool that contains pairs of the set
("+dup") against the best clean metric, and every metric against the human leave-one-out
score on the rows that have one (tau), or the human holding the other half of the
annotations (order).

    pairwise.py [resamples=10000] [repeats=20] [seeds=12] [workers=48]

seeds is a comma-separated list of the DENEB and Pearl retrainings.  With several seeds a
retrained metric is the mean of its seeds: every quantity is computed per seed on the same
resample and averaged, so the table ranks the expected outcome of one training, and the
per-seed values are kept next to it.  A seed that is not registered for a metric is left out
of that metric's mean (printed).  The interval still covers the evaluation sample only.

Run with PYTHONPATH=Polos-revised.  Writes runs/metrics/pairwise_s<seeds joined by ->.json.
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, norm

STEP2 = Path((_ROOT + '/analysis/probe'))
sys.path.insert(0, str(STEP2))
import discrimination  # noqa: E402
import validity  # noqa: E402
from analyze_effects import finite  # noqa: E402
from effects import Benchmark, bootstrap  # noqa: E402
from lib import ROOT, write_json  # noqa: E402
from protocol import CLEANED, SETS, metrics_for  # noqa: E402

LEVEL = 0.05
CHUNK = 50
STATE = {}   # filled before the pool forks: task name -> (benchmark, columns, selection)


def draw_chunk(job):
    task, count, seed = job
    benchmark, columns, selection = STATE[task]
    return bootstrap(benchmark, columns, selection, count, seed)


def tau_draws(pool, task, resamples, seeds):
    jobs = [(task, min(CHUNK, resamples - start), seeds.spawn(1)[0])
            for start in range(0, resamples, CHUNK)]
    return np.concatenate(pool.map(draw_chunk, jobs))


def order_draws(data, labels, resamples, rng):
    """Overall ordering accuracy per resample.  It is a ratio of per-picture sums, so a
    resample of pictures is a weighted sum of those."""
    keep = data['gap'] != 0
    unit, units = data['unit'][keep], data['units']
    total = np.bincount(unit, minlength=units).astype(np.float64)
    hits = np.stack([np.bincount(unit, weights=data['correct'][label][keep], minlength=units)
                     for label in labels], 1)
    weights = rng.multinomial(units, np.full(units, 1 / units), size=resamples).astype(np.float64)
    return 100 * hits.sum(0) / total.sum(), 100 * (weights @ hits) / (weights @ total)[:, None]


def contrast(point, draws):
    """A difference: point estimate, percentile interval, and normal p-value from the bootstrap SE."""
    low, high = np.percentile(draws, [2.5, 97.5])
    spread = draws.std(ddof=1)
    p = 2 * norm.sf(abs(point) / spread) if spread > 0 else float(point == 0)
    return dict(mean=float(point), low=float(low), high=float(high), p=float(p))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    out = np.empty(len(p))
    out[order] = adjusted
    return out


def gather(name, seeds):
    """Metrics of a set over the seeds: display names, clean flags, the distinct specs to
    score, and per metric the positions of its seeds' specs among them."""
    names, clean, specs, members, used = [], [], [], [], {}
    for seed in seeds:
        for display, spec, duplicated in metrics_for(name, seed):
            if display not in names:
                names.append(display)
                clean.append(not duplicated)
                members.append([])
            if spec not in specs:
                specs.append(spec)
            position = specs.index(spec)
            if position not in members[names.index(display)]:
                members[names.index(display)].append(position)
                used.setdefault(display, []).append(seed)
    return names, clean, specs, members, {display: used[display] for display, m in zip(names, members) if len(m) > 1}


def reduce(values, members):
    """(..., specs) -> (..., metrics): the mean over each metric's seeds."""
    return np.stack([values[..., member].mean(-1) for member in members], -1)


def rank_agreement(a, b):
    """Kendall tau between the rankings two quantities give the same clean metrics."""
    common = sorted(set(a) & set(b))
    return float(kendalltau([a[m] for m in common], [b[m] for m in common])[0]) if len(common) > 2 else None


def summarize(names, clean, point, draws, seed_points=None):
    """Rank intervals, all pairwise contrasts of the clean metrics, and the +dup contrasts."""
    point, clean = np.asarray(point, float), np.asarray(clean, bool)
    index = np.flatnonzero(clean)
    index = index[np.argsort(-point[index])]                      # clean metrics, best first
    ranks = (-draws[:, index]).argsort(1).argsort(1) + 1
    pairs = [(i, j) for a, i in enumerate(index) for j in index[a + 1:]]
    cells = [contrast(point[i] - point[j], draws[:, i] - draws[:, j]) for i, j in pairs]
    for cell, adjusted in zip(cells, holm([cell['p'] for cell in cells])):
        cell['p_holm'] = float(adjusted)
    separated = {(i, j): cell['p_holm'] < LEVEL for (i, j), cell in zip(pairs, cells)}
    apart = lambda i, j: separated.get((i, j), separated.get((j, i), False))  # noqa: E731
    rows = []
    for position, i in enumerate(index):
        low, high = np.percentile(ranks[:, position], [2.5, 97.5])
        span = np.percentile(draws[:, i], [2.5, 97.5])
        rows.append(dict(metric=names[i], score=float(point[i]), low=float(span[0]), high=float(span[1]),
                         seeds=None if seed_points is None or len(seed_points[i]) < 2 else seed_points[i],
                         rank=position + 1,
                         rank_low=int(low), rank_high=int(high), first=float(np.mean(ranks[:, position] == 1)),
                         separated_from=int(sum(apart(i, j) for j in index if j != i))))
    best = index[0]
    out = dict(rows=rows,
               pairs=[dict(a=names[i], b=names[j], **cell) for (i, j), cell in zip(pairs, cells)],
               pair_count=len(pairs),
               separated_holm=int(sum(separated.values())),
               separated_interval=int(sum(cell['low'] > 0 or cell['high'] < 0 for cell in cells)),
               top_group=[names[i] for i in index if i == best or not apart(best, i)],
               adjacent=[dict(a=names[i], b=names[j], **contrast(point[i] - point[j], draws[:, i] - draws[:, j]))
                         for i, j in zip(index[:-1], index[1:])],
               duplicated=[])
    for i in np.flatnonzero(~clean):
        base = names[i].removesuffix(' +dup').removesuffix(CLEANED)
        entry = dict(metric=names[i], score=float(point[i]),
                     seeds=None if seed_points is None or len(seed_points[i]) < 2 else seed_points[i],
                     minus_best_clean=dict(metric=names[best], **contrast(point[i] - point[best], draws[:, i] - draws[:, best])),
                     first=float(np.mean(draws[:, i] > draws[:, index].max(1))))
        if base in names:
            own = names.index(base)
            entry['minus_ranked_version'] = contrast(point[i] - point[own], draws[:, i] - draws[:, own])
        out['duplicated'].append(entry)
    return out


def to_human(names, clean, point, draws, human_point, human_draws):
    return dict(human=float(human_point),
                rows=[dict(metric=name, duplicated=not clean[i], score=float(point[i]),
                           **contrast(point[i] - human_point, draws[:, i] - human_draws))
                      for i, name in enumerate(names)])


def show(title, table):
    print(f"\n  {title}: {table['separated_holm']} of {table['pair_count']} pairs of clean metrics separated "
          f"(Holm), {table['separated_interval']} by the unadjusted interval; "
          f"{len(table['top_group'])} cannot be told from the best")
    print(f"  {'rank':>4s} {'metric':36s} {'score':>7s} {'rank 95%':>9s} {'P(first)':>9s} {'separated from':>15s}"
          f" {'minus next [95%]':>24s}")
    gaps = table['adjacent'] + [None]
    for row, gap in zip(table['rows'], gaps):
        step = f"{gap['mean']:+6.2f} [{gap['low']:+5.2f},{gap['high']:+5.2f}]" if gap else ''
        print(f"  {row['rank']:4d} {row['metric']:36s} {row['score']:7.2f} "
              f"{'%d-%d' % (row['rank_low'], row['rank_high']):>9s} {row['first']:9.3f} "
              f"{'%d of %d' % (row['separated_from'], len(table['rows']) - 1):>15s} {step:>24s}")
    for entry in table['duplicated']:
        best, own = entry['minus_best_clean'], entry.get('minus_ranked_version')
        print(f"  {'+d':>4s} {entry['metric']:36s} {entry['score']:7.2f}   above every clean metric in "
              f"{100 * entry['first']:.1f}% of resamples; minus best clean ({best['metric']}) "
              f"{best['mean']:+.2f} [{best['low']:+.2f},{best['high']:+.2f}]"
              + (f"; minus its ranked version {own['mean']:+.2f} [{own['low']:+.2f},{own['high']:+.2f}]" if own else ''))


def show_human(title, table):
    rows = [row for row in table['rows'] if not row['duplicated']]
    above = [row for row in rows if row['low'] > 0]
    below = [row for row in rows if row['high'] < 0]
    print(f"\n  {title}: human {table['human']:.2f}; of {len(rows)} clean metrics {len(above)} above it, "
          f"{len(below)} below, {len(rows) - len(above) - len(below)} not separated "
          f"(unadjusted 95% interval of the paired difference)")
    for row in sorted(table['rows'], key=lambda row: -row['mean'])[:8]:
        print(f"       {row['metric']:36s} {row['score']:7.2f}  minus human {row['mean']:+6.2f} "
              f"[{row['low']:+5.2f},{row['high']:+5.2f}] p={row['p']:.4f}")


def main(resamples=10000, repeats=20, seeds='12', workers=48):
    resamples, repeats, workers = int(resamples), int(repeats), int(workers)
    seeds = [int(seed) for seed in str(seeds).split(',')]
    spawner = np.random.SeedSequence(20261114)
    out = {}
    for name in SETS:
        names, clean, specs, members, averaged = gather(name, seeds)
        benchmark = Benchmark(name)
        groups, leave_one_out, counts = validity.masks(benchmark.rows)
        predictions = [column for _, column in validity.columns(specs, name)]
        judged = np.isfinite(leave_one_out)
        STATE.clear()
        for mask in ('all', 'above_floor'):
            STATE[mask] = (benchmark, predictions, benchmark.select(groups[mask]))
            if judged.any():
                STATE[mask + '_judged'] = (benchmark, predictions + [leave_one_out],
                                           benchmark.select(groups[mask] & judged))
        print(f"\n# {name}: {counts['rows']} rows, {counts['floor']} at the floor, "
              f"{int(judged.sum())} with a human leave-one-out score; {sum(clean)} clean metrics, "
              f"{resamples} resamples; seed means: "
              + ('; '.join(f'{display} {used}' for display, used in averaged.items()) or 'none'), flush=True)
        entry = dict(averaged=averaged)
        with Pool(workers) as pool:
            for task, (_, columns, selection) in STATE.items():
                raw = np.array([benchmark.measure(column, selection.items) for column in columns])
                raw_draws = tau_draws(pool, task, resamples, spawner)
                point, draws = reduce(raw[:len(specs)], members), reduce(raw_draws[:, :len(specs)], members)
                per_seed = [raw[member].tolist() for member in members]
                if task.endswith('_judged'):
                    entry[task] = dict(items=int(len(selection.items)),
                                       **to_human(names, clean, point, draws, raw[-1], raw_draws[:, -1]))
                    show_human(f'{task} ({len(selection.items)} rows)', entry[task])
                else:
                    entry[task] = dict(items=int(len(selection.items)),
                                       **summarize(names, clean, point, draws, per_seed))
                    show(f'{task} ({len(selection.items)} rows)', entry[task])
                sys.stdout.flush()
        data = discrimination.outcomes(name, specs, repeats, np.random.default_rng(0))
        if data is not None:
            raw, raw_draws = order_draws(data, specs + ['human'], resamples,
                                         np.random.default_rng(spawner.spawn(1)[0]))
            point, draws = reduce(raw[:-1], members), reduce(raw_draws[:, :-1], members)
            entry['order'] = dict(items=data['pairs'], **summarize(names, clean, point, draws,
                                                                    [raw[member].tolist() for member in members]))
            show(f"order ({data['pairs']} caption pairs)", entry['order'])
            entry['order_human'] = dict(items=data['pairs'],
                                        **to_human(names, clean, point, draws, raw[-1], raw_draws[:, -1]))
            show_human('order against the human with the other half', entry['order_human'])
        ranks = {q: {row['metric']: row['rank'] for row in entry[q]['rows']} for q in ('all', 'above_floor', 'order')
                 if q in entry}
        entry['rank_agreement'] = {q: rank_agreement(ranks['all'], ranks[q]) for q in ranks if q != 'all'}
        print('  rank agreement with the all-rows ranking: '
              + ', '.join(f'{q} {value:+.2f}' for q, value in entry['rank_agreement'].items()))
        out[name] = entry
        sys.stdout.flush()
    tag = '-'.join(map(str, seeds))
    write_json(ROOT / f'runs/metrics/pairwise_s{tag}.json',
               finite(dict(sets=SETS, seeds=seeds, resamples=resamples, repeats=repeats, level=LEVEL, benchmarks=out)))


if __name__ == '__main__':
    main(*sys.argv[1:])
