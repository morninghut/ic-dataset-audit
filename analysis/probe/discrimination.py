"""Can a metric order two captions of the same picture as well as a human annotator can?

For every picture the evaluation set contains several judged captions. Each judged caption
("sample") carries k annotations, so its annotations can be split into two halves of the same
size (one annotation is dropped when k is odd): one half is the reference the ordering is
scored against, the other half is a human competitor holding exactly as much information as
the reference. A metric and that human are measured on the same caption pairs against the same
reference half, which makes the human accuracy the ceiling of the comparison rather than an
unreachable "true" ordering. A competitor that cannot separate the two captions counts as half
a success, the value of guessing, so the human is not punished for the ties a coarse scale
produces.

Pairs are grouped by how far apart the reference half puts the two captions, measured as a
fraction of that set's own score range, because an
evaluation set whose correlation comes from easy pairs looks strong while being unable to
separate two metrics on anything else. Both halves are used in turn and the split is repeated,
with picture-clustered bootstrap over the repeated splits.

Usage: discrimination.py [resamples] [repeats] [dir[:label] ...]
"""
import sys

import numpy as np

from analyze_effects import finite
from lib import CACHE, RUNS, load_rows, sample_structure, write_json
from validity import columns

SETS = ['composite', 'flickr8k_expert', 'flickr8k_cf', 'polaris_test', 'nebula_test', 'spica_test']
DEFAULT = ['baselines/none:clipscore', 'baselines/none:refclipscore', 'learned']
# The trained metric shown on a set is always one whose training pool shares no
# (picture, candidate) pair with that set, so the ceiling comparison is not read off
# a memorized label.
LEARNED = {'composite': 'siglip/polaris_train:released',
           'flickr8k_expert': 'siglip/polaris_train:released',
           'flickr8k_cf': 'siglip/polaris_train:released',
           'polaris_test': 'siglip/polaris_train:released',
           'nebula_test': 'siglip/spica_train:released',
           'spica_test': 'siglip/spica_train:released'}
EDGES = [0.0, 0.1, 0.2, 0.3, 0.5, 1.01]
REPEATS = 20
MINIMUM = 2


def samples(name):
    """Per sample: picture, first row (for predictions), and the annotations it carries."""
    rows = load_rows(name, CACHE)
    structure = sample_structure(rows)
    index = structure['sample']
    order = np.argsort(index, kind='stable')
    starts = np.flatnonzero(np.diff(index[order], prepend=-1))
    groups = np.split(order, starts[1:])
    keep = [group for group in groups if len(group) >= MINIMUM]
    first = np.array([group[0] for group in keep])
    target = rows['target'].astype(np.float64)
    return rows, first, [target[group] for group in keep]


def pairs_of(pictures):
    """All unordered pairs of samples that belong to the same picture."""
    left, right = [], []
    order = np.argsort(pictures, kind='stable')
    starts = np.flatnonzero(np.diff(pictures[order], prepend=pictures[order][0] - 1))
    for group in np.split(order, starts[1:]):
        if len(group) < 2:
            continue
        a, b = np.triu_indices(len(group), 1)
        left.append(group[a])
        right.append(group[b])
    return (np.concatenate(left), np.concatenate(right)) if left else (np.empty(0, int),) * 2


def halves(annotations, rng):
    """Two disjoint equal-sized means per sample from a random split of its annotations."""
    first = np.empty(len(annotations))
    second = np.empty(len(annotations))
    for position, values in enumerate(annotations):
        draw = rng.permutation(len(values))
        cut = len(values) // 2
        first[position] = values[draw[:cut]].mean()
        second[position] = values[draw[cut:2 * cut]].mean()
    return first, second


def outcome(difference, gap):
    """1 when the ordering agrees with the reference, 0.5 when the competitor cannot separate."""
    sign = np.sign(difference)
    return np.where(sign == 0, 0.5, (sign == np.sign(gap)).astype(np.float64))


def accuracies(gap, correct, bins, weights):
    """Weighted accuracy per gap bin and over all pairs."""
    index = np.digitize(np.abs(gap), bins) - 1
    out = np.empty(len(bins))
    for position in range(len(bins) - 1):
        keep = (index == position) & (gap != 0)
        total = weights[keep].sum()
        out[position] = (weights[keep] * correct[keep]).sum() / total if total else np.nan
    keep = gap != 0
    out[-1] = (weights[keep] * correct[keep]).sum() / weights[keep].sum()
    return out


def outcomes(name, specs, repeats, rng):
    """Every same-picture caption pair under every split: how far the reference half puts the
    two captions apart, and whether each competitor ordered them like the reference.

    Returns None when the set has no picture with two multiply judged captions.  "unit" is the
    picture of each entry of gap, as an index into the pictures that carry pairs.
    """
    rows, first, annotations = samples(name)
    if not len(first):
        return None
    target = rows['target'].astype(np.float64)
    scale = float(target.max() - target.min()) or 1.0
    pictures = rows['image'][first]
    left, right = pairs_of(pictures)
    if not len(left):
        return None
    unit_ids, inverse = np.unique(pictures[left], return_inverse=True)
    labels, predictions = zip(*columns([LEARNED[name] if spec == 'learned' else spec
                                        for spec in specs], name))
    scores = [column[first] for column in predictions]
    gaps, correct = [], {label: [] for label in list(labels) + ['human']}
    for _ in range(repeats):
        a, b = halves(annotations, rng)
        for reference, competitor in ((a, b), (b, a)):
            gap = (reference[left] - reference[right]) / scale
            gaps.append(gap)
            correct['human'].append(outcome(competitor[left] - competitor[right], gap))
            for label, column in zip(labels, scores):
                correct[label].append(outcome(column[left] - column[right], gap))
    names = list(labels) + ['human']
    return dict(labels=names, gap=np.concatenate(gaps), unit=np.tile(inverse, 2 * repeats),
                units=len(unit_ids), correct={label: np.concatenate(correct[label]) for label in names},
                pairs=int(len(left)), samples=int(len(first)), scale=scale,
                annotations=float(np.mean([len(values) for values in annotations])))


def analyze(name, specs, repeats, resamples):
    rng = np.random.default_rng(0)
    data = outcomes(name, specs, repeats, rng)
    if data is None:
        return None
    names, gap, tile, stacked = data['labels'], data['gap'], data['unit'], data['correct']
    unit_count = data['units']
    ones = np.ones(len(gap))
    point = {label: accuracies(gap, stacked[label], EDGES, ones) for label in names}
    place = np.digitize(np.abs(gap), EDGES) - 1
    counts = np.array([int(((place == position) & (gap != 0)).sum() / (2 * repeats))
                       for position in range(len(EDGES) - 1)] + [int((gap != 0).sum() / (2 * repeats))])
    spans = {label: None for label in names}
    gaps_to_human = {}
    if resamples:
        draws = {label: np.empty((resamples, len(EDGES))) for label in names}
        for r in range(resamples):
            weights = np.bincount(rng.integers(unit_count, size=unit_count),
                                  minlength=unit_count).astype(np.float64)[tile]
            for label in names:
                draws[label][r] = accuracies(gap, stacked[label], EDGES, weights)
        spans = {label: (np.nanpercentile(draws[label], 2.5, axis=0),
                         np.nanpercentile(draws[label], 97.5, axis=0)) for label in names}
        gaps_to_human = {label: (np.nanmean(draws[label] - draws['human'], axis=0),
                                 np.nanpercentile(draws[label] - draws['human'], 2.5, axis=0),
                                 np.nanpercentile(draws[label] - draws['human'], 97.5, axis=0))
                         for label in names if label != 'human'}
    return dict(labels=names, point=point, spans=spans, counts=counts, to_human=gaps_to_human,
                pairs=data['pairs'], samples=data['samples'], pictures=unit_count, scale=data['scale'],
                annotations=data['annotations'])


def main(resamples=0, repeats=REPEATS, *specs):
    resamples, repeats = int(resamples), int(repeats)
    specs = list(specs) or DEFAULT
    out = {}
    headers = [f'{EDGES[i]:.2f}-{EDGES[i + 1]:.2f}' for i in range(len(EDGES) - 1)] + ['all']
    for name in SETS:
        entry = analyze(name, specs, repeats, resamples)
        if entry is None:
            continue
        print(f"\n# {name}: {entry['pairs']} caption pairs over {entry['pictures']} pictures, "
              f"{entry['samples']} multiply judged captions with {entry['annotations']:.1f} "
              f"annotations on average, score range {entry['scale']:g}, {repeats} splits")
        print(f"{'column':44s} " + ' '.join(f'{h:>18s}' for h in headers))
        print(f"{'pairs per split':44s} " + ' '.join(f'{c:>18d}' for c in entry['counts']))
        for label in entry['labels']:
            cells = []
            for position in range(len(EDGES)):
                value = entry['point'][label][position]
                span = entry['spans'][label]
                cells.append(f'{100 * value:5.1f}' + (f' [{100 * span[0][position]:.1f},'
                                                     f'{100 * span[1][position]:.1f}]' if span else ''))
            print(f'{label:44s} ' + ' '.join(f'{cell:>18s}' for cell in cells))
        for label, (mean, low, high) in entry['to_human'].items():
            cells = [f'{100 * mean[i]:+5.1f} [{100 * low[i]:+.1f},{100 * high[i]:+.1f}]'
                     for i in range(len(EDGES))]
            print(f'{label + "  - human":44s} ' + ' '.join(f'{cell:>18s}' for cell in cells))
        out[name] = dict(pairs=entry['pairs'], pictures=entry['pictures'], samples=entry['samples'],
                         annotations=entry['annotations'], scale=entry['scale'],
                         counts=entry['counts'].tolist(), bins=headers,
                         accuracy={label: entry['point'][label].tolist() for label in entry['labels']},
                         minus_human={label: dict(mean=value[0].tolist(), low=value[1].tolist(),
                                                  high=value[2].tolist())
                                      for label, value in entry['to_human'].items()},
                         interval={label: None if entry['spans'][label] is None else
                                   dict(low=entry['spans'][label][0].tolist(),
                                        high=entry['spans'][label][1].tolist())
                                   for label in entry['labels']})
    write_json(RUNS / 'discrimination.json', finite(dict(sets=list(out), specs=specs, bins=headers,
                                                         repeats=repeats, resamples=resamples,
                                                         benchmarks=out)))


if __name__ == '__main__':
    main(*sys.argv[1:])
