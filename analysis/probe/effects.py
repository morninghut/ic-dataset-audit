"""Metrics on arbitrary row subsets, and picture-clustered bootstrap of differences.

Every benchmark is reduced to items (rows, or pairs for the pairwise protocols) grouped into
independent units (one picture). A Selection is any subset of items together with the units
that carry them; both the point estimate and the resampling work on its units, so a picture is
never split between resamples and the pairwise protocols stay intact.
"""
import numpy as np

from lib import CACHE, load_rows, read_json, row_pictures, tau

STRICT = {'foil_1ref': True, 'foil_4ref': True, 'pascal50s': False}
TAU_B = {'flickr8k_cf'}


class Selection:
    """Items of chosen units, stored ragged so that resampling units is a single gather."""

    def __init__(self, units, groups):
        self.units = units
        self.lengths = np.array([len(g) for g in groups], np.int64)
        self.offsets = np.concatenate(([0], np.cumsum(self.lengths)[:-1])) if len(groups) else np.empty(0, np.int64)
        self.items = np.concatenate(groups) if len(groups) else np.empty(0, np.int64)

    def __len__(self):
        return len(self.units)

    def resample(self, rng):
        draw = rng.integers(len(self.lengths), size=len(self.lengths))
        counts = self.lengths[draw]
        total = int(counts.sum())
        position = np.arange(total) - np.repeat(np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
        return self.items[np.repeat(self.offsets[draw], counts) + position]


class Benchmark:
    """Items of one evaluation set grouped by picture, with the protocol used to report it."""

    def __init__(self, name):
        self.name = name
        self.rows = load_rows(name, CACHE)
        self.pairwise = name in STRICT
        pictures = row_pictures(self.rows)
        if self.pairwise:
            self.records = read_json(CACHE / f'{name}_pairs.json')
            if 2 * len(self.records) != len(pictures):
                raise ValueError(f'{name}: pairs and rows disagree')
            self.pictures = pictures[::2]
            self.preferred = np.array([r['preferred'] for r in self.records])
            keys = [f"{r.get('trial', 0)}\t{r['category']}" for r in self.records]
            names, self.group = np.unique(keys, return_inverse=True)
            self.group = self.group.ravel()
            categories = np.array([key.split('\t')[1] for key in names])
            self.category, self.group_category = np.unique(categories, return_inverse=True)
            self.group_category = self.group_category.ravel()
        else:
            self.pictures = pictures
            self.target = self.rows['target'].astype(np.float64)
        self.item_count = len(self.pictures)
        order = np.argsort(self.pictures, kind='stable')
        self.units, starts = np.unique(self.pictures[order], return_index=True)
        self.groups = np.split(order, starts[1:])

    def select(self, items=None, units=None):
        """Items kept by a boolean item mask, restricted to units kept by a boolean unit mask."""
        keep = np.ones(self.item_count, bool) if items is None else items
        chosen, groups = [], []
        for index, group in enumerate(self.groups):
            if units is not None and not units[index]:
                continue
            group = group[keep[group]]
            if len(group):
                chosen.append(self.units[index])
                groups.append(group)
        return Selection(np.array(chosen, self.units.dtype), groups)

    def measure(self, prediction, items):
        """The benchmark's own protocol, restricted to these items: 100 x tau, or accuracy.

        Pairwise accuracy keeps the published aggregation: mean over categories of the mean
        over trials, so a resample that drops a category simply averages the rest.
        """
        if not len(items):
            return float('nan')
        if not self.pairwise:
            x, y = prediction[items], self.target[items]
            if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                return float('nan')
            return 100 * tau(x, y, 'b' if self.name in TAU_B else 'c')
        a, b = prediction[2 * items], prediction[2 * items + 1]
        correct = (a > b) if STRICT[self.name] else (np.where(a > b, 0, 1) == self.preferred[items])
        group = self.group[items]
        size = len(self.group_category)
        counts = np.bincount(group, minlength=size)
        totals = np.bincount(group, weights=correct.astype(np.float64), minlength=size)
        present = counts > 0
        rate = np.divide(totals, counts, out=np.zeros(size), where=present)
        category_sum = np.bincount(self.group_category[present], weights=rate[present],
                                   minlength=len(self.category))
        category_count = np.bincount(self.group_category[present], minlength=len(self.category))
        seen = category_count > 0
        return 100 * float(np.mean(category_sum[seen] / category_count[seen]))

    def unit_mask(self, pictures):
        return np.isin(self.units, pictures)


def bootstrap(benchmark, columns, selection, resamples=1000, seed=0):
    """Resample the selection's units with replacement; the statistic of every column."""
    rng = np.random.default_rng(seed)
    out = np.empty((resamples, len(columns)))
    for r in range(resamples):
        items = selection.resample(rng)
        out[r] = [benchmark.measure(column, items) for column in columns]
    return out


def interval(values, level=95):
    values = np.asarray(values, float)
    low, high = np.nanpercentile(values, [(100 - level) / 2, 100 - (100 - level) / 2])
    return dict(mean=float(np.nanmean(values)), low=float(low), high=float(high),
                positive_fraction=float(np.mean(values > 0)))
