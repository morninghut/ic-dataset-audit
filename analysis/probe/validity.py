"""What the benchmarks measure once their trivial rows are removed.

The audit showed that the rows scored at the bottom of each scale are separated from the
rest by a plain text similarity with AUC ~0.98, so they carry the easy part of the
correlation. Three row sets are reported for every evaluation set: all rows, the rows above
the floor of the scale, and the rows whose sample has more than one annotation, which are
also the only rows where the human leave-one-out ceiling is defined.

All model columns are resampled together on the same pictures, so the reported difference
to the first column is paired and its interval answers "can this benchmark tell the two
metrics apart at all". The human column is resampled on its own rows.

Usage: validity.py [resamples] [dir[:label] ...]
Directories are relative to runs/probe/models, e.g. baselines/none, siglip/polaris_train.
"""
import sys

import numpy as np

from analyze_effects import finite
from effects import Benchmark, bootstrap, interval
from lib import RUNS, read_json, sample_structure, write_json

SETS = ['composite', 'flickr8k_expert', 'flickr8k_cf', 'polaris_test', 'nebula_test', 'spica_test']
DEFAULT = ['baselines/none', 'siglip/polaris_train:released', 'siglip/spica_train:released']
MASKS = ['all', 'above_floor', 'multi']


def columns(specs, name):
    """[(label, prediction)] for every requested model directory and label."""
    out = []
    for spec in specs:
        directory, _, wanted = spec.partition(':')
        path = RUNS / 'models' / directory
        labels = read_json(path / 'report.json')['labels']
        predictions = np.load(path / 'predictions' / f'{name}.npy')
        for index, label in enumerate(labels):
            if wanted and label != wanted:
                continue
            out.append((f'{directory}:{label}', predictions[:, index].astype(np.float64)))
    return out


def human_leave_one_out(rows):
    """Mean score of the other annotators of the same sample; nan where there is only one."""
    structure = sample_structure(rows)
    size, mean = structure['size'], structure['mean']
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(size > 1, (mean * size - rows['target']) / np.maximum(size - 1, 1), np.nan), size


def masks(rows):
    """Row sets: everything, above the floor of the score scale, and multiply judged samples."""
    leave_one_out, size = human_leave_one_out(rows)
    floor = rows['target'] <= rows['target'].min()
    return (dict(all=np.ones(len(size), bool), above_floor=~floor, multi=size > 1), leave_one_out,
            dict(rows=len(size), floor=int(floor.sum()), floor_score=float(rows['target'].min()),
                 multi=int((size > 1).sum())))


def evaluate(benchmark, labels, predictions, keep, resamples):
    """Score of every column on the same items, with paired intervals and differences."""
    selection = benchmark.select(keep)
    out = {label: dict(score=benchmark.measure(column, selection.items), items=int(len(selection.items)))
           for label, column in zip(labels, predictions)}
    if resamples and len(selection) > 1:
        draws = bootstrap(benchmark, predictions, selection, resamples)
        for index, label in enumerate(labels):
            out[label]['bootstrap'] = interval(draws[:, index])
            if index:
                out[label]['difference_to_first'] = interval(draws[:, index] - draws[:, 0])
    return out


def main(resamples=0, *specs):
    resamples = int(resamples)
    specs = list(specs) or DEFAULT
    out = {}
    for name in SETS:
        benchmark = Benchmark(name)
        groups, leave_one_out, counts = masks(benchmark.rows)
        pairs = columns(specs, name)
        labels, predictions = [label for label, _ in pairs], [column for _, column in pairs]
        entry = dict(counts=counts, scores={})
        print(f"\n# {name}: {counts['rows']} rows, {counts['floor']} at the floor score "
              f"{counts['floor_score']:g}, {counts['multi']} multiply judged")
        print(f"{'column':44s} " + ' '.join(f'{m:>24s}' for m in MASKS))
        for mask in MASKS:
            cells = evaluate(benchmark, labels, predictions, groups[mask], resamples)
            human = benchmark.select(groups[mask] & np.isfinite(leave_one_out))
            cells['human_leave_one_out'] = dict(score=benchmark.measure(leave_one_out, human.items),
                                                items=int(len(human.items)))
            if resamples and len(human) > 1:
                cells['human_leave_one_out']['bootstrap'] = interval(
                    bootstrap(benchmark, [leave_one_out], human, resamples)[:, 0])
            entry['scores'][mask] = cells
        for label in labels + ['human_leave_one_out']:
            cells = [entry['scores'][mask][label] for mask in MASKS]
            print(f'{label:44s} ' + ' '.join(f'{show(cell):>24s}' for cell in cells))
        out[name] = entry
    write_json(RUNS / 'validity.json', finite(dict(sets=SETS, specs=specs, masks=MASKS,
                                                   resamples=resamples, benchmarks=out)))


def show(cell):
    if not np.isfinite(cell['score']):
        return '-'
    span = cell.get('bootstrap')
    return f"{cell['score']:6.2f}" + (f" [{span['low']:5.2f},{span['high']:5.2f}]" if span else '')


if __name__ == '__main__':
    main(*sys.argv[1:])
