"""Evaluation scores of every epoch of the Pearl trainings scored by epoch_sweep.py.

    epoch_sweep_report.py [split ...]

For each run under runs/metrics/pearl with an epoch_scores directory: the validation
score the trainer monitored (read from the scoring logs), which epoch it selected, and
100 x tau of every epoch on each evaluation set; on sets with duplicated pairs also on the
rows whose pair was in the released pool and on the rest.  Then, per variant, the spread
over epochs and seeds, and how the variant contrasts change when every run is read at the
same epoch instead of the selected one.

With splits given, only those are read and nothing is written (a look at a sweep that is
still running).  A checkpoint that lacks one of the splits is left out.

Run with PYTHONPATH=Polos-revised.  Writes runs/metrics/epoch_sweep.json.
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probe'))
from analyze_effects import item_classes, pool_keys  # noqa: E402
from effects import Benchmark  # noqa: E402
from lib import ROOT, write_json  # noqa: E402

RUNS = ROOT / 'runs/metrics'
ALL_SPLITS = ['composite', 'flickr8k_expert', 'polaris_test', 'spica_test', 'nebula_test']
SPLITS = sys.argv[1:] or ALL_SPLITS
VARIANTS = ['released', 'pairclean', 'randomdrop', 'matcheddrop']
LINE = re.compile(r'epoch=(\d+)-step=\d+\.ckpt: monitored ([0-9.]+)')


def monitored(run):
    """{epoch: validation score} from the scoring log of this run (local or copied from the other machine)."""
    found = {}
    for log in [RUNS / 'logs' / f'score_pearl_{run}.log', RUNS / 'logs/remote' / f'score_pearl_{run}.log']:
        if log.exists():
            for epoch, score in LINE.findall(log.read_text(errors='replace')):
                found[int(epoch)] = float(score)
    return found


def main():
    pictures, keys = pool_keys('spica_train', every_row=True)      # Pearl is trained on every released row
    benchmarks = {split: Benchmark(split) for split in SPLITS}
    groups = {}
    for split, benchmark in benchmarks.items():
        classes = item_classes(benchmark, pictures, keys)
        groups[split] = {'all': np.flatnonzero(classes >= 0)}
        if (classes == 2).sum() >= 200:
            groups[split]['pair'] = np.flatnonzero(classes == 2)
            groups[split]['not pair'] = np.flatnonzero((classes == 0) | (classes == 1))
    columns = [(split, group) for split in SPLITS for group in groups[split]]

    out = {}
    for directory in sorted((RUNS / 'pearl').glob('*_s*/epoch_scores')):
        run = directory.parent.name
        validation = monitored(run)
        epochs = {}
        for checkpoint in sorted(directory.glob('epoch=*'), key=lambda p: int(p.name.split('=')[1].split('-')[0])):
            if not all((checkpoint / f'{split}.npz').exists() for split in SPLITS):
                continue
            epoch = int(checkpoint.name.split('=')[1].split('-')[0])
            values = {}
            for split in SPLITS:
                with np.load(checkpoint / f'{split}.npz') as f:
                    prediction = f['score'].astype(np.float64)
                for group, items in groups[split].items():
                    values[f'{split}/{group}'] = benchmarks[split].measure(prediction, items)
            epochs[epoch] = dict(validation=validation.get(epoch), tau=values)
        if not epochs:
            continue
        scored = [e for e in epochs if epochs[e]['validation'] is not None]
        selected = max(scored, key=lambda e: epochs[e]['validation']) if scored else None
        out[run] = dict(selected=selected, epochs=epochs)

    print('100 x tau by epoch; * = the epoch validation selected')
    print(f"{'run':18s} {'ep':>3s} {'valid':>7s} " + ' '.join(f'{s[:12] + ":" + g[:8]:>21s}' for s, g in columns))
    for run, entry in out.items():
        for epoch, row in entry['epochs'].items():
            mark = '*' if epoch == entry['selected'] else ' '
            valid = f"{row['validation']:.4f}" if row['validation'] is not None else '   -   '
            print(f'{run:18s} {epoch:2d}{mark} {valid:>7s} '
                  + ' '.join(f"{row['tau'][f'{s}/{g}']:21.2f}" for s, g in columns))
        print()

    def table(pick, title):
        """Per variant: mean over runs of the value at the epoch chosen by pick(entry)."""
        print(title)
        print(f"{'variant':12s} {'runs':>4s} " + ' '.join(f'{s[:12] + ":" + g[:8]:>21s}' for s, g in columns))
        result = {}
        for variant in VARIANTS:
            rows = []
            for run, entry in out.items():
                epoch = pick(entry)
                if run.startswith(variant + '_s') and epoch in entry['epochs']:
                    rows.append([entry['epochs'][epoch]['tau'][f'{s}/{g}'] for s, g in columns])
            if rows:
                rows = np.array(rows)
                result[variant] = dict(runs=len(rows), mean=rows.mean(0).tolist(),
                                       low=rows.min(0).tolist(), high=rows.max(0).tolist())
                print(f'{variant:12s} {len(rows):4d} '
                      + ' '.join(f'{m:8.2f} [{a:5.2f},{b:5.2f}]' for m, a, b in zip(rows.mean(0), rows.min(0), rows.max(0))))
        print()
        return result

    summary = {'selected': table(lambda e: e['selected'], 'variant means at the selected epoch [min, max over runs]')}
    last = max(max(entry['epochs']) for entry in out.values())
    for epoch in range(last + 1):
        summary[f'epoch {epoch}'] = table(lambda e, k=epoch: k, f'variant means with every run read at epoch {epoch}')

    print('spread within a run (max - min over its epochs)')
    print(f"{'run':18s} {'eps':>3s} " + ' '.join(f'{s[:12] + ":" + g[:8]:>21s}' for s, g in columns))
    for run, entry in out.items():
        rows = np.array([[row['tau'][f'{s}/{g}'] for s, g in columns] for row in entry['epochs'].values()])
        print(f'{run:18s} {len(rows):3d} ' + ' '.join(f'{v:21.2f}' for v in rows.max(0) - rows.min(0)))

    if SPLITS == ALL_SPLITS:
        write_json(RUNS / 'epoch_sweep.json', dict(columns=[f'{s}/{g}' for s, g in columns], runs=out, summary=summary))


if __name__ == '__main__':
    main()
