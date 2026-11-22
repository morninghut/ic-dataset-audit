"""How much of Rigel's score on the Polaris-family test sets comes from rows it was trained on?

Rigel's only public checkpoint was trained on the Spica pool, which holds 68.4% of the Polaris
test rows as judged pairs (2.2% of Spica test, 0.1% of Nebula test).  There is no training
code, so there is no cleaned retraining to compare with.  What can be measured is the contrast
inside one evaluation set: rows whose (picture, candidate) pair is in the pool against rows
whose picture the pool never held, with a zero-shot judge as the yardstick for how hard each
group of rows is.  The judge has no training pool, so

    (Rigel - judge on pair rows) - (Rigel - judge on absent rows)

is what Rigel gains on the rows it saw beyond what its fine-tuning gains on rows it did not.
The same line is printed for the retrained Pearl (released pool, mean of the seeds), where the
cleaned retraining gives the answer independently (leakage_seeds.py), as a check of the method.

    rigel_groups.py [resamples=2000]

Run with PYTHONPATH=Polos-revised.  Writes runs/metrics/rigel_groups.json.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probe'))
from analyze_effects import finite, item_classes, pool_keys  # noqa: E402
from effects import Benchmark, bootstrap, interval  # noqa: E402
from lib import ROOT, RUNS, read_json, write_json  # noqa: E402

SETS = ['polaris_test', 'spica_test', 'nebula_test']
SEEDS = [12, 13, 14]
JUDGE = ('vlm_internvl8b_nothink', 'refs')
JUDGE_FREE = ('vlm_internvl8b_nothink', 'free')
GROUPS = dict(all=None, pair=2, picture_only=1, absent=0, not_pair=(0, 1))
MINIMUM_ITEMS = 200


def column(name, label, split):
    path = RUNS / 'models/external' / name
    labels = read_json(path / 'report.json')['labels']
    return np.load(path / 'predictions' / f'{split}.npy').astype(np.float64)[:, labels.index(label)]


def main(resamples=2000):
    resamples = int(resamples)
    pictures, keys = pool_keys('spica_train', every_row=True)      # Rigel is trained on every released row
    out = {}
    for split in SETS:
        benchmark = Benchmark(split)
        classes = item_classes(benchmark, pictures, keys)
        columns = [column('rigel', 'refs', split), column('rigel', 'free', split),
                   column(*JUDGE, split), column(*JUDGE_FREE, split)]
        columns += [column(f'pearl_s{seed}', 'released', split) for seed in SEEDS]
        columns += [column(f'pearl_s{seed}', 'pairclean', split) for seed in SEEDS]

        def derive(table):
            """(n, columns) -> named quantities per resample."""
            rigel, rigel_free, judge, judge_free = table[:, 0], table[:, 1], table[:, 2], table[:, 3]
            pearl, clean = table[:, 4:4 + len(SEEDS)].mean(1), table[:, 4 + len(SEEDS):].mean(1)
            return {'Rigel + refs': rigel, 'Rigel': rigel_free, 'judge + refs': judge, 'judge': judge_free,
                    'Pearl +dup': pearl, 'Pearl clean': clean,
                    'Rigel + refs - judge + refs': rigel - judge, 'Rigel - judge': rigel_free - judge_free,
                    'Pearl +dup - judge + refs': pearl - judge, 'Pearl clean - judge + refs': clean - judge}

        print(f'\n# {split}: {benchmark.item_count} rows; judge = InternVL 8B zero-shot')
        names = None
        entry, kept = {}, {}
        for position, (group, wanted) in enumerate(GROUPS.items()):
            mask = classes >= 0 if wanted is None else np.isin(classes, wanted)
            if mask.sum() < MINIMUM_ITEMS:
                continue
            selection = benchmark.select(mask)
            point = derive(np.array([[benchmark.measure(c, selection.items) for c in columns]]))
            draws = derive(bootstrap(benchmark, columns, selection, resamples, seed=position))
            if names is None:
                names = list(point)
                print(f"{'rows':13s} {'items':>6s} " + ' '.join(f'{n[:21]:>21s}' for n in names))
            kept[group] = (point, draws)
            entry[group] = dict(items=int(len(selection.items)))
            cells = []
            for n in names:
                span = interval(draws[n])
                entry[group][n] = dict(mean=float(point[n][0]), low=span['low'], high=span['high'])
                cells.append(f"{point[n][0]:6.2f} [{span['low']:5.2f},{span['high']:5.2f}]" if ' - ' in n
                             else f'{point[n][0]:21.2f}')
            print(f"{group:13s} {entry[group]['items']:6d} " + ' '.join(f'{c:>21s}' for c in cells), flush=True)
        if 'pair' in kept and 'absent' in kept:
            entry['pair-absent'] = {}
            line = []
            for n in names:
                if ' - ' not in n:
                    continue
                # The two groups share no picture, so their resamples are independent.
                span = interval(kept['pair'][1][n] - kept['absent'][1][n])
                value = float(kept['pair'][0][n][0] - kept['absent'][0][n][0])
                entry['pair-absent'][n] = dict(mean=value, low=span['low'], high=span['high'])
                line.append(f"{n}: {value:+.2f} [{span['low']:+.2f},{span['high']:+.2f}]")
            print('pair - absent   ' + ';  '.join(line))
        out[split] = entry
    write_json(ROOT / 'runs/metrics/rigel_groups.json', finite(dict(sets=SETS, judge=list(JUDGE), resamples=resamples,
                                                                        benchmarks=out)))


if __name__ == '__main__':
    main(*sys.argv[1:])
