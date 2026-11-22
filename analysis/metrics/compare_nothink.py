"""InternVL judge: scores conditional on answering directly, against scores with an empty think block prefilled.

    compare_nothink.py

InternVL3.5 opens a reasoning block on part of the prompts, so its first-token probability on
the ten digits is well below 1 and score_vlm.py's rating is conditional on it answering
directly.  The check run (VLM_PREFILL closes an empty think block) makes the first token a
digit almost always.  For every split and mode both runs scored, this prints the digit mass
of each, 100 x tau of each against the human scores (all rows), their difference with a
picture-clustered bootstrap interval, and the rank correlation between the two score lists;
then tau of the conditional scores by tercile of digit mass.

Run with PYTHONPATH=Polos-revised.  Writes runs/metrics/nothink_check.json.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probe'))
from effects import Benchmark, bootstrap, interval  # noqa: E402
from lib import ROOT, write_json  # noqa: E402

VLM = ROOT / 'runs/metrics/vlm'
PLAIN, CHECK = VLM / 'InternVL3_5-8B-HF', VLM / 'InternVL3_5-8B-HF-nothink'
SPLITS = ['polaris_test', 'spica_test', 'nebula_test', 'flickr8k_expert', 'composite', 'flickr8k_cf']


def main():
    out = {}
    print(f"{'mode':5s} {'split':16s} {'mass':>6s} {'mass*':>6s} {'tau':>7s} {'tau*':>7s}  {'tau* - tau':>22s} {'rho':>6s}"
          '   tau by tercile of mass (low, mid, high)        (* = empty think block prefilled)')
    for mode in ['refs', 'free']:
        for split in SPLITS:
            a, b = PLAIN / mode / f'{split}.npz', CHECK / mode / f'{split}.npz'
            if not (a.exists() and b.exists()):
                continue
            with np.load(a) as f:
                plain, mass = f['score'].astype(np.float64), f['mass'].astype(np.float64)
            with np.load(b) as f:
                check, check_mass = f['score'].astype(np.float64), f['mass'].astype(np.float64)
            benchmark = Benchmark(split)
            selection = benchmark.select()
            point = [benchmark.measure(c, selection.items) for c in (plain, check)]
            draws = bootstrap(benchmark, [plain, check], selection, 1000, seed=0)
            span = interval(draws[:, 1] - draws[:, 0])
            edges = np.quantile(mass, [1 / 3, 2 / 3])
            tercile = np.digitize(mass, edges)
            by_mass = [benchmark.measure(plain, np.flatnonzero(tercile == k)) for k in range(3)]
            by_mass_check = [benchmark.measure(check, np.flatnonzero(tercile == k)) for k in range(3)]
            rho = float(spearmanr(plain, check).statistic)
            out[f'{mode}/{split}'] = dict(mass=float(mass.mean()), mass_check=float(check_mass.mean()),
                                          tau=point[0], tau_check=point[1], difference=point[1] - point[0],
                                          low=span['low'], high=span['high'], spearman=rho,
                                          tau_by_mass_tercile=by_mass, tau_check_by_mass_tercile=by_mass_check)
            print(f'{mode:5s} {split:16s} {mass.mean():6.3f} {check_mass.mean():6.3f} {point[0]:7.2f} {point[1]:7.2f}  '
                  f"{point[1] - point[0]:+6.2f} [{span['low']:+.2f},{span['high']:+.2f}] {rho:6.3f}   "
                  + ' '.join(f'{v:6.2f}' for v in by_mass) + '   * ' + ' '.join(f'{v:6.2f}' for v in by_mass_check),
                  flush=True)
    write_json(ROOT / 'runs/metrics/nothink_check.json', out)


if __name__ == '__main__':
    main()
