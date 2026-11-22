#!/usr/bin/env python3
"""Recompute the result files of the paper from the stored per-sample scores and check them.

    python reproduce.py files               # every shipped file is present and has its SHA-256
    python reproduce.py tables              # result files -> every number and table of the paper
    python reproduce.py analysis            # per-sample scores -> result files
    python reproduce.py figures             # result files -> figures
    python reproduce.py all

files     compares every file listed in MANIFEST.json with its SHA-256 (repository files and whichever
          archives are unpacked).
tables    runs paper/make_tables.py and compares paper/generated with paper/expected, the files
          the manuscript was compiled from.  Seconds; needs only the repository.
analysis  reruns the statistics (bootstrap over images, paired tests, differences between groups,
          per-epoch curves) and compares every number with the shipped result file.  Needs the
          per-sample scores and `python setup_data.py`.  The bootstrap seeds are fixed, so the
          numbers must agree to the last digit.
figures   redraws the figures that show numbers only (not those that show dataset images) and
          compares the PNG files with those of the manuscript by SHA-256.

--only NAME limits `analysis` to one step, --workers N sets the processes of the paired tests
(the result does not depend on it), --keep-going continues after a failed step.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / 'MANIFEST.json').read_text())
PARTS = {'repository': 'repository', 'bundle': 'per-sample scores (per_sample_scores.tar)', 'extra': 'further scores (further_scores.tar)'}
EXPECTED = ROOT / 'expected_results'
METRICS, SUMMARY = 'analysis/metrics', 'analysis/summary'
# name, directory, command, result file, what it holds
STEPS = [
    ('leakage_pearl', METRICS, ['leakage_seeds.py', 'pearl', 'spica_train', '12', '13', '14'], 'runs/metrics/leakage_pearl.json',
     'Pearl: released, deduplicated, random and matched training sets, by group of samples'),
    ('leakage_deneb', METRICS, ['leakage_seeds.py', 'deneb', 'nebula_train', '12', '13', '14'], 'runs/metrics/leakage_deneb.json',
     'DENEB: the same contrasts'),
    ('epoch_sweep', METRICS, ['epoch_sweep_report.py'], 'runs/metrics/epoch_sweep.json', 'Pearl: every epoch of every training'),
    ('deneb_epochs', SUMMARY, ['deneb_epoch_report.py'], 'runs/summary/deneb_epoch_sweep.json', 'DENEB: every epoch of every training'),
    ('rigel_groups', METRICS, ['rigel_groups.py', '2000'], 'runs/metrics/rigel_groups.json', 'Rigel against a zero-shot judge, by group'),
    ('nothink', METRICS, ['compare_nothink.py'], 'runs/metrics/nothink_check.json', 'InternVL judge with and without a prefilled think block'),
    ('pairwise', METRICS, ['pairwise.py', '10000', '20', '12,13,14', '{workers}'], 'runs/metrics/pairwise_s12-13-14.json',
     'ranking of all metrics on six evaluation sets with paired tests'),
    ('components', SUMMARY, ['components.py'], 'runs/summary/components.json', 'detection and ranking terms of Kendall tau'),
    ('groups', SUMMARY, ['groups.py'], 'runs/summary/groups.json', 'groups of samples, label agreement, cross-fit effects'),
]
# output name, command
FIGURES = [(name, [f'{name}.py']) for name in ('fig_overview', 'fig_duplication', 'fig_groups', 'fig_crossfit', 'fig_epochs', 'fig_components',
                                               'fig_power', 'fig_heatmap', 'fig_leaderboard', 'fig_all_above')] + \
          [('fig_epochs_deneb', ['fig_epochs.py', 'deneb'])]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def files(_):
    """Every file of MANIFEST.json: present and unchanged.  A part (archive) of which no file is present is reported as absent."""
    failed = []
    for part in ('repository', 'bundle', 'extra'):
        listed = {name: entry for name, entry in MANIFEST.items() if entry['part'] == part}
        present = [name for name in listed if (ROOT / name).exists() or (EXPECTED / name).exists()]
        if not present:
            print(f'absent  files          {PARTS[part]}: none of its {len(listed)} files is here')
            continue
        # a result file that `analysis` has rewritten is kept unchanged under expected_results/
        bad = [name for name in listed
               if name not in present or sha256(EXPECTED / name if (EXPECTED / name).exists() else ROOT / name) != listed[name]['sha256']]
        print(f'{"ok    " if not bad else "DIFFER"}  files          {PARTS[part]}: {len(listed) - len(bad)} of {len(listed)} files match'
              + (f'; first differences: {bad[:5]}' if bad else ''))
        failed += bad
    return failed


def differences(a, b, path=''):
    """Largest absolute difference between two JSON values, and where; inf for a structural difference."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return math.inf, path
        return max((differences(a[key], b[key], f'{path}/{key}') for key in a), default=(0.0, path))
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return math.inf, path
        return max((differences(x, y, f'{path}[{i}]') for i, (x, y) in enumerate(zip(a, b))), default=(0.0, path))
    number = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool)
    if number(a) and number(b):
        return abs(a - b), path
    return (0.0, path) if a == b else (math.inf, path)


def environment():
    env = dict(os.environ, CAPAUDIT_ROOT=str(ROOT), PYTHONUTF8='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1')
    return env


def analysis(args):
    if not (ROOT / 'feature_cache_v2' / 'polaris_test.npz').exists():
        sys.exit('feature_cache_v2/ is missing: run `python setup_data.py` first')
    if not (ROOT / 'runs/probe/models/external/pearl_s12/predictions').exists():
        sys.exit('the per-sample scores are missing: unpack per_sample_scores.tar in the repository root (see README.md)')
    steps = [step for step in STEPS if args.only in (None, step[0])]
    if not steps:
        sys.exit(f'unknown step {args.only}; choose from {[step[0] for step in STEPS]}')
    (ROOT / 'logs').mkdir(exist_ok=True)
    running = []
    for name, directory, command, result, _ in steps:
        shipped = EXPECTED / result
        if not shipped.exists():                       # keep the shipped file to compare against
            if sha256(ROOT / result) != MANIFEST[result]['sha256']:
                sys.exit(f'{result} is not the shipped file any more; restore it (git checkout) and rerun')
            shipped.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / result, shipped)
        command = [sys.executable] + [part.format(workers=args.workers) for part in command]
        log = open(ROOT / 'logs' / f'{name}.log', 'w')
        running.append((name, result, time.time(), log,
                        subprocess.Popen(command, cwd=ROOT / directory, env=environment(), stdout=log, stderr=subprocess.STDOUT)))
        print(f'started {name}: {" ".join(command[1:])}', flush=True)
    failed = []
    for name, result, started, log, process in running:
        code = process.wait()
        log.close()
        if code:
            print(f'FAILED  {name}: exit code {code}, see logs/{name}.log')
            failed.append(name)
            if not args.keep_going:
                break
            continue
        size, where = differences(json.loads((EXPECTED / result).read_text()), json.loads((ROOT / result).read_text()))
        ok = size <= args.tolerance
        print(f'{"ok    " if ok else "DIFFER"}  {name:14s} {result}  largest difference {size:.2g}' + ('' if ok else f' at {where}'), flush=True)
        if not ok:
            failed.append(name)
    return failed


def tables(_):
    subprocess.run([sys.executable, 'make_tables.py'], cwd=ROOT / 'paper', env=environment(), check=True, stdout=subprocess.DEVNULL)
    failed = []
    for expected in sorted((ROOT / 'paper/expected').iterdir()):
        made = ROOT / 'paper/generated' / expected.name
        if not made.exists() or made.read_bytes() != expected.read_bytes():
            failed.append(expected.name)
    count = len(list((ROOT / 'paper/expected').iterdir()))
    print(f'{"ok    " if not failed else "DIFFER"}  tables         {count - len(failed)} of {count} files in paper/generated are identical to paper/expected'
          + (f'; different: {failed}' if failed else ''))
    return failed


def figures(_):
    expected = json.loads((ROOT / 'paper/figures/expected.json').read_text())
    failed, same = [], 0
    for name, command in FIGURES:
        code = subprocess.run([sys.executable] + command, cwd=ROOT / 'paper/figures', env=environment(), stdout=subprocess.DEVNULL).returncode
        if code:
            failed.append(name)
        else:
            same += sha256(ROOT / 'paper/figures' / f'{name}.png') == expected[f'{name}.png']
    print(f'{"ok    " if not failed else "FAILED"}  figures        {len(FIGURES) - len(failed)} of {len(FIGURES)} written to paper/figures'
          + (f'; failed: {failed}' if failed else '') + f'; {same} PNG files identical to those of the manuscript'
          + ('' if same == len(FIGURES) else ' (another matplotlib or FreeType version changes pixels, not numbers)'))
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split('\n\n', 1)[1])
    parser.add_argument('what', choices=['files', 'tables', 'analysis', 'figures', 'all'])
    parser.add_argument('--only')
    parser.add_argument('--workers', type=int, default=max(1, min(44, (os.cpu_count() or 2) - 2)))
    parser.add_argument('--tolerance', type=float, default=1e-9)
    parser.add_argument('--keep-going', action='store_true')
    args = parser.parse_args()
    failed = []
    for name, run in (('files', files), ('analysis', analysis), ('tables', tables), ('figures', figures)):
        if args.what in (name, 'all'):
            failed += run(args)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
