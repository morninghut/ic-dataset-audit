"""Kendall tau_c on the Polaris test set after every training epoch.

Rows: Pearl (twelve trainings on four versions of the Spica training set) and, once its sweep
has been scored, DENEB (nine trainings on three versions of the Nebula training set).  Left: the
duplicated samples.  Right: all other samples.  Thin lines are single trainings, thick lines the
mean of the trainings whose training set holds the duplicates (warm) or is deduplicated (cool);
numbers are the gap between the two means.  All panels span the same number of tau points.
Sources: runs/metrics/epoch_sweep.json, runs/summary/deneb_epoch_sweep.json.
"""
import numpy as np

from common import COOL, COOL_SOFT, GREY, INK, RUNS5, WARM, WARM_SOFT, WIDTH, plt, read, save

SPAN = 15.5
COLUMNS = [('polaris_test/pair', 'duplicated samples'), ('polaris_test/not pair', 'other samples')]

import sys

# fig_epochs.py          -> fig_epochs (Pearl, main text)
# fig_epochs.py deneb    -> fig_epochs_deneb (appendix; needs runs/summary/deneb_epoch_sweep.json)
if sys.argv[1:] == ['deneb']:
    sweeps, NAME = [('DENEB', read(RUNS5 / 'deneb_epoch_sweep.json')['runs'])], 'fig_epochs_deneb'
else:
    sweeps, NAME = [('Pearl', read('epoch_sweep.json')['runs'])], 'fig_epochs'

figure, grid = plt.subplots(len(sweeps), 2, figsize=(WIDTH, 2.1), squeeze=False,
                            gridspec_kw={'wspace': 0.3, 'hspace': 0.42})
for row, (metric, runs) in enumerate(sweeps):
    groups = {False: [], True: []}
    for name, run in runs.items():
        epochs = {int(e): cell for e, cell in run['epochs'].items()}
        groups[name.startswith('pairclean')].append(epochs)
    last = max(max(epochs) for group in groups.values() for epochs in group) + 1
    means = {}
    for axis, (column, title) in zip(grid[row], COLUMNS):
        for clean, colour, soft in ((False, WARM, WARM_SOFT), (True, COOL, COOL_SOFT)):
            for epochs in groups[clean]:
                axis.plot([e + 1 for e in sorted(epochs)], [epochs[e]['tau'][column] for e in sorted(epochs)],
                          color=soft, linewidth=0.7, zorder=2)
            mean = [np.mean(cell) if cell else np.nan
                    for cell in ([epochs[e]['tau'][column] for epochs in groups[clean] if e in epochs] for e in range(last))]
            means[column, clean] = mean
            axis.plot(range(1, last + 1), mean, color=colour, linewidth=1.5, marker='o', markersize=3,
                      markeredgecolor='white', markeredgewidth=0.6, zorder=3)
        centre = np.nanmean([means[column, False], means[column, True]])
        bottom = np.floor(2 * (centre - SPAN / 2)) / 2
        axis.set_ylim(bottom, bottom + SPAN)
        axis.set_xlim(0.7, last + 0.75)
        axis.set_xticks(range(1, last + 1))
        axis.grid(axis='y')
        axis.set_axisbelow(True)
        if row == 0:
            axis.set_title(title, fontweight='medium', pad=5)
        if row == len(sweeps) - 1:
            axis.set_xlabel('Training epoch')
    grid[row][0].set_ylabel(f'{metric}: Kendall $\\tau$')
    held, clean = means[COLUMNS[0][0], False], means[COLUMNS[0][0], True]
    axis = grid[row][0]
    for epoch in range(last):
        gap = held[epoch] - clean[epoch]
        if np.isnan(gap):
            continue
        if gap > 1.6:
            axis.plot([epoch + 1] * 2, [clean[epoch] + 0.5, held[epoch] - 0.5], color=GREY, linewidth=0.5,
                      linestyle=(0, (1, 2)), zorder=1)
        axis.text(epoch + (1.09 if gap > 1.6 else 1), (held[epoch] + clean[epoch]) / 2 if gap > 1.6 else held[epoch] + 1.3,
                  f'+{gap:.1f}' if gap >= 0 else f'−{-gap:.1f}', ha='left' if gap > 1.6 else 'center', va='center',
                  fontsize=6.4 if last <= 5 else 5.0, fontweight='semibold', color=INK, zorder=4,
                  bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.6})
    if row == 0:
        low, high = axis.get_ylim()
        axis.text(0.85, high - 0.9, 'with duplicates', ha='left', va='center', fontsize=6.2, color=WARM)
        axis.text(0.85, low + 0.9, 'deduplicated', ha='left', va='center', fontsize=6.2, color=COOL)
    print(metric, 'runs per epoch (holding / deduplicated):',
          ['/'.join(str(sum(e in epochs for epochs in groups[clean])) for clean in (False, True)) for e in range(last)],
          'gaps', [round(h - c, 2) for h, c in zip(held, clean)])

save(figure, NAME)
