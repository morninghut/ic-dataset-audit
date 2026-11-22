"""The gain from keeping the duplicates, within each group of Polaris test samples.

Difference in Kendall tau_c between a model whose training set holds the duplicated samples and
the same model trained on the deduplicated set; mean of three seeds, 95% bootstrap interval over
images.  Filled bars: all samples of the group.  Open bars: samples above the lowest human score.
Source: runs/summary/groups.json (analysis/summary/groups.py).
"""
from matplotlib.patches import Patch

from common import ASH, GREY, INK, MIST, RUNS5, WARM, WIDTH, plt, read, save

PANELS = [('Pearl\nreleased − deduplicated', 'pearl', 'released-pairclean'),
          ('Pearl\nmatched − deduplicated', 'pearl', 'matcheddrop-pairclean'),
          ('DENEB\nreleased − deduplicated', 'deneb', 'released-pairclean')]
GROUPS = [('duplicated', 'duplicated'), ('image_only', 'image\nonly'), ('unseen', 'unseen\nimage')]
BAR = 0.34

data = read(RUNS5 / 'groups.json')['retrain']
figure, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.15), sharey=True, gridspec_kw={'wspace': 0.08, 'top': 0.8})
for axis, (title, family, key) in zip(axes, PANELS):
    axis.axhline(0, color=MIST, linewidth=0.7, zorder=1)
    for x, (group, _) in enumerate(GROUPS):
        colour = WARM if group == 'duplicated' else GREY
        for subset, offset, filled in (('all', -BAR / 2 - 0.02, True), ('above', BAR / 2 + 0.02, False)):
            cell = data[family][f'{group}/{subset}'][key]
            axis.bar(x + offset, cell['mean'], width=BAR, facecolor=colour if filled else 'white', edgecolor=colour,
                     linewidth=0.8, zorder=2)
            axis.plot([x + offset] * 2, [cell['low'], cell['high']], color=INK, linewidth=0.7, zorder=3)
    axis.set_title(title, fontsize=6.4, fontweight='medium', pad=5, loc='center', linespacing=1.25)
    axis.set_xticks(range(len(GROUPS)))
    axis.set_xticklabels([label for _, label in GROUPS], color=INK, fontsize=5.6, linespacing=1.1)
    axis.set_xlim(-0.6, len(GROUPS) - 0.4)
    axis.set_ylim(-1.6, 9.2)
    axis.set_yticks(range(0, 9, 2))
    axis.grid(axis='y')
    axis.set_axisbelow(True)
    axis.spines['bottom'].set_visible(False)
for axis in axes[1:]:
    axis.spines['left'].set_visible(False)
axes[0].set_ylabel('Difference in Kendall $\\tau$')
figure.legend(handles=[Patch(facecolor=ASH, edgecolor=ASH, label='all samples'),
                       Patch(facecolor='white', edgecolor=GREY, label='above lowest score')],
              loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03), handlelength=0.9, handleheight=0.9,
              handletextpad=0.45, columnspacing=1.6, fontsize=6.2)
save(figure, 'fig_groups')
