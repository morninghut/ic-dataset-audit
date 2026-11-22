"""The ranking on the Polaris test set and on Composite, as two leaderboards.

Kendall tau_c over all samples of the sixteen best ranked metrics, coloured by family.  The bracket
spans the metrics that the paired test does not separate from the first one.  The outlined bar on
the Polaris test set is Pearl trained on the released training set, which holds duplicated samples of
that set and is not ranked.  The vertical axis does not start at zero.
Source: runs/metrics/pairwise_s12-13-14.json.
"""
from matplotlib.patches import Patch

from common import DUPLICATED, FAMILIES, GREY, INK, METRICS, WARM, WIDTH, colour, entry, plt, ranking, save

SETS = [('polaris_test', 'Polaris test')]
TOP, FLOOR = 16, 50

benchmarks = ranking()
figure, axes = plt.subplots(len(SETS), 1, figsize=(WIDTH, 1.85), squeeze=False)
axes = axes[:, 0]
for axis, (name, title) in zip(axes, SETS):
    part = benchmarks[name]['all']
    bars = [(row['metric'], row['score'], True) for row in sorted(part['rows'], key=lambda row: row['rank'])[:TOP]]
    held = entry(part, 'Pearl (retrained)' + DUPLICATED)
    if held:
        place = sum(score > held['score'] for _, score, _ in bars)
        bars.insert(place, ('Pearl (retrained)', held['score'], False))
    ceiling = max(score for _, score, _ in bars)
    for x, (metric, score, ranked) in enumerate(bars):
        if ranked:
            axis.bar(x, score - FLOOR, bottom=FLOOR, width=0.72, color=colour(metric), linewidth=0)
        else:
            axis.bar(x, score - FLOOR, bottom=FLOOR, width=0.66, facecolor='white', edgecolor=WARM, linewidth=0.9,
                     linestyle=(0, (2.2, 1.6)))
        axis.text(x, score + 0.35, f'{score:.1f}', ha='center', va='bottom', fontsize=5.9,
                  color=INK if ranked else WARM, fontweight='medium')
    tied = [x for x, (metric, _, ranked) in enumerate(bars) if ranked and metric in part['top_group']]
    level = ceiling + 2.6
    axis.plot([tied[0] - 0.36, tied[0] - 0.36, tied[-1] + 0.36, tied[-1] + 0.36], [level - 0.5, level, level, level - 0.5],
              color=GREY, linewidth=0.6, clip_on=False)
    axis.set_xlim(-0.65, len(bars) - 0.35)
    axis.set_ylim(FLOOR, ceiling + 3.4)
    axis.set_yticks(range(FLOOR, int(ceiling) + 1, 5))
    axis.set_xticks(range(len(bars)))
    axis.set_xticklabels([METRICS[metric][0] + ('' if ranked else ' (with duplicates)') for metric, _, ranked in bars],
                         rotation=38, ha='right', rotation_mode='anchor', fontsize=6.1)
    for tick, (_, _, ranked) in zip(axis.get_xticklabels(), bars):
        tick.set_color(INK if ranked else WARM)
    axis.grid(axis='y')
    axis.set_axisbelow(True)
    axis.spines['left'].set_visible(False)
    axis.set_ylabel('Kendall $\\tau$')

handles = [Patch(facecolor=shade, label=label) for label, shade in FAMILIES.values()
           if any(colour(metric) == shade for name, _ in SETS
                  for metric in (row['metric'] for row in sorted(benchmarks[name]['all']['rows'], key=lambda r: r['rank'])[:TOP]))]
figure.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.13), handlelength=0.9,
              handleheight=0.9, handletextpad=0.45, columnspacing=1.3, fontsize=6.2)
save(figure, 'fig_leaderboard')
