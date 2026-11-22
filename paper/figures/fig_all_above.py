"""What is left of the correlation over all samples once the lowest-score samples are set aside.

For three evaluation sets with repeated annotation: Kendall tau_c of a metric over all samples (wide
light bar) and over the samples whose human score is above the lowest (narrow bar in the colour of
the family), and the agreement of one rating with the others on those samples (vertical line).  A
hatched bar marks an entry whose training set holds duplicated samples of that set.  Source: runs/metrics/pairwise_s12-13-14.json.
"""
from common import (DUPLICATED, GREY, INK, METRICS, MIST, SET_NAMES, WIDTH, colour, entry, plt, ranking, save)

SETS = ['polaris_test', 'spica_test', 'flickr8k_expert']
SHOWN = ['Rigel (official ckpt) + refs', 'VLM judge (InternVL 8B) + refs', 'VLM judge (Qwen 8B)',
         'Pearl (retrained)', 'DENEB (retrained)', 'Polos (retrained)', 'ridge SigLIP, Polaris pool (ours)',
         'RefPAC-S++ (ViT-L/14)', 'CLIP-S', 'CIDEr', 'BLEU-4']

benchmarks = ranking()
figure, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.45), sharey=True, gridspec_kw={'wspace': 0.12, 'left': 0.27})
for axis, name in zip(axes, SETS):
    parts = benchmarks[name]
    human = parts['above_floor_judged']['human']
    axis.axvline(human, color=INK, linewidth=0.7, linestyle=(0, (2, 2)), zorder=5)
    axis.text(human, -0.95, 'human', ha='center', va='center', fontsize=6, color=INK,
              bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
    for y, metric in enumerate(SHOWN):
        ranked = entry(parts['all'], metric) is not None
        key = metric if ranked else metric + DUPLICATED
        whole, above = entry(parts['all'], key), entry(parts['above_floor'], key)
        axis.barh(y, whole['score'], height=0.68, color=MIST, linewidth=0, zorder=2)
        axis.barh(y, above['score'], height=0.36, color=colour(metric) if ranked else 'white', edgecolor=colour(metric),
                  linewidth=0 if ranked else 0.5, hatch=None if ranked else '//////', zorder=3)
    floor = 1 - parts['above_floor']['items'] / parts['all']['items']
    axis.set_title(f"{SET_NAMES[name]}" + (' test' if name.endswith('test') else ''), pad=15)
    axis.text(0, 1.035, f'{100 * floor:.0f}% lowest score', transform=axis.transAxes, ha='left', va='bottom',
              fontsize=6.2, color=GREY)
    axis.set_xlim(0, 68)
    axis.set_xticks([0, 20, 40, 60])
    axis.grid(axis='x')
    axis.set_axisbelow(True)
    axis.spines['left'].set_visible(False)
axes[0].set_ylim(len(SHOWN) - 0.4, -1.45)
axes[0].set_yticks(range(len(SHOWN)))
axes[0].set_yticklabels([METRICS[m][0] for m in SHOWN], fontsize=6.6, color=INK)
axes[1].set_xlabel('Kendall $\\tau$')
save(figure, 'fig_all_above')
