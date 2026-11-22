"""The re-ranking as a heat map: Kendall tau (number) and rank within the set (shade).

One row per metric, two columns per evaluation set: tau over all samples and tau above the lowest
human score.  A darker cell is a better rank among the metrics ranked on that set.  Bold: not
separated from the first by the paired test.  A dot: the ranked model was retrained on the
deduplicated training set, because the released training set holds samples of that set.
Parentheses on white: only a model whose training set holds samples of the set exists, so the
entry is not ranked.  Last row: the human reference (one rating against the mean of the others).
Source: runs/metrics/pairwise_s12-13-14.json.
"""
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from common import COOL, DUPLICATED, GREY, INK, METRICS, MIST, PAPER, SET_NAMES, WIDTH, card, entry, plt, ranking, save

SETS = ['polaris_test', 'spica_test', 'nebula_test', 'composite', 'flickr8k_expert', 'flickr8k_cf']
PARTS = [('all', 'all'), ('above_floor', 'above')]
GROUPS = [
    ['Rigel (official ckpt) + refs', 'Rigel (official ckpt)'],
    ['VLM judge (InternVL 8B) + refs', 'VLM judge (Qwen 8B) + refs', 'VLM judge + refs',
     'VLM judge (InternVL 8B)', 'VLM judge (Qwen 8B)', 'VLM judge'],
    ['Pearl (retrained)', 'DENEB (retrained)', 'Polos (retrained)'],
    ['ridge SigLIP, Polaris pool (ours)', 'ridge SigLIP, Spica pool (ours)', 'ridge CLIP, Polaris pool (ours)'],
    ['RefPAC-S++ (ViT-L/14)', 'PAC-S++ (ViT-L/14)', 'RefCLIP-S', 'CLIP-S'],
    ['CIDEr', 'BLEU-4'],
]
CELL, TALL, GAP, SETGAP, LABEL = 5.75, 2.9, 0.9, 1.0, 26.0
LEGEND_H = 12.0
SHADES = LinearSegmentedColormap.from_list('rank', ['#0a4a94', COOL, '#9cc8f5', '#e6f1fd', '#fbfcfe'])

benchmarks = ranking()
rows = sum(len(group) for group in GROUPS)
body_height = (rows + 1) * TALL + (len(GROUPS)) * GAP + 9.5
height = body_height + LEGEND_H
figure = plt.figure(figsize=(WIDTH, WIDTH * height / 100))
axis = figure.add_axes([0, 0, 1, 1])
axis.set_xlim(0, 100)
axis.set_ylim(0, height)
axis.set_axis_off()


def left_of(index, part):
    return LABEL + index * (2 * CELL + SETGAP) + part * CELL


def draw_colorbar(x0, y0, w, h, steps=96):
    """Vector gradient bar matching the exact SHADES(depth ** 0.75) rank mapping."""
    dx = w / steps
    for i in range(steps):
        depth = 1.0 - (i + 0.5) / steps
        seg_w = dx * 1.15 if i < steps - 1 else dx
        axis.add_patch(Rectangle((x0 + i * dx, y0), seg_w, h, facecolor=SHADES(depth ** 0.75),
                                 edgecolor='none', zorder=2))
    axis.add_patch(Rectangle((x0, y0), w, h, facecolor='none', edgecolor=MIST, linewidth=0.45, zorder=3))


# Top visual legend: within-set rank colorbar (row 1) + miniature cell-marker key (row 2).
card(axis, 3.8, height - 10.6, 96.1, 10.3, colour=PAPER, radius=1.2, zorder=1)

y_cb = height - 3.3
axis.text(6.0, y_cb, 'Within-set rank', ha='left', va='center', fontsize=5.7, color=INK, fontweight='medium', zorder=3)
axis.text(35.0, y_cb, 'Lower rank', ha='right', va='center', fontsize=5.3, color=GREY, zorder=3)
draw_colorbar(36.6, y_cb - 1.15, 45.6, 2.3)
axis.text(97.7, y_cb, 'Higher rank (1st)', ha='right', va='center', fontsize=5.3, color=INK, fontweight='medium', zorder=3)

y_sym = height - 7.8
sw_w, sw_h = 5.5, 2.75
for sx, kind, desc in (
    (6.0, 'bold', 'Not separated from 1st'),
    (36.6, 'dot', 'Retrained (deduplicated)'),
    (66.2, 'paren', 'Not ranked (duplicates in train)'),
):
    if kind == 'bold':
        axis.add_patch(Rectangle((sx, y_sym - sw_h / 2), sw_w, sw_h, facecolor=SHADES(0.0), edgecolor='none', zorder=2))
        axis.text(sx + sw_w / 2, y_sym - 0.03, 'Bold', ha='center', va='center', fontsize=5.2, color='white',
                  fontweight='bold', zorder=3)
    elif kind == 'dot':
        axis.add_patch(Rectangle((sx, y_sym - sw_h / 2), sw_w, sw_h, facecolor=SHADES(0.45), edgecolor='none', zorder=2))
        axis.text(sx + sw_w / 2 - 0.2, y_sym - 0.03, '\u03c4', ha='center', va='center', fontsize=5.5, color=INK, zorder=3)
        axis.plot(sx + sw_w - 0.70, y_sym + sw_h / 2 - 0.65, 'o', markersize=1.6, color=INK, zorder=4)
    else:
        axis.add_patch(Rectangle((sx, y_sym - sw_h / 2), sw_w, sw_h, facecolor='white', edgecolor=MIST,
                                 linewidth=0.45, zorder=2))
        axis.text(sx + sw_w / 2, y_sym - 0.03, '(\u03c4)', ha='center', va='center', fontsize=5.3, color=GREY, zorder=3)
    axis.text(sx + sw_w + 1.2, y_sym, desc, ha='left', va='center', fontsize=5.3, color=GREY, zorder=3)


top = body_height - 8.8
for index, name in enumerate(SETS):
    x = left_of(index, 0)
    axis.text(x + CELL, body_height - 2.0, SET_NAMES[name].replace('Flickr8k-', 'F8k-'), ha='center', va='center', fontsize=6.2,
              color=INK, fontweight='medium')
    axis.plot([x + 0.3, x + 2 * CELL - 0.3], [body_height - 4.0, body_height - 4.0], color=MIST, linewidth=0.5)
    for part, (_, label) in enumerate(PARTS):
        axis.text(x + (part + 0.5) * CELL, body_height - 6.1, label, ha='center', va='center', fontsize=5.4, color=GREY)

y = top
for group in GROUPS:
    for metric in group:
        axis.text(LABEL - 1.2, y - TALL / 2, METRICS[metric][0], ha='right', va='center', fontsize=5.8, color=INK)
        for index, name in enumerate(SETS):
            for part, (key, _) in enumerate(PARTS):
                table = benchmarks[name][key]
                x = left_of(index, part)
                row = entry(table, metric)
                held = entry(table, metric + DUPLICATED)
                if row is not None and 'rank' in row:
                    depth = (row['rank'] - 1) / (len(table['rows']) - 1)
                    axis.add_patch(plt.Rectangle((x + 0.12, y - TALL + 0.12), CELL - 0.24, TALL - 0.24,
                                                 facecolor=SHADES(depth ** 0.75), edgecolor='none'))
                    axis.text(x + CELL / 2, y - TALL / 2 - 0.03, f"{row['score']:.1f}", ha='center', va='center',
                              fontsize=5.5, color='white' if depth ** 0.75 < 0.36 else INK,
                              fontweight='bold' if metric in table['top_group'] else 'regular')
                    if held is not None:
                        axis.plot(x + CELL - 0.75, y - 0.72, 'o', markersize=1.5,
                                  color='white' if depth ** 0.75 < 0.36 else INK)
                elif held is not None:
                    axis.add_patch(plt.Rectangle((x + 0.12, y - TALL + 0.12), CELL - 0.24, TALL - 0.24, facecolor='white',
                                                 edgecolor=MIST, linewidth=0.4))
                    axis.text(x + CELL / 2, y - TALL / 2 - 0.03, f"({held['score']:.1f})", ha='center', va='center',
                              fontsize=5.3, color=GREY)
        y -= TALL
    y -= GAP
axis.text(LABEL - 1.2, y - TALL / 2, 'Human reference', ha='right', va='center', fontsize=5.8, color=GREY)
for index, name in enumerate(SETS):
    judged = benchmarks[name].get('above_floor_judged')
    axis.text(left_of(index, 1) + CELL / 2, y - TALL / 2, f"{judged['human']:.1f}" if judged else '–', ha='center',
              va='center', fontsize=5.5, color=GREY)

save(figure, 'fig_heatmap')
