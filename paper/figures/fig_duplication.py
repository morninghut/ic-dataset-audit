"""How much of each evaluation set is already inside each training set.

One cell per (training set, evaluation set).  The bar is the whole evaluation set; the dark part
is the share of its samples whose image, candidate caption and human score are all in the training
set, the light part the share whose image is in the training set with other candidates.  Source:
runs/metrics/near_duplicates.json (exact matches).
"""
from matplotlib.patches import FancyBboxPatch

from common import GREY, INK, MIST, PAPER, WARM, WARM_SOFT, WIDTH, plt, read, save

POOLS = [('polaris_train', 'Polaris train', 'Polos'), ('nebula_train', 'Nebula train', 'DENEB'),
         ('spica_train', 'Spica train', 'Pearl, Rigel')]
FAMILY = [('polaris_test', 'Polaris\ntest'), ('nebula_test', 'Nebula\ntest'), ('spica_test', 'Spica\ntest')]
EARLIER = [('composite', 'Composite'), ('flickr8k_expert', 'Flickr8k\nExpert'), ('flickr8k_cf', 'Flickr8k\nCF')]

data = read('near_duplicates.json')['pools']
figure, axis = plt.subplots(figsize=(WIDTH, 2.05))
axis.set_axis_off()
axis.set_xlim(0, 100)
axis.set_ylim(-2, 44)

LEFT, CELL, GAP, GROUP_GAP = 23.5, 9.9, 2.4, 5.0
columns = []
x = LEFT
for group in (FAMILY, EARLIER):
    for name, label in group:
        columns.append((name, label, x))
        x += CELL + GAP
    x += GROUP_GAP - GAP

for (start, end), title in (((0, 2), 'Recent collections'), ((3, 5), 'Earlier benchmarks')):
    x0, x1 = columns[start][2], columns[end][2] + CELL
    axis.text((x0 + x1) / 2, 42.5, title, ha='center', va='center', fontsize=6.5, color=GREY)
    axis.plot([x0, x1], [40.2, 40.2], color=MIST, linewidth=0.6)
for name, label, x in columns:
    axis.text(x + CELL / 2, 36, label, ha='center', va='center', fontsize=6.3, color=INK,
              fontweight='medium', linespacing=1.15)

for row, (pool, title, metrics) in enumerate(POOLS):
    y = 26.5 - row * 9.6
    axis.text(0, y - 0.4, title, ha='left', va='center', fontsize=6.8, color=INK, fontweight='medium')
    for name, label, x in columns:
        cell = data[pool][name]
        pair, picture = (100 * cell[key] / cell['rows'] for key in ('pair', 'picture_only'))
        axis.add_patch(FancyBboxPatch((x, y - 3.1), CELL, 1.5, boxstyle='round,pad=0,rounding_size=0.7',
                                      facecolor=PAPER, edgecolor='none'))
        for share, colour in ((pair + picture, WARM_SOFT), (pair, WARM)):
            if share >= 0.5:
                axis.add_patch(FancyBboxPatch((x, y - 3.1), CELL * share / 100, 1.5,
                                              boxstyle=f'round,pad=0,rounding_size={min(0.7, CELL * share / 200)}',
                                              facecolor=colour, edgecolor='none'))
        strong = pair >= 1
        text = '0' if not cell['pair'] else f'{pair:.1f}%' if pair >= 0.05 else '<0.1%'
        axis.text(x + CELL / 2, y + 1.4, text, ha='center', va='center',
                  fontsize=8 if strong else 6.8, fontweight='semibold' if strong else 'regular',
                  color=WARM if strong else GREY)

for x0, colour, text in ((LEFT, WARM, 'duplicated sample'), (LEFT + 34, WARM_SOFT, 'image only')):
    y = -0.6
    axis.add_patch(FancyBboxPatch((x0, y - 0.75), 3.2, 1.5, boxstyle='round,pad=0,rounding_size=0.7',
                                  facecolor=colour, edgecolor='none'))
    axis.text(x0 + 4.4, y, text, ha='left', va='center', fontsize=6.2, color=GREY)

save(figure, 'fig_duplication')
