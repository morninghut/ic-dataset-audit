"""What the evaluation sets can tell apart.

(a) Of the 28 pairs among the eight best ranked metrics of a set, how many a paired bootstrap test
    separates after Holm correction, for Kendall tau over all samples, tau above the lowest score
    and order accuracy (only sets that record several ratings per pair have the last).
(b) Paired differences between the three learned regression metrics (tau over all samples, 95%
    interval, whisker); filled bars are significant after Holm correction over all pairs of the set.
Source: runs/metrics/pairwise_s12-13-14.json.
"""
import itertools

from common import ASH, GREY, INK, MIST, SET_NAMES, WIDTH, plt, ranking, save

SETS = ['polaris_test', 'spica_test', 'nebula_test', 'composite', 'flickr8k_expert', 'flickr8k_cf']
PARTS = [('all', 'all samples', ASH), ('above_floor', 'above lowest score', INK), ('order', 'order accuracy', 'white')]
LEARNED = [('Polos (retrained)', 'Polos'), ('DENEB (retrained)', 'DENEB'), ('Pearl (retrained)', 'Pearl')]
TOP = 8

benchmarks = ranking()


def separated(part):
    top = [row['metric'] for row in sorted(part['rows'], key=lambda row: row['rank'])[:TOP]]
    pairs = {frozenset((pair['a'], pair['b'])): pair for pair in part['pairs']}
    return sum(pairs[frozenset(couple)]['p_holm'] < 0.05 for couple in itertools.combinations(top, 2))


def difference(part, a, b):
    for pair in part['pairs']:
        if {pair['a'], pair['b']} == {a, b}:
            sign = 1 if pair['a'] == a else -1
            return (sign * pair['mean'], min(sign * pair['low'], sign * pair['high']),
                    max(sign * pair['low'], sign * pair['high']), pair['p_holm'] < 0.05)
    raise KeyError((a, b))


figure = plt.figure(figsize=(WIDTH, 2.2))
left = figure.add_axes([0.2, 0.2, 0.25, 0.63])
total = TOP * (TOP - 1) // 2
for y, name in enumerate(SETS):
    for offset, (key, _, shade) in zip((-0.26, 0, 0.26), PARTS):
        part = benchmarks[name].get(key)
        if not part:
            continue
        count = separated(part)
        if count:
            left.barh(y + offset, count, height=0.22, color=shade, edgecolor=INK if shade == 'white' else 'none', linewidth=0.5)
            left.text(count + 0.8, y + offset, str(count), ha='left', va='center', fontsize=5.4, color=GREY)
        else:   # no bar to draw: the number alone, darker than the others so that it is not overlooked
            left.text(0.8, y + offset, '0', ha='left', va='center', fontsize=5.4, color=INK)
left.axvline(total, color=MIST, linewidth=0.6)
left.set_yticks(range(len(SETS)))
left.set_yticklabels([SET_NAMES[name] for name in SETS], color=INK)
left.set_ylim(len(SETS) - 0.45, -0.55)
left.set_xlim(0, total + 3)
left.set_xticks([0, 14, 28])
left.set_xlabel(f'Separated pairs (of {total})')
left.spines['left'].set_visible(False)
for position, (_, label, shade) in enumerate(PARTS):
    x = [0.04, 0.17, 0.375][position]
    figure.patches.append(plt.Rectangle((x, 0.905), 0.016, 0.03, transform=figure.transFigure, facecolor=shade,
                                        edgecolor=INK if shade == 'white' else 'none', linewidth=0.5))
    figure.text(x + 0.024, 0.92, label, ha='left', va='center', fontsize=6.0, color=GREY)
figure.text(0.0, 0.98, '(a)', ha='left', va='center', fontsize=6.8, fontweight='semibold', color=INK)
figure.text(0.555, 0.98, '(b)', ha='left', va='center', fontsize=6.8, fontweight='semibold', color=INK)

couples = list(itertools.combinations(LEARNED, 2))
for column, ((a, name_a), (b, name_b)) in enumerate(couples):
    axis = figure.add_axes([0.575 + column * 0.145, 0.2, 0.125, 0.63])
    axis.axvline(0, color=MIST, linewidth=0.7, zorder=1)
    for y, name in enumerate(SETS):
        mean, low, high, clear = difference(benchmarks[name]['all'], a, b)
        axis.barh(y, mean, height=0.5, facecolor=INK if clear else 'white', edgecolor=INK, linewidth=0.5, zorder=2)
        axis.plot([low, high], [y, y], color=GREY, linewidth=0.7, zorder=3, solid_capstyle='butt')
    axis.set_ylim(len(SETS) - 0.45, -0.55)
    axis.set_xlim(-5.2, 5.2)
    axis.set_xticks([-4, 0, 4])
    axis.set_xticklabels(['−4', '0', '4'])
    axis.set_yticks([])
    axis.spines['left'].set_visible(False)
    axis.set_title(f'{name_a} − {name_b}', fontsize=6.2, fontweight='medium', pad=4, loc='center')
    if column == 1:
        axis.set_xlabel('Difference in Kendall $\\tau$')
    if column == 0:
        axis.set_yticks(range(len(SETS)))
        axis.set_yticklabels([SET_NAMES[name].replace('Flickr8k', 'F8k') for name in SETS], color=INK, fontsize=5.6)

save(figure, 'fig_power')
