"""Figure 1: schematic of the study.

(a) Audit: which test sets have samples in which training sets (line width and label: share of
    the test set, from runs/metrics/near_duplicates.json).
(b) Controlled retraining: the same metric trained on the released and on the deduplicated
    training set, compared within three groups of test samples.
(c) Protocol: the three rules that follow, and what they are applied to.
Only the shares in (a) are data; (b) and (c) are diagrams.
"""
import numpy as np
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as Curve
from matplotlib.patheffects import withStroke

from common import GREY, INK, MIST, PAPER, WARM, WARM_SOFT, WIDTH, plt, read, save

HEIGHT = 2.3
UNITS = 100 * HEIGHT / WIDTH          # the canvas is 100 units wide with square units
LINE = 0.55

figure = plt.figure(figsize=(WIDTH, HEIGHT))
canvas = figure.add_axes([0, 0, 1, 1])
canvas.set_xlim(0, 100)
canvas.set_ylim(0, UNITS)
canvas.set_axis_off()


def box(x, y, width, height, text='', fill='white', edge=INK, style='-', size=6.0, ink=INK):
    canvas.add_patch(Rectangle((x, y), width, height, facecolor=fill, edgecolor=edge, linewidth=LINE, linestyle=style,
                               zorder=3))
    canvas.text(x + width / 2, y + height / 2, text, ha='center', va='center', fontsize=size, color=ink, zorder=4)


def arrow(start, end):
    canvas.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=5, color=INK, linewidth=LINE,
                                     shrinkA=0, shrinkB=0, zorder=2))


def panel(x, y, label):
    canvas.text(x, y, f'({label})', ha='left', va='center', fontsize=6.8, fontweight='semibold', color=INK)


def bezier(points, t):
    p0, p1, p2, p3 = (np.array(p) for p in points)
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


# ---------------------------------------------------------------------------------------------
# (a) Audit: test sets on the left, training sets on the right, a line where samples are shared.
data = read('near_duplicates.json')['pools']
panel(0, UNITS - 1.8, 'a')
NAMES = ['polaris', 'nebula', 'spica']
LEVEL = {'polaris': UNITS - 8.0, 'nebula': UNITS - 17.0, 'spica': UNITS - 26.0}
NODE, TALL, LEFT, RIGHT = 12.5, 5.0, 0.4, 30.6
for name in NAMES:
    box(LEFT, LEVEL[name] - TALL / 2, NODE, TALL, f'{name.capitalize()} test')
    box(RIGHT, LEVEL[name] - TALL / 2, NODE, TALL, f'{name.capitalize()} train')
links = []
for test in NAMES:
    for train in NAMES:
        cell = data[f'{train}_train'][f'{test}_test']
        share = 100 * cell['pair'] / cell['rows']
        if share >= 1:
            links.append((test, train, share))
# Lines leave and enter a box at separate heights, ordered so that they do not cross at the box.
for test, train, share in links:
    out = sorted((l for l in links if l[0] == test), key=lambda l: -LEVEL[l[1]])
    into = sorted((l for l in links if l[1] == train), key=lambda l: -LEVEL[l[0]])
    y0 = LEVEL[test] + 1.5 * ((len(out) - 1) / 2 - out.index((test, train, share)))
    y1 = LEVEL[train] + 1.5 * ((len(into) - 1) / 2 - into.index((test, train, share)))
    x0, x1 = LEFT + NODE, RIGHT
    major = share >= 10
    points = [(x0, y0), ((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1), (x1, y1)]
    canvas.add_patch(PathPatch(Curve(points, [Curve.MOVETO, Curve.CURVE4, Curve.CURVE4, Curve.CURVE4]), facecolor='none',
                               edgecolor=WARM, linewidth=max(0.5, share / 28), capstyle='butt', zorder=2,
                               linestyle='-' if major else (0, (2, 2))))
    # The label sits on the line near the end where the line is alone: the training set for
    # lines that leave the Polaris test set, the test set otherwise.
    at = bezier(points, 0.76 if sum(l[2] >= 10 for l in out) > 1 else 0.5 if not major else 0.2)
    canvas.text(at[0], at[1], f'{share:.0f}%', ha='center', va='center', fontsize=5.8, color=WARM, zorder=5,
                fontweight='semibold' if major else None,
                path_effects=[withStroke(linewidth=3, foreground='white', joinstyle='round')])

# ---------------------------------------------------------------------------------------------
# (b) Controlled retraining: two training sets, two models, one test set in three groups.
X = 52.0
panel(X, UNITS - 1.8, 'b')
BAR, HALF = 19.0, 3.4
left, right = X + 6.0, X + 29.0
top = UNITS - 9.7
for x, title, kept in ((left, 'released', True), (right, 'deduplicated', False)):
    canvas.text(x + BAR / 2, top + HALF + 1.9, title, ha='center', va='center', fontsize=5.8, color=INK)
    box(x, top, BAR * 0.72, HALF, fill=PAPER)
    box(x + BAR * 0.72, top, BAR * 0.28, HALF, fill=WARM if kept else 'white', edge=INK if kept else GREY,
        style='-' if kept else (0, (2, 2)))
    arrow((x + BAR / 2, top), (x + BAR / 2, top - 4.0))
    box(x + 3.5, top - 8.4, BAR - 7.0, 4.4, 'model ' + ('A' if kept else 'B'), size=5.8)
    arrow((x + BAR / 2, top - 8.4), (x + BAR / 2, top - 12.4))
canvas.text(X + 2.6, top + HALF / 2, 'train', ha='center', va='center', fontsize=5.8, color=GREY)
base = top - 12.4 - HALF - 0.6
span = right + BAR - left
canvas.text(X + 2.6, base + (HALF + 0.6) / 2, 'test', ha='center', va='center', fontsize=5.8, color=GREY)
x = left
for label, part, fill, ink in (('duplicated', 0.44, WARM, 'white'), ('image only', 0.28, WARM_SOFT, INK),
                               ('unseen', 0.28, 'white', INK)):
    box(x, base, span * part, HALF + 0.6, label, fill=fill, ink=ink, size=5.6)
    canvas.text(x + span * part / 2, base - 2.4, '$\\tau_A - \\tau_B$', ha='center', va='center', fontsize=6.0, color=INK)
    x += span * part

# ---------------------------------------------------------------------------------------------
# (c) Protocol.
panel(0, 12.7, 'c')
STEPS = ['Deduplicate', '$\\tau$: all samples / above lowest score', 'Paired tests', 'Re-ranking']
WIDTHS = [17.0, 37.0, 16.0, 15.0]
GAP = (100 - 4.4 - sum(WIDTHS)) / (len(STEPS) - 1)
x = 4.4
for index, (text, wide) in enumerate(zip(STEPS, WIDTHS)):
    box(x, 10.2, wide, 5.0, text, fill=PAPER if index == len(STEPS) - 1 else 'white', size=5.8)
    if index:
        arrow((x - GAP, 12.7), (x, 12.7))
    x += wide + GAP
canvas.set_ylim(9.5, UNITS)
figure.set_size_inches(WIDTH, HEIGHT * (UNITS - 9.5) / UNITS)

save(figure, 'fig_overview')
