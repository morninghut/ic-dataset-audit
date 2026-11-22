"""Cross-fit effect of having trained on a sample, by learner.

For each learner and each (training set, test set) that share samples: mean Kendall tau_c of the
four models that saw a sample minus tau_c of the model that did not, one model against one
model, on the duplicated samples (orange bar) and on the samples whose image is not in the training set
at all (grey bar; a placebo).  Whisker: 95% bootstrap interval over images; networks are means of three
seeds.  Source: runs/summary/groups.json (analysis/summary/groups.py).
"""
from common import ASH, GREY, INK, MIST, RUNS5, WARM, WIDTH, plt, read, save

LEARNERS = [('ridge_siglip', 'Ridge regression'), ('mlp_siglip', 'Three-layer network'), ('polos_arch', 'Polos architecture')]
PAIRS = [('spica_train', 'polaris_test', 'Spica → Polaris'), ('nebula_train', 'polaris_test', 'Nebula → Polaris'),
         ('polaris_train', 'spica_test', 'Polaris → Spica'), ('polaris_train', 'nebula_test', 'Polaris → Nebula')]

data = read(RUNS5 / 'groups.json')['crossfit']
figure, axes = plt.subplots(1, 3, figsize=(WIDTH, 1.75), sharey=True, gridspec_kw={'wspace': 0.1, 'left': 0.215})
for axis, (learner, title) in zip(axes, LEARNERS):
    axis.axvline(0, color=MIST, linewidth=0.7, zorder=1)
    for y, (pool, name, _) in enumerate(PAIRS):
        cell = data.get(f'{learner}|{pool}>{name}')
        if not cell:
            axis.text(12, y, 'not trained', ha='center', va='center', fontsize=5.4, color=ASH)
            continue
        for group, offset, colour in (('duplicated/all', -0.16, WARM), ('unseen/all', 0.16, ASH)):
            part = cell[group]
            axis.barh(y + offset, part['effect'], height=0.3, color=colour, linewidth=0, zorder=2)
            axis.plot([part['low'], part['high']], [y + offset] * 2, color=INK, linewidth=0.6, zorder=3,
                      solid_capstyle='butt')
    axis.set_title(title, fontsize=6.4, fontweight='medium', pad=4, loc='center')
    axis.set_xlim(-3, 25)
    axis.set_xticks([0, 10, 20])
    axis.set_ylim(len(PAIRS) - 0.5, -0.5)
    axis.spines['left'].set_visible(False)
    axis.grid(axis='x')
    axis.set_axisbelow(True)
axes[0].set_yticks(range(len(PAIRS)))
axes[0].set_yticklabels([label for _, _, label in PAIRS], color=INK, fontsize=6.2)
axes[1].set_xlabel('Seen − unseen, Kendall $\\tau$')
save(figure, 'fig_crossfit')
