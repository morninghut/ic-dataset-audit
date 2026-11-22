"""Kendall's tau over all samples is the sum of a detection term and a ranking term.

With L the lowest-score samples and A the others,
    tau_c(all) = w_det (2 AUC - 1) + w_rank tau_c(A),
where AUC is the area under the ROC curve of the metric for telling A from L, and the weights
depend only on the share of lowest-score samples (analysis/summary/components.py).

Left: the two terms for the best ranked metric of every evaluation set.  Right: ten ranked
metrics of the Polaris test set by their two abilities, the AUC of the detection term (bars start
at chance, 0.5) and tau above the lowest score.
Source: runs/summary/components.json.
"""
from common import ASH, GREY, INK, METRICS, RUNS5, SET_NAMES, WIDTH, colour, plt, read, save

SHOWN = ['VLM judge (InternVL 8B) + refs', 'VLM judge (Qwen 8B)', 'ridge SigLIP, Polaris pool (ours)', 'Polos (retrained)',
         'DENEB (retrained)', 'Pearl (retrained)', 'RefPAC-S++ (ViT-L/14)', 'CLIP-S', 'CIDEr', 'BLEU-4']
SETS = ['polaris_test', 'spica_test', 'nebula_test', 'composite', 'flickr8k_expert', 'flickr8k_cf']
data = read(RUNS5 / 'components.json')

figure = plt.figure(figsize=(WIDTH, 2.1))
left = figure.add_axes([0.17, 0.17, 0.3, 0.72])

# Left: decomposition for the best ranked metric of each set.
for y, name in enumerate(SETS):
    part = data[name]
    best = max((cell for cell in part['metrics'] if cell['ranked']), key=lambda cell: cell['tau_all'])
    # cross_share is the share of (concordant - discordant) pairs that pair a lowest-score sample
    # with another one; for tau_c it equals w_det (2 AUC - 1) / tau, and it also holds for tau_b.
    detection = best['cross_share'] * best['tau_all']
    ranking = best['tau_all'] - detection
    if part['variant'] == 'c':
        assert abs(100 * part['weight_detection'] * (2 * best['auc'] - 1) - detection) < 0.05, name
    left.barh(y, detection, height=0.56, color=ASH, linewidth=0)
    left.barh(y, ranking, left=detection, height=0.56, color=INK, linewidth=0)
    left.text(detection / 2, y, f'{100 * detection / (detection + ranking):.0f}%', ha='center', va='center',
              fontsize=6.2, color='white', fontweight='semibold')
    left.text(detection + ranking + 1.2, y, f"{best['tau_all']:.0f}", ha='left', va='center', fontsize=6.2, color=GREY)
left.set_yticks(range(len(SETS)))
left.set_yticklabels([SET_NAMES[name] for name in SETS], color=INK)
left.set_ylim(len(SETS) - 0.45, -0.55)
left.set_xlim(0, 76)
left.set_xticks([0, 20, 40, 60])
left.set_xlabel('Kendall $\\tau$, all samples')
left.spines['left'].set_visible(False)
for x0, shade, label in ((0.17, ASH, 'detection'), (0.31, INK, 'ranking')):
    figure.patches.append(plt.Rectangle((x0 + 0.005, 0.948), 0.016, 0.03, transform=figure.transFigure, color=shade))
    figure.text(x0 + 0.03, 0.963, label, ha='left', va='center', fontsize=6.2, color=GREY)
figure.text(0.0, 0.963, '(a)', ha='left', va='center', fontsize=6.8, fontweight='semibold', color=INK)
figure.text(0.545, 0.963, '(b)', ha='left', va='center', fontsize=6.8, fontweight='semibold', color=INK)

# Right: the two abilities of ten ranked metrics on the Polaris test set, as two columns of bars.
part = data['polaris_test']
cells = {cell['metric']: cell for cell in part['metrics'] if cell['ranked']}
abilities = [('auc', 'Detection', 'AUC', (0.5, 1.0), [0.5, 1.0], '{:.2f}'),
             ('tau_above', 'Ranking', 'Kendall $\\tau$', (0, 42), [0, 20, 40], '{:.0f}')]
for column, (key, title, label, limits, ticks, form) in enumerate(abilities):
    axis = figure.add_axes([0.745 + column * 0.135, 0.17, 0.105, 0.72])
    for y, metric in enumerate(SHOWN):
        value = cells[metric][key]
        axis.barh(y, value - limits[0], left=limits[0], height=0.62, color=colour(metric), linewidth=0)
        axis.text(value + 0.03 * (limits[1] - limits[0]), y, form.format(value), ha='left', va='center', fontsize=5.4,
                  color=GREY)
    axis.set_xlim(limits[0], limits[0] + 1.28 * (limits[1] - limits[0]))
    axis.set_xticks(ticks)
    axis.set_ylim(len(SHOWN) - 0.45, -0.55)
    axis.set_yticks(range(len(SHOWN)) if column == 0 else [])
    axis.set_yticklabels([METRICS[m][0] for m in SHOWN] if column == 0 else [], color=INK, fontsize=5.8)
    axis.set_xlabel(label)
    axis.set_title(title, fontsize=6.2, fontweight='medium', pad=3, loc='left', color=GREY)
    axis.spines['left'].set_visible(False)

save(figure, 'fig_components')
print({name: (round(data[name]['weight_detection'], 3), round(data[name]['weight_ranking'], 3)) for name in SETS})
