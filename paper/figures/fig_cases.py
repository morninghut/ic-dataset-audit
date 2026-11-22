"""Case studies: one image-caption pair per row, with where the humans and the metrics place it.

Left: the image.  Middle: the candidate caption, one reference, and the human score (rescaled to
0-1) with the number of ratings.  Right: the percentile of the pair under each metric (bar) and
under the human score (vertical line), both taken among the pairs of the same evaluation set
that are rated above the lowest score.  Writes fig_cases (main text) and fig_cases_more_1,
fig_cases_more_2 (appendix).  Source: runs/summary/cases.json (analysis/summary/cases.py).
"""
import textwrap

from PIL import Image

from common import DUPLICATED, GREY, INK, MIST, PAPER, RUNS5, SET_NAMES, WARM, WIDTH, colour, plt, read, save

# Label, metric on sets where its released model is ranked, fallback on sets where it is not.
SHOWN = [('CIDEr', 'CIDEr'), ('CLIP-S', 'CLIP-S'), ('RefPAC-S++', 'RefPAC-S++ (ViT-L/14)'), ('Polos', 'Polos (retrained)'),
         ('DENEB', 'DENEB (retrained)'), ('Pearl', 'Pearl (retrained)'), ('VLM judge', 'VLM judge (InternVL 8B) + refs')]
EXTRA = ('Pearl (dup.)', 'Pearl (retrained)' + DUPLICATED)
# The main text shows two of the four cases in cases.json (page limit): a success of the judge and a duplicated sample.
MAIN = ('learned and embedding metrics too high', 'duplicated sample')
ROW, PHOTO_W, PHOTO_H, PLOT = 17.6, 20.5, 15.6, 33.0


def photo(figure, units, path, x, y, width, height):
    image = Image.open(path).convert('RGB')
    target = width / height
    w, h = image.size
    if w / h > target:
        crop = int(h * target)
        image = image.crop(((w - crop) // 2, 0, (w - crop) // 2 + crop, h))
    else:
        crop = int(w / target)
        image = image.crop((0, (h - crop) // 2, w, (h - crop) // 2 + crop))
    image.thumbnail((480, 480))
    axis = figure.add_axes([x / 100, y / units, width / 100, height / units])
    axis.imshow(image, aspect='auto')
    axis.set_axis_off()


def draw(cases, name, letters=True, sets=False):
    units = ROW * len(cases) + 3.2
    figure = plt.figure(figsize=(WIDTH, WIDTH * units / 100))
    canvas = figure.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 100)
    canvas.set_ylim(0, units)
    canvas.set_axis_off()
    for position, case in enumerate(cases):
        top = units - 0.4 - position * ROW
        photo(figure, units, case['image'], 3.6 if letters else 0, top - PHOTO_H, PHOTO_W, PHOTO_H)
        if letters:
            canvas.text(0, top - 1.2, f"({'abcdefgh'[position]})", ha='left', va='center', fontsize=6.8,
                        fontweight='semibold', color=INK)
        x = (3.6 if letters else 0) + PHOTO_W + 1.6
        lines = textwrap.wrap(f"“{case['candidate'].strip()}”", 25)[:3]
        canvas.text(x, top - 0.3, '\n'.join(lines), ha='left', va='top', fontsize=5.8, color=INK, fontstyle='italic',
                    linespacing=1.25)
        reference = textwrap.wrap('Ref.: ' + case['references'][0].strip(), 29) if case['references'] else []
        room = (4 if sets else 5) - len(lines)
        if len(reference) > room:
            reference = reference[:room]
            reference[-1] = reference[-1][:26] + ' …'
        canvas.text(x, top - 1.0 - 2.75 * len(lines), '\n'.join(reference), ha='left', va='top', fontsize=5.3,
                    color=GREY, linespacing=1.25)
        count = len(case['ratings'])
        if sets:
            canvas.text(x, top - PHOTO_H + 3.5, SET_NAMES[case['set']] + (' test' if case['set'].endswith('_test') else ''),
                        ha='left', va='center', fontsize=5.3, color=GREY)
        canvas.text(x, top - PHOTO_H + 0.9, f"human {case['human']:.2f} ({count} rating{'s' if count > 1 else ''})",
                    ha='left', va='center', fontsize=5.8, color=INK)
        shown = [(label, metric) for label, metric in SHOWN if metric in case['percentile']]
        if case['note'] == 'duplicated sample' and EXTRA[1] in case['percentile']:
            shown.insert([label for label, _ in shown].index('Pearl') + 1, EXTRA)
        if 'Rigel (official ckpt) + refs' in case['percentile']:
            shown.append(('Rigel', 'Rigel (official ckpt) + refs'))
        left = 100 - PLOT
        axis = figure.add_axes([left / 100, (top - PHOTO_H + 0.4) / units, (100 - left - 1.2) / 100, (PHOTO_H - 0.4) / units])
        axis.axvline(case['human_percentile'], color=INK, linewidth=0.8, linestyle=(0, (2, 1.5)), zorder=4)
        for y, (label, metric) in enumerate(shown):
            value = case['percentile'][metric]
            held = metric.endswith(DUPLICATED)
            axis.barh(y, 100, height=0.62, color=PAPER, linewidth=0, zorder=1)
            axis.barh(y, value, height=0.62, color=WARM if held else colour(metric), linewidth=0, zorder=3)
        axis.set_yticks(range(len(shown)))
        axis.set_yticklabels([label for label, _ in shown], fontsize=5.6, color=INK)
        axis.set_ylim(len(shown) - 0.4, -0.6)
        axis.set_xlim(0, 101)
        last = position == len(cases) - 1
        axis.set_xticks([0, 50, 100])
        axis.set_xticklabels(['0', '50', '100'] if last else [])
        axis.tick_params(axis='x', labelsize=5.6, pad=1.5)
        for side in ('left', 'bottom'):
            axis.spines[side].set_visible(False)
        if last:
            axis.set_xlabel('Percentile within the evaluation set', fontsize=5.8, labelpad=1.5)
        if position == 0:
            axis.text(case['human_percentile'], -0.95, 'human', ha='right' if case['human_percentile'] > 85 else 'center',
                      va='bottom', fontsize=5.4, color=INK)
    save(figure, name)


if __name__ == '__main__':
    data = read(RUNS5 / 'cases.json')
    draw([case for case in data['main'] if case['note'] in MAIN], 'fig_cases')
    half = (len(data['more']) + 1) // 2
    draw(data['more'][:half], 'fig_cases_more_1', sets=True)
    draw(data['more'][half:], 'fig_cases_more_2', sets=True)
