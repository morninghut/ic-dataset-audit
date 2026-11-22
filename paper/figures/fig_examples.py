"""Three kinds of duplicate, each with one real example.

Exact: the same image file, the same candidate caption and the same list of ratings in the
Polaris test set and in the Spica training set; the Nebula training set stores their mean.
Re-encoded image: one photograph saved as two files whose pixels differ, found only by feature
similarity.  Near-identical caption: the same image with a caption that differs in a few characters.
Source: runs/summary/dup_examples.json (analysis/summary/dup_examples.py).
"""
import textwrap

from PIL import Image

from common import GREY, INK, MIST, RUNS5, WARM, WIDTH, card, plt, read, save

EXACT, IMAGE, CAPTION = 3, 2, 3
HEIGHT = 2.12
WRAP = 33
UNITS = 100 * HEIGHT / WIDTH
CARD, STEP = 35.4, 34.2

data = read(RUNS5 / 'dup_examples.json')
figure = plt.figure(figsize=(WIDTH, HEIGHT))
canvas = figure.add_axes([0, 0, 1, 1])
canvas.set_xlim(0, 100)
canvas.set_ylim(0, UNITS)
canvas.set_axis_off()


def photo(path, x, y, width, height):
    """The image, centre-cropped to the box."""
    image = Image.open(path).convert('RGB')
    target = width / height
    w, h = image.size
    if w / h > target:
        crop = int(h * target)
        image = image.crop(((w - crop) // 2, 0, (w - crop) // 2 + crop, h))
    else:
        crop = int(w / target)
        image = image.crop((0, (h - crop) // 2, w, (h - crop) // 2 + crop))
    image.thumbnail((520, 520))
    axis = figure.add_axes([x / 100, y / UNITS, width / 100, height / UNITS])
    axis.imshow(image, aspect='auto')
    axis.set_axis_off()


def title(index):
    x = index * STEP - 2.4
    canvas.text(x + 2.4, UNITS - 1.8, f"({'abc'[index]})", ha='left', va='center', fontsize=6.8, fontweight='semibold',
                color=INK)
    return x


def ratings(x, y, label, values, colour=INK):
    """A labelled row of squares, one per rating, shaded by the rating."""
    canvas.text(x, y, label, ha='left', va='center', fontsize=5.8, color=colour)
    left = x + 11.6
    for position, value in enumerate(values):
        shade = tuple(1 - value * (1 - c) for c in plt.matplotlib.colors.to_rgb(WARM))
        card(canvas, left + position * 2.05, y - 0.9, 1.8, 1.8, colour=shade, radius=0.35, edgecolor=MIST, linewidth=0.3)
    return left + len(values) * 2.05


def quoted(x, y, label, text, colour=INK):
    canvas.text(x, y, label, ha='left', va='center', fontsize=5.8, color=colour)
    canvas.text(x, y - 1.7, textwrap.fill(f'“{text}”', WRAP), ha='left', va='top', fontsize=5.9, color=colour,
                fontstyle='italic', linespacing=1.25)


PHOTO_TOP, PHOTO_HEIGHT = UNITS - 4.2, 23.0

# Exact duplicate.
item = data['exact'][EXACT]
assert item['polaris_test'] == item['spica_train']
x = title(0)
photo(item['image'], x + 2.4, PHOTO_TOP - PHOTO_HEIGHT, CARD - 4.8, PHOTO_HEIGHT)
lines = textwrap.wrap(f"“{item['candidate']}”", WRAP)
canvas.text(x + 2.4, PHOTO_TOP - PHOTO_HEIGHT - 1.9, '\n'.join(lines), ha='left', va='top', fontsize=5.9, color=INK,
            fontstyle='italic', linespacing=1.25)
y = PHOTO_TOP - PHOTO_HEIGHT - 4.4 - 3.0 * len(lines)
ratings(x + 2.4, y, 'Polaris test', item['polaris_test'])
ratings(x + 2.4, y - 3.3, 'Spica train', item['spica_train'], WARM)
end = ratings(x + 2.4, y - 6.6, 'Nebula train', item['nebula_train'], WARM)
canvas.text(end + 0.6, y - 6.6, f"mean {item['nebula_train'][0]:.2f}", ha='left', va='center', fontsize=5.6,
            color=GREY)

# Re-encoded image.
item = data['image'][IMAGE]
assert item['test_size'] == item['train_size']
x = title(1)
half = (CARD - 4.8 - 1.0) / 2
photo(item['spica_test_file'], x + 2.4, PHOTO_TOP - PHOTO_HEIGHT, half, PHOTO_HEIGHT)
photo(item['spica_train_file'], x + 2.4 + half + 1.0, PHOTO_TOP - PHOTO_HEIGHT, half, PHOTO_HEIGHT)
y = PHOTO_TOP - PHOTO_HEIGHT - 3.2
for offset, label, size, colour in ((0, 'Spica test', item['test_bytes'], INK),
                                    (half + 1.0, 'Spica train', item['train_bytes'], WARM)):
    canvas.text(x + 2.4 + offset, y, label, ha='left', va='center', fontsize=5.8, color=colour)
    canvas.text(x + 2.4 + offset, y - 3.0, f'{size / 1000:.0f} kB', ha='left', va='center', fontsize=5.8, color=GREY)
canvas.text(x + 2.4, y - 6.4, f"feature similarity {item['cosine']:.3f}", ha='left', va='center', fontsize=5.8,
            color=GREY)

# Near-identical caption.
item = data['caption'][CAPTION]
x = title(2)
photo(item['image'], x + 2.4, PHOTO_TOP - PHOTO_HEIGHT, CARD - 4.8, PHOTO_HEIGHT)
y = PHOTO_TOP - PHOTO_HEIGHT - 3.2
quoted(x + 2.4, y, 'Polaris test', item['polaris_test_caption'])
quoted(x + 2.4, y - 9.6, 'Spica train', item['spica_train_caption'], WARM)

save(figure, 'fig_examples')
