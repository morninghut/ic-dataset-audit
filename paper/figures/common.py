"""Shared look and data access for the paper's figures.

Every figure script reads the result files under runs/ and writes <name>.pdf (for the paper) and
<name>.png (for a quick look) next to itself.  No number is typed into a figure script.

Pattern Recognition asks for a single column with 4.8 cm side margins, so the text block is
11.4 cm wide; WIDTH below is that width in inches and every figure is drawn at its final size.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / 'runs' / 'metrics'
RUNS5 = ROOT / 'runs' / 'summary'
WIDTH = 11.4 / 2.54

# One neutral scale and two colours with a fixed meaning throughout the paper:
# WARM = the training set holds duplicated samples of the evaluation set; COOL = it does not.
INK, GREY, MIST, PAPER = '#1d1d1f', '#6e6e73', '#d2d2d7', '#f5f5f7'
WARM, COOL = '#f56300', '#0071e3'
WARM_SOFT, COOL_SOFT = '#fbd0b3', '#b3d4f7'
VIOLET, TEAL = '#7d4ddb', '#1a9e96'

for path in sorted((HERE.parent / 'fonts' / 'extras' / 'ttf').glob('Inter-*.ttf')):
    font_manager.fontManager.addfont(str(path))

plt.rcParams.update({
    'font.family': 'Inter', 'font.size': 7, 'text.color': INK,
    'axes.edgecolor': MIST, 'axes.labelcolor': GREY, 'axes.labelsize': 7, 'axes.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 7.5,
    'axes.titleweight': 'semibold', 'axes.titlelocation': 'left', 'axes.titlepad': 6,
    'xtick.color': GREY, 'ytick.color': GREY, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'xtick.major.pad': 3, 'ytick.major.pad': 3,
    'grid.color': PAPER, 'grid.linewidth': 0.8, 'legend.frameon': False, 'legend.fontsize': 6.5,
    'lines.solid_capstyle': 'round', 'pdf.fonttype': 42, 'savefig.dpi': 400,
})

SET_NAMES = {
    'polaris_test': 'Polaris', 'spica_test': 'Spica', 'nebula_test': 'Nebula',
    'flickr8k_expert': 'Flickr8k-Expert', 'composite': 'Composite', 'flickr8k_cf': 'Flickr8k-CF',
}


def read(name):
    path = Path(name)
    return json.loads((path if path.is_absolute() else RUNS / name).read_text())


def save(figure, name):
    for suffix in ('pdf', 'png'):
        figure.savefig(HERE / f'{name}.{suffix}', bbox_inches='tight', pad_inches=0.02)
    plt.close(figure)
    print('wrote', name)

# Names used in the paper for the entries of the ranking, their family, and the family colours.
ASH = '#a1a1a6'
FAMILIES = {
    'tuned': ('VLM judge, fine-tuned on human scores', VIOLET),
    'judge': ('VLM judge, zero-shot', COOL),
    'learned': ('Learned regression metric', INK),
    'probe': ('Ridge regression on frozen features', TEAL),
    'embedding': ('Embedding similarity', GREY),
    'text': ('Text overlap', ASH),
}
METRICS = {
    'Rigel (official ckpt) + refs': ('Rigel +ref', 'tuned'),
    'Rigel (official ckpt)': ('Rigel', 'tuned'),
    'VLM judge (InternVL 8B) + refs': ('Judge InternVL-8B +ref', 'judge'),
    'VLM judge (Qwen 8B) + refs': ('Judge Qwen-8B +ref', 'judge'),
    'VLM judge + refs': ('Judge Qwen-4B +ref', 'judge'),
    'VLM judge (InternVL 8B)': ('Judge InternVL-8B', 'judge'),
    'VLM judge (Qwen 8B)': ('Judge Qwen-8B', 'judge'),
    'VLM judge': ('Judge Qwen-4B', 'judge'),
    'Polos (retrained)': ('Polos', 'learned'),
    'Polos (official ckpt)': ('Polos (official)', 'learned'),
    'DENEB (retrained)': ('DENEB', 'learned'),
    'Pearl (retrained)': ('Pearl', 'learned'),
    'ridge SigLIP, Polaris pool (ours)': ('Ridge SigLIP (Polaris)', 'probe'),
    'ridge SigLIP, Spica pool (ours)': ('Ridge SigLIP (Spica)', 'probe'),
    'ridge CLIP, Polaris pool (ours)': ('Ridge CLIP (Polaris)', 'probe'),
    'ridge CLIP, Spica pool (ours)': ('Ridge CLIP (Spica)', 'probe'),
    'RefPAC-S++ (ViT-L/14)': ('RefPAC-S++ (L/14)', 'embedding'),
    'RefPAC-S (ViT-L/14)': ('RefPAC-S (L/14)', 'embedding'),
    'RefPAC-S++': ('RefPAC-S++', 'embedding'),
    'RefPAC-S': ('RefPAC-S', 'embedding'),
    'RefCLIP-S': ('RefCLIP-S', 'embedding'),
    'PAC-S++ (ViT-L/14)': ('PAC-S++ (L/14)', 'embedding'),
    'PAC-S (ViT-L/14)': ('PAC-S (L/14)', 'embedding'),
    'PAC-S++': ('PAC-S++', 'embedding'),
    'PAC-S': ('PAC-S', 'embedding'),
    'CLIP-S': ('CLIP-S', 'embedding'),
    'text only (SimCSE)': ('Text-only SimCSE', 'text'),
    'CIDEr': ('CIDEr', 'text'), 'SPICE': ('SPICE', 'text'), 'METEOR': ('METEOR', 'text'),
    'ROUGE-L': ('ROUGE-L', 'text'), 'BLEU-4': ('BLEU-4', 'text'), 'BLEU-1': ('BLEU-1', 'text'),
}
DUPLICATED, CLEANED = ' +dup', ' (cleaned pool)'


def ranking():
    return read('pairwise_s12-13-14.json')['benchmarks']


def entry(part, name):
    """The row of a metric in one part of the ranking, ranked or listed as unranked; None if absent."""
    for row in part['rows'] + part.get('duplicated', []):
        if row['metric'] == name:
            return row
    return None


def colour(name):
    return FAMILIES[METRICS[name.removesuffix(DUPLICATED).removesuffix(CLEANED)][1]][1]


def minus(text):
    """Typographic minus sign for numbers printed in a figure."""
    return text.replace('-', '\u2212')


def card(axis, x, y, width, height, colour=PAPER, radius=1.2, **kwargs):
    from matplotlib.patches import FancyBboxPatch
    patch = FancyBboxPatch((x, y), width, height, boxstyle=f'round,pad=0,rounding_size={radius}', facecolor=colour,
                           edgecolor=kwargs.pop('edgecolor', 'none'), **kwargs)
    axis.add_patch(patch)
    return patch
