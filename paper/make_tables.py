"""Write every table and every number of the paper from the result files.

    ../.venv/bin/python make_tables.py

Reads runs/metrics/*.json and runs/summary/*.json and writes
generated/tab_*.tex (one tabular each, \\input by main.tex and appendix.tex) and generated/numbers.tex
(one \\newcommand per number quoted in the running text).  The numbers of the paper come from
this file's output; the exceptions are the published results of other papers in PUBLISHED below
and the few counts marked "typed" in comments of the .tex files.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'figures'))
from common import CLEANED, DUPLICATED, METRICS, ROOT, RUNS, RUNS5, entry, read  # noqa: E402

OUT = HERE / 'generated'
OUT.mkdir(exist_ok=True)
SETS = ['polaris_test', 'spica_test', 'nebula_test', 'flickr8k_expert', 'composite', 'flickr8k_cf']
HEAD = {'polaris_test': 'Polaris', 'spica_test': 'Spica', 'nebula_test': 'Nebula',
        'flickr8k_expert': 'F8k-Exp.', 'composite': 'Compos.', 'flickr8k_cf': 'F8k-CF'}
LONG = {'polaris_test': 'Polaris test', 'polaris_val': 'Polaris val', 'spica_test': 'Spica test',
        'spica_val': 'Spica val', 'nebula_test': 'Nebula test', 'nebula_valid': 'Nebula val',
        'flickr8k_expert': 'Flickr8k-Expert', 'composite': 'Composite', 'flickr8k_cf': 'Flickr8k-CF',
        'pascal50s': 'PASCAL-50S', 'foil_1ref': 'FOIL'}
POOLS = {'polaris_train': 'Polaris train', 'nebula_train': 'Nebula train', 'spica_train': 'Spica train'}
GROUPS = [('duplicated', 'duplicated'), ('image_only', 'image only'), ('unseen', 'unseen image')]
# Results printed in the papers of the metrics, for the reproduction table (tau_c; tau_b on Flickr8k-CF).
PUBLISHED = {
    'DENEB (retrained)': dict(composite=58.2, flickr8k_expert=56.8, flickr8k_cf=38.3, nebula_test=54.3),
    'Pearl (retrained)': dict(composite=60.4, flickr8k_expert=58.6, flickr8k_cf=38.6, nebula_test=55.4),
    'Rigel (official ckpt)': dict(composite=64.3, flickr8k_expert=59.9, flickr8k_cf=40.4, nebula_test=55.1),
    'Rigel (official ckpt) + refs': dict(composite=66.1, flickr8k_expert=59.0, flickr8k_cf=40.4, nebula_test=57.5),
    'Polos (official ckpt)': dict(composite=57.6, flickr8k_expert=56.4, flickr8k_cf=37.8, polaris_test=57.8),
}

ranking = read('pairwise_s12-13-14.json')['benchmarks']
near = read('near_duplicates.json')['pools']
pearl = read('leakage_pearl.json')['benchmarks']
deneb = read('leakage_deneb.json')['benchmarks']
sweep = read('epoch_sweep.json')
rigel = read('rigel_groups.json')['benchmarks']
nothink = read('nothink_check.json')
structure = read(RUNS5 / 'structure.json')
components = read(RUNS5 / 'components.json')
groups = read(RUNS5 / 'groups.json')

numbers = {}


def define(name, text):
    assert name.isalpha() and name not in numbers, name
    numbers[name] = text


def signed(value, digits=2):
    return f'{value:+.{digits}f}'.replace('-', '$-$').replace('+', '$+$')


def interval(cell, digits=2):
    return f"[{signed(cell['low'], digits)}, {signed(cell['high'], digits)}]"


def effect(cell, digits=2):
    return f"{signed(cell['mean'], digits)} {interval(cell, digits)}"


def name_of(metric):
    base = metric.removesuffix(DUPLICATED).removesuffix(CLEANED)
    text = METRICS[base][0].replace('+ref', '+\\,ref')
    if metric.endswith(DUPLICATED):
        text += '$^{\\dagger}$'
    if metric.endswith(CLEANED):
        text += '$^{\\ddagger}$'
    return text


def write(name, lines):
    (OUT / f'{name}.tex').write_text('\n'.join(lines) + '\n')
    print('wrote', name)


def tabular(spec, head, body):
    return [f'\\begin{{tabular}}{{{spec}}}', '\\toprule', *head, '\\midrule', *body, '\\bottomrule', '\\end{tabular}']


# ---------------------------------------------------------------------------------------------
# Table 1: the evaluation sets.
body = []
for name in SETS:
    parts = ranking[name]
    rows, above = parts['all']['items'], parts['above_floor']['items']
    human = parts.get('above_floor_judged')
    order = parts.get('order')
    cells = [LONG[name], f'{rows:,}', f'{100 * (1 - above / rows):.1f}',
             f"{human['human']:.1f}" if human else '--', f"{order['items']:,}" if order else '--']
    for pool in POOLS:
        share = 100 * near[pool][name]['pair'] / near[pool][name]['rows']
        cells.append('0' if not share else f'{share:.1f}' if share >= 0.05 else '$<$0.1')
    body.append(' & '.join(cells) + ' \\\\')
    if name == 'nebula_test':
        body.append('\\addlinespace')
write('tab_sets', tabular(
    'lrrrrrrr',
    ['& & Lowest & Human & Same-image & \\multicolumn{3}{c}{Duplicated samples (\\%) in} \\\\',
     '\\cmidrule(l){6-8}',
     'Evaluation set & Samples & score (\\%) & $\\tau$ & pairs & Polaris tr. & Nebula tr. & Spica tr. \\\\'],
    body))
for name in SETS:
    parts = ranking[name]
    define('Floor' + HEAD[name].replace('8', 'Eight').replace('-', '').replace('.', ''),
           f"{100 * (1 - parts['above_floor']['items'] / parts['all']['items']):.1f}")
for pool, short in (('spica_train', 'Spica'), ('nebula_train', 'Nebula'), ('polaris_train', 'Polaris')):
    for name, target in (('polaris_test', 'Polaris'), ('spica_test', 'Spica'), ('nebula_test', 'Nebula')):
        cell = near[pool][name]
        if cell['pair'] > 100:
            define(f'Dup{short}{target}', f"{100 * cell['pair'] / cell['rows']:.1f}")
            define(f'DupUpper{short}{target}', f"{100 * cell['pair_upper'] / cell['rows']:.1f}")
            define(f'DupRows{short}{target}', f"{cell['pair']:,}")
define('PolarisRows', f"{near['spica_train']['polaris_test']['rows']:,}")
define('PolarisPictureOnly', f"{100 * near['spica_train']['polaris_test']['picture_only'] / near['spica_train']['polaris_test']['rows']:.1f}")
define('CompositePictureSpica', f"{100 * near['spica_train']['composite']['picture_only'] / near['spica_train']['composite']['rows']:.1f}")
define('SpicaOwnDup', f"{100 * near['spica_train']['spica_test']['pair'] / near['spica_train']['spica_test']['rows']:.1f}")
define('SpicaOwnDupUpper', f"{100 * near['spica_train']['spica_test']['pair_upper'] / near['spica_train']['spica_test']['rows']:.1f}")
define('SpicaReencoded', str(near['spica_train']['spica_test']['absent_pictures_matching_a_pool_picture']['0.99']['pictures']))
define('SpicaReencodedRows', str(near['spica_train']['spica_test']['absent_pictures_matching_a_pool_picture']['0.99']['rows']))

# ---------------------------------------------------------------------------------------------
# Table 2: the re-ranking.  One row per metric, per set tau over all rows and over rows above the floor.
SHOWN = ['Rigel (official ckpt) + refs', 'Rigel (official ckpt)',
         'VLM judge (InternVL 8B) + refs', 'VLM judge (Qwen 8B) + refs', 'VLM judge + refs',
         'VLM judge (InternVL 8B)', 'VLM judge (Qwen 8B)', 'VLM judge',
         'Pearl (retrained)', 'DENEB (retrained)', 'Polos (retrained)',
         'ridge SigLIP, Polaris pool (ours)', 'ridge SigLIP, Spica pool (ours)', 'ridge CLIP, Polaris pool (ours)',
         'RefPAC-S++ (ViT-L/14)', 'RefCLIP-S', 'CLIP-S', 'CIDEr', 'BLEU-4']
BREAKS = {'VLM judge (InternVL 8B) + refs', 'Pearl (retrained)', 'ridge SigLIP, Polaris pool (ours)',
          'RefPAC-S++ (ViT-L/14)'}


def ranking_cell(part, metric):
    row = entry(part, metric)
    if row is not None and 'rank' in row:
        text = f"{row['score']:.1f}"
        return f'\\textbf{{{text}}}' if metric in part['top_group'] else text
    row = entry(part, metric + DUPLICATED)
    return f"({row['score']:.1f})" if row else '--'


body = []
for metric in SHOWN:
    if metric in BREAKS:
        body.append('\\addlinespace')
    cells = [name_of(metric)]
    for name in SETS:
        cells += [ranking_cell(ranking[name][part], metric) for part in ('all', 'above_floor')]
    body.append(' & '.join(cells) + ' \\\\')
human = ['Human, leave one out']
for name in SETS:
    judged = ranking[name].get('above_floor_judged')
    human += ['', f"{judged['human']:.1f}" if judged else '--']
body += ['\\midrule', ' & '.join(human) + ' \\\\']
write('tab_ranking', tabular(
    'l' + 'r@{\\hspace{3pt}}r' * 6,
    [' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{HEAD[name]}}}' for name in SETS) + ' \\\\',
     ' '.join(f'\\cmidrule(l{{2pt}}r{{2pt}}){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(6)),
     'Metric' + ' & all & above' * 6 + ' \\\\'],
    body))

# ---------------------------------------------------------------------------------------------
# Table 3: how many pairs of metrics each set separates.


def top_separated(part, count=8):
    top = [row['metric'] for row in sorted(part['rows'], key=lambda row: row['rank'])[:count]]
    pairs = {frozenset((pair['a'], pair['b'])): pair for pair in part['pairs']}
    return sum(pairs[frozenset(couple)]['p_holm'] < 0.05 for couple in itertools.combinations(top, 2))


body = []
for name in SETS:
    cells = [LONG[name], str(len(ranking[name]['all']['rows']))]
    for key in ('all', 'above_floor', 'order'):
        part = ranking[name].get(key)
        cells += ([f"{100 * part['separated_holm'] / part['pair_count']:.0f}", str(top_separated(part)),
                   str(len(part['top_group']))] if part else ['--', '--', '--'])
        if part:
            tag = HEAD[name].replace('8', 'Eight').replace('-', '').replace('.', '') + {'all': 'All', 'above_floor': 'Above', 'order': 'Order'}[key]
            define('TopSep' + tag, str(top_separated(part)))
            define('TopGroup' + tag, str(len(part['top_group'])))
    body.append(' & '.join(cells) + ' \\\\')
    if name == 'nebula_test':
        body.append('\\addlinespace')
write('tab_power', tabular(
    'lr' + 'rrr' * 3,
    ['& & \\multicolumn{3}{c}{$\\tau$, all samples} & \\multicolumn{3}{c}{$\\tau$, above lowest} & \\multicolumn{3}{c}{Order accuracy} \\\\',
     '\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(l){9-11}',
     'Evaluation set & $n$ & \\% & top 8 & tied & \\% & top 8 & tied & \\% & top 8 & tied \\\\'],
    body))

# ---------------------------------------------------------------------------------------------
# Numbers for the running text: Pearl and DENEB retraining.
polaris = pearl['polaris_test']
for key, tag in (('released-pairclean', 'Rel'), ('matcheddrop-pairclean', 'Match'), ('randomdrop-pairclean', 'Rand')):
    for group, label in (('all', 'All'), ('pair', 'Pair'), ('picture_only', 'Picture'), ('absent', 'Absent'),
                         ('pair-absent', 'PairAbsent')):
        define(f'Pearl{tag}{label}', signed(polaris[group][key]['mean']))
        define(f'Pearl{tag}{label}CI', interval(polaris[group][key]))
define('PearlRelMatchPair', effect(polaris['pair']['released-matcheddrop']))
for name, tag in (('composite', 'Composite'), ('flickr8k_expert', 'Expert'), ('nebula_test', 'Nebula'), ('spica_test', 'Spica')):
    for key, label in (('released-pairclean', 'Rel'), ('matcheddrop-pairclean', 'Match'), ('released-matcheddrop', 'RelMatch')):
        define(f'Pearl{tag}{label}', effect(pearl[name]['all'][key]))
define('PearlSpicaAbsentRel', effect(pearl['spica_test']['absent']['released-pairclean']))
define('PearlSpicaAbsentRelMatch', effect(pearl['spica_test']['absent']['released-matcheddrop']))
for group, label in (('all', 'All'), ('pair', 'Pair'), ('picture_only', 'Picture'), ('absent', 'Absent'), ('pair-absent', 'PairAbsent')):
    define(f'Deneb{label}', signed(deneb['polaris_test'][group]['released-pairclean']['mean']))
    define(f'Deneb{label}CI', interval(deneb['polaris_test'][group]['released-pairclean']))
others = [deneb[name]['all']['released-pairclean']['mean'] for name in ('nebula_test', 'spica_test', 'composite', 'flickr8k_expert')]
define('DenebOtherLow', signed(min(others), 1))
define('DenebOtherHigh', signed(max(others), 1))
define('PearlPolarisReleased', f"{entry(ranking['polaris_test']['all'], 'Pearl (retrained)' + DUPLICATED)['score']:.2f}")
define('PearlPolarisCleaned', f"{entry(ranking['polaris_test']['all'], 'Pearl (retrained)')['score']:.2f}")
define('PearlPolarisRank', str(entry(ranking['polaris_test']['all'], 'Pearl (retrained)')['rank']))
held = entry(ranking['polaris_test']['all'], 'Pearl (retrained)' + DUPLICATED)
define('PearlDupFirst', f"{100 * held['first']:.1f}")
define('PearlDupOverBest', effect(held['minus_best_clean']))
define('PolarisRanked', str(len(ranking['polaris_test']['all']['rows'])))
define('CompositeRanked', str(len(ranking['composite']['all']['rows'])))

# Epoch sweep.
runs = sweep['runs']
columns = {'pair': 'polaris_test/pair', 'rest': 'polaris_test/not pair', 'composite': 'composite/all',
           'expert': 'flickr8k_expert/all'}
variants = ['released', 'randomdrop', 'matcheddrop', 'pairclean']
last = max(len(run['epochs']) for run in runs.values())


def epoch_mean(names, epoch, column):
    values = [runs[n]['epochs'][str(epoch)]['tau'][column] for n in names if str(epoch) in runs[n]['epochs']]
    return (float(np.mean(values)), len(values)) if values else (None, 0)


holding = [n for n in runs if not n.startswith('pairclean')]
cleaned = [n for n in runs if n.startswith('pairclean')]
words = ['One', 'Two', 'Three', 'Four', 'Five']
gaps = []
for epoch in range(last):
    high, low = epoch_mean(holding, epoch, columns['pair'])[0], epoch_mean(cleaned, epoch, columns['pair'])[0]
    gaps.append(high - low)
    define('EpochGap' + words[epoch], f'{high - low:.1f}')
    define('EpochHeld' + words[epoch], f'{high:.1f}')
    define('EpochClean' + words[epoch], f'{low:.1f}')
rest = [run['epochs'][e]['tau'][columns['rest']] for run in runs.values() for e in run['epochs']]
define('EpochRestLow', f'{min(rest):.1f}')
define('EpochRestHigh', f'{max(rest):.1f}')
define('EpochCheckpoints', str(len(rest)))
define('EpochRuns', str(len(runs)))
spread = [max(e['validation'] for e in run['epochs'].values()) - min(e['validation'] for e in run['epochs'].values())
          for run in runs.values()]
define('ValidationSpread', f'{100 * max(spread):.1f}')
expert = [max(e['tau'][columns['expert']] for e in run['epochs'].values()) - min(e['tau'][columns['expert']] for e in run['epochs'].values())
          for run in runs.values()]
define('ExpertSpreadLow', f'{min(expert):.1f}')
define('ExpertSpreadHigh', f'{max(expert):.1f}')
define('ExpertSpreadMedian', f'{float(np.median(expert)):.1f}')
# The general cost of deduplication: released minus deduplicated where nothing is duplicated, and
# the same gap (trainings holding the duplicates minus deduplicated) on Composite after each epoch.
general = [pearl[name]['all']['released-pairclean']['mean'] for name in ('composite', 'flickr8k_expert', 'nebula_test', 'spica_test')]
define('PearlOtherLow', signed(min(general), 1))
define('PearlOtherHigh', signed(max(general), 1))
flat = [epoch_mean(holding, epoch, columns['composite'])[0] - epoch_mean(cleaned, epoch, columns['composite'])[0]
        for epoch in range(last)]
define('CompositeEpochGapLow', f'{min(flat):.1f}')
define('CompositeEpochGapHigh', f'{max(flat):.1f}')

body = []
for label, column in (('Polaris test, duplicated samples', 'pair'), ('Polaris test, all other samples', 'rest'),
                      ('Composite, all samples', 'composite'), ('Flickr8k-Expert, all samples', 'expert')):
    body.append(f'\\multicolumn{{6}}{{l}}{{\\emph{{{label}}}}} \\\\')
    for variant in variants:
        names = [n for n in runs if n.startswith(variant)]
        cells = [{'released': 'Released', 'randomdrop': 'Random removal', 'matcheddrop': 'Matched removal', 'pairclean': 'Deduplicated'}[variant]]
        for epoch in range(last):
            mean, count = epoch_mean(names, epoch, columns[column])
            cells.append(f'{mean:.2f}\\,({count})' if count else '--')
        body.append('\\quad ' + ' & '.join(cells) + ' \\\\')
    if column != 'expert':
        body.append('\\addlinespace')
write('tab_epochs', tabular('lrrrrr', ['Training set & Epoch 1 & Epoch 2 & Epoch 3 & Epoch 4 & Epoch 5 \\\\'], body))

# ---------------------------------------------------------------------------------------------
# Numbers: judges, Rigel, humans, learned metrics among themselves.
JUDGES = ['VLM judge (InternVL 8B) + refs', 'VLM judge (Qwen 8B) + refs', 'VLM judge + refs']
for name, tag in (('polaris_test', 'Polaris'), ('spica_test', 'Spica'), ('flickr8k_expert', 'Expert')):
    judged = ranking[name]['above_floor_judged']
    define(f'Human{tag}', f"{judged['human']:.1f}")
    rows = {row['metric']: row for row in judged['rows']}
    ranked = [row for row in judged['rows'] if not row['duplicated']]
    define(f'Below{tag}', str(sum(row['high'] < 0 for row in ranked)))
    define(f'Above{tag}', str(sum(row['low'] > 0 for row in ranked)))
    define(f'Judged{tag}', str(len(ranked)))
    gains = [rows[j]['mean'] for j in JUDGES]
    define(f'JudgeHuman{tag}Low', signed(min(gains), 1))
    define(f'JudgeHuman{tag}High', signed(max(gains), 1))
best = max((row for row in ranking['flickr8k_expert']['above_floor_judged']['rows'] if not row['duplicated']), key=lambda row: row['mean'])
define('ExpertBestHuman', f"{name_of(best['metric'])}, {effect(best)}")


def pair_difference(part, a, b):
    for pair in part['pairs']:
        if {pair['a'], pair['b']} == {a, b}:
            sign = 1 if pair['a'] == a else -1
            cell = {'mean': sign * pair['mean'], 'low': min(sign * pair['low'], sign * pair['high']),
                    'high': max(sign * pair['low'], sign * pair['high'])}
            return cell, pair['p_holm'] < 0.05
    raise KeyError((a, b))


for name, tag in (('flickr8k_expert', 'Expert'), ('composite', 'Composite'), ('flickr8k_cf', 'CF')):
    part = ranking[name]['all']
    for metric, label in (('Rigel (official ckpt)', 'Rigel'), ('Rigel (official ckpt) + refs', 'RigelRef')):
        row = entry(part, metric)
        define(f'{label}{tag}', f"{row['score']:.1f}")
        define(f'{label}{tag}Rank', str(row['rank']))
cell, clear = pair_difference(ranking['composite']['all'], 'Rigel (official ckpt) + refs', 'VLM judge (Qwen 8B) + refs')
define('RigelCompositeOverJudge', effect(cell))
cell, clear = pair_difference(ranking['flickr8k_expert']['all'], 'Rigel (official ckpt)', 'VLM judge (InternVL 8B)')
define('RigelExpertOverJudge', effect(cell, 1))
for name, tag in (('polaris_test', 'Polaris'), ('spica_test', 'Spica'), ('nebula_test', 'Nebula')):
    define(f'RigelAll{tag}', effect(rigel[name]['all']['Rigel + refs - judge + refs']))
    define(f'RigelRest{tag}', effect(rigel[name]['not_pair']['Rigel + refs - judge + refs']))
    define(f'RigelRefScore{tag}', f"{rigel[name]['all']['Rigel + refs']['mean']:.1f}")
    best = max(ranking[name]['all']['rows'], key=lambda row: row['score'])
    define(f'BestRanked{tag}', f"{best['score']:.1f}")
define('RigelPairAbsent', effect(rigel['polaris_test']['pair-absent']['Rigel + refs - judge + refs']))
define('RigelYardstickPearl', effect(rigel['polaris_test']['pair-absent']['Pearl +dup - judge + refs']))
define('RigelFreePair', effect(rigel['polaris_test']['pair']['Rigel - judge']))

LEARNED = ['Polos (retrained)', 'DENEB (retrained)', 'Pearl (retrained)']
body = []
for name in SETS:
    for key, label in (('all', 'all'), ('above_floor', 'above lowest')):
        part = ranking[name][key]
        cells = [LONG[name] if key == 'all' else '', label]
        for a, b in itertools.combinations(LEARNED, 2):
            cell, clear = pair_difference(part, a, b)
            mark = '$^{*}$' if clear else ''
            cells.append(f"{signed(cell['mean'])}{mark}")
        body.append(' & '.join(cells) + ' \\\\')
write('tab_learned', tabular('llrrr', ['Evaluation set & Samples & Polos $-$ DENEB & Polos $-$ Pearl & DENEB $-$ Pearl \\\\'], body))
for sets, tag in ((SETS[:3], 'Family'), (SETS[3:], 'Earlier')):
    define(f'LearnedClear{tag}', str(sum(pair_difference(ranking[name]['all'], a, b)[1]
                                         for name in sets for a, b in itertools.combinations(LEARNED, 2))))
RIDGE = 'ridge SigLIP, Polaris pool (ours)'
verdicts = [pair_difference(ranking[name]['all'], RIDGE, metric) for name in SETS for metric in LEARNED]
define('RidgeComparisons', str(len(verdicts)))
define('RidgeAbove', str(sum(clear and cell['mean'] > 0 for cell, clear in verdicts)))
define('RidgeBelow', str(sum(clear and cell['mean'] < 0 for cell, clear in verdicts)))
recent = {entry(ranking[name]['all'], RIDGE)['rank'] for name in SETS[:3]}
assert len(recent) == 1, recent                      # the text says "on all three"
assert all(pair_difference(ranking[name]['all'], RIDGE, metric)[0]['mean'] > 0 for name in SETS[:3] for metric in LEARNED)
define('RidgeRecentRank', str(recent.pop()))
gaps = [pair_difference(ranking[name]['all'], RIDGE, metric)[0]['mean'] for name in SETS[:3] for metric in LEARNED]
define('RidgeGapLow', f'{min(gaps):.1f}')
define('RidgeGapHigh', f'{max(gaps):.1f}')
# The same ridge regression on the CLIP features that Polos reads, trained on the same Polaris training set.
RIDGE_CLIP, POLOS = 'ridge CLIP, Polaris pool (ours)', 'Polos (retrained)'
clip_recent = [pair_difference(ranking[name]['all'], RIDGE_CLIP, POLOS) for name in SETS[:3]]
assert not any(clear for _, clear in clip_recent)    # the text says "not separated on the recent test sets"
define('RidgeClipPolosLow', signed(min(cell['mean'] for cell, _ in clip_recent), 1))
define('RidgeClipPolosHigh', signed(max(cell['mean'] for cell, _ in clip_recent), 1))
clip_earlier = {name: pair_difference(ranking[name]['all'], RIDGE_CLIP, POLOS) for name in SETS[3:]}
assert [clear and cell['mean'] > 0 for cell, clear in clip_earlier.values()] == [True, False, True], clip_earlier
assert clip_earlier['composite'][1] and clip_earlier['composite'][0]['mean'] < 0   # above on both Flickr8k sets, below on Composite
siglip_over_clip = [pair_difference(ranking[name]['all'], RIDGE, RIDGE_CLIP)[0]['mean'] for name in SETS[:3]]
define('RidgeEncoderGapLow', f'{min(siglip_over_clip):.1f}')
define('RidgeEncoderGapHigh', f'{max(siglip_over_clip):.1f}')
learned_ranks = [entry(ranking[name]['all'], metric)['rank'] for name in SETS[:3] for metric in LEARNED]
define('LearnedRankBest', str(min(learned_ranks)))
define('LearnedRankWorst', str(max(learned_ranks)))
for name, tag in (('polaris_test', 'Polaris'), ('spica_test', 'Spica')):
    differences = [abs(pair_difference(ranking[name]['all'], a, b)[0]['mean']) for a, b in itertools.combinations(LEARNED, 2)]
    define(f'LearnedMaxGap{tag}', f'{max(differences):.1f}')
    define(f'LearnedClear{tag}', str(sum(pair_difference(ranking[name]['all'], a, b)[1] for a, b in itertools.combinations(LEARNED, 2))))
for name, tag in (('polaris_test', 'Polaris'), ('spica_test', 'Spica'), ('nebula_test', 'Nebula')):
    part = ranking[name]['above_floor']
    for metric, label in (('CLIP-S', 'Clip'), ('BLEU-1', 'Bleu')):
        define(f'{label}Above{tag}', f"{entry(part, metric)['score']:.1f}")
ridge = entry(ranking['polaris_test']['all'], 'ridge SigLIP, Polaris pool (ours)')
define('RidgePolarisRank', str(ridge['rank']))
define('RidgePolarisScore', f"{ridge['score']:.1f}")
scores = [row['score'] for row in ranking['polaris_test']['above_floor']['rows']]
define('AboveFloorPolarisBest', f'{max(scores):.1f}')
define('AboveFloorPolarisWorst', f'{min(scores):.1f}')
for name, tag in (('polaris_test', 'Polaris'), ('spica_test', 'Spica'), ('composite', 'Composite')):
    define(f'RankAgreement{tag}', f"{ranking[name]['rank_agreement']['above_floor']:.2f}")
for name, tag in (('polaris_test', 'Polaris'), ('spica_test', 'Spica'), ('flickr8k_expert', 'Expert')):
    define(f'OrderPairs{tag}', f"{ranking[name]['order']['items']:,}")
define('ExpertJudgeRankLow', str(min(entry(ranking['flickr8k_expert']['all'], j)['rank'] for j in JUDGES + [j.removesuffix(' + refs') for j in JUDGES])))
define('ExpertJudgeRankHigh', str(max(entry(ranking['flickr8k_expert']['all'], j)['rank'] for j in JUDGES + [j.removesuffix(' + refs') for j in JUDGES])))

# ---------------------------------------------------------------------------------------------
# Appendix: full duplication matrix.
body = []
for pool, title in POOLS.items():
    body.append(f'\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}} \\\\')
    for name, label in LONG.items():
        cell = near[pool][name]
        def share(count):
            value = f"{100 * count / cell['rows']:.1f}"
            return '$<$0.1' if count and value == '0.0' else value
        body.append(f"\\quad {label} & {cell['rows']:,} & {cell['pair']:,} & {share(cell['pair'])} & "
                    f"{share(cell['pair_upper'])} & {share(cell['picture_only'])} \\\\")
    if pool != 'spica_train':
        body.append('\\addlinespace')
write('tab_matrix', tabular('lrrrrr', ['& & \\multicolumn{3}{c}{Duplicated samples} & Image \\\\', '\\cmidrule(lr){3-5}',
                                      'Training set / evaluation set & Samples & exact & \\% & \\% upper & only \\% \\\\'], body))

# Cross-fit effects, one model against one model (runs/summary/groups.json).
LEARNERS = [('ridge_siglip', 'Ridge regression, SigLIP features'), ('ridge_clip', 'Ridge regression, CLIP features'),
            ('mlp_siglip', 'Three-layer network, SigLIP features'), ('polos_arch', 'Polos architecture, CLIP features')]
CROSS = [('spica_train', 'polaris_test'), ('nebula_train', 'polaris_test'), ('polaris_train', 'spica_test'),
         ('polaris_train', 'nebula_test')]
crossfit = groups['crossfit']


def short(pool, name):
    return f"{POOLS[pool].replace(' train', '')} $\\rightarrow$ {LONG[name].replace(' test', '')}"


body, full, placebo = [], [], []
for learner, title in LEARNERS:
    main = learner != 'ridge_clip'            # the CLIP ridge rows are in the appendix table only
    if main:
        body.append(f'\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\')
    full.append(f'\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\')
    for pool, name in CROSS:
        cell = crossfit.get(f'{learner}|{pool}>{name}')
        if not cell:
            continue
        part = cell['duplicated/all']
        mark = '$^{*}$' if not learner.startswith('ridge') and part['models'] < 3 else ''
        if main:
            body.append(f"\\quad {short(pool, name)}{mark} & {part['unseen']:.1f} & {part['seen']:.1f} & "
                        f"{signed(part['effect'], 1)} [{signed(part['low'], 1)}, {signed(part['high'], 1)}] & "
                        f"{signed(part['arm_min'], 1)} to {signed(part['arm_max'], 1)} \\\\")
        cells = [f'\\quad {short(pool, name)}{mark}']
        for key in ('duplicated/all', 'duplicated/above', 'image_only/all', 'image_only/above', 'unseen/all', 'unseen/above'):
            item = cell.get(key)
            cells.append(f"{signed(item['effect'], 1)} [{signed(item['low'], 1)}, {signed(item['high'], 1)}]" if item else '--')
        full.append(' & '.join(cells) + ' \\\\')
        placebo.append(abs(cell['unseen/all']['effect']))
write('tab_crossfit', tabular('@{}lrrrr@{}', ['Learner, training $\\rightarrow$ test & Unseen & Seen & $E$ [95\\% CI] & Single models \\\\'], body))
write('tab_crossfit_full', tabular(
    'lrrrrrr',
    ['& \\multicolumn{2}{c}{Duplicated} & \\multicolumn{2}{c}{Image only} & \\multicolumn{2}{c}{Unseen image} \\\\',
     '\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(l){6-7}',
     'Learner, training $\\rightarrow$ test set & all & above & all & above & all & above \\\\'], full))
for learner, tag in (('ridge_siglip', 'Ridge'), ('mlp_siglip', 'Mlp'), ('polos_arch', 'Polosarch')):
    values = [cell['duplicated/all']['effect'] for key, cell in crossfit.items() if key.startswith(learner + '|')]
    define(f'Crossfit{tag}Low', signed(min(values), 1))
    define(f'Crossfit{tag}High', signed(max(values), 1))
define('CrossfitPlaceboMax', f'{max(placebo):.1f}')
define('MlpSpicaPolaris', f"{crossfit['mlp_siglip|spica_train>polaris_test']['duplicated/all']['effect']:.1f}")
define('MlpNebulaPolaris', f"{crossfit['mlp_siglip|nebula_train>polaris_test']['duplicated/all']['effect']:.1f}")
seeds = crossfit['mlp_siglip|nebula_train>polaris_test']['duplicated/all']['models']
numbers['MlpNebulaNote'] = '' if seeds == 3 else (' In this draft the three-layer network on the Nebula training set is from '
                                                  f'{seeds} seed{"s" if seeds > 1 else ""}; the others are means of three.')
define('MlpNebulaSeeds', str(crossfit['mlp_siglip|nebula_train>polaris_test']['duplicated/all']['models']))
define('RidgeSpicaPolaris', f"{crossfit['ridge_siglip|spica_train>polaris_test']['duplicated/all']['effect']:.1f}")
define('RidgeNebulaPolaris', f"{crossfit['ridge_siglip|nebula_train>polaris_test']['duplicated/all']['effect']:.1f}")

# Appendix: retraining contrasts.
body = []
for title, data, keys in (('Pearl, Spica training set', pearl, ['released-pairclean', 'matcheddrop-pairclean', 'released-matcheddrop']),
                          ('DENEB, Nebula training set', deneb, ['released-pairclean', 'randomdrop-pairclean', 'released-randomdrop'])):
    heads = {'released-pairclean': 'released $-$ dedup.', 'matcheddrop-pairclean': 'matched $-$ dedup.',
             'released-matcheddrop': 'released $-$ matched', 'randomdrop-pairclean': 'random $-$ dedup.',
             'released-randomdrop': 'released $-$ random'}
    body.append(f"\\multicolumn{{2}}{{l}}{{\\emph{{{title}}}}} & " + ' & '.join(heads[k] for k in keys) + ' \\\\')
    for group, label in (('all', 'Polaris test, all samples'), ('pair', '\\quad duplicated'),
                         ('picture_only', '\\quad image only'), ('absent', '\\quad unseen image'),
                         ('pair-absent', '\\quad duplicated $-$ unseen')):
        cell = data['polaris_test'][group]
        body.append(f"{label} & {cell.get('items', ''):,} & ".replace(' & , &', ' & &') if 'items' in cell else f'{label} & & ')
        body[-1] += ' & '.join(effect(cell[k]) for k in keys) + ' \\\\'
    for name in ('spica_test', 'nebula_test', 'composite', 'flickr8k_expert'):
        cell = data[name]['all']
        body.append(f"{LONG[name]}, all samples & {cell['items']:,} & " + ' & '.join(effect(cell[k]) for k in keys) + ' \\\\')
    body.append('\\addlinespace')
write('tab_retrain', tabular('lrrrr', ['Samples & $n$ & \\multicolumn{3}{c}{Difference in $\\tau$ [95\\% CI], mean of three seeds} \\\\'], body[:-1]))

# Appendix: the full ranking, three quantities.


def full_table(key, sets, human_key=None):
    names = []
    for name in sets:
        part = ranking[name][key]
        for row in part['rows'] + part['duplicated']:
            if row['metric'] not in names:
                names.append(row['metric'])
    first = ranking[sets[0]][key]

    def position(metric):
        row = entry(first, metric)
        kind = metric.endswith(DUPLICATED) + 2 * metric.endswith(CLEANED)
        return (kind, row['rank'] if row and 'rank' in row else 99, names.index(metric))

    body = []
    previous = 0
    for metric in sorted(names, key=position):
        kind = position(metric)[0]
        if kind != previous:
            body.append('\\addlinespace')
            previous = kind
        cells = [name_of(metric)]
        for name in sets:
            row = entry(ranking[name][key], metric)
            if row is None:
                cells.append('--')
            elif 'rank' in row:
                cells.append(f"{row['score']:.2f}\\enspace{row['rank']}\\,[{row['rank_low']},\\,{row['rank_high']}]")
            else:
                cells.append(f"{row['score']:.2f}")
        body.append(' & '.join(cells) + ' \\\\')
    if human_key:
        body += ['\\midrule', 'Human reference & ' + ' & '.join(f"{ranking[name][human_key]['human']:.2f}" for name in sets) + ' \\\\']
    return tabular('l' + 'r' * len(sets), ['Metric & ' + ' & '.join(LONG[name] for name in sets) + ' \\\\'], body)


write('tab_full_all', full_table('all', SETS))
write('tab_full_above', full_table('above_floor', SETS))
write('tab_full_order', full_table('order', [name for name in SETS if 'order' in ranking[name]], 'order_human'))

# Appendix: entries that are not ranked.
body = []
for name in SETS:
    first = True
    for row in ranking[name]['all']['duplicated']:
        own = row.get('minus_ranked_version')
        body.append(' & '.join([LONG[name] if first else '', name_of(row['metric']), f"{row['score']:.2f}",
                                effect(row['minus_best_clean']), f"{100 * row['first']:.1f}",
                                effect(own) if own else '--']) + ' \\\\')
        first = False
    body.append('\\addlinespace')
write('tab_unranked', tabular('llrrrr', ['Set & Entry & $\\tau$ & $-$ best ranked & first (\\%) & $-$ own ranked version \\\\'], body[:-1]))
define('PolosSpicaDupFirst', f"{100 * entry(ranking['spica_test']['all'], 'Polos (retrained)' + DUPLICATED)['first']:.0f}")
define('PolosSpicaDupGain', effect(entry(ranking['spica_test']['all'], 'Polos (retrained)' + DUPLICATED)['minus_ranked_version']))
define('PolosNebulaDupGain', effect(entry(ranking['nebula_test']['all'], 'Polos (retrained)' + DUPLICATED)['minus_ranked_version']))

# Appendix: against the human reference, rows above the floor.
body = []
for name in ('polaris_test', 'spica_test', 'flickr8k_expert'):
    judged = ranking[name]['above_floor_judged']
    ranked = sorted((row for row in judged['rows'] if not row['duplicated']), key=lambda row: -row['mean'])
    body.append(f"\\multicolumn{{3}}{{l}}{{\\emph{{{LONG[name]}}} ({judged['items']:,} samples; human {judged['human']:.2f})}} \\\\")
    for row in ranked[:6]:
        body.append(f"\\quad {name_of(row['metric'])} & {row['score']:.2f} & {effect(row)} \\\\")
    body.append('\\addlinespace')
write('tab_human', tabular('lrr', ['Metric & $\\tau$ & Metric $-$ human [95\\% CI] \\\\'], body[:-1]))

# Appendix: Rigel by kind of row.
body = []
for name in ('polaris_test', 'spica_test', 'nebula_test'):
    cells = [LONG[name]]
    for group in ('all', 'pair', 'picture_only', 'absent'):
        cell = rigel[name].get(group)
        cells.append(effect(cell['Rigel + refs - judge + refs'], 1) if cell and cell['items'] >= 1000 else '--')
    body.append(' & '.join(cells) + ' \\\\')
body.append('\\addlinespace')
cells = ['Pearl$^{\\dagger}$, Polaris test']
for group in ('all', 'pair', 'picture_only', 'absent'):
    cells.append(effect(rigel['polaris_test'][group]['Pearl +dup - judge + refs'], 1))
body.append(' & '.join(cells) + ' \\\\')
write('tab_rigel', tabular('lrrrr', ['Rigel\\,+\\,ref $-$ judge\\,+\\,ref & All samples & Duplicated & Image only & Unseen image \\\\'], body))

# Appendix: the two ways of reading a score from InternVL.
body = []
for name in SETS:
    cells = [LONG[name]]
    for mode in ('refs', 'free'):
        cell = nothink[f'{mode}/{name}']
        # tau is read from the digit probabilities renormalised after a possible <think> token
        # ("conditional"); tau_check prefills an empty thinking block first ("prefilled").
        cells += [f"{cell['tau']:.2f}", f"{cell['tau_check']:.2f}", signed(cell['difference'])]
    body.append(' & '.join(cells) + ' \\\\')
write('tab_nothink', tabular('lrrrrrr', ['& \\multicolumn{3}{c}{With references} & \\multicolumn{3}{c}{Without references} \\\\',
                                        '\\cmidrule(lr){2-4} \\cmidrule(l){5-7}',
                                        'Evaluation set & conditional & prefilled & diff. & conditional & prefilled & diff. \\\\'], body))
expert = [abs(nothink[f'{mode}/flickr8k_expert']['difference']) for mode in ('refs', 'free')]
rest = [abs(nothink[f'{mode}/{name}']['difference']) for mode in ('refs', 'free') for name in SETS if name != 'flickr8k_expert']
define('NothinkExpertLow', f'{min(expert):.1f}')
define('NothinkExpertHigh', f'{max(expert):.1f}')
define('NothinkRestMax', f'{max(rest):.2f}')

# Appendix: reproduction of published numbers.
body = []
for metric, published in PUBLISHED.items():
    cells = [name_of(metric)]
    for name in ('flickr8k_expert', 'flickr8k_cf', 'composite', 'nebula_test', 'polaris_test'):
        row = entry(ranking[name]['all'], metric) or entry(ranking[name]['all'], metric + DUPLICATED)
        if metric == 'Pearl (retrained)' and name == 'nebula_test':
            row = entry(ranking[name]['all'], metric + DUPLICATED)
        cells.append(f"{row['score']:.1f} / {published[name]:.1f}" if name in published and row else '--')
    body.append(' & '.join(cells) + ' \\\\')
write('tab_repro', tabular('lrrrrr', ['Metric & Flickr8k-Expert & Flickr8k-CF & Composite & Nebula test & Polaris test \\\\'], body))

# Numbers of the provenance audit (analysis/audit).
AUDIT = ROOT / 'analysis' / 'audit'
same = json.loads((AUDIT / 'eval_leakage.json').read_text())['polaris_test']
define('SameScoreSpicaPolaris', f"{100 * same['spica_train']['same_score_row_share']:.1f}")
define('SameScoreNebulaPolaris', f"{100 * same['nebula_train']['same_score_row_share']:.1f}")
release = json.loads((AUDIT / 'csv_audit.json').read_text())['composition']
define('SpicaRows', f"{release['total_rows']:,}")
define('SpicaIds', f"{release['total_images']:,}")
define('SpicaNoRefs', f"{release['rows_without_references']:,}")
pictures = {item['pixel_sha1'] for item in json.loads((AUDIT / 'inventory_spica.json').read_text())}
define('SpicaPictures', f'{len(pictures):,}')
shared = json.loads((AUDIT / 'overlap.json').read_text())['cross_dataset']['spica_train_vs_spica_test']
define('SpicaTestInTrain', f"{shared['shared_images']:,}")
define('SpicaTestPictures', f"{shared['images_b']:,}")
define('SpicaTestInTrainShare', f"{100 * shared['share_of_b']:.1f}")

# ---------------------------------------------------------------------------------------------
# How each set stores judgments (runs/summary/structure.json).
STRUCTURE = [('polaris_train', 'Polaris train', 'rating'), ('polaris_test', 'Polaris test', 'rating'),
             ('nebula_train', 'Nebula train', 'pair'), ('nebula_test', 'Nebula test', 'pair'),
             ('spica_train', 'Spica train', 'rating'), ('spica_test', 'Spica test', 'rating'),
             ('flickr8k_expert', 'Flickr8k-Expert', 'rating'), ('flickr8k_cf', 'Flickr8k-CF', 'pair'),
             ('composite', 'Composite', 'pair')]
body = []
for name, label, unit in STRUCTURE:
    cell = structure[name]
    body.append(f"{label} & {unit} & {cell['images']:,} & {cell['pairs']:,} & {cell['ratings']:,} & "
                f"{cell['ratings_per_pair']:.1f} & {100 * cell['lowest_share_ratings']:.0f} \\\\")
    if 'deduplicated' in cell:
        kept = cell['deduplicated']
        body.append(f"\\quad deduplicated & & {kept['images']:,} & {kept['pairs']:,} & {kept['ratings']:,} & "
                    f"{kept['ratings_per_pair']:.1f} & {100 * kept['lowest_share_ratings']:.0f} \\\\")
    if name in ('spica_test',):
        body.append('\\addlinespace')
write('tab_structure', tabular('@{}llrrrrr@{}', ['& A sample & & & & Ratings & Lowest \\\\',
                                               'Set & is one & Images & Pairs & Samples & per pair & (\\%) \\\\'], body))
for name, tag in (('spica_train', 'Spica'), ('nebula_train', 'Nebula'), ('polaris_train', 'Polaris')):
    cell = structure[name]
    define(f'RemovedShare{tag}', f"{100 * (1 - cell['deduplicated']['ratings'] / cell['ratings']):.1f}")
common = structure['spica_test_vs_nebula_test']
define('SpicaNebulaCommonPairs', f"{common['common_pairs']:,}")
define('SpicaOnlyPairs', f"{common['spica_pairs_only']:,}")
define('SpicaOnlyLowest', f"{100 * common['spica_only_lowest']:.0f}")
define('SpicaCommonRatings', f"{common['spica_ratings_on_common_pairs']:,}")

# Agreement of the two copies of a duplicated pair, and shares counted in pairs.
labels = groups['labels']
body = []
for key, tag in (('spica_train>polaris_test', 'SpicaPolaris'), ('nebula_train>polaris_test', 'NebulaPolaris'),
                 ('polaris_train>spica_test', 'PolarisSpica'), ('polaris_train>nebula_test', 'PolarisNebula')):
    cell = labels[key]
    pool, name = key.split('>')
    define(f'PairShare{tag}', f"{100 * cell['duplicated_pairs'] / cell['pairs']:.0f}")
    define(f'SameLabel{tag}', f"{100 * cell['same_mean_share']:.1f}")
    define(f'Copies{tag}', f"{cell['ratings_per_pair_training']:.1f}")
    body.append(f"{short(pool, name)} & {cell['duplicated_rows']:,} & {100 * cell['duplicated_rows'] / cell['rows']:.1f} & "
                f"{cell['duplicated_pairs']:,} & {100 * cell['duplicated_pairs'] / cell['pairs']:.1f} & "
                f"{cell['ratings_per_pair_training']:.1f} & {100 * cell['same_mean_share']:.1f} & {cell['correlation']:.3f} \\\\")
write('tab_labels', tabular(
    'lrrrrrrr',
    ['& \\multicolumn{2}{c}{Duplicated samples} & \\multicolumn{2}{c}{Duplicated pairs} & Training & Same & \\\\',
     '\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}',
     'Training $\\rightarrow$ test set & count & \\% & count & \\% & samples per pair & mean (\\%) & $r$ \\\\'], body))

# The three groups of Polaris test samples relative to the Spica training set.
description = groups['description']['spica_train>polaris_test']
retrain = groups['retrain']
body = []
for group, label in GROUPS:
    cell, above = description[f'{group}/all'], description[f'{group}/above']
    gain = retrain['pearl'][f'{group}/all']['released-pairclean']
    gain_above = retrain['pearl'][f'{group}/above']['released-pairclean']
    body.append(f"{label.capitalize()} & {cell['rows']:,} & {100 * cell['lowest_share']:.0f} & {100 * cell['single_rating_share']:.0f} & "
                f"{100 * cell['tied_pair_share']:.0f} & {signed(gain['mean'], 1)} & {signed(gain_above['mean'], 1)} \\\\")
    tag = {'duplicated': 'Dup', 'image_only': 'Image', 'unseen': 'Unseen'}[group]
    define(f'PearlAbove{tag}', f"{signed(gain_above['mean'], 1)} [{signed(gain_above['low'], 1)}, {signed(gain_above['high'], 1)}]")
    define(f'{tag}Lowest', f"{100 * cell['lowest_share']:.0f}")
define('PearlAboveImageRows', f"{description['image_only/above']['rows']:,}")
define('PearlAboveImageImages', f"{description['image_only/above']['images']:,}")
write('tab_groups', tabular('@{}lrrrrrr@{}', ['& & Lowest & Single & Tied & \\multicolumn{2}{c}{$\\Delta$ (Pearl)} \\\\',
                                            '\\cmidrule(l){6-7}',
                                            'Group & Samples & (\\%) & (\\%) & (\\%) & all & above \\\\'], body))

# Appendix: descriptives and contrasts of every group, all samples and above the lowest score.
body = []
for family, title, pool, keys in (('pearl', 'Pearl, Spica training set', 'spica_train', ['released-pairclean', 'matcheddrop-pairclean']),
                                  ('deneb', 'DENEB, Nebula training set', 'nebula_train', ['released-pairclean', 'randomdrop-pairclean'])):
    heads = {'released-pairclean': 'released $-$ dedup.', 'matcheddrop-pairclean': 'matched $-$ dedup.',
             'randomdrop-pairclean': 'random $-$ dedup.'}
    body.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}} & " + ' & '.join(heads[k] for k in keys) + ' \\\\')
    describe = groups['description'][f'{pool}>polaris_test']
    for group, label in GROUPS:
        for subset, note in (('all', ''), ('above', ', above lowest')):
            cell, gain = describe[f'{group}/{subset}'], retrain[family][f'{group}/{subset}']
            level = gain['level']['pairclean']
            body.append(f"\\quad {label}{note} & {cell['rows']:,} & {cell['images']:,} & {100 * cell['tied_pair_share']:.0f} & {level:.1f} & "
                        + ' & '.join(effect(gain[k]) for k in keys) + ' \\\\')
    for other, label in (('image_only', 'duplicated $-$ image only'), ('unseen', 'duplicated $-$ unseen image')):
        for subset, note in (('all', ''), ('above', ', above lowest')):
            gain = retrain[family][f'duplicated-{other}/{subset}']
            body.append(f"\\quad {label}{note} & & & & & " + ' & '.join(effect(gain[k]) for k in keys) + ' \\\\')
    body.append('\\addlinespace')
write('tab_groups_full', tabular('lrrrrrr', ['Group of Polaris test samples & Samples & Images & Tied (\\%) & $\\tau$ dedup. & '
                                            '\\multicolumn{2}{c}{Difference in $\\tau$ [95\\% CI]} \\\\'], body[:-1]))

# Detection and ranking terms of tau.
body = []
for name in SETS:
    part = components[name]
    ranked = [cell for cell in part['metrics'] if cell['ranked']]
    best = max(ranked, key=lambda cell: cell['tau_all'])
    modern = [cell for cell in ranked if METRICS.get(cell['metric'], ('', 'text'))[1] != 'text']
    body.append(f"{LONG[name]} & {100 * part['lowest_share']:.0f} & "
                + (f"{part['weight_detection']:.2f} & {part['weight_ranking']:.2f}" if part['variant'] == 'c' else '-- & --')
                + f" & {name_of(best['metric'])} & {best['tau_all']:.1f} & {100 * best['cross_share']:.0f} & {best['auc']:.3f} & "
                f"{min(c['auc'] for c in modern):.2f}--{max(c['auc'] for c in modern):.2f} & "
                f"{min(c['tau_above'] for c in modern):.0f}--{max(c['tau_above'] for c in modern):.0f} \\\\")
    tag = HEAD[name].replace('8', 'Eight').replace('-', '').replace('.', '')
    define(f'DetectionShare{tag}', f"{100 * best['cross_share']:.0f}")
write('tab_components', tabular(
    'lrrrlrrrrr',
    ['& Lowest & & & \\multicolumn{4}{c}{Best ranked metric} & \\multicolumn{2}{c}{Range, non-text metrics} \\\\',
     '\\cmidrule(lr){5-8} \\cmidrule(l){9-10}',
     'Evaluation set & (\\%) & $w_{\\text{det}}$ & $w_{\\text{rank}}$ & name & $\\tau$ & det. (\\%) & AUC & AUC & $\\tau(A)$ \\\\'], body))

# DENEB by epoch (runs/summary/deneb_epoch_sweep.json, written once the sweep has finished).
path = RUNS5 / 'deneb_epoch_sweep.json'
sentence = ''
if path.exists():
    druns = read(path)['runs']
    held = [run['epochs'] for name, run in druns.items() if not name.startswith('pairclean')]
    clean = [run['epochs'] for name, run in druns.items() if name.startswith('pairclean')]
    if held and clean:
        last = max(len(epochs) for epochs in held + clean)
        body, gaps = [], []
        for label, column in (('Duplicated samples', 'polaris_test/pair'), ('All other samples', 'polaris_test/not pair')):
            body.append(f'\\multicolumn{{{last + 1}}}{{l}}{{\\emph{{{label}}}}} \\\\')
            for title, group in (('With duplicates', held), ('Deduplicated', clean)):
                cells = [title]
                for epoch in range(last):
                    values = [epochs[str(epoch)]['tau'][column] for epochs in group if str(epoch) in epochs]
                    cells.append(f'{np.mean(values):.2f}\\,({len(values)})' if values else '--')
                body.append('\\quad ' + ' & '.join(cells) + ' \\\\')
            if column.endswith('/pair'):
                for epoch in range(last):
                    a = [epochs[str(epoch)]['tau'][column] for epochs in held if str(epoch) in epochs]
                    b = [epochs[str(epoch)]['tau'][column] for epochs in clean if str(epoch) in epochs]
                    if a and b:
                        gaps.append(float(np.mean(a) - np.mean(b)))
                body.append('\\addlinespace')
        write('tab_deneb_epochs', tabular('l' + 'r' * last, ['Training set & ' + ' & '.join(f'Epoch {e + 1}' for e in range(last)) + ' \\\\'], body))
        define('DenebEpochGapLow', f'{min(gaps):.1f}')
        define('DenebEpochGapHigh', f'{max(gaps):.1f}')
        define('DenebEpochCheckpoints', str(sum(len(epochs) for epochs in held + clean)))
        rest = [np.mean([e[str(k)]['tau']['polaris_test/not pair'] for e in held if str(k) in e])
                - np.mean([e[str(k)]['tau']['polaris_test/not pair'] for e in clean if str(k) in e]) for k in range(len(gaps))]
        define('DenebEpochRestMax', f'{max(abs(value) for value in rest):.2f}')
        define('DenebEpochCompared', str(len(gaps)))
        assert gaps[0] == min(gaps) and max(gaps) < 1, gaps      # the sentence below says exactly this
        sentence = (' For DENEB the gap on duplicated samples also opens during the first epochs, but it stays below one point '
                    f'throughout: {min(gaps):.1f} after the first epoch and at most {max(gaps):.1f} later, while all other samples differ '
                    'by at most \\DenebEpochRestMax{} points (\\ref{app:epochs}).')
numbers['DenebEpochSentence'] = sentence
if not sentence:
    write('tab_deneb_epochs', ['\\emph{The DENEB checkpoints had not all been scored when this draft was compiled.}'])

write('numbers', [f'\\newcommand{{\\{name}}}{{{text}}}' for name, text in sorted(numbers.items())])
print(len(numbers), 'numbers')
