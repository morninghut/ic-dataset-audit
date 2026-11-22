"""The duplication-free, floor-aware protocol applied to every metric we can score.

For each evaluation set three quantities are reported per metric, all with a
picture-clustered bootstrap:

  all      100 x Kendall tau over every row (the number the papers print)
  floor    the same over the rows scored above the floor of the scale
  order    accuracy of ordering two captions of the same picture, against one
           half of the annotators, next to a human who holds the other half

A trained metric enters a set only through a model whose training pool shares no
labelled (picture, candidate) pair with that set.  Where the pool as released shares none,
that is the model trained on the released pool, and the model trained on the cleaned pool is
listed next to it, marked "(cleaned pool)".  Where the released pool does share pairs with the
set, the cleaned-pool model is the one ranked and the released-pool model is listed next to it,
marked "+dup", so the table shows what duplication does to the ranking.  Marked rows are not
ranked.

    protocol.py [resamples] [repeats] [seed]

seed picks the DENEB and Pearl retrainings (external/<family>_s<seed>); 12 by default.

Run with PYTHONPATH=Polos-revised from analysis/probe's interpreter.
Writes runs/metrics/protocol.json (protocol_s<seed>.json for another seed) and prints the tables.
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

STEP2 = Path((_ROOT + '/analysis/probe'))
sys.path.insert(0, str(STEP2))
import discrimination  # noqa: E402
import validity  # noqa: E402
from analyze_effects import finite  # noqa: E402
from effects import Benchmark  # noqa: E402
from lib import ROOT, RUNS, write_json  # noqa: E402

SETS = ['polaris_test', 'spica_test', 'nebula_test', 'flickr8k_expert', 'composite', 'flickr8k_cf']
POLARIS_FAMILY = ['polaris_test', 'spica_test', 'nebula_test']

# name -> spec, or {set: spec} when the model depends on the set.  A missing set means the
# metric has no duplication-free model for it.
UNTRAINED = {
    'BLEU-1': 'external/classic:bleu1',
    'BLEU-4': 'external/classic:bleu4',
    'ROUGE-L': 'external/classic:rougel',
    'METEOR': 'external/classic:meteor',
    'CIDEr': 'external/classic:cider',
    'SPICE': 'external/classic:spice',
    'CLIP-S': 'baselines/none:clipscore',
    'RefCLIP-S': 'baselines/none:refclipscore',
    'text only (SimCSE)': 'baselines/none:textonly_simcse',
    'PAC-S': 'external/pacs:pacs',
    'RefPAC-S': 'external/pacs:refpacs',
    'PAC-S (ViT-L/14)': 'external/pacs_l14:pacs',
    'RefPAC-S (ViT-L/14)': 'external/pacs_l14:refpacs',
    'PAC-S++': 'external/pacspp:pacs',
    'RefPAC-S++': 'external/pacspp:refpacs',
    'PAC-S++ (ViT-L/14)': 'external/pacspp_l14:pacs',
    'RefPAC-S++ (ViT-L/14)': 'external/pacspp_l14:refpacs',
    'VLM judge': 'external/vlm:free',
    'VLM judge + refs': 'external/vlm:refs',
    'VLM judge (Qwen 8B)': 'external/vlm_qwen8b:free',
    'VLM judge (Qwen 8B) + refs': 'external/vlm_qwen8b:refs',
    # InternVL3.5 opens a reasoning block on part of the prompts; the rating is read after an empty
    # one is prefilled (digit mass 0.95-0.98).  Without the prefill (external/vlm_internvl8b, mass
    # 0.65-0.96) tau differs by at most 0.55, except on Flickr8k-Expert, where it is 2.4-3.0 lower
    # (compare_nothink.py).
    'VLM judge (InternVL 8B)': 'external/vlm_internvl8b_nothink:free',
    'VLM judge (InternVL 8B) + refs': 'external/vlm_internvl8b_nothink:refs',
}
# (clean spec per set, released spec, sets whose pairs the released pool contains)
TRAINED = {
    'Polos (official ckpt)': (
        {name: 'mlp_legacy_cached_clip_s42/polaris_train:official'
         for name in ['polaris_test', 'flickr8k_expert', 'composite', 'flickr8k_cf']},
        'mlp_legacy_cached_clip_s42/polaris_train:official', ['spica_test', 'nebula_test']),
    'Polos (retrained)': (
        'mlp_legacy_cached_clip_s42_variants/polaris_train:pairclean',
        'mlp_legacy_cached_clip_s42_variants/polaris_train:released', ['spica_test', 'nebula_test']),
    # Released checkpoint only (no training code): trained on the Spica pool, so clean where that
    # pool holds no judged pair of the set.
    'Rigel (official ckpt)': (
        {name: 'external/rigel:free' for name in ['flickr8k_expert', 'composite', 'flickr8k_cf']},
        'external/rigel:free', ['polaris_test', 'spica_test', 'nebula_test']),
    'Rigel (official ckpt) + refs': (
        {name: 'external/rigel:refs' for name in ['flickr8k_expert', 'composite', 'flickr8k_cf']},
        'external/rigel:refs', ['polaris_test', 'spica_test', 'nebula_test']),
    'DENEB (retrained)': (
        'external/deneb_s{seed}:pairclean', 'external/deneb_s{seed}:released', ['polaris_test']),
    'Pearl (retrained)': (
        'external/pearl_s{seed}:pairclean', 'external/pearl_s{seed}:released',
        ['polaris_test', 'spica_test', 'nebula_test']),
    'ridge SigLIP, Polaris pool (ours)': (
        'ridge_variant_siglip/polaris_train:pairclean', 'ridge_variant_siglip/polaris_train:released',
        ['spica_test', 'nebula_test']),
    'ridge SigLIP, Spica pool (ours)': (
        'ridge_variant_siglip/spica_train:pairclean', 'ridge_variant_siglip/spica_train:released',
        ['polaris_test', 'spica_test', 'nebula_test']),
    # The same ridge regression on the CLIP features that Polos reads: separates the encoder from the architecture.
    'ridge CLIP, Polaris pool (ours)': (
        'ridge_variant_clip/polaris_train:pairclean', 'ridge_variant_clip/polaris_train:released',
        ['spica_test', 'nebula_test']),
    'ridge CLIP, Spica pool (ours)': (
        'ridge_variant_clip/spica_train:pairclean', 'ridge_variant_clip/spica_train:released',
        ['polaris_test', 'spica_test', 'nebula_test']),
}


CLEANED = ' (cleaned pool)'


def available(spec):
    directory, _, label = spec.partition(':')
    path = RUNS / 'models' / directory / 'report.json'
    return path.exists() and label in validity.read_json(path)['labels']


def metrics_for(name, seed):
    """[(display name, spec, duplicated)] of the metrics that can be scored on this set."""
    out = [(display, spec, False) for display, spec in UNTRAINED.items() if available(spec)]
    for display, (clean, released, duplicated) in TRAINED.items():
        clean = clean.get(name) if isinstance(clean, dict) else clean
        clean, released = clean and clean.format(seed=seed), released.format(seed=seed)
        if name in duplicated:
            if clean and available(clean):
                out.append((display, clean, False))
            if available(released):
                out.append((display + ' +dup', released, True))
        elif available(released):
            # No pair of this set is in the released pool: the model as its authors trained it is
            # clean here and is the one ranked.  The model from the cleaned pool is listed unranked,
            # to show what the cleaning costs on a set it was not needed for.
            out.append((display, released, False))
            if clean and clean != released and available(clean):
                out.append((display + CLEANED, clean, True))
    return out


def ranks(values, eligible):
    """Rank 1 = best among the eligible entries; nan elsewhere."""
    values = np.asarray(values, float)
    out = np.full(len(values), np.nan)
    keep = eligible & np.isfinite(values)
    out[keep] = (-values[keep]).argsort().argsort() + 1
    return out


def main(resamples=1000, repeats=20, seed=12):
    resamples, repeats, seed = int(resamples), int(repeats), int(seed)
    out = {}
    for name in SETS:
        entries = metrics_for(name, seed)
        names = [display for display, _, _ in entries]
        specs = [spec for _, spec, _ in entries]
        clean = np.array([not duplicated for _, _, duplicated in entries])
        benchmark = Benchmark(name)
        groups, leave_one_out, counts = validity.masks(benchmark.rows)
        columns = validity.columns(specs, name)
        labels, predictions = [label for label, _ in columns], [column for _, column in columns]
        if labels != specs:
            raise ValueError(f'{name}: columns came back as {labels}')
        table = {}
        for mask in ('all', 'above_floor'):
            cells = validity.evaluate(benchmark, labels, predictions, groups[mask], resamples)
            human = benchmark.select(groups[mask] & np.isfinite(leave_one_out))
            table[mask] = dict(metrics=[cells[label] for label in labels],
                               human=benchmark.measure(leave_one_out, human.items))
        order = discrimination.analyze(name, specs, repeats, resamples)

        print(f"\n# {name}: {counts['rows']} rows, {counts['floor']} at the floor "
              f"({100 * counts['floor'] / counts['rows']:.1f}%)"
              + (f", {order['pairs']} same-picture caption pairs" if order else ', no repeated judgements'))
        print(f"{'metric':30s} {'all rows':>22s} {'above floor':>22s} {'ordering':>20s} "
              f"{'ordering - human':>22s}   ranks")
        all_scores = [cell['score'] for cell in table['all']['metrics']]
        floor_scores = [cell['score'] for cell in table['above_floor']['metrics']]
        order_scores = [100 * order['point'][spec][-1] if order else np.nan for spec in specs]
        rank = [ranks(all_scores, clean), ranks(floor_scores, clean), ranks(order_scores, clean)]
        rows = []
        for index, display in enumerate(names):
            cells = [validity.show(table[mask]['metrics'][index]) for mask in ('all', 'above_floor')]
            ordering, gap = '-', '-'
            row = dict(metric=display, spec=specs[index], duplicated=bool(not clean[index]),
                       all=table['all']['metrics'][index], above_floor=table['above_floor']['metrics'][index],
                       rank_all=rank[0][index], rank_above_floor=rank[1][index], rank_order=rank[2][index])
            if order:
                span = order['spans'][specs[index]]
                ordering = f'{order_scores[index]:5.1f}' + (f' [{100 * span[0][-1]:.1f},{100 * span[1][-1]:.1f}]'
                                                            if span else '')
                row['order'] = dict(score=order_scores[index])
                if span:
                    mean, low, high = (100 * v[-1] for v in order['to_human'][specs[index]])
                    gap = f'{mean:+5.1f} [{low:+.1f},{high:+.1f}]'
                    row['order'].update(low=100 * span[0][-1], high=100 * span[1][-1],
                                        minus_human=dict(mean=mean, low=low, high=high))
            shown = ' '.join('  -' if np.isnan(r[index]) else f'{int(r[index]):3d}' for r in rank)
            print(f'{display:30s} {cells[0]:>22s} {cells[1]:>22s} {ordering:>20s} {gap:>22s}   {shown}')
            rows.append(row)
        human_order = 100 * order['point']['human'][-1] if order else np.nan
        print(f"{'human':30s} {table['all']['human']:22.2f} {table['above_floor']['human']:22.2f} "
              f'{human_order:20.1f}')
        agreement = {}
        for label, other in (('above_floor', rank[1]), ('order', rank[2])):
            keep = np.isfinite(rank[0]) & np.isfinite(other)
            if keep.sum() > 2:
                agreement[label] = float(kendalltau(rank[0][keep], other[keep])[0])
        print('rank agreement with the all-rows ranking (Kendall tau over clean metrics): '
              + ', '.join(f'{k} {v:+.2f}' for k, v in agreement.items()))
        out[name] = dict(counts=counts, rows=rows, rank_agreement=agreement,
                         human=dict(all=table['all']['human'], above_floor=table['above_floor']['human'],
                                    order=human_order),
                         pairs=order['pairs'] if order else 0)
    write_json(ROOT / 'runs/metrics' / ('protocol.json' if seed == 12 else f'protocol_s{seed}.json'),
               finite(dict(sets=SETS, seed=seed, resamples=resamples, repeats=repeats, benchmarks=out)))


if __name__ == '__main__':
    main(*sys.argv[1:])
