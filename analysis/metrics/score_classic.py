"""BLEU, ROUGE-L, CIDEr, METEOR and SPICE for every split, with the pycocoevalcap scorers.

This is the protocol of the released correlation scripts (clipscore, pacscore
compute_correlations.py): Stanford PTB tokenization, one item per benchmark row,
per-item scores.  CIDEr therefore takes its document frequencies from the
references of the rows of the split it scores.  Only the bookkeeping differs:
BLEU, ROUGE-L, METEOR and SPICE of an item do not depend on the other items, so
each distinct (candidate, references) pair is scored once, METEOR in parallel
processes and SPICE in one JVM with its own parse cache.

    score_classic.py <ngram|meteor|spice> [splits...]

  ngram   bleu1, bleu4, rougel, cider
  meteor  meteor
  spice   spice (needs the CoreNLP 3.6.0 jars next to pycocoevalcap's spice-1.0.jar)

Writes runs/metrics/classic/<group>/<split>.npz in CSV order.  A candidate
that is empty after tokenization scores 0 on every metric.
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import ast
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(_ROOT)
DATA = ROOT / 'pearl_data'
OUT = ROOT / 'runs/metrics/classic'
CORE = ['spica_test', 'polaris_test', 'nebula_test', 'flickr8k_expert', 'composite', 'flickr8k_cf',
        'pascal50s', 'polaris_val', 'nebula_valid', 'spica_val']
METEOR_WORKERS = 32

csv.field_size_limit(10 ** 9)


def read(split):
    """[(candidate, (references...))] in CSV order."""
    with open(DATA / 'csv' / f'{split}.csv', newline='') as f:
        return [(str(row['mt']), tuple(ast.literal_eval(row['refs']))) for row in csv.DictReader(f)]


def tokenize(texts):
    """PTB-tokenize distinct texts -> {text: tokenized text}."""
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    texts = sorted(texts)
    tokens = PTBTokenizer().tokenize({i: [{'caption': text}] for i, text in enumerate(texts)})
    if len(tokens) != len(texts):
        raise ValueError(f'tokenizer returned {len(tokens)} of {len(texts)} texts')
    return {text: tokens[i][0] for i, text in enumerate(texts)}


def prepare(split):
    """Tokenized rows, the distinct non-empty items and the row -> item index (-1 = empty candidate)."""
    rows = read(split)
    table = tokenize({text for candidate, refs in rows for text in (candidate,) + refs})
    rows = [(table[candidate], tuple(table[ref] for ref in refs)) for candidate, refs in rows]
    items, index, inverse = [], {}, np.empty(len(rows), np.int64)
    for i, row in enumerate(rows):
        if not row[0].strip():
            inverse[i] = -1
            continue
        if row not in index:
            index[row] = len(items)
            items.append(row)
        inverse[i] = index[row]
    return rows, items, inverse


def expand(values, inverse):
    return np.where(inverse >= 0, np.asarray(values, np.float64)[np.maximum(inverse, 0)], 0.0).astype(np.float32)


def ngram(rows, items, inverse):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    gts = {i: list(refs) for i, (_, refs) in enumerate(items)}
    res = {i: [candidate] for i, (candidate, _) in enumerate(items)}
    _, bleu = Bleu(4).compute_score(gts, res, verbose=0)
    _, rouge = Rouge().compute_score(gts, res)
    # CIDEr over the rows themselves: its document frequencies count rows, as in the released scripts
    keep = np.flatnonzero(inverse >= 0)
    _, cider_kept = Cider().compute_score({int(i): list(rows[i][1]) for i in keep},
                                          {int(i): [rows[i][0]] for i in keep})
    cider = np.zeros(len(rows), np.float32)
    cider[keep] = cider_kept
    return dict(bleu1=expand(bleu[0], inverse), bleu4=expand(bleu[3], inverse),
                rougel=expand(rouge, inverse), cider=cider)


def meteor_chunk(chunk):
    from pycocoevalcap.meteor.meteor import Meteor
    scorer = Meteor()
    clean = lambda text: text.replace('|||', ' ')
    gts = {i: [clean(ref) for ref in refs] for i, (_, refs) in enumerate(chunk)}
    res = {i: [clean(candidate)] for i, (candidate, _) in enumerate(chunk)}
    _, scores = scorer.compute_score(gts, res)
    return scores


def meteor(rows, items, inverse):
    size = max(1, -(-len(items) // METEOR_WORKERS))  # one JVM per worker: starting one costs seconds
    chunks = [items[start:start + size] for start in range(0, len(items), size)]
    with ProcessPoolExecutor(METEOR_WORKERS) as pool:
        scores = [score for part in pool.map(meteor_chunk, chunks) for score in part]
    return dict(meteor=expand(scores, inverse))


def spice(rows, items, inverse):
    import pycocoevalcap.spice.spice as module
    directory = Path(module.__file__).parent
    cache = ROOT / 'runs/metrics/classic/spice_cache'
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        with open(Path(work) / 'in.json', 'w') as f:
            json.dump([dict(image_id=i, test=candidate, refs=list(refs))
                       for i, (candidate, refs) in enumerate(items)], f)
        subprocess.check_call(['java', '-Xmx64G', '-jar', 'spice-1.0.jar', str(Path(work) / 'in.json'),
                               '-cache', str(cache), '-out', str(Path(work) / 'out.json'), '-subset',
                               '-silent'], cwd=directory)
        with open(Path(work) / 'out.json') as f:
            results = json.load(f)
    scores = np.zeros(len(items))
    for item in results:
        value = item['scores']['All']['f']
        scores[item['image_id']] = 0.0 if value is None or np.isnan(float(value)) else float(value)
    if len(results) != len(items):
        raise ValueError(f'SPICE returned {len(results)} of {len(items)} items')
    return dict(spice=expand(scores, inverse))


def main(group, *splits):
    scorer = dict(ngram=ngram, meteor=meteor, spice=spice)[group]
    (OUT / group).mkdir(parents=True, exist_ok=True)
    for split in splits or CORE:
        target = OUT / group / f'{split}.npz'
        if target.exists():
            continue
        started = time.time()
        rows, items, inverse = prepare(split)
        scores = scorer(rows, items, inverse)
        np.savez(target, **scores)
        print(f'{split}: {len(rows)} rows, {len(items)} distinct items, {(inverse < 0).sum()} empty, '
              + ', '.join(f'{k} mean {v.mean():.4f}' for k, v in scores.items())
              + f', {time.time() - started:.0f}s', flush=True)


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main(*sys.argv[1:])
