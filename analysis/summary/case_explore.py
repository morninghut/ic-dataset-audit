"""List candidate case studies by how metric percentiles deviate from the human percentile."""
import sys
import numpy as np
from case_data import load

name, pool = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else None)
minimum = int(sys.argv[3]) if len(sys.argv) > 3 else 5
t = load(name, pool)
M = t['metrics']
want = {'Judge': 'VLM judge (InternVL 8B) + refs', 'JudgeQ': 'VLM judge (Qwen 8B) + refs', 'JudgeNoRef': 'VLM judge (InternVL 8B)',
        'Pearl': 'Pearl (retrained)', 'DENEB': 'DENEB (retrained)', 'Polos': 'Polos (retrained)', 'RefPAC': 'RefPAC-S++ (ViT-L/14)',
        'PAC': 'PAC-S++ (ViT-L/14)', 'CLIP-S': 'CLIP-S', 'CIDEr': 'CIDEr', 'Pearl+dup': 'Pearl (retrained) +dup',
        'Rigel': 'Rigel (official ckpt) + refs', 'Rigel+dup': 'Rigel (official ckpt) + refs +dup'}
idx = {k: M.index(v) for k, v in want.items() if v in M}
P = t['percentile']
keep = (t['count'] >= minimum) & (t['mean'] > t['low'])
above = t['mean'] > t['low']
ref = np.sort(t['mean'][above])
H = 100 * (np.searchsorted(ref, t['mean'], side='left') + np.searchsorted(ref, t['mean'], side='right')) / 2 / above.sum()
D = {k: P[:, j] - H for k, j in idx.items()}


def show(i):
    group = f" g={t['group'][i]}" if 'group' in t else ''
    return (f"[{i}] h={t['mean'][i]:.2f} (pct {H[i]:.0f}) n={t['count'][i]}{group} | " + ' '.join(f"{k}:{P[i, j]:.0f}" for k, j in idx.items())
            + f"\n      CAND: {t['candidate'][i]}\n      REFS: " + ' / '.join(t['references'][i][:3]) + f"\n      IMG: {t['image'][i]}")


def top(title, score, count=int(sys.argv[4]) if len(sys.argv) > 4 else 8):
    order = np.argsort(-np.where(keep, score, -1e9))[:count]
    print('=====', title)
    for i in order:
        print(show(i))


learned = [k for k in ('Pearl', 'DENEB', 'Polos') if k in D]
emb = [k for k in ('RefPAC', 'PAC', 'CLIP-S') if k in D]
top('all agree with humans (small max deviation), high score', -np.max(np.abs(np.stack([D[k] for k in ['Judge'] + learned + ['RefPAC', 'CIDEr']])), 0) + (H > 80) * 1000)
top('all agree with humans, low score', -np.max(np.abs(np.stack([D[k] for k in ['Judge'] + learned + ['RefPAC', 'CIDEr']])), 0) + (H < 20) * 1000)
top('learned+embedding overrate, judge right', np.min(np.stack([D[k] for k in learned + ['RefPAC']]), 0) - np.abs(D['Judge']))
top('learned+embedding underrate, judge right', np.min(np.stack([-D[k] for k in learned + ['RefPAC']]), 0) - np.abs(D['Judge']))
top('judge overrates, learned right', D['Judge'] - np.max(np.abs(np.stack([D[k] for k in learned])), 0))
top('judge underrates, learned right', -D['Judge'] - np.max(np.abs(np.stack([D[k] for k in learned])), 0))
top('everything overrates', np.min(np.stack([D[k] for k in ['Judge'] + learned + ['RefPAC', 'CIDEr']]), 0))
top('everything underrates', np.min(np.stack([-D[k] for k in ['Judge'] + learned + ['RefPAC', 'CIDEr']]), 0))
top('CIDEr underrates, others right', -D['CIDEr'] - np.max(np.abs(np.stack([D[k] for k in ['Judge'] + learned])), 0))
top('reference-free CLIP-S overrates, with refs right', D['CLIP-S'] - np.abs(D['Judge']) - np.abs(D['RefPAC']))
if 'Pearl+dup' in D and 'group' in t:
    dup = t['group'] == 2
    top('duplicated: Pearl+dup close to human, Pearl far (overrated by clean)', np.where(dup, np.abs(D['Pearl']) - np.abs(D['Pearl+dup']), -1e9))
