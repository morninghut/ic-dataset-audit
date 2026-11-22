"""Write the case studies shown in the paper to runs/summary/cases.json.

Each case is one image-caption pair of an evaluation set, chosen by hand from the lists that
case_explore.py prints (the indices below are positions in case_data.load(<set>)).  Stored per
case: image file, candidate, references, the individual human ratings, the human score rescaled
to 0-1, its percentile, and the percentile of every metric.  Percentiles are taken among the
pairs of the set that are rated above the lowest score.

    cases.py        (PYTHONPATH=Polos-revised, from analysis/summary)
"""
import json

import numpy as np

from case_data import ROOT, load

# (set, index, what the case shows)
MAIN = [('polaris_test', 8473, 'agreement'), ('polaris_test', 7687, 'learned and embedding metrics too high'),
        ('polaris_test', 7802, 'judge too low'), ('polaris_test', 7566, 'duplicated sample')]
MORE = [('polaris_test', 3147, ''), ('polaris_test', 3493, ''), ('polaris_test', 5188, ''), ('polaris_test', 4768, ''),
        ('polaris_test', 2916, ''), ('polaris_test', 3112, ''), ('polaris_test', 8445, ''),
        ('flickr8k_expert', 3949, ''), ('flickr8k_expert', 442, ''), ('flickr8k_expert', 692, ''),
        ('composite', 3172, ''), ('composite', 5714, '')]
POOL = {'polaris_test': 'spica_train'}


def main():
    tables, out = {}, {'main': [], 'more': []}
    for key, cases in (('main', MAIN), ('more', MORE)):
        for name, index, note in cases:
            if name not in tables:
                tables[name] = load(name, POOL.get(name))
            t = tables[name]
            above = t['mean'] > t['low']
            reference = np.sort(t['mean'][above])
            human = 100 * (np.searchsorted(reference, t['mean'][index], side='left')
                           + np.searchsorted(reference, t['mean'][index], side='right')) / 2 / above.sum()
            out[key].append(dict(
                set=name, index=index, note=note, image=t['image'][index], candidate=t['candidate'][index],
                references=t['references'][index], ratings=t['ratings'][index],
                human=float((t['mean'][index] - t['low']) / (t['high'] - t['low'])), human_percentile=float(human),
                group=int(t['group'][index]) if 'group' in t else None,
                percentile={metric: float(value) for metric, value in zip(t['metrics'], t['percentile'][index])}))
            print(name, index, f"human {out[key][-1]['human']:.2f} pct {human:.0f}", t['candidate'][index][:70], flush=True)
    (ROOT / 'runs/summary/cases.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
