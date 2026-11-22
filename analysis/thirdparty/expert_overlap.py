"""Count the test samples of Polaris and Nebula that are in the training files released with EXPERT.

Usage: python expert_overlap.py /path/to/EXPERT   (https://github.com/hjkim811/EXPERT, commit 7fbfb00)

A sample is matched by its image identifier and the exact text of its candidate caption; both come from
the files of the EXPERT repository itself, so no image is decoded.
"""
import collections
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
load = lambda name: json.loads((root / name).read_text())
key = lambda row: (row['imgid'], row['mt'])

training = {'nebula_exp_train': load('exp_datasets/nebula_exp_train.json'),
            'polaris_exp_train': load('exp_datasets/polaris_exp_train.json')}
training['union'] = training['nebula_exp_train'] + training['polaris_exp_train']
tests = {'polaris_test': load('annotations/polaris_test.json'), 'nebula_test': load('annotations/nebula_test.json')}

result = {}
for train_name, train in training.items():
    score = {key(row): row['score'] for row in train}
    images = {row['imgid'] for row in train}
    for test_name, test in tests.items():
        ratings = collections.defaultdict(list)
        for row in test:
            if key(row) in score:
                ratings[key(row)].append(row['score'])
        same = sum(len(values) for values in ratings.values())
        result[f'{train_name}->{test_name}'] = {
            'test_samples': len(test),
            'duplicated_samples': same,
            'duplicated_share': same / len(test),
            'image_only_share': sum(key(row) not in score and row['imgid'] in images for row in test) / len(test),
            'duplicated_pairs': len(ratings),
            'pairs_with_equal_mean_score': sum(abs(sum(v) / len(v) - score[k]) < 1e-6 for k, v in ratings.items()),
        }
print(json.dumps(result, indent=2))
