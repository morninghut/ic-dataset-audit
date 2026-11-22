"""A zero-shot vision-language judge of caption quality, for every evaluation split.

The judge sees the picture and one candidate caption and is asked for a rating
from 0 to 9 (the rubric of FLEUR, Lee et al., ACL 2024, on a one-digit scale).
Nothing is generated: one forward pass gives the next-token distribution after
the prompt, and the score is the expected rating under that distribution
restricted to the ten digit tokens, divided by 9.  Taking the expectation rather
than the arg-max digit is FLEUR's score smoothing; it removes most ties.

    VLM_MODEL=<hub id> score_vlm.py <free|refs> [batch size] [splits...]

The judge is Qwen/Qwen3-VL-4B-Instruct unless VLM_MODEL names another one; every judge
gets one picture of at most 256 visual tokens and writes under its own directory.
VLM_PREFILL is text put at the start of the answer, before the rating is read; it is for
models that would otherwise open a reasoning block (InternVL3.5 starts about a third of its
answers with "<think>": an empty block, "<think>\n\n</think>\n\n", makes it answer directly).
Runs with a prefill write under <model>-<VLM_TAG>.

  free   the judge sees picture and candidate only
  refs   the judge also sees the split's reference captions (at most five)

A (picture, candidate[, references]) triple is scored once and shared by all the
rows and splits that contain it.  Finished triples are kept in
runs/metrics/vlm/<model>/<mode>/table.npz, so an interrupted run resumes.
Per split the script writes <split>.npz in CSV order with

    score   expected rating / 9
    argmax  most likely digit
    mass    probability the model puts on the ten digits (a sanity check: a low
            value means the model wanted to answer with something else)
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import ast
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(_ROOT)
DATA = ROOT / 'pearl_data'
MODEL = os.environ.get('VLM_MODEL', 'Qwen/Qwen3-VL-4B-Instruct')
PREFILL = os.environ.get('VLM_PREFILL', '').encode().decode('unicode_escape')
TAG = os.environ.get('VLM_TAG', '')
if bool(PREFILL) != bool(TAG):
    raise SystemExit('VLM_PREFILL and VLM_TAG go together')
OUT = ROOT / 'runs/metrics/vlm' / (MODEL.split('/')[-1] + (f'-{TAG}' if TAG else ''))
PIXELS = 512 * 512          # at most 256 visual tokens per picture
# How each family is held to that budget.  InternVL resizes every picture to one 448 x 448
# tile of 256 tokens when it is not allowed to cut the picture into several tiles.
PROCESSOR = {'Qwen': dict(min_pixels=64 * 32 * 32, max_pixels=PIXELS),
             'InternVL': dict(crop_to_patches=False)}
MAX_REFS = 5
ORDER = ['nebula_test', 'spica_test', 'composite', 'flickr8k_expert', 'polaris_test', 'pascal50s',
         'flickr8k_cf', 'nebula_valid', 'spica_val', 'polaris_val', 'foil_1ref', 'foil_4ref']

RUBRIC = ('Your task is to evaluate and rate the caption on a scale of 0 to 9 based on the given '
          'Grading Criteria. (Print the score ONLY)\n\n'
          'Grading Criteria:\n\n'
          '0: The caption does not describe the image at all.\n'
          '9: The caption accurately and clearly describes the image.\n\n')

csv.field_size_limit(10 ** 9)


def prompt(candidate, references):
    text = RUBRIC
    if references:
        text += ('Reference captions written by people for this image:\n'
                 + ''.join(f'- {r.strip()}\n' for r in references) + '\n')
    return text + f'Caption: {candidate.strip()}\n\nScore (Choose a rating from 0 to 9):'


def key_of(picture, candidate, references):
    return hashlib.sha1('\x1f'.join([picture, candidate, *references]).encode()).hexdigest()


def read_split(name, mode):
    """Per row: (key, picture, candidate, references shown to the judge)."""
    with open(DATA / 'csv' / f'{name}.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        references = tuple(ast.literal_eval(row['refs'])[:MAX_REFS]) if mode == 'refs' else ()
        candidate = str(row['mt'])
        out.append((key_of(row['imgid'], candidate, references), row['imgid'], candidate, references))
    return out


class Table:
    """Finished triples: key -> (score, argmax, mass)."""

    def __init__(self, path):
        self.path, self.values = path, {}
        if path.exists():
            with np.load(path) as f:
                self.values = dict(zip(f['keys'].tolist(), f['values']))

    def save(self):
        keys = np.array(list(self.values))
        values = np.array([self.values[k] for k in keys], np.float32).reshape(len(keys), 3)
        temporary = self.path.with_suffix('.tmp.npz')
        np.savez(temporary, keys=keys, values=values)
        temporary.replace(self.path)


def main(mode='free', batch_size=32, *splits):
    batch_size = int(batch_size)
    splits = list(splits) or ORDER
    out = OUT / mode
    out.mkdir(parents=True, exist_ok=True)
    table = Table(out / 'table.npz')

    family = next((name for name in PROCESSOR if name in MODEL), None)
    if family is None:
        raise ValueError(f'no picture budget is defined for {MODEL}')
    processor = AutoProcessor.from_pretrained(MODEL, **PROCESSOR[family])
    processor.tokenizer.padding_side = 'left'
    digits = [processor.tokenizer.encode(str(d), add_special_tokens=False) for d in range(10)]
    if any(len(d) != 1 for d in digits):
        raise ValueError('a digit is not a single token')
    digits = torch.tensor([d[0] for d in digits])
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                        device_map='cuda').eval()
    rating = torch.arange(10, dtype=torch.float32, device='cuda')

    class Pending(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            key, picture, candidate, references = self.items[index]
            image = Image.open(DATA / 'images' / picture).convert('RGB')
            message = [{'role': 'user', 'content': [{'type': 'image'},
                                                    {'type': 'text', 'text': prompt(candidate, references)}]}]
            return key, image, processor.apply_chat_template(message, add_generation_prompt=True) + PREFILL

    def collate(batch):
        keys, images, texts = zip(*batch)
        return list(keys), processor(text=list(texts), images=list(images), padding=True,
                                     return_tensors='pt')

    for name in splits:
        rows = read_split(name, mode)
        pending, seen = [], set()
        for item in rows:
            if item[0] not in table.values and item[0] not in seen:
                seen.add(item[0])
                pending.append(item)
        pending.sort(key=lambda item: (item[1], len(item[2])))
        print(f'{name}: {len(rows)} rows, {len(pending)} triples to score', flush=True)
        loader = torch.utils.data.DataLoader(Pending(pending), batch_size=batch_size, num_workers=12,
                                             collate_fn=collate)
        started, done = time.time(), 0
        for step, (keys, inputs) in enumerate(loader):
            inputs = inputs.to('cuda')
            with torch.no_grad():
                logits = model(**inputs).logits[:, -1].float()
            probability = logits.softmax(-1)[:, digits.to('cuda')]
            mass = probability.sum(-1)
            score = (probability / mass[:, None]) @ rating / 9
            values = torch.stack((score, probability.argmax(-1).float(), mass), 1).cpu().numpy()
            for key, value in zip(keys, values):
                table.values[key] = value
            done += len(keys)
            if step % 50 == 0:
                rate = done / (time.time() - started)
                print(f'[{name}] {done}/{len(pending)} {rate:.1f}/s eta '
                      f'{(len(pending) - done) / rate / 60:.1f} min', flush=True)
            if step % 500 == 499:
                table.save()
        table.save()
        values = np.array([table.values[item[0]] for item in rows], np.float32).reshape(len(rows), 3)
        np.savez(out / f'{name}.npz', score=values[:, 0], argmax=values[:, 1], mass=values[:, 2])
        print(f'{name}: score mean {values[:, 0].mean():.4f}, digit mass mean {values[:, 2].mean():.4f}, '
              f'min {values[:, 2].min():.4f}', flush=True)

    with open(out / 'scoring.json', 'w') as f:
        json.dump(dict(model=MODEL, mode=mode, prefill=PREFILL, processor=PROCESSOR[family], max_references=MAX_REFS,
                       rubric=RUBRIC, splits=splits), f, indent=1)


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
