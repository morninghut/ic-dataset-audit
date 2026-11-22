"""Extend feature_cache_v1 with the Spica and Nebula rows (catalog + row files only).

The text and image lists are append-only, so every index of the v1 cache stays
valid and the already extracted encoder outputs are copied as a prefix.  New
pictures are interned by decoded-pixel hash, so a Spica/Nebula picture that is
already in the v1 catalog (Polaris, COCO val2014, ...) reuses its features, and
`picture_ids.npy` gives one integer identity per picture across all datasets.

After this script, run the unchanged extractor on the new cache:
    python -m revision.cache_features --part {simcse,clip,siglip} --cache feature_cache_v2
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import ast
import csv
import io
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(_ROOT)
sys.path.insert(0, str(ROOT / 'analysis/audit'))
from common import NEBULA, SPICA, pixel_hash, read_json, write_json  # noqa: E402

OLD, NEW = ROOT / 'feature_cache_v1', ROOT / 'feature_cache_v2'
AUDIT = ROOT / 'analysis/audit'
MAX_REFS = 10
csv.field_size_limit(10 ** 9)


def hash_path(path):
    with Image.open(path) as im:
        return pixel_hash(im)


def parse_refs(text):
    """Reference lists are repr strings; some train rows are quoted twice."""
    text = text.strip()
    for _ in range(2):
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(s) for s in parsed]
        text = str(parsed)
    raise ValueError(text[:80])


def main():
    NEW.mkdir(exist_ok=True)
    old = read_json(OLD / 'catalog.json')
    texts, images = list(old['texts']), list(old['images'])
    text_ids = {t: i for i, t in enumerate(texts)}
    with Pool(48) as pool:
        hashes = pool.map(hash_path, images, chunksize=64)
    first = {}
    for index, h in enumerate(hashes):
        first.setdefault(h, index)
    print('v1 images', len(images), 'distinct pictures', len(first), flush=True)

    def text(value):
        if value not in text_ids:
            text_ids[value] = len(texts)
            texts.append(value)
        return text_ids[value]

    def picture(h, path_factory):
        if h not in first:
            first[h] = len(images)
            images.append(str(path_factory()))
            hashes.append(h)
        return first[h]

    datasets = dict(old['datasets'])

    def save(name, records, source):
        n = len(records)
        cand, img = np.empty(n, np.int32), np.empty(n, np.int32)
        refs, counts = np.full((n, MAX_REFS), -1, np.int32), np.empty(n, np.int16)
        target, ids = np.empty(n, np.float32), []
        for i, row in enumerate(records):
            cand[i], img[i], counts[i] = text(row['caption']), row['image'], len(row['references'])
            refs[i, :counts[i]] = [text(s) for s in row['references']]
            target[i] = row['score']
            ids.append(row['id'])
        np.savez(NEW / f'{name}.npz', candidate=cand, image=img, references=refs, counts=counts,
                 target=target, ids=np.array(ids))
        datasets[name] = dict(count=n, original_count=n, source=str(source), pairwise=False,
                              text_prefix_count=len(texts), image_prefix_count=len(images))
        print(name, n, len(texts), len(images), flush=True)

    spica_hash = {r['name']: r['pixel_sha1'] for r in read_json(AUDIT / 'inventory_spica.json')}
    for split in ['train', 'val', 'test']:
        source = SPICA / f'spica_{split}.csv'
        with_refs, without = [], []
        with source.open(newline='') as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                imgid = row['imgid']
                record = dict(id=f'{split}:{index}', caption=row['mt'], references=parse_refs(row['refs']),
                              score=float(row['score']),
                              image=picture(spica_hash[imgid], lambda: SPICA / 'images' / imgid))
                if not row['mt'].strip() or not np.isfinite(record['score']):
                    raise ValueError(f'bad row {split}:{index}')
                (with_refs if record['references'] else without).append(record)
        save(f'spica_{split}', with_refs, source)
        if without:
            save(f'spica_{split}_noref', without, source)

    nebula_hash = {(r['split'], r['name']): r['pixel_sha1'] for r in read_json(AUDIT / 'inventory_nebula.json')}
    out = NEBULA / 'images'
    out.mkdir(exist_ok=True)
    for split in ['train', 'valid', 'test']:
        records, index = [], 0
        for path in sorted((NEBULA / 'data').glob(f'{split}-*.parquet')):
            for row in pq.read_table(path).to_pylist():
                h = nebula_hash[(split, row['file_name'])]

                def write(h=h, row=row):
                    target = out / f'{h}.jpg'
                    data = row['image']['bytes']
                    with Image.open(io.BytesIO(data)) as im:
                        assert pixel_hash(im) == h
                    target.write_bytes(data)
                    return target
                records.append(dict(id=f'{split}:{index}', caption=row['mt'], references=list(row['refs']),
                                    score=float(row['human_score']), image=picture(h, write)))
                index += 1
        save(f'nebula_{split}', records, NEBULA / 'data')

    ordered = {}
    picture_ids = np.array([ordered.setdefault(h, len(ordered)) for h in hashes], np.int32)
    np.save(NEW / 'picture_ids.npy', picture_ids)
    write_json(NEW / 'pictures.json', list(ordered))
    catalog = dict(old, texts=texts, images=images, datasets=datasets, version=2, built_at=time.time(),
                   extends='feature_cache_v1', v1_texts=len(old['texts']), v1_images=len(old['images']))
    (NEW / 'catalog.json').write_text(__import__('json').dumps(catalog, ensure_ascii=False, indent=2) + '\n')
    print('texts', len(old['texts']), '->', len(texts), 'images', len(old['images']), '->', len(images),
          'pictures', len(ordered), flush=True)

    copy_prefix(old, texts, images)


def copy_prefix(old, texts, images):
    """Copy v1 row files and encoder outputs; the extractor resumes after the prefix."""
    import shutil
    for path in OLD.glob('*'):
        if path.suffix in ['.npz'] or path.name.endswith('_pairs.json') or '_stats' in path.name:
            shutil.copy2(path, NEW / path.name)
    nt, ni, ot, oi = len(texts), len(images), len(old['texts']), len(old['images'])
    shapes = dict(simcse_cls=(nt, 13, 768), simcse_mean=(nt, 13), simcse_var=(nt, 13), simcse_count=(nt,),
                  clip_text=(nt, 512), siglip_text=(nt, 1152), clip_image=(ni, 512), siglip_image=(ni, 1152))
    for key, shape in shapes.items():
        source = np.load(OLD / f'{key}.npy', mmap_mode='r')
        target = np.lib.format.open_memmap(NEW / f'{key}.npy', mode='w+', dtype=np.float32, shape=shape)
        for start in range(0, len(source), 8192):
            end = min(start + 8192, len(source))
            target[start:end] = source[start:end]
        target.flush()
        print('copied', key, source.shape, '->', shape, flush=True)
    for part in ['simcse', 'clip', 'siglip']:
        write_json(NEW / f'{part}_status.json', dict(completed_texts=ot, completed_images=0 if part == 'simcse' else oi,
                                                     status='extending', v1_texts=ot, v1_images=oi))


if __name__ == '__main__':
    if sys.argv[1:] == ['--copy-only']:
        new = read_json(NEW / 'catalog.json')
        copy_prefix(read_json(OLD / 'catalog.json'), new['texts'], new['images'])
    else:
        main()
