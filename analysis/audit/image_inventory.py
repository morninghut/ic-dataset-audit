"""Inventory every image file: file md5, decoded-pixel hash, format and size.

Covers the released Spica images, the Nebula parquet image bytes and the
benchmark image directories already used by the Polos reproduction, so later
steps can match images across datasets by exact pixel content.

Usage: image_inventory.py <group> [...]   groups: spica nebula polaris flickr8k composite pascal foil
"""
import io
import os
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, NEBULA, OUT, SPICA, md5, pixel_hash, write_json

Image.MAX_IMAGE_PIXELS = None


def describe(name, data):
    row = dict(name=name, bytes=len(data), file_md5=md5(data))
    try:
        with Image.open(io.BytesIO(data)) as im:
            row.update(format=im.format, mode=im.mode, width=im.width, height=im.height)
            row['pixel_sha1'] = pixel_hash(im)
    except Exception as error:  # unreadable or truncated file
        row['error'] = f'{type(error).__name__}: {error}'
    return row


def describe_path(path):
    return describe(Path(path).name, Path(path).read_bytes())


def file_group(paths, workers=32):
    with Pool(workers) as pool:
        return pool.map(describe_path, [str(p) for p in paths], chunksize=64)


def nebula_group():
    import pyarrow.parquet as pq
    rows, seen = [], set()
    for path in sorted((NEBULA / 'data').glob('*.parquet')):
        split = path.name.split('-')[0]
        table = pq.read_table(path, columns=['file_name', 'image'])
        names = table.column('file_name').to_pylist()
        images = table.column('image').to_pylist()
        for name, image in zip(names, images):
            key = (split, name)
            if key in seen:
                continue
            seen.add(key)
            row = describe(name, image['bytes'] if isinstance(image, dict) else image)
            row['split'] = split
            rows.append(row)
        del table, images
    return rows


def files(root):
    return sorted(p for p in Path(root).rglob('*') if p.is_file())


GROUPS = {
    'spica': lambda: file_group(files(SPICA / 'images')),
    'nebula': nebula_group,
    'polaris': lambda: file_group(files(DATA / 'polaris/images')),
    'flickr8k': lambda: file_group(files(DATA / 'flickr8k/images')),
    'composite': lambda: file_group(files(DATA / 'composite/images')),
    'pascal': lambda: file_group(files(DATA / 'pascal/VOCdevkit')),
    'coco_val2014': lambda: file_group(files(DATA / 'coco/val2014')),  # FOIL images
}

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for group in sys.argv[1:]:
        rows = GROUPS[group]()
        write_json(OUT / f'inventory_{group}.json', rows)
        print(group, len(rows), 'unique pixel hashes', len({r.get('pixel_sha1') for r in rows}), flush=True)
