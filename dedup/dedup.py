#!/usr/bin/env python3
"""Remove from Polaris, Nebula or Spica the training samples that are also evaluation samples.

The three collections share samples across their splits: the same image with the same candidate
caption and the same human ratings is, for example, in the Spica training set and in the Polaris
test set.  This tool writes a training file without them.  It needs the released files and the
manifests next to this script; it decodes no image and downloads nothing unless asked to.

    python dedup.py list
    python dedup.py clean spica_train --data /data/Spica --out cleaned/
    python dedup.py clean nebula_train --download --out cleaned/
    python dedup.py clean polaris_train --data /data/polaris --against nebula_test,spica_test
    python dedup.py groups polaris_test --data /data/polaris --train spica_train --out groups.csv
    python dedup.py clean-any my_training.csv --images /data/images --out cleaned/

clean   writes the training file without the rows whose (image, candidate) pair is in an
        evaluation set.  --variant relaxed also removes near-duplicates (see below).  --variant
        random or matched writes the control sets of the paper, which remove the same number of
        rows.  --against limits the evaluation sets considered.
groups  writes, for every sample of an evaluation set, whether its image-candidate pair is in the
        given training set (duplicated), only its image (image_only), or neither (unseen_image).
        Report a metric trained on that training set on the unseen_image samples, or train it on
        the cleaned file.

clean-any  does the same for any training file (CSV, JSON, JSON lines or parquet), for example a
        mixture of several collections or a later revision of a dataset.  It decodes every image
        once and matches by content, so it needs the images and Pillow.

clean and groups identify a row by its position in the released file and check every row against
a fingerprint before anything is written, so a file from another revision of a dataset is refused
instead of being filtered wrongly; use clean-any for such a file.

Two samples are the same if their images have identical decoded pixels and their candidate
captions are identical strings.  With --variant relaxed they are also the same if their images
have a SigLIP cosine of at least 0.97 (the same photograph stored twice, or another shot of the
same scene) and their captions are equal after lower-casing and removing punctuation, or have a
character similarity of at least 0.9.  The paper's experiments use the exact rule.

Requires Python 3.9+ and numpy; pyarrow for parquet files; Pillow for clean-any.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MANIFESTS = HERE / 'manifests'
GROUPS = ['unseen_image', 'image_only', 'duplicated']
REVISIONS = {'spica': ('hiranohachiman/Spica', 'c382725'),
             'nebula': ('Ka2ukiMatsuda/Nebula', 'a4c99b14818d834bca2b9a8b1fa3741deda94b47'),
             'polaris': ('yuwd/Polaris', 'e527aa82a79f08cd97792baf624853b80aecc81f')}
PARQUET_PARTS = {'data/train-*.parquet': 10, 'data/valid-*.parquet': 2, 'data/test-*.parquet': 2}

csv.field_size_limit(2 ** 31 - 1)


def fingerprint(identifier, candidate):
    return int.from_bytes(hashlib.sha1(f'{identifier}\x1f{candidate}'.encode()).digest()[:8], 'big')


def files_of(pattern):
    if '*' not in pattern:
        return [pattern]
    count = PARQUET_PARTS[pattern]
    return [pattern.replace('*', f'{part:05d}-of-{count:05d}') for part in range(count)]


def download(dataset, pattern, directory):
    repository, revision = REVISIONS[dataset]
    for name in files_of(pattern):
        target = directory / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/datasets/{repository}/resolve/{revision}/{name}'
        print('downloading', url, file=sys.stderr)
        partial = target.with_name(target.name + '.part')
        urllib.request.urlretrieve(url, partial)
        partial.rename(target)


def locate(directory, pattern):
    """The released files of one split, also when the directory is the folder that holds them."""
    paths = []
    for name in files_of(pattern):
        found = [p for p in (directory / name, directory / Path(name).name) if p.exists()]
        if not found:
            sys.exit(f'{directory / name} not found.  Point --data at the dataset directory or use --download.')
        paths.append(found[0])
    return paths


def column(header, name):
    if name not in header:
        sys.exit(f'no column {name!r}; the file has {header}.  Set --image-column / --caption-column.')
    return header.index(name)


class Csv:
    """A released CSV file (imgid, refs, mt, score), kept as parsed rows so they are written unchanged."""

    def __init__(self, path, image='imgid', candidate='mt'):
        with open(path, 'rb') as f:
            self.terminator = '\r\n' if f.readline().endswith(b'\r\n') else '\n'
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            self.header = next(reader)
            self.rows = list(reader)
        self.path = path
        image, candidate = column(self.header, image), column(self.header, candidate)
        self.keys = [(row[image], row[candidate]) for row in self.rows]

    def write(self, keep, directory, suffix):
        target = directory / f'{self.path.stem}_{suffix}.csv'
        with open(target, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, lineterminator=self.terminator)
            writer.writerow(self.header)
            writer.writerows(row for row, kept in zip(self.rows, keep) if kept)
        return [target]


class Parquet:
    """The parquet files of one Nebula split; images stay inside the files and are not decoded."""

    def __init__(self, paths, image='file_name', candidate='mt'):
        import pyarrow.parquet as pq
        self.pq, self.paths, self.keys, self.lengths = pq, paths, [], []
        for path in paths:
            table = pq.read_table(path, columns=[image, candidate])
            self.keys += list(zip(table[image].to_pylist(), table[candidate].to_pylist()))
            self.lengths.append(table.num_rows)

    def write(self, keep, directory, suffix):
        import pyarrow as pa
        targets, start = [], 0
        for path, length in zip(self.paths, self.lengths):
            table = self.pq.read_table(path).filter(pa.array(keep[start:start + length]))
            targets.append(directory / path.name.replace('-', f'_{suffix}-', 1))
            self.pq.write_table(table, targets[-1])
            start += length
        return targets


def load(name, kind, args):
    """The manifest of a released split, its description and the released rows, checked row by row."""
    index = json.loads((MANIFESTS / 'index.json').read_text())
    if name not in index[kind]:
        sys.exit(f'unknown {kind} split {name!r}; see `dedup.py list`')
    spec = index[kind][name]
    if 'file' not in spec:
        sys.exit(f'{name} is matched by sample_id, not by a released file; read manifests/{kind}_{name}.npz directly')
    manifest = np.load(MANIFESTS / f'{kind}_{name}.npz')
    if args.data is None and not args.download:
        sys.exit('give --data <directory of the released dataset> or --download')
    directory = Path(args.data) if args.data else Path(args.cache) / spec['dataset']
    if args.download:
        download(spec['dataset'], spec['file'], directory)
    paths = locate(directory, spec['file'])
    released = Csv(paths[0]) if spec['file'].endswith('.csv') else Parquet(paths)
    expected = manifest['fingerprint']
    if len(released.keys) != len(expected):
        sys.exit(f'{name}: {len(released.keys):,} rows found, {len(expected):,} expected.  '
                 f'The manifest was made for {" @ ".join(REVISIONS[spec["dataset"]])}.')
    found = np.fromiter((fingerprint(*key) for key in released.keys), np.uint64, len(expected))
    if not np.array_equal(found, expected):
        sys.exit(f'{name}: {int((found != expected).sum()):,} rows differ from the revision the manifest was made for '
                 f'({" @ ".join(REVISIONS[spec["dataset"]])}); nothing was written.')
    return index, spec, manifest, released


def clean(args):
    index, spec, manifest, released = load(args.split, 'training', args)
    sets = index['evaluation_sets']
    against = spec['against'] if args.against is None else args.against.split(',')
    unknown = sorted(set(against) - set(sets))
    if unknown:
        sys.exit(f'unknown evaluation sets {unknown}; choose from {sets}')
    if args.variant in ('dedup', 'relaxed'):
        if f'duplicated_in_{args.variant}' not in manifest and args.variant == 'relaxed':
            sys.exit(f'{args.split}: the manifests have no relaxed rule; update them')
        field = manifest['duplicated_in' if args.variant == 'dedup' else 'duplicated_in_relaxed']
        bits = sum(1 << sets.index(name) for name in against)
        keep = (field & np.uint16(bits)) == 0
    else:
        if args.against is not None:
            sys.exit('--against applies to the deduplicated set only; the control sets are fixed')
        if f'keep_{args.variant}' not in manifest:
            sys.exit(f'{args.split} has no {args.variant} control; available: '
                     f'{[k[5:] for k in manifest.files if k.startswith("keep_")]}')
        keep = manifest[f'keep_{args.variant}']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = released.write(keep.tolist(), out, args.variant)
    duplicated = manifest['duplicated_in']
    report = dict(split=args.split, variant=args.variant, rows=int(len(keep)), rows_kept=int(keep.sum()),
                  rows_removed=int((~keep).sum()), deduplicated_against=against if args.variant in ('dedup', 'relaxed') else None,
                  rows_duplicated_in={name: int(((duplicated >> bit) & 1).sum()) for bit, name in enumerate(sets)
                                      if ((duplicated >> bit) & 1).any()},
                  written=[str(path) for path in written])
    (out / f'{args.split}_{args.variant}.report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def groups(args):
    index, spec, manifest, released = load(args.split, 'evaluation', args)
    column = f'group_{args.train}'
    if column not in manifest:
        sys.exit(f'--train must be one of {[k[6:] for k in manifest.files if k.startswith("group_")]}')
    group = manifest[column]
    with (open(args.out, 'w', newline='') if args.out else sys.stdout) as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['row', 'group'])
        writer.writerows((row, GROUPS[value]) for row, value in enumerate(group))
    print({name: int((group == value).sum()) for value, name in enumerate(GROUPS)}, file=sys.stderr)


class Records:
    """A JSON or JSON-lines file holding one object per sample."""

    def __init__(self, path, image, candidate):
        text = path.read_text(encoding='utf-8')
        self.lines = path.suffix == '.jsonl'
        self.rows = [json.loads(line) for line in text.splitlines() if line.strip()] if self.lines else json.loads(text)
        self.path = path
        if not self.rows or image not in self.rows[0] or candidate not in self.rows[0]:
            sys.exit(f'{path}: expected a list of objects with the keys {image!r} and {candidate!r}')
        self.keys = [(row[image], row[candidate]) for row in self.rows]

    def write(self, keep, directory, suffix):
        target = directory / f'{self.path.stem}_{suffix}{self.path.suffix}'
        kept = [row for row, flag in zip(self.rows, keep) if flag]
        target.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in kept) + '\n' if self.lines
                          else json.dumps(kept, ensure_ascii=False), encoding='utf-8')
        return [target]


def pixel_hash(data):
    """sha1 of the decoded RGB pixels, so two encodings of one image get the same hash."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert('RGB')
    digest = hashlib.sha1(f'{rgb.width}x{rgb.height}|'.encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def hash_file(path):
    return pixel_hash(Path(path).read_bytes())


def image_hashes(identifiers, directory, workers):
    """Pixel hash of every distinct image; a file is found by its name with or without an extension."""
    by_name = {}
    for path in Path(directory).rglob('*'):
        if path.is_file():
            by_name.setdefault(path.name, path)
            by_name.setdefault(path.stem, path)
    names = sorted(set(identifiers))
    missing = [name for name in names if Path(str(name)).name not in by_name and Path(str(name)).stem not in by_name]
    if missing:
        sys.exit(f'{len(missing):,} of {len(names):,} images not found under {directory}, for example {missing[:3]}')
    paths = [str(by_name.get(Path(str(name)).name) or by_name[Path(str(name)).stem]) for name in names]
    with Pool(workers) as pool:
        return dict(zip(names, pool.map(hash_file, paths, chunksize=64)))


def clean_any(args):
    path = Path(args.file)
    index = json.loads((MANIFESTS / 'index.json').read_text())
    sets = index['evaluation_sets'] if args.against is None else args.against.split(',')
    content = np.load(MANIFESTS / 'evaluation_content.npz')
    unknown = sorted(name for name in sets if f'pairs_{name}' not in content)
    if unknown:
        sys.exit(f'unknown evaluation sets {unknown}; choose from {index["evaluation_sets"]}')
    if path.suffix == '.parquet':
        released = Parquet([path], args.image_column or 'file_name', args.caption_column)
        import pyarrow.parquet as pq
        images = pq.read_table(path, columns=['image'])['image'].to_pylist()
        hashes = {}
        for (name, _), image in zip(released.keys, images):
            if name not in hashes:
                hashes[name] = pixel_hash(image['bytes'])
    else:
        if args.images is None:
            sys.exit('--images <directory of the image files> is required for CSV and JSON files')
        image = args.image_column or 'imgid'
        released = Csv(path, image, args.caption_column) if path.suffix == '.csv' else Records(path, image, args.caption_column)
        hashes = image_hashes([name for name, _ in released.keys], args.images, args.workers)

    def key(name, candidate):
        digest = hashlib.sha1(f'{hashes[name]}\x1f{candidate}'.encode()).digest()
        return int.from_bytes(digest[:8], 'big'), int.from_bytes(digest[8:16], 'big')

    pairs = np.array([key(*row) for row in released.keys], np.uint64).view([('a', np.uint64), ('b', np.uint64)]).ravel()
    pictures = np.array([int(hashes[name][:16], 16) for name, _ in released.keys], np.uint64)
    duplicated, shared_image = {}, {}
    for name in sets:
        known = np.ascontiguousarray(content[f'pairs_{name}']).view([('a', np.uint64), ('b', np.uint64)]).ravel()
        duplicated[name] = np.isin(pairs, known)
        shared_image[name] = np.isin(pictures, content[f'images_{name}'])
    keep = ~np.logical_or.reduce(list(duplicated.values()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = released.write(keep.tolist(), out, 'dedup')
    report = dict(file=str(path), rows=int(len(keep)), rows_kept=int(keep.sum()), rows_removed=int((~keep).sum()),
                  images=len(hashes), distinct_images=len(set(hashes.values())), deduplicated_against=sets,
                  rows_duplicated_in={name: int(hit.sum()) for name, hit in duplicated.items() if hit.any()},
                  rows_kept_whose_image_is_in={name: int((hit & keep).sum()) for name, hit in shared_image.items() if (hit & keep).any()},
                  written=[str(target) for target in written])
    (out / f'{path.stem}_dedup.report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def listing(_):
    index = json.loads((MANIFESTS / 'index.json').read_text())
    print('training splits (clean):')
    for name, spec in index['training'].items():
        print(f"  {name:14s} {spec['rows']:>8,} rows   kept: " + ', '.join(f'{k} {v:,}' for k, v in spec['kept'].items()))
    print('evaluation splits (groups), share of duplicated samples per training set:')
    for name, spec in index['evaluation'].items():
        shares = ', '.join(f"{pool} {100 * g['duplicated'] / spec['samples']:.1f}%" for pool, g in spec['groups'].items())
        print(f"  {name:16s} {spec['samples']:>7,} samples   {shares}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split('\n\n', 2)[2])
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='the splits covered by the manifests').set_defaults(run=listing)
    for name, run, text in (('clean', clean, 'write a training file without evaluation samples'),
                            ('groups', groups, 'label the samples of an evaluation set')):
        command = commands.add_parser(name, help=text)
        command.add_argument('split')
        command.add_argument('--data', help='directory of the released dataset')
        command.add_argument('--download', action='store_true', help='fetch the released files at the pinned revision')
        command.add_argument('--cache', default='downloads', help='where --download stores files (default: downloads)')
        command.set_defaults(run=run)
        if name == 'clean':
            command.add_argument('--out', default='cleaned', help='output directory (default: cleaned)')
            command.add_argument('--variant', choices=['dedup', 'relaxed', 'random', 'matched'], default='dedup')
            command.add_argument('--against', help='comma-separated evaluation sets (default: those used in the paper)')
        else:
            command.add_argument('--train', required=True, help='training split the metric was trained on')
            command.add_argument('--out', help='CSV file to write (default: standard output)')
    command = commands.add_parser('clean-any', help='clean any training file by content (decodes the images)')
    command.add_argument('file', help='CSV, JSON, JSON-lines or parquet file with one sample per row')
    command.add_argument('--images', help='directory of the image files named in the image column')
    command.add_argument('--image-column', help='default: imgid (file_name for parquet)')
    command.add_argument('--caption-column', default='mt', help='column of the candidate caption (default: mt)')
    command.add_argument('--against', help='comma-separated evaluation sets (default: all)')
    command.add_argument('--out', default='cleaned', help='output directory (default: cleaned)')
    command.add_argument('--workers', type=int, default=min(32, os.cpu_count() or 1))
    command.set_defaults(run=clean_any)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
