#!/usr/bin/env python3
"""Put the human scores into the row files, so that the analysis can run.

The repository holds, for every evaluation and training set, one row file (rows/<set>.npz) with the image and caption
indices of each sample but without its human score: the scores belong to the datasets, which we
do not redistribute.  This script reads them from the official releases at the pinned revisions
and writes the complete row files to feature_cache_v2/.

    python setup_data.py                    # download what is needed (about 190 MB of traffic)
    python setup_data.py --polaris DIR --spica DIR --nebula DIR --composite FILE --flickr8k DIR
    python setup_data.py --labels DIR       # read labels/<set>.npy instead, if you have that directory

Only score columns are read.  For the parquet files of Nebula and Composite, which hold the images,
that is done with HTTP range requests, so the images are not downloaded.  Every vector of scores is
compared with a SHA-256 digest stored in rows/index.json; a set whose scores differ is refused.

Requires numpy; pyarrow for Nebula and Composite.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROWS = HERE / 'rows'
OUT = HERE / 'feature_cache_v2'
HUB = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/') + '/datasets/{}/resolve/{}/{}'
SOURCES = {
    'polaris': ('yuwd/Polaris', 'e527aa82a79f08cd97792baf624853b80aecc81f'),
    'spica': ('hiranohachiman/Spica', 'c382725'),
    'nebula': ('Ka2ukiMatsuda/Nebula', 'a4c99b14818d834bca2b9a8b1fa3741deda94b47'),
    'composite': ('yuwd/Composite', 'd6c63cbe948daa123550d8b6a206c323c5cdde3f'),
}
# Flickr8k-Expert and Flickr8k-CF in the format prepared by the PAC-S authors (Sarto et al., 2023),
# which the Polos repository points to; this public copy is the one in the EXPERT repository.
FLICKR8K = 'https://raw.githubusercontent.com/hjkim811/EXPERT/7fbfb00ae8d6cb214ab55c067880ce7acf730efe/annotations/{}'
NEBULA_PARTS = {'train': 10, 'valid': 2, 'test': 2}

csv.field_size_limit(2 ** 31 - 1)


def request(url, headers=None, method='GET', attempts=6):
    """The body (or, for HEAD, the headers) of a URL; connection errors are retried."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}, method=method), timeout=120) as response:
                return response.headers if method == 'HEAD' else response.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code < 500 or attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))


class HttpFile(io.RawIOBase):
    """A remote file read with range requests, enough for pyarrow to fetch single parquet columns."""

    def __init__(self, url):
        self.url, self.position, self.blocks = url, 0, []
        self.size = int(request(url, method='HEAD')['Content-Length'])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def load(self, start, end):
        data = request(self.url, headers={'Range': f'bytes={start}-{end - 1}'})
        self.blocks.append((start, data))
        return data

    def prefetch(self, ranges, threads=16):
        """Fetch several byte ranges at once; later reads inside them need no request."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(lambda r: self.load(*r), ranges))

    def read(self, size=-1):
        end = self.size if size is None or size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b''
        for start, data in self.blocks:
            if start <= self.position and end <= start + len(data):
                break
        else:
            start, data = self.position, self.load(self.position, end)
        data = data[self.position - start:end - start]
        self.position = end
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def remote_column(url, column):
    """One column of a remote parquet file: the footer, then the chunks of that column in parallel."""
    import pyarrow.parquet as pq
    file = HttpFile(url)
    file.load(max(0, file.size - (1 << 20)), file.size)
    parquet = pq.ParquetFile(file)
    ranges = []
    for group in range(parquet.metadata.num_row_groups):
        for index in range(parquet.metadata.num_columns):
            chunk = parquet.metadata.row_group(group).column(index)
            if chunk.path_in_schema == column:
                start = min(offset for offset in (chunk.dictionary_page_offset, chunk.data_page_offset) if offset)
                ranges.append((start, start + chunk.total_compressed_size))
    file.prefetch(ranges)
    return parquet.read(columns=[column])[column].to_pylist()


def fetch(url):
    print('  reading', url, file=sys.stderr)
    return request(url)


def csv_scores(local, dataset, name):
    """The score column of a released CSV file (imgid, refs, mt, score), in file order."""
    text = (Path(local) / name).read_text(encoding='utf-8') if local else fetch(HUB.format(*SOURCES[dataset], name)).decode()
    return [float(row['score']) for row in csv.DictReader(io.StringIO(text, newline=''))]


def parquet_scores(local, dataset, names, column):
    import pyarrow.parquet as pq
    scores = []
    for name in names:
        if local:
            path = Path(local)
            source = path if path.is_file() else next(p for p in (path / name, path / Path(name).name) if p.exists())
            scores += pq.read_table(source, columns=[column])[column].to_pylist()
        else:
            print('  reading the score column of', HUB.format(*SOURCES[dataset], name), file=sys.stderr)
            scores += remote_column(HUB.format(*SOURCES[dataset], name), column)
    return scores


def flickr8k_scores(local, name):
    data = json.loads((Path(local) / name).read_text() if local else fetch(FLICKR8K.format(name)))
    return {f'{key}:{index}': float(judgment['rating'])
            for key, item in data.items() for index, judgment in enumerate(item['human_judgement'])}


class Official:
    """Human scores of the official releases, read once per file."""

    def __init__(self, args):
        self.args, self.cache = args, {}

    def table(self, family, split):
        key = family, split
        if key not in self.cache:
            a = self.args
            if family == 'polaris':
                self.cache[key] = csv_scores(a.polaris, 'polaris', f'polaris_{split}.csv')
            elif family == 'spica':
                self.cache[key] = csv_scores(a.spica, 'spica', f'spica_{split}.csv')
            elif family == 'nebula':
                count = NEBULA_PARTS[split]
                names = [f'data/{split}-{part:05d}-of-{count:05d}.parquet' for part in range(count)]
                self.cache[key] = parquet_scores(a.nebula, 'nebula', names, 'human_score')
            elif family == 'composite':
                self.cache[key] = parquet_scores(a.composite, 'composite', ['data/test-00000-of-00001.parquet'], a.composite_column)
            else:
                self.cache[key] = flickr8k_scores(a.flickr8k, 'flickr8k.json' if split == 'expert' else 'crowdflower_flickr8k.json')
        return self.cache[key]

    def scores(self, source, ids):
        family, split = source
        table = self.table(family, split)
        if family == 'flickr8k':
            return [table[i] for i in ids]
        if family == 'composite':
            return [table[int(i.split('_')[1])] for i in ids]
        return [table[int(i.split(':')[-1])] for i in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--polaris', help='directory with polaris_{train,val,test}.csv')
    parser.add_argument('--spica', help='directory with spica_{train,val,test}.csv')
    parser.add_argument('--nebula', help='directory of the Nebula release (holds data/*.parquet)')
    parser.add_argument('--composite', help='the parquet file of yuwd/Composite, or its directory')
    parser.add_argument('--composite-column', default='human_score')
    parser.add_argument('--flickr8k', help='directory with flickr8k.json and crowdflower_flickr8k.json (PAC-S format)')
    parser.add_argument('--labels', help='directory with <set>.npy score vectors, used instead of the official releases')
    args = parser.parse_args()

    index = json.loads((ROWS / 'index.json').read_text())
    official = Official(args)
    OUT.mkdir(exist_ok=True)
    failed = []
    for name, spec in index['sets'].items():
        with np.load(ROWS / f'{name}.npz') as f:
            rows = {key: f[key] for key in f.files}
        if spec['source'] is None:                       # pairwise sets carry no score
            target = np.full(spec['rows'], np.nan, np.float32)
        elif args.labels:
            target = np.load(Path(args.labels) / f'{name}.npy')
        else:
            target = np.array(official.scores(spec['source'], rows['ids'].tolist()), np.float32)
        if spec['source'] is not None and hashlib.sha256(np.ascontiguousarray(target, np.float32).tobytes()).hexdigest() != spec['sha256']:
            failed.append(name)
            print(f'{name:28s} MISMATCH: the scores differ from those used in the paper; row file not written')
            continue
        np.savez(OUT / f'{name}.npz', **rows, target=target)
        print(f'{name:28s} {spec["rows"]:>7,} rows  ok')
    for path in ROWS.iterdir():                          # picture identities and pair lists, used as they are
        if path.suffix in ('.npy', '.json') and path.name != 'index.json':
            (OUT / path.name).write_bytes(path.read_bytes())
    if failed:
        sys.exit(f'{len(failed)} sets failed: {failed}')
    print('row files written to', OUT)


if __name__ == '__main__':
    main()
