"""Verify every picture the Pearl CSVs reference opens, once, in parallel.

Pearl's dataset normally runs this check per row at the start of every run; the
training runs skip it (PEARL_SKIP_IMAGES=1) and rely on this instead.
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

DATA = Path((_ROOT + '/pearl_data/images'))


def check(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except Exception as error:
        return f'{path}: {error}'


def main():
    paths = sorted(DATA.iterdir())
    with Pool(32) as pool:
        bad = [r for r in pool.imap_unordered(check, paths, chunksize=64) if r]
    print('pictures', len(paths), 'unreadable', len(bad))
    for line in bad[:20]:
        print(line)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
