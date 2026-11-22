"""Shared paths and helpers for the Spica / Nebula data audit."""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import hashlib
import json
from pathlib import Path

ROOT = Path(_ROOT)
OUT = ROOT / 'analysis/audit'
SPICA = ROOT / 'datasets_spica'
NEBULA = ROOT / 'datasets_nebula'
DATA = Path((_POLOS_HOME + '/data_en'))
CACHE = ROOT / 'feature_cache_v1'


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True))
    return obj


def read_json(path):
    return json.loads(Path(path).read_text())


def md5(data):
    return hashlib.md5(data).hexdigest()


def pixel_hash(image):
    """Hash of decoded RGB pixels, so re-encoded copies of one picture collide."""
    rgb = image.convert('RGB')
    h = hashlib.sha1()
    h.update(f'{rgb.width}x{rgb.height}|'.encode())
    h.update(rgb.tobytes())
    return h.hexdigest()
