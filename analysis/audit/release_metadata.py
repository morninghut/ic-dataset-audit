"""Record how each dataset and code repository is licensed and pinned.

Licensing matters for the paper: without a license we may publish filter lists,
hashes and scripts, but not redistribute the data itself.
"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, write_json

DATASETS = {
    'spica': 'hiranohachiman/Spica',
    'nebula': 'Ka2ukiMatsuda/Nebula',
    'polaris': 'yuwd/Polaris',
    'composite': 'yuwd/Composite',
}
MODELS_AND_CODE = {
    'pearl_code': ('https://api.github.com/repos/hiranohachiman/Pearl', 'github'),
    'deneb_code': ('https://api.github.com/repos/Ka2ukiMatsuda/DENEB', 'github'),
    'polos_code': ('https://api.github.com/repos/keio-smilab24/Polos', 'github'),
}


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'spica-audit'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    report = {}
    for name, repo in DATASETS.items():
        try:
            info = fetch(f'https://huggingface.co/api/datasets/{repo}')
            report[name] = dict(
                kind='huggingface_dataset', repo=repo, sha=info.get('sha'),
                license=(info.get('cardData') or {}).get('license'),
                tags=[t for t in info.get('tags', []) if t.startswith('license')],
                gated=info.get('gated'), downloads=info.get('downloads'),
                last_modified=info.get('lastModified'),
                files=[s['rfilename'] for s in info.get('siblings', [])][:40])
        except Exception as error:
            report[name] = dict(kind='huggingface_dataset', repo=repo, error=repr(error))
    for name, (url, kind) in MODELS_AND_CODE.items():
        try:
            info = fetch(url)
            report[name] = dict(kind=kind, url=info.get('html_url'),
                                license=(info.get('license') or {}).get('spdx_id'),
                                pushed_at=info.get('pushed_at'), stars=info.get('stargazers_count'))
        except Exception as error:
            report[name] = dict(kind=kind, url=url, error=repr(error))
    write_json(OUT / 'release_metadata.json', report)
    for name, entry in report.items():
        print(f"{name:14s} license={entry.get('license')!r} {entry.get('error', '')}")


if __name__ == '__main__':
    main()
