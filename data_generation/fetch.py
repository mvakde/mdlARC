#!/usr/bin/env python3
"""Download the data-generation Release files from GitHub and verify SHA-256.

Files are fetched from
  https://github.com/mvakde/mdlARC/releases/download/<tag>/<name>
Each download is checked against the hashes pinned below and against the
SHA256SUMS file published with the Release. Existing files with the right
hash are kept. Standard library only.

  python fetch.py                       # both ZIPs into assets_tmp/ (where training reads the package)
  python fetch.py --name mdlarc-generators-v3.1.zip --dest downloads
  python fetch.py --name mdlarc-d1024-e440.pt     # the 37.08 Kaggle model's checkpoint (910 MB)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
DEFAULT_TAG = 'v3.1-nov2026'
BASE_URL = 'https://github.com/mvakde/mdlARC/releases/download'
PACKAGE = 'arc2-family-full-counts-v1-20261024-r1.zip'
GENERATORS = 'mdlarc-generators-v3.1.zip'
CHECKPOINT = 'mdlarc-d1024-e440.pt'
# Pinned hashes. The package hash is also pinned in manifest/release.json and
# in the mdlARC training spec that used it.
KNOWN_SHA256 = {
    PACKAGE: '51d8450883db3467d738b5dbebcd60a763e55fac91335b0afcf0d91f79a3a809',
    GENERATORS: '5e3ecd55f98c5b5b77b90990104f9943c0706df3e8251cbffe8c5035d03eba99',
    CHECKPOINT: '328f80bc60ba8105455fdd5f0eed921afdbfb22a84880d63e250b75dc3a38cfc',
}
USER_AGENT = 'mdlarc-fetch/1.0'


def url_for(base, tag, name):
    return f'{base.rstrip("/")}/{tag}/{name}'


def open_url(url):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    return urllib.request.urlopen(request, timeout=60)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def published_sums(base, tag):
    try:
        with open_url(url_for(base, tag, 'SHA256SUMS')) as response:
            text = response.read().decode()
    except OSError as error:
        print(f'SHA256SUMS not available ({error}); using pinned hashes only', flush=True)
        return {}
    sums = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and len(parts[0]) == 64:
            sums[parts[1].lstrip('*')] = parts[0].lower()
    return sums


def expected_hash(name, sums):
    pinned = KNOWN_SHA256.get(name)
    if pinned and len(pinned) != 64:
        pinned = None
    manifest = HERE / 'manifest' / 'release.json'
    if name == PACKAGE and manifest.is_file():
        recorded = json.loads(manifest.read_text())['package']['sha256']
        if pinned and recorded != pinned:
            raise SystemExit('Pinned package hash differs from manifest/release.json')
        pinned = recorded
    published = sums.get(name)
    if pinned and published and pinned != published:
        raise SystemExit(f'{name}: SHA256SUMS ({published}) differs from the pinned hash ({pinned})')
    return pinned or published


def download(base, tag, name, dest, expected):
    target = dest / name
    if target.is_file() and expected and file_sha(target) == expected:
        print(f'{name}: already present and verified', flush=True)
        return target
    pending = dest / (name + '.part')
    digest, size = hashlib.sha256(), 0
    url = url_for(base, tag, name)
    print(f'Downloading {url}', flush=True)
    with open_url(url) as response, pending.open('wb') as handle:
        total = int(response.headers.get('Content-Length') or 0)
        next_report = 0
        for block in iter(lambda: response.read(1 << 20), b''):
            handle.write(block)
            digest.update(block)
            size += len(block)
            if total and size >= next_report:
                print(f'  {size / 1e6:.1f} / {total / 1e6:.1f} MB', flush=True)
                next_report += max(total // 10, 1 << 20)
    actual = digest.hexdigest()
    if expected and actual != expected:
        pending.unlink()
        raise SystemExit(f'{name}: SHA-256 mismatch (expected {expected}, got {actual})')
    pending.replace(target)
    status = 'verified' if expected else 'NOT verified (no hash available)'
    print(f'{name}: {size} bytes, sha256 {actual} {status}', flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', default=DEFAULT_TAG)
    parser.add_argument('--dest', type=Path, default=HERE.parent / 'assets_tmp')
    parser.add_argument('--name', action='append', help=f'file to fetch (default: {PACKAGE} and {GENERATORS})')
    parser.add_argument('--base-url', default=BASE_URL)
    parser.add_argument('--require-hash', action='store_true', help='refuse files without a known hash')
    args = parser.parse_args()
    names = args.name or [PACKAGE, GENERATORS]
    args.dest.mkdir(parents=True, exist_ok=True)
    sums = published_sums(args.base_url, args.tag)
    for name in names:
        if '/' in name or name in ('', '.', '..'):
            raise SystemExit(f'Invalid file name: {name}')
        expected = expected_hash(name, sums)
        if not expected and args.require_hash:
            raise SystemExit(f'{name}: no pinned or published hash')
        download(args.base_url, args.tag, name, args.dest, expected)
    return 0


if __name__ == '__main__':
    sys.exit(main())
