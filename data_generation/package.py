#!/usr/bin/env python3
"""Build the training package ZIP from regenerated samples.

The output has the layout mdlARC's family loader reads
(dataset_building_scripts/family_data.py, kind "family_sampled"):

  family_catalog.json          catalog (schema arc-family-catalog-v1)
  instances/<sha256>.json      one member per puzzle, named by its SHA-256

Families, instance records, ancestors, sampling omissions and the catalog
header are rebuilt from manifest/; puzzle bytes come from the sample.py
output and the original ARC-AGI-2 tasks. The released package also pinned
SHA-256 hashes of the private authoring records (catalog key
"metadata_sha256"). Those records are not published, so by default this
field pins the public generator files instead; pass --metadata-from with the
released package to copy the original pins (this reproduces the released
family_catalog.json byte for byte when every sample matches).

Standard library + NumPy only (NumPy is not used here).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sample  # noqa: E402

GROUPS = {'arc2_train': 'arc2/training', 'arc2_eval': 'arc2/evaluation',
          'conceptarc': 'conceptarc/training', 'arc1_filtered': 'arc1-filtered/training'}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_member(archive, name, raw):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, raw)


class Originals:
    """Original ARC-AGI-2 task bytes by family id, from data/ or a package ZIP."""

    def __init__(self, path):
        self.path = Path(path)
        self.archive = None if self.path.is_dir() else zipfile.ZipFile(self.path)
        self.catalog = json.loads(self.archive.read('family_catalog.json')) if self.archive else None

    def bytes_for(self, family, expected):
        if self.archive is None:
            source, task = family.split('/', 1)
            split = {'arc2_train': 'training', 'arc2_eval': 'evaluation'}[source]
            data = json.loads((self.path / split / f'{task}.json').read_text())
            return json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
        return self.archive.read(f'instances/{expected}.json')


def family_relative(record, release_name):
    return (f'data/compositions/releases/{release_name}/families/'
            + hashlib.sha256(record['id'].encode()).hexdigest()[:24] + '.zip')


def build_instances(record, header, release_name, sample_archive, originals, allow_mismatch, problems):
    instances, payloads = [], []
    if record['kind'] == 'variation':
        original = record['original']
        raw = originals.bytes_for(record['id'], original['sha256'])
        if sha256(raw) != original['sha256']:
            problems.append(f"{record['id']}: original task bytes differ from the manifest")
        puzzle = json.loads(raw)
        instances.append({'family': record['id'], 'kind': 'original', 'origin': 'original',
                          'pair_count': len(puzzle['train']) + len(puzzle['test']),
                          'path': f'instances/{sha256(raw)}.json', 'seed': None, 'sha256': sha256(raw)})
        payloads.append(raw)
    with zipfile.ZipFile(sample_archive) as archive:
        report = json.loads(archive.read('records.json'))
        produced = {r['seed']: r for r in report['records']}
        if report.get('valid') is not True or report.get('generator_sha256') != record['generator']['sha256']:
            problems.append(f"{record['id']}: sample report is not valid for the pinned generator")
        for seed, demos, queries, expected in record['samples']:
            r = produced.get(seed)
            if r is None:
                problems.append(f"{record['id']}: seed {seed} missing from samples")
                continue
            raw = archive.read(r['member'])
            digest = sha256(raw)
            if digest != expected:
                problems.append(f"{record['id']}: seed {seed} bytes differ from the manifest")
                if not allow_mismatch:
                    continue
            puzzle = json.loads(raw)
            pairs = len(puzzle['train']) + len(puzzle['test'])
            if record['kind'] == 'variation':
                source, task = record['id'].split('/', 1)
                item = {'kind': 'variation', 'origin': 'synth', 'pair_count': pairs,
                        'path': f'instances/{digest}.json',
                        'relative': f'results/task-variations/{GROUPS[source]}/{task}/samples/seed-{seed:010d}.json',
                        'seed': seed, 'sha256': digest}
            else:
                item = {'archive_member': f'seed-{seed:010d}.json',
                        'composition_release': header['discovery']['composition_release'],
                        'generator_sha256': record['generator']['sha256'], 'kind': 'composition',
                        'num_test': len(puzzle['test']), 'num_train': len(puzzle['train']), 'origin': 'synth',
                        'pair_count': pairs, 'path': f'instances/{digest}.json',
                        'relative': family_relative(record, release_name), 'seed': seed, 'sha256': digest,
                        'stage': record['generator']['stage']}
            instances.append(item)
            payloads.append(raw)
    entry = {'ancestors': record['ancestors'], 'instances': instances}
    if record['kind'] == 'composition' and record['sampling']['omitted']:
        entry['sampling_omissions'] = [{**o, 'family': record['id']} for o in record['sampling']['omitted']]
    return entry, payloads


def public_pins(root, families):
    pins = {'release:manifest/families.jsonl': file_sha(root / 'manifest/families.jsonl'),
            'release:manifest/release.json': file_sha(root / 'manifest/release.json')}
    for record in families:
        pins['release:' + record['generator']['path']] = record['generator']['sha256']
    return pins


def compare(new_path, reference_path):
    with zipfile.ZipFile(new_path) as new, zipfile.ZipFile(reference_path) as ref:
        a, b = json.loads(new.read('family_catalog.json')), json.loads(ref.read('family_catalog.json'))
        keys = sorted(set(a) | set(b))
        differing_keys = [k for k in keys if a.get(k) != b.get(k)]
        fam_a, fam_b = a['families'], b['families']
        differing_families = [f for f in sorted(set(fam_a) | set(fam_b)) if fam_a.get(f) != fam_b.get(f)]
        names_a, names_b = set(new.namelist()), set(ref.namelist())
        shared = sorted((names_a & names_b) - {'family_catalog.json'})
        byte_diffs = [n for n in shared if new.read(n) != ref.read(n)]
        return {'catalog_keys_differing': differing_keys,
                'families_new': len(fam_a), 'families_reference': len(fam_b),
                'families_differing': len(differing_families), 'first_differing_families': differing_families[:5],
                'instances_new': len(names_a) - 1, 'instances_reference': len(names_b) - 1,
                'instances_only_in_new': len(names_a - names_b), 'instances_only_in_reference': len(names_b - names_a),
                'instance_bytes_differing': len(byte_diffs),
                'catalog_bytes_identical': new.read('family_catalog.json') == ref.read('family_catalog.json'),
                'zip_bytes_identical': file_sha(new_path) == file_sha(reference_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=HERE)
    parser.add_argument('--samples', type=Path, default=HERE / 'samples', help='output folder of sample.py')
    parser.add_argument('--originals', type=Path, required=True,
                        help='ARC-AGI-2 data/ folder (training/, evaluation/) or the released package ZIP')
    parser.add_argument('--out', type=Path, required=True, help='new package ZIP (must not exist)')
    parser.add_argument('--metadata-from', type=Path, help='released package ZIP: copy its metadata_sha256 pins')
    parser.add_argument('--compare', type=Path, help='released package ZIP to compare with')
    parser.add_argument('--allow-mismatch', action='store_true',
                        help='package regenerated bytes even when they differ from the manifest')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; choose a new path')
    families, release = sample.load_manifest(root)
    header = release['catalog']
    selected = sorted((r for r in families if r['in_package']), key=lambda r: r['id'])
    originals = Originals(args.originals)
    samples_root = args.samples.resolve()
    absent = [r['id'] for r in selected if not sample.output_path(samples_root, r).is_file()]
    if absent:
        raise SystemExit(f'{len(absent)} in-package families have not been sampled yet, e.g. {absent[:3]}')
    problems = []
    catalog = {k: v for k, v in header.items()}
    catalog['families'] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent, prefix='.package-') as temporary:
        pending = Path(temporary) / 'package.zip'
        with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            written = set()
            for record in selected:
                entry, payloads = build_instances(record, header, release['release'],
                                                  sample.output_path(samples_root, record), originals,
                                                  args.allow_mismatch, problems)
                for item, raw in zip(entry['instances'], payloads):
                    if item['path'] not in written:
                        write_member(archive, item['path'], raw)
                        written.add(item['path'])
                catalog['families'][record['id']] = entry
            summary = catalog['summary']
            rebuilt = {'instances_per_epoch': len(catalog['families']),
                       'candidate_instances': sum(len(f['instances']) for f in catalog['families'].values()),
                       'saved_composition_instances': sum(len(f['instances']) for k, f in catalog['families'].items()
                                                          if k.startswith('composition/'))}
            for key, value in rebuilt.items():
                if summary.get(key) != value:
                    problems.append(f'summary {key}: manifest {summary.get(key)}, rebuilt {value}')
            if args.metadata_from:
                with zipfile.ZipFile(args.metadata_from) as reference:
                    catalog['metadata_sha256'] = json.loads(reference.read('family_catalog.json'))['metadata_sha256']
            else:
                catalog['metadata_sha256'] = public_pins(root, families)
            write_member(archive, 'family_catalog.json', encoded(catalog))
        if problems and not args.allow_mismatch:
            for p in problems[:20]:
                print('PROBLEM', p)
            raise SystemExit(f'{len(problems)} problems; package not written')
        pending.replace(args.out)
    result = {'package': str(args.out), 'bytes': args.out.stat().st_size, 'sha256': file_sha(args.out),
              'families': len(catalog['families']), 'instances': len(written), 'problems': len(problems),
              'spec_catalog': {'archive_path': 'assets_tmp/' + args.out.name, 'size': args.out.stat().st_size,
                               'sha256': file_sha(args.out)}}
    print(json.dumps(result, indent=1), flush=True)
    if args.compare:
        print(json.dumps({'comparison': compare(args.out, args.compare)}, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
