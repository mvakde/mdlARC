"""Read an explicit arc-synth-data composition release without generator execution."""
import json
from pathlib import Path
import re
import zipfile


def validate_identity(identity):
    if not isinstance(identity, dict) or set(identity) != {'path', 'sha256'}:
        raise ValueError('composition_release requires path and sha256')
    p = Path(identity['path'])
    if (p.is_absolute() or '..' in p.parts or '\\' in str(p) or
            p.parts[:3] != ('data', 'compositions', 'releases') or p.name != 'release.json'
            or not re.fullmatch('[0-9a-f]{64}', identity['sha256'])):
        raise ValueError('Invalid composition release identity')


def load_release(synth, identity, pins):
    from .family_data import regular, sha, file_sha
    validate_identity(identity)
    path = regular(synth, identity['path']); base = path.parent
    raw = path.read_bytes()
    if sha(raw) != identity['sha256']:
        raise ValueError('Composition release pin changed')
    summary = json.loads(raw)
    inventory_path = base / 'inventory.json'; raw_inventory = inventory_path.read_bytes()
    if (summary.get('schema') not in {'arc-composition-resample-v1', 'arc-composition-resample-v2'} or summary.get('ready') is not True
            or summary.get('failures') or sha(raw_inventory) != summary['inventory_sha256']):
        raise ValueError('Incomplete or corrupt composition release')
    inventory = json.loads(raw_inventory)
    if (inventory.get('schema') != 'arc-composition-resample-v1' or inventory.get('release') != summary['release']
            or set(inventory['families']) != set(summary['families'])
            or len(inventory['families']) != inventory['counts']['families']):
        raise ValueError('Composition release inventory differs')
    omissions = {}
    expected_counts = json.loads(json.dumps(inventory['counts'], separators=(',', ':')))
    if summary['schema'] == 'arc-composition-resample-v2':
        exception = summary['selection_exceptions']
        if set(exception) != {'path', 'sha256'} or exception['path'] != 'omissions.json':
            raise ValueError('Invalid composition omission identity')
        omission_path = regular(base, exception['path'])
        raw_omissions = omission_path.read_bytes()
        if sha(raw_omissions) != exception['sha256'] or summary['requested_counts'] != inventory['counts']:
            raise ValueError('Composition omission pin changed')
        record = json.loads(raw_omissions)
        if record['inventory_sha256'] != summary['inventory_sha256']:
            raise ValueError('Omissions reference another inventory')
        limits = record.get('family_omission_limits', {})
        default_limit = record.get('default_family_omission_limit', 1)
        if default_limit not in (1, 2):
            raise ValueError('Invalid general family omission limit')
        if any(family not in inventory['families'] or limit != 2 for family, limit in limits.items()):
            raise ValueError('Invalid explicit family omission limits')
        seen_seeds = set()
        for item in record['omissions']:
            family = item['family']
            if family not in inventory['families'] or len(omissions.get(family, [])) >= limits.get(family, default_limit):
                raise ValueError('Family omission limit exceeded')
            row = [item['seed'], item['num_train'], item['num_test']]
            if (family, item['seed']) in seen_seeds or row not in inventory['families'][family]['schedule']:
                raise ValueError('Omitted seed/counts differ from source schedule')
            seen_seeds.add((family, item['seed']))
            omissions.setdefault(family, []).append(item)
        pins['synth:' + omission_path.relative_to(synth).as_posix()] = exception['sha256']
        expected_counts['samples'] -= len(record['omissions'])
        for item in record['omissions']:
            for role, key in [('demos', 'num_train'), ('queries', 'num_test')]:
                expected_counts['distribution'][role][str(item[key])] -= 1
                if expected_counts['distribution'][role][str(item[key])] == 0:
                    expected_counts['distribution'][role].pop(str(item[key]))
    if summary['counts'] != expected_counts:
        raise ValueError('Composition selected counts differ from frozen inventory/authorized omissions')
    pins['synth:' + identity['path']] = identity['sha256']
    pins['synth:' + inventory_path.relative_to(synth).as_posix()] = sha(raw_inventory)
    for relative, expected in inventory['source_sha256'].items():
        if file_sha(regular(synth, relative)) != expected:
            raise ValueError('Frozen composition source changed: ' + relative)
        pins['synth:' + relative] = expected
    families = {}; total = 0
    for family, source in inventory['families'].items():
        record = summary['families'][family]
        omitted = omissions.get(family, [])
        omitted_seeds = {item['seed'] for item in omitted}
        expected_schedule = [row for row in source['schedule'] if row[0] not in omitted_seeds]

        archive = regular(base, record['archive'])
        if archive.stat().st_size != record['archive_bytes'] or file_sha(archive) != record['archive_sha256']:
            raise ValueError('Composition family archive changed: ' + family)
        relative = archive.relative_to(synth).as_posix()
        pins['synth:' + relative] = record['archive_sha256']
        with zipfile.ZipFile(archive) as handle:
            report = json.loads(handle.read('records.json'))
            if (report.get('valid') is not True or report['family'] != family
                    or report['generator_sha256'] != source['generator_sha256'] or report['stage'] != source['stage']
                    or [[r['seed'], r['num_train'], r['num_test']] for r in report['records']] != expected_schedule
                    or [r[0] for r in source['schedule']] != source['seeds']
                    or len(handle.namelist()) != len(report['records']) + 1
                    or len(handle.namelist()) != len(set(handle.namelist()))):
                raise ValueError('Composition sample schedule differs: ' + family)
        items = []
        for r in report['records']:
            if (not (2 <= r['num_train'] <= 9 and 1 <= r['num_test'] <= 3 and r['num_train'] > r['num_test'])
                    or r['member'] != f"seed-{r['seed']:010d}.json" or not re.fullmatch('[0-9a-f]{64}', r['sha256'])):
                raise ValueError('Invalid composition sample record')
            items.append({'origin': 'synth', 'relative': relative, 'archive_member': r['member'],
                'sha256': r['sha256'], 'seed': r['seed'], 'kind': 'composition', 'stage': source['stage'],
                'num_train': r['num_train'], 'num_test': r['num_test'], 'generator_sha256': source['generator_sha256'],
                'composition_release': identity})
        total += len(items)
        families[family] = {'source': source, 'instances': items, 'omissions': omitted}
    if total != summary['counts']['samples']:
        raise ValueError('Composition release sample total differs')
    return families


def puzzle_bytes(synth, item):
    from .family_data import regular, sha
    with zipfile.ZipFile(regular(synth, item['relative'])) as archive:
        raw = archive.read(item['archive_member'])
    if sha(raw) != item['sha256']:
        raise ValueError('Composition sample changed')
    task = json.loads(raw)
    if set(task) != {'train', 'test'} or any(len(task[s]) != item[k] for s, k in [('train', 'num_train'), ('test', 'num_test')]):
        raise ValueError('Composition sample pair counts changed')
    return raw
