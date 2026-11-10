#!/usr/bin/env python3
"""Explicitly inventory saved families; optionally package a complete corpus and pin a spec."""
import argparse
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_building_scripts.family_data import discover, normalize_config, package_catalog, encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--synth-root', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True, help='Fresh ignored results directory; never reused for resume')
    parser.add_argument('--package', type=Path, help='Fresh assets_tmp ZIP; fails if any selected payload is absent')
    parser.add_argument('--output-spec', type=Path, help='Fresh ignored specs/temp specs execution config')
    parser.add_argument('--run-name', help='Fresh run identity for the emitted execution spec')
    parser.add_argument('--recovery-cache', type=Path, help='Ignored results directory for exact recovered bytes; reusable across inventories')
    parser.add_argument('--recovery-archive', type=Path, action='append', default=[], help='Existing puzzle ZIP to search by checksum')
    parser.add_argument('--regenerate-missing', action='store_true', help='Run saved hash-pinned generators only for still-missing recorded seeds')
    parser.add_argument('--recovery-workers', type=int, default=2, help='Bounded local generator subprocesses (default: 2)')
    args = parser.parse_args()
    if bool(args.package) != bool(args.output_spec):
        parser.error('--package and --output-spec must be provided together')
    if args.run_name and (not args.output_spec or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name)):
        parser.error('--run-name requires --output-spec and a safe nonempty run name')
    if (args.recovery_archive or args.regenerate_missing) and not args.recovery_cache:
        parser.error('Recovery archives/generation require --recovery-cache')
    if args.recovery_workers < 1 or args.recovery_workers > 8:
        parser.error('--recovery-workers must be 1..8')
    if args.recovery_cache and not args.recovery_cache.resolve().is_relative_to(ROOT / 'results'):
        parser.error('--recovery-cache must be under ignored results/')
    for path, prefix in ((args.audit_dir, ROOT / 'results'), (args.package, ROOT / 'assets_tmp'),
                         (args.output_spec, ROOT / 'specs/temp specs')):
        if path is not None and (not path.resolve().is_relative_to(prefix) or path.exists()):
            parser.error(f'Use a fresh ignored path under {prefix}: {path}')
    spec = json.loads(args.spec.read_text())
    if args.run_name:
        spec['run_name'] = args.run_name
    config = normalize_config(spec['dataset'], ROOT)
    print('Auditing saved family membership, lineage and puzzle bytes', flush=True)
    cache = None
    if args.recovery_cache:
        from dataset_building_scripts.family_recovery import RecoveryCache, recover_catalog
        cache = RecoveryCache(args.recovery_cache)
    catalog, payload = discover(ROOT, args.synth_root, config, recovery=cache)
    if cache and (catalog['missing'] or catalog['missing_lineage_payloads']):
        result = recover_catalog(catalog, args.synth_root, cache, archives=args.recovery_archive,
                                 regenerate=args.regenerate_missing, workers=args.recovery_workers)
        print(json.dumps({'recovery': result}, separators=(',', ':')), flush=True)
        print('Re-auditing the complete recovered corpus, including held-out pair checks', flush=True)
        catalog, payload = discover(ROOT, args.synth_root, config, recovery=cache)
    args.audit_dir.mkdir(parents=True)
    (args.audit_dir / 'catalog.json').write_bytes(encoded(catalog))
    (args.audit_dir / 'summary.json').write_bytes(encoded(catalog['summary']))
    print(json.dumps(catalog['summary'], separators=(',', ':')))
    if args.package:
        identity = package_catalog(catalog, payload, args.package.resolve())
        identity['archive_path'] = args.package.resolve().relative_to(ROOT).as_posix()
        spec['dataset']['pretrain_data']['catalog'] = identity
        args.output_spec.parent.mkdir(parents=True, exist_ok=True)
        args.output_spec.write_bytes(encoded(spec))
        print(json.dumps({'pinned_spec': str(args.output_spec), 'catalog': identity}, separators=(',', ':')))


if __name__ == '__main__':
    main()
