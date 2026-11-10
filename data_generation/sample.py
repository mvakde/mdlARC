#!/usr/bin/env python3
"""Re-run the released generators on their recorded seeds and check the bytes.

Each family runs in a fresh Python process, as in the original release run:
the generator is loaded once, then every recorded seed is generated in the
recorded order with the recorded demo/query counts. Each puzzle is checked
the way the original pipeline checked it, serialised exactly as the original
files were, and its SHA-256 is compared with the manifest.

Output: one ZIP per family under OUT/families/ (seed-XXXXXXXXXX.json members
plus records.json) and a line per family in OUT/sample-report.jsonl.

Standard library + NumPy only. Use the pinned runtime (see
manifest/release.json) for byte-identical output; PYTHONHASHSEED is forced
to 0 for every generator process.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import threading
import time
import types
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
DEFAULT_TIMEOUT = 1800


# --------------------------------------------------------------------------
# Serialisation and checks, ported from the original samplers.
# --------------------------------------------------------------------------

def composition_bytes(puzzle):
    """Composition sample bytes (composition-counts-v1 release)."""
    return (json.dumps(puzzle, separators=(',', ':'), ensure_ascii=True) + '\n').encode()


def variation_bytes(puzzle):
    """Variation sample bytes (task-variations sampler)."""
    return (json.dumps(puzzle, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def grid(value):
    import numpy as np
    array = np.asarray(value)
    assert array.ndim == 2 and all(1 <= n <= 30 for n in array.shape), "Grid dimensions must be 1..30"
    assert np.issubdtype(array.dtype, np.integer), "Grid cells must be integers"
    assert np.all((0 <= array) & (array <= 9)), "Grid colors must be 0..9"
    return array.astype(int).tolist()


def encode(task, num_train=4, num_test=1):
    """Exact counts, grid schema and unique inputs; keeps the generator's key order."""
    assert isinstance(task, dict) and set(task) == {"train", "test"}, "Puzzle must have exactly train/test keys"
    result, seen = {}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        assert isinstance(task[split], list) and len(task[split]) == count, f"Wrong {split} pair count"
        result[split] = []
        for pair in task[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}, "Pair must have exactly input/output keys"
            pair = {k: grid(v) for k, v in pair.items()}
            key = json.dumps(pair["input"], separators=(",", ":"))
            assert key not in seen, "Duplicate input within a puzzle"
            seen.add(key)
            result[split].append(pair)
    return result


def json_value(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Generated puzzle dictionaries must use string keys")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("Generated puzzle contains unsupported value %r" % type(value).__name__)


def check_json_grid(grid_value, label):
    if not isinstance(grid_value, list) or not grid_value:
        raise ValueError("%s must be a non-empty list of rows" % label)
    if any(not isinstance(row, list) or not row for row in grid_value):
        raise ValueError("%s rows must be non-empty lists" % label)
    width = len(grid_value[0])
    if any(len(row) != width for row in grid_value):
        raise ValueError("%s must be rectangular" % label)
    for row in grid_value:
        for color in row:
            if isinstance(color, bool) or not isinstance(color, int) or not 0 <= color <= 9:
                raise ValueError("%s must contain integer ARC colors from 0 through 9" % label)


def check_arc_json(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("Generated puzzle must contain exactly train and test")
    for split in ("train", "test"):
        pairs = puzzle[split]
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("Generated puzzle %s must be a non-empty list" % split)
        for index, pair in enumerate(pairs):
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError("Generated %s pair %d must contain exactly input and output" % (split, index))
            check_json_grid(pair["input"], "%s[%d].input" % (split, index))
            check_json_grid(pair["output"], "%s[%d].output" % (split, index))
    return puzzle


# --------------------------------------------------------------------------
# Worker: one family per fresh process.
# --------------------------------------------------------------------------

def load_composition(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("composed_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__dict__


def load_variation(path):
    module = types.ModuleType("_task_variations_generated")
    module.__file__ = str(path)
    code = compile(Path(path).read_text(encoding="utf-8"), str(path), "exec")
    exec(code, module.__dict__)
    return module.__dict__


def write_member(archive, name, raw):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, raw)


def run_composition(namespace, request, archive):
    import numpy as np
    for name in ('generate_puzzle', 'solve', 'validate_generated_puzzle', 'self_test'):
        if not callable(namespace.get(name)):
            raise ValueError('Missing callable ' + name)
    generate, validate, solve = namespace['generate_puzzle'], namespace['validate_generated_puzzle'], namespace['solve']
    signatures, records, changes = set(), [], 0
    for seed, demos, queries in request['schedule']:
        if not (2 <= demos <= 10 and 1 <= queries <= 3 and demos > queries):
            raise ValueError('Illegal count schedule')
        begin = time.monotonic()
        try:
            task = generate(seed, num_train=demos, num_test=queries)
            puzzle = encode(task, demos, queries)
            if validate(task) is False:
                raise ValueError('Generator validator returned False')
            for split in ('train', 'test'):
                for pair in puzzle[split]:
                    if grid(solve(np.array(pair['input'], dtype=int))) != pair['output']:
                        raise ValueError('Input-only solver mismatch')
                    changes += pair['input'] != pair['output']
        except Exception as error:
            raise RuntimeError(f'seed {seed} ({demos}/{queries}): {type(error).__name__}: {error}') from error
        raw = composition_bytes(puzzle)
        digest = sha256(raw)
        if digest in signatures:
            raise ValueError(f'Repeated puzzle across recorded seeds at seed {seed}')
        signatures.add(digest)
        name = f'seed-{seed:010d}.json'
        records.append({'seed': seed, 'num_train': demos, 'num_test': queries, 'member': name,
                        'sha256': digest, 'bytes': len(raw), 'seconds': round(time.monotonic() - begin, 6)})
        write_member(archive, name, raw)
    if not changes:
        raise ValueError('Entire family is identity')
    return records, ['exact_counts', 'schema_grid_bounds_colors', 'unique_inputs', 'generator_validator',
                     'input_only_generator_solver', 'unique_puzzles', 'not_all_identity']


def run_variation(namespace, request, archive):
    for name in ('generate_puzzle', 'validate_generated_puzzle', 'self_test'):
        if not callable(namespace.get(name)):
            raise ValueError('Missing callable ' + name)
    # The original sampler ran self_test() in the same process before sampling.
    try:
        namespace['self_test']()
    except Exception as error:
        raise RuntimeError(f'self_test failed: {type(error).__name__}: {error}') from error
    records = []
    for seed in request['seeds']:
        begin = time.monotonic()
        try:
            puzzle = namespace['generate_puzzle'](seed, num_train=None, num_test=None)
            namespace['validate_generated_puzzle'](puzzle)
            serialised = check_arc_json(json_value(puzzle))
        except Exception as error:
            raise RuntimeError(f'seed {seed}: {type(error).__name__}: {error}') from error
        raw = variation_bytes(serialised)
        name = f'seed-{seed:010d}.json'
        records.append({'seed': seed, 'num_train': len(serialised['train']), 'num_test': len(serialised['test']),
                        'member': name, 'sha256': sha256(raw), 'bytes': len(raw),
                        'seconds': round(time.monotonic() - begin, 6)})
        write_member(archive, name, raw)
    return records, ['self_test', 'generator_validator', 'schema_grid_bounds_colors']


def worker(request):
    started = time.monotonic()
    path = Path(request['generator'])
    raw = path.read_bytes()
    if sha256(raw) != request['generator_sha256']:
        raise ValueError('Generator bytes differ from the manifest pin')
    loader = load_composition if request['kind'] == 'composition' else load_variation
    runner = run_composition if request['kind'] == 'composition' else run_variation
    with zipfile.ZipFile(request['archive'], 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        namespace = loader(path)
        records, checks = runner(namespace, request, archive)
        report = {'valid': True, 'family': request['family'], 'generator_sha256': request['generator_sha256'],
                  'kind': request['kind'], 'checks': checks, 'records': records,
                  'seconds': round(time.monotonic() - started, 6)}
        write_member(archive, 'records.json', (json.dumps(report, separators=(',', ':')) + '\n').encode())
    return report


def worker_main():
    request = json.load(sys.stdin)
    try:
        # Generators may print diagnostics; keep none of it, as the original runs did.
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            report = worker(request)
    except BaseException as error:  # noqa: BLE001 - report everything to the parent
        report = {'valid': False, 'family': request.get('family'), 'error': f'{type(error).__name__}: {error}'[-2000:]}
    Path(request['report']).write_text(json.dumps(report, separators=(',', ':')) + '\n')


# --------------------------------------------------------------------------
# Parent: selection, scheduling, comparison.
# --------------------------------------------------------------------------

def load_manifest(root):
    families = []
    with (root / 'manifest' / 'families.jsonl').open() as handle:
        for line in handle:
            if line.strip():
                families.append(json.loads(line))
    release = json.loads((root / 'manifest' / 'release.json').read_text())
    return families, release


def stratum(record):
    if record['kind'] == 'composition':
        return 'composition/' + record['pairing']
    return 'variation/' + record['source']


def select(families, args):
    chosen = families
    if args.family:
        wanted = set(args.family)
        chosen = [r for r in chosen if r['id'] in wanted]
        missing = wanted - {r['id'] for r in chosen}
        if missing:
            raise SystemExit(f'Unknown family ids: {sorted(missing)[:5]}')
    if not args.all and not args.family:
        chosen = [r for r in chosen if r['in_package']]
    if args.kind:
        chosen = [r for r in chosen if r['kind'] == args.kind]
    if args.pairing:
        chosen = [r for r in chosen if r.get('pairing') == args.pairing]
    if args.source:
        chosen = [r for r in chosen if r.get('source') == args.source]
    if args.stratified:
        groups = {}
        for r in chosen:
            groups.setdefault(stratum(r), []).append(r)
        picker = random.Random(args.select_seed)
        per = max(1, args.stratified // max(1, len(groups)))
        picked = []
        for key in sorted(groups):
            pool = sorted(groups[key], key=lambda r: r['id'])
            picked.extend(picker.sample(pool, min(per, len(pool))))
        chosen = picked
    if args.limit:
        chosen = chosen[:args.limit]
    return chosen


def output_path(out, record):
    relative = Path(record['generator']['path']).relative_to('generators').with_suffix('.zip')
    return out / 'families' / relative


def expected_rows(record):
    return [tuple(row) for row in record['samples']]


def compare(record, report):
    expected = expected_rows(record)
    actual = [(r['seed'], r['num_train'], r['num_test'], r['sha256']) for r in report['records']]
    matched = sum(1 for e, a in zip(expected, actual)
                  if e[0] == a[0] and e[3] == a[3] and (e[1] is None or e[1] == a[1]) and (e[2] is None or e[2] == a[2]))
    first = next((i for i, (e, a) in enumerate(zip(expected, actual))
                  if not (e[0] == a[0] and e[3] == a[3] and (e[1] is None or e[1] == a[1]) and (e[2] is None or e[2] == a[2]))), None)
    return {'expected': len(expected), 'generated': len(actual), 'matched': matched,
            'first_mismatch_seed': None if first is None else expected[first][0]}


def build_request(root, record, archive, report):
    request = {'family': record['id'], 'kind': record['kind'],
               'generator': str((root / record['generator']['path']).resolve()),
               'generator_sha256': record['generator']['sha256'],
               'archive': str(archive), 'report': str(report)}
    if record['kind'] == 'composition':
        request['schedule'] = [row[:3] for row in record['samples']]
    else:
        request['seeds'] = [row[0] for row in record['samples']]
    return request


def child_environment():
    env = {**os.environ, 'PYTHONHASHSEED': '0', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    return env


def run_one(root, out, record, timeout):
    target = output_path(out, record)
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='.sample-', dir=target.parent) as temporary:
        temp = Path(temporary)
        work = temp / 'cwd'
        work.mkdir()
        request = build_request(root, record, temp / 'family.zip', temp / 'report.json')
        try:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '_worker'],
                                     input=json.dumps(request).encode(), stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, env=child_environment(), cwd=work, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {'id': record['id'], 'ok': False, 'error': f'timeout after {timeout}s',
                    'seconds': round(time.monotonic() - started, 3)}
        report_path = temp / 'report.json'
        if not report_path.is_file():
            return {'id': record['id'], 'ok': False,
                    'error': (process.stderr.decode(errors='replace')[-2000:] or f'exit {process.returncode}'),
                    'seconds': round(time.monotonic() - started, 3)}
        report = json.loads(report_path.read_text())
        if not report.get('valid'):
            return {'id': record['id'], 'ok': False, 'error': report.get('error'),
                    'seconds': round(time.monotonic() - started, 3)}
        result = compare(record, report)
        (temp / 'family.zip').replace(target)
    return {'id': record['id'], 'ok': True, **result, 'seconds': round(time.monotonic() - started, 3)}


def reuse(out, record):
    target = output_path(out, record)
    if not target.is_file():
        return None
    try:
        with zipfile.ZipFile(target) as archive:
            report = json.loads(archive.read('records.json'))
            if report.get('valid') is not True or report['generator_sha256'] != record['generator']['sha256']:
                return None
            for r in report['records']:
                if sha256(archive.read(r['member'])) != r['sha256']:
                    return None
    except (zipfile.BadZipFile, KeyError, ValueError):
        return None
    return {'id': record['id'], 'ok': True, **compare(record, report), 'seconds': 0.0, 'reused': True}


def runtime_status(release):
    import numpy
    pins = release['runtime']
    actual = {'python': platform.python_version(), 'implementation': platform.python_implementation(),
              'numpy': numpy.__version__}
    differences = {k: (pins[k], v) for k, v in actual.items() if pins.get(k) != v}
    return actual, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', nargs='?', default='run', choices=['run', '_worker'])
    parser.add_argument('--root', type=Path, default=HERE, help='release tree (holds manifest/ and generators/)')
    parser.add_argument('--out', type=Path, default=HERE / 'samples', help='output directory')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='seconds per family')
    parser.add_argument('--family', action='append', help='family id (repeatable)')
    parser.add_argument('--kind', choices=['variation', 'composition'])
    parser.add_argument('--pairing', choices=['eval-eval', 'train-train', 'mixed'])
    parser.add_argument('--source', choices=['arc2_train', 'arc2_eval', 'arc1_filtered', 'conceptarc'])
    parser.add_argument('--all', action='store_true',
                        help='also sample the 187 variation families that are not in the training package')
    parser.add_argument('--stratified', type=int, help='pick about N families spread evenly over the strata')
    parser.add_argument('--select-seed', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true', help='keep verified family ZIPs already in OUT')
    parser.add_argument('--strict-runtime', action='store_true', help='refuse to run outside the pinned runtime')
    args = parser.parse_args()
    if args.action == '_worker':
        worker_main()
        return 0
    if os.environ.get('PYTHONHASHSEED') != '0':
        # Generators run in children with PYTHONHASHSEED=0 anyway; re-exec so the
        # parent's own behaviour is pinned as well.
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                  {**os.environ, 'PYTHONHASHSEED': '0'})
    root = args.root.resolve()
    families, release = load_manifest(root)
    actual, differences = runtime_status(release)
    print(f'Runtime: {actual["implementation"]} {actual["python"]}, NumPy {actual["numpy"]}', flush=True)
    if differences:
        message = 'Runtime differs from the pins: ' + ', '.join(f'{k} pinned {p}, found {a}' for k, (p, a) in differences.items())
        if args.strict_runtime:
            raise SystemExit(message)
        print('WARNING: ' + message + '. Bytes may differ.', flush=True)
    chosen = select(families, args)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'sample-report.jsonl'
    lock = threading.Lock()
    totals = {'families': 0, 'ok': 0, 'failed': 0, 'samples': 0, 'matched': 0, 'mismatched_families': 0}
    start = time.monotonic()

    def record_result(result, record):
        with lock:
            with report_path.open('a') as handle:
                handle.write(json.dumps({**result, 'kind': record['kind'],
                                         'stratum': stratum(record)}, separators=(',', ':')) + '\n')
            totals['families'] += 1
            if result['ok']:
                totals['ok'] += 1
                totals['samples'] += result['expected']
                totals['matched'] += result['matched']
                if result['matched'] != result['expected'] or result['generated'] != result['expected']:
                    totals['mismatched_families'] += 1
            else:
                totals['failed'] += 1
            done = totals['families']
            if done % 25 == 0 or done == len(chosen) or not result['ok']:
                elapsed = time.monotonic() - start
                eta = elapsed / done * (len(chosen) - done)
                status = '' if result['ok'] else f' FAILED {result["id"]}: {str(result.get("error"))[:300]}'
                print(f'{done}/{len(chosen)} families; samples matched {totals["matched"]}/{totals["samples"]}; '
                      f'failed {totals["failed"]}; elapsed {elapsed:.0f}s; ETA {eta:.0f}s{status}', flush=True)

    pending = []
    for record in chosen:
        reused = reuse(out, record) if args.resume else None
        if reused:
            record_result(reused, record)
        else:
            pending.append(record)
    # Compositions take longer on average; start them first.
    pending.sort(key=lambda r: (r['kind'] != 'composition', r['id']))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, root, out, r, args.timeout): r for r in pending}
        for future in as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001
                result = {'id': record['id'], 'ok': False, 'error': f'{type(error).__name__}: {error}'}
            record_result(result, record)
    summary = {**totals, 'seconds': round(time.monotonic() - start, 1), 'runtime': actual,
               'byte_identical': totals['failed'] == 0 and totals['mismatched_families'] == 0}
    print(json.dumps(summary), flush=True)
    return 0 if summary['byte_identical'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
