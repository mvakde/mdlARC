#!/usr/bin/env python3
"""Check regenerated samples and the recorded schedules.

Checks, per family (from the ZIPs written by sample.py):
  * every recorded sample is present, its SHA-256 matches the manifest and
    sample.py's generator-owned checks passed (validator, input-only solver
    for compositions, self_test for variations);
  * grid schema: exactly train/test, pairs with exactly input/output,
    rectangular grids of 1..30 by 1..30 integer colours 0..9;
  * pair counts equal the recorded counts and are legal for the protocol;
  * no duplicate input or pair inside a puzzle, no repeated puzzle inside a
    family, and no family made only of identity pairs (compositions);
  * no exact copy of a parent pair (compositions; the original pipeline's
    parent snapshot pairs are recorded as hashes in the manifest).
Warnings (checked at authoring time on other seeds, not at sampling time):
  * a variation pair that copies a source-task pair (needs --originals);
  * fewer than 40 distinct puzzles among a variation family's 42 seeds.
Release-wide:
  * the composition count schedule is recomputed from the count policy and
    the recorded seeds and compared with the recorded rows;
  * duplicate puzzles across families are counted (informational).
With --solver, every composition generator's input-only solve() is run
again, in a fresh process, on every stored pair.

Standard library + NumPy only.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sample  # noqa: E402  (same folder: shared loaders and serialisers)


# --------------------------------------------------------------------------
# composition-counts-v1 schedule (ported unchanged from the release code)
# --------------------------------------------------------------------------

def policy_rng(seed, label):
    return random.Random(int.from_bytes(hashlib.sha256(f'{seed}:{label}'.encode()).digest(), 'big'))


def rounded(total, weights):
    result = {k: total * p // 100 for k, p in weights.items()}
    order = sorted(weights, key=lambda k: (-(total * weights[k] % 100), k))
    for k in order[:total - sum(result.values())]:
        result[k] += 1
    return result


def quotas(sizes, weights, targets, randomizer):
    """Controlled rounding with exact row/column sums (integral max flow)."""
    rows, cols = list(sizes), list(weights)
    result = {r: {c: sizes[r] * weights[c] // 100 for c in cols} for r in rows}
    graph = [[] for _ in range(len(rows) + len(cols) + 2)]
    source, sink = len(graph) - 2, len(graph) - 1

    def edge(a, b, cap):
        forward = [b, cap, len(graph[b])]
        graph[a].append(forward)
        graph[b].append([a, 0, len(graph[a]) - 1])
        return forward
    randomizer.shuffle(rows)
    randomizer.shuffle(cols)
    links = {}
    required = 0
    for i, r in enumerate(rows):
        n = sizes[r] - sum(result[r].values())
        required += n
        edge(source, i, n)
        order = list(enumerate(cols))
        randomizer.shuffle(order)
        for j, c in order:
            if sizes[r] * weights[c] % 100:
                links[r, c] = edge(i, len(rows) + j, 1)
    for j, c in enumerate(cols):
        need = targets[c] - sum(result[r][c] for r in rows)
        if need < 0:
            raise ValueError('Infeasible column quota')
        edge(len(rows) + j, sink, need)
    flow = 0
    while True:
        level = [-1] * len(graph)
        level[source] = 0
        queue = deque([source])
        while queue:
            a = queue.popleft()
            for b, cap, _ in graph[a]:
                if cap and level[b] < 0:
                    level[b] = level[a] + 1
                    queue.append(b)
        if level[sink] < 0:
            break
        offsets = [0] * len(graph)

        def push(a, amount):
            if a == sink:
                return amount
            while offsets[a] < len(graph[a]):
                e = graph[a][offsets[a]]
                b, cap, reverse = e
                if cap and level[b] == level[a] + 1:
                    used = push(b, min(cap, amount))
                    if used:
                        e[1] -= used
                        graph[b][reverse][1] += used
                        return used
                offsets[a] += 1
            return 0
        while (used := push(source, required)):
            flow += used
    if flow != required:
        raise ValueError('Infeasible balanced count schedule')
    for (r, c), e in links.items():
        result[r][c] += 1 - e[1]
    if any(sum(result[r][c] for r in rows) != targets[c] for c in cols):
        raise ValueError('Column total differs')
    return result


def schedule(families, seed, demo_weights, query_weights):
    """Return {family: [[seed, demos, queries], ...]} for composition-counts-v1."""
    cohorts = {}
    for family, record in sorted(families.items()):
        cohorts.setdefault(record['dataset'] + ':' + record['stage'], []).append(family)
    sizes = {c: sum(len(families[f]['seeds']) for f in fs) for c, fs in cohorts.items()}
    total = sum(sizes.values())
    tables = []
    for role, weights in [('demo', demo_weights), ('query', query_weights)]:
        cohort_quota = quotas(sizes, weights, rounded(total, weights), policy_rng(seed, role + ':cohorts'))
        table = {}
        for cohort, fs in sorted(cohorts.items()):
            table.update(quotas({f: len(families[f]['seeds']) for f in fs}, weights,
                                cohort_quota[cohort], policy_rng(seed, role + ':' + cohort)))
        tables.append(table)
    result = {}
    for family, record in sorted(families.items()):
        randomizer = policy_rng(seed, family)
        demos = [n for n, count in tables[0][family].items() for _ in range(count)]
        pairs = []
        for q, count in sorted(tables[1][family].items(), reverse=True):
            for _ in range(count):
                eligible = [i for i, d in enumerate(demos) if d > q]
                if not eligible:
                    raise ValueError('Infeasible legal joint quota')
                d = demos.pop(randomizer.choice(eligible))
                pairs.append((d, q))
        randomizer.shuffle(pairs)
        result[family] = [[s, d, q] for s, (d, q) in zip(record['seeds'], pairs, strict=True)]
    return result


def recorded_schedule(record):
    rows = {row[0]: row[:3] for row in record['samples']}
    rows.update({o['seed']: [o['seed'], o['num_train'], o['num_test']] for o in record['sampling']['omitted']})
    return [rows[s] for s in record['sampling']['seeds']]


def check_schedule(families, release):
    policy = release['composition_counts']
    demo = {int(k): v for k, v in policy['weights_percent']['demos'].items()}
    query = {int(k): v for k, v in policy['weights_percent']['queries'].items()}
    compositions = {r['id']: {'dataset': r['dataset'], 'stage': r['generator']['stage'], 'seeds': r['sampling']['seeds']}
                    for r in families if r['kind'] == 'composition'}
    computed = schedule(compositions, policy['schedule_seed'], demo, query)
    by_id = {r['id']: r for r in families}
    differing = [f for f, rows in computed.items() if rows != recorded_schedule(by_id[f])]
    omitted = sum(len(r['sampling']['omitted']) for r in families if r['kind'] == 'composition')
    demos = Counter(row[1] for rows in computed.values() for row in rows)
    queries = Counter(row[2] for rows in computed.values() for row in rows)
    return {'families': len(computed), 'rows': sum(map(len, computed.values())), 'omitted_rows': omitted,
            'differing_families': len(differing), 'first_differing': differing[:5],
            'demos': dict(sorted(demos.items())), 'queries': dict(sorted(queries.items()))}


# --------------------------------------------------------------------------
# Static sample checks
# --------------------------------------------------------------------------

def schema_errors(puzzle):
    errors = []
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        return ['puzzle must have exactly train and test']
    for split in ('train', 'test'):
        pairs = puzzle[split]
        if not isinstance(pairs, list) or not pairs:
            errors.append(f'{split} must be a non-empty list')
            continue
        for i, pair in enumerate(pairs):
            if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
                errors.append(f'{split}[{i}] must have exactly input and output')
                continue
            for role in ('input', 'output'):
                g = pair[role]
                label = f'{split}[{i}].{role}'
                if not isinstance(g, list) or not 1 <= len(g) <= 30 or any(not isinstance(r, list) for r in g):
                    errors.append(f'{label}: needs 1..30 rows')
                    continue
                width = len(g[0])
                if not 1 <= width <= 30 or any(len(r) != width for r in g):
                    errors.append(f'{label}: needs rectangular rows of width 1..30')
                    continue
                if any(isinstance(c, bool) or not isinstance(c, int) or not 0 <= c <= 9 for r in g for c in r):
                    errors.append(f'{label}: colours must be integers 0..9')
    return errors


def pair_hash(pair):
    return hashlib.sha256(json.dumps(pair, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_originals(path):
    """Original ARC-AGI-2 tasks keyed 'arc2_train/<id>' / 'arc2_eval/<id>'.

    PATH is either the ARC-AGI-2 repository's data/ folder (training/ and
    evaluation/ inside) or the released training package ZIP.
    """
    if path is None:
        return {}
    path = Path(path)
    tasks = {}
    if path.is_dir():
        for split, source in (('training', 'arc2_train'), ('evaluation', 'arc2_eval')):
            for file in sorted((path / split).glob('*.json')):
                tasks[f'{source}/{file.stem}'] = json.loads(file.read_text())
        return tasks
    with zipfile.ZipFile(path) as archive:
        catalog = json.loads(archive.read('family_catalog.json'))
        for family, record in catalog['families'].items():
            for item in record['instances']:
                if item['kind'] == 'original':
                    tasks[family] = json.loads(archive.read(item['path']))
    return tasks


def original_bytes(task):
    return json.dumps(task, sort_keys=True, separators=(',', ':')).encode()


def check_family(record, archive_path, originals, catalogue):
    problems, warnings = [], []
    if not archive_path.is_file():
        return None
    with zipfile.ZipFile(archive_path) as archive:
        report = json.loads(archive.read('records.json'))
        if report.get('valid') is not True:
            problems.append('sample.py report is not valid')
        if report.get('generator_sha256') != record['generator']['sha256']:
            problems.append('generator hash differs from the manifest')
        produced = {r['seed']: r for r in report.get('records', [])}
        puzzles, identity_only, counted = [], True, 0
        forbidden = set(record['sampling'].get('parent_pair_sha256', []))
        source_pairs = set()
        if record['kind'] == 'variation' and record['id'] in originals:
            task = originals[record['id']]
            source_pairs = {pair_hash(p) for s in ('train', 'test') for p in task[s]}
            if record['original'] and hashlib.sha256(original_bytes(task)).hexdigest() != record['original']['sha256']:
                problems.append('original task bytes differ from the manifest')
        for seed, demos, queries, digest in record['samples']:
            r = produced.get(seed)
            if r is None:
                problems.append(f'seed {seed}: missing')
                continue
            raw = archive.read(r['member'])
            actual = hashlib.sha256(raw).hexdigest()
            if actual != digest:
                problems.append(f'seed {seed}: sha256 differs from the manifest')
            puzzle = json.loads(raw)
            errors = schema_errors(puzzle)
            if errors:
                problems.append(f'seed {seed}: ' + '; '.join(errors[:3]))
                continue
            nt, nq = len(puzzle['train']), len(puzzle['test'])
            if (demos is not None and nt != demos) or (queries is not None and nq != queries):
                problems.append(f'seed {seed}: counts {nt}/{nq} differ from recorded {demos}/{queries}')
            legal_q = 3 if record['kind'] == 'composition' else 4
            if not (2 <= nt <= 10 and 1 <= nq <= legal_q and nt > nq):
                problems.append(f'seed {seed}: illegal counts {nt}/{nq}')
            inputs, pairs = set(), set()
            for split in ('train', 'test'):
                for pair in puzzle[split]:
                    key_in = json.dumps(pair['input'], separators=(',', ':'))
                    key_pair = pair_hash(pair)
                    if key_in in inputs:
                        problems.append(f'seed {seed}: duplicate input')
                    if key_pair in pairs:
                        problems.append(f'seed {seed}: duplicate pair')
                    inputs.add(key_in)
                    pairs.add(key_pair)
                    if key_pair[:16] in forbidden:
                        problems.append(f'seed {seed}: exact copy of a parent pair')
                    if key_pair in source_pairs:
                        # Enforced by the authoring-time host checks, not at sampling time.
                        warnings.append(f'seed {seed}: exact copy of a source-task pair')
                    identity_only &= pair['input'] == pair['output']
            puzzles.append(actual)
            counted += 1
            catalogue.setdefault(actual, set()).add(record['id'])
        distinct = len(set(puzzles))
        if record['kind'] == 'composition':
            if distinct != len(puzzles):
                problems.append('repeated puzzle across recorded seeds')
            if puzzles and identity_only:
                problems.append('entire family is identity')
        elif distinct < min(40, len(puzzles)):
            # The authoring host checks required 40 distinct puzzles on other seeds.
            warnings.append(f'only {distinct} distinct puzzles from {len(puzzles)} seeds')
    return {'id': record['id'], 'samples': counted, 'problems': problems, 'warnings': warnings}


# --------------------------------------------------------------------------
# Optional: re-run composition solvers in fresh processes
# --------------------------------------------------------------------------

def solver_worker():
    request = json.load(sys.stdin)
    import contextlib
    import numpy as np
    result = {'id': request['family'], 'pairs': 0, 'mismatches': []}
    try:
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            raw = Path(request['generator']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != request['generator_sha256']:
                raise ValueError('generator hash differs from the manifest')
            namespace = sample.load_composition(Path(request['generator']))
            solve = namespace['solve']
            with zipfile.ZipFile(request['archive']) as archive:
                report = json.loads(archive.read('records.json'))
                for r in report['records']:
                    puzzle = json.loads(archive.read(r['member']))
                    for split in ('train', 'test'):
                        for i, pair in enumerate(puzzle[split]):
                            result['pairs'] += 1
                            try:
                                predicted = sample.grid(solve(np.array(pair['input'], dtype=int)))
                            except Exception as error:  # noqa: BLE001
                                predicted = f'{type(error).__name__}: {error}'
                            if predicted != pair['output']:
                                result['mismatches'].append(f'seed {r["seed"]} {split}[{i}]')
    except Exception as error:  # noqa: BLE001
        result['error'] = f'{type(error).__name__}: {error}'[-1000:]
    Path(request['report']).write_text(json.dumps(result))


def run_solver(root, record, archive_path, timeout):
    with tempfile.TemporaryDirectory(prefix='.solver-') as temporary:
        temp = Path(temporary)
        request = {'family': record['id'], 'generator': str((root / record['generator']['path']).resolve()),
                   'generator_sha256': record['generator']['sha256'], 'archive': str(archive_path),
                   'report': str(temp / 'report.json')}
        try:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '_solver'], input=json.dumps(request).encode(),
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=sample.child_environment(),
                           cwd=temp, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {'id': record['id'], 'error': f'timeout after {timeout}s'}
        if not (temp / 'report.json').is_file():
            return {'id': record['id'], 'error': 'solver process failed'}
        return json.loads((temp / 'report.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', nargs='?', default='run', choices=['run', '_solver'])
    parser.add_argument('--root', type=Path, default=HERE)
    parser.add_argument('--samples', type=Path, default=HERE / 'samples', help='output folder of sample.py')
    parser.add_argument('--originals', type=Path,
                        help='ARC-AGI-2 data/ folder or the training package ZIP (enables source-copy checks)')
    parser.add_argument('--family', action='append')
    parser.add_argument('--require-all', action='store_true',
                        help='fail if any in-package family has not been sampled')
    parser.add_argument('--solver', action='store_true', help='re-run composition solvers on stored pairs')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--timeout', type=int, default=sample.DEFAULT_TIMEOUT)
    parser.add_argument('--skip-schedule', action='store_true')
    parser.add_argument('--report', type=Path, help='write the full JSON report here')
    args = parser.parse_args()
    if args.action == '_solver':
        solver_worker()
        return 0
    root = args.root.resolve()
    families, release = sample.load_manifest(root)
    result = {}
    if not args.skip_schedule:
        result['schedule'] = check_schedule(families, release)
        print('Schedule:', json.dumps(result['schedule']), flush=True)
    selected = [r for r in families if not args.family or r['id'] in set(args.family)]
    originals = load_originals(args.originals)
    samples_root = args.samples.resolve()
    catalogue, checked, failures, missing = {}, [], [], []
    for record in selected:
        archive_path = sample.output_path(samples_root, record)
        outcome = check_family(record, archive_path, originals, catalogue)
        if outcome is None:
            if record['in_package']:
                missing.append(record['id'])
            continue
        checked.append(outcome)
        if outcome['problems']:
            failures.append(outcome)
    cross = [sorted(ids) for ids in catalogue.values() if len(ids) > 1]
    result['samples'] = {
        'families_checked': len(checked), 'samples_checked': sum(c['samples'] for c in checked),
        'families_with_problems': len(failures), 'problems': failures[:50],
        'families_with_warnings': sum(bool(c['warnings']) for c in checked),
        'warnings': [c for c in checked if c['warnings']][:50],
        'in_package_families_not_sampled': len(missing),
        'puzzles_shared_across_families': len(cross), 'examples_shared': cross[:5],
        'originals_loaded': len(originals),
    }
    if args.solver:
        targets = [r for r in selected if r['kind'] == 'composition'
                   and sample.output_path(samples_root, r).is_file()]
        outcomes = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_solver, root, r, sample.output_path(samples_root, r), args.timeout) for r in targets]
            for future in as_completed(futures):
                outcomes.append(future.result())
        bad = [o for o in outcomes if o.get('error') or o.get('mismatches')]
        result['solver'] = {'families': len(outcomes), 'pairs': sum(o.get('pairs', 0) for o in outcomes),
                            'families_with_mismatches': len(bad), 'examples': bad[:10]}
    print(json.dumps({k: v for k, v in result['samples'].items() if k not in ('problems', 'warnings')}), flush=True)
    for w in result['samples']['warnings'][:10]:
        print('WARNING', w['id'], '; '.join(w['warnings'][:3]), flush=True)
    for f in failures[:20]:
        print('PROBLEM', f['id'], '; '.join(f['problems'][:3]), flush=True)
    if 'solver' in result:
        print('Solver:', json.dumps({k: v for k, v in result['solver'].items() if k != 'examples'}), flush=True)
    if args.report:
        args.report.write_text(json.dumps(result, indent=1))
    ok = (not failures and (not args.require_all or not missing)
          and ('schedule' not in result or result['schedule']['differing_families'] == 0)
          and ('solver' not in result or result['solver']['families_with_mismatches'] == 0))
    print('OK' if ok else 'FAILED', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
