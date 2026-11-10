"""Recover exact saved bytes into a separate cache; never rewrite synth history.

This is a local data preparation operation, not an authoring pipeline. Only
hash-pinned generators and recorded seeds may run, in bounded subprocesses.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import ModuleType
import zipfile

from .family_data import encoded, file_sha, regular, sha

WORKER_MODULE = 'dataset_building_scripts.family_recovery'

class RecoveryCache:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, digest):
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid recovery hash')
        return regular(self.directory, 'objects/' + digest)

    def read(self, digest):
        path = self.path(digest)
        if not path.exists():
            return None
        raw = path.read_bytes()
        if sha(raw) != digest:
            raise ValueError('Corrupt recovery object: ' + digest)
        return raw

    def save(self, raw, digest):
        if sha(raw) != digest:
            return False
        if self.read(digest) is not None:
            return True
        path = self.path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            pending = Path(handle.name)
            handle.write(raw)
        try:
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
        return True


def puzzle_text(puzzle, label):
    """Historical v2/v3 rendering; accept it only if the protected hash matches."""
    sections = ['Puzzle: ' + label] if label else []
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split], 1):
            lines = [f'{split.title()} pair {index}']
            for role in ('input', 'output'):
                if role in pair:
                    lines += [role.title() + ':']
                    lines += [''.join(str(c) for c in row) for row in pair[role]]
            sections.append('\n'.join(lines))
    return ('\n\n'.join(sections) + '\n').encode()


def recover_evidence(records, synth, cache):
    """Rebuild redundant renderings/copies; recorded hashes are the authority."""
    plans = {}
    for record in records:
        if cache.read(record['sha256']) is not None:
            continue
        relative = Path(record['path'])
        # Restrict reconstruction to the established protected-input layout.
        if len(relative.parts) < 7 or relative.parts[3] != 'llm-inputs' or relative.parts[5] != 'inputs':
            continue
        job = regular(synth, Path(*relative.parts[:3]))
        stage = relative.parts[4]
        name = '/'.join(relative.parts[6:])
        plan_path = job.parent / 'plan.json'
        if plan_path not in plans:
            plans[plan_path] = {j['name']: j for j in json.loads(plan_path.read_bytes())['jobs']}
        plan = plans[plan_path][job.name]
        inputs = job / 'llm-inputs' / stage / 'inputs'
        candidates = []
        if name.startswith('parents/') and relative.suffix == '.txt':
            label = relative.parts[7]
            parent = plan['parents'][0 if label == 'A' else 1]
            path = regular(inputs, name).with_suffix('.json')
            if not path.is_file():
                path = job / 'parents' / label / path.name
            if path.is_file():
                if relative.name == 'task.txt':
                    labels = [parent, parent.split('/')[-1]]
                else:
                    selection = plan.get('parent_samples', {}).get(label, {})
                    labels = [f"parent {p}, seed {selection.get('seed')}" for p in (parent, parent.split('/')[-1])]
                puzzle = json.loads(path.read_bytes())
                candidates.extend(puzzle_text(puzzle, label) for label in labels)
        elif name.startswith('examples/'):
            suffix = name[len('examples/'):]
            source = regular(synth, 'pipelines/task-variations/examples/review/' + suffix)
            if source.is_file():
                candidates.append(source.read_bytes())
                if source.suffix == '.json':
                    candidates.append((json.dumps(json.loads(source.read_bytes()), separators=(',', ':')) + '\n').encode())
            if relative.suffix == '.txt' and source.with_suffix('.json').is_file():
                candidates.append(puzzle_text(json.loads(source.with_suffix('.json').read_bytes()),
                                              str(Path(suffix).with_suffix('.json'))))
            if suffix == 'README.md':
                candidates.append(('Study these FOUR worked human reviews: ' + ', '.join(plan['review_examples']) +
                    '.\nRead task.txt, summary, generator_before.py, manual_review.md, generator_after.py, '
                    'and before/after sample .txt files. They are review-style examples, not parents.\n').encode())
            if suffix == 'manifest.json':
                protected = json.loads((job / 'llm-inputs' / stage / 'protected.json').read_bytes())
                hashes = {k[len('examples/'):]: v for k, v in sorted(protected.items())
                          if k.startswith('examples/') and k != 'examples/manifest.json'}
                candidates.append((json.dumps({'example_ids': plan['review_examples'], 'files': hashes},
                                               separators=(',', ':')) + '\n').encode())
        for raw in candidates:
            if cache.save(raw, record['sha256']):
                break


def generator_request(family, instances, synth, catalog):
    first = instances[0]
    relative = Path(first['relative'])
    if first['kind'] == 'variation':
        job = regular(synth, relative.parent.parent)
        manifest = json.loads((job / 'samples/manifest').read_bytes())
        generator = regular(job, manifest['generator']['path'])
        expected = catalog['metadata_sha256']['synth:' + str(generator.relative_to(synth))]
        kwargs = {k: manifest['sampling'].get(k) for k in ('num_train', 'num_test')}
    else:
        job = regular(synth, Path(*relative.parts[:3]))
        name = 'generator_v1.py' if first['stage'] == 'author' else 'generator_v2.py'
        checkpoint = json.loads((job / 'checkpoints.json').read_bytes())
        expected = checkpoint[first['stage']][name]
        generator = regular(job, name)
        kwargs = {'num_train': 4, 'num_test': 1}
    if file_sha(generator) != expected:
        raise ValueError('Saved generator changed: ' + str(generator))
    return {'family': family, 'generator': str(generator), 'generator_sha256': expected,
            'instances': instances, 'kwargs': kwargs}


def json_value(value):
    if hasattr(value, 'tolist'):
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def generate(request, cache):
    """Worker entry point. A mismatch is reported, never blessed as a new pin."""
    generator = Path(request['generator'])
    if file_sha(generator) != request['generator_sha256']:
        raise ValueError('Generator changed before execution')
    module = ModuleType('_family_recovery_generator')
    module.__file__ = str(generator)
    exec(compile(generator.read_text(), str(generator), 'exec'), module.__dict__)
    recovered, unresolved = 0, []
    checked_repeat = False
    for item in request['instances']:
        if cache.read(item['sha256']) is not None:
            recovered += 1
            continue
        task = module.generate_puzzle(item['seed'], **request['kwargs'])
        module.validate_generated_puzzle(task)
        task = json_value(task)
        if not checked_repeat:
            repeated = json_value(module.generate_puzzle(item['seed'], **request['kwargs']))
            if task != repeated:
                raise ValueError('Nondeterministic generator: ' + request['family'])
            checked_repeat = True
        matched = False
        # Historical writers differ in key ordering/newline, never in puzzle content.
        for sort in (True, False):
            for newline in ('\n', ''):
                raw = (json.dumps(task, sort_keys=sort, separators=(',', ':')) + newline).encode()
                if cache.save(raw, item['sha256']):
                    matched = True
                    break
            if matched:
                break
        if matched:
            recovered += 1
        else:
            unresolved.append(item['seed'])
    trials = 0
    if unresolved and request.get('try_hash_seeds', True):
        # Some historical generators iterate string sets. Their seed is not a
        # complete runtime identity: process hash randomization also changes
        # iteration order. Recover only outputs matching the original byte pins.
        deadline = time.monotonic() + 120
        for hash_seed in range(1, 129):
            pending = [i for i in request['instances'] if i['seed'] in unresolved]
            if not pending or time.monotonic() >= deadline:
                break
            child = {**request, 'instances': pending, 'try_hash_seeds': False,
                     'cache': str(cache.directory)}
            with tempfile.TemporaryDirectory(dir=cache.directory, prefix='.hash-worker-') as folder:
                path = Path(folder) / 'request.json'
                path.write_bytes(encoded(child))
                try:
                    subprocess.run([sys.executable, '-B', '-m', WORKER_MODULE, str(path)],
                        cwd=Path(__file__).resolve().parents[1], stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': str(hash_seed)},
                        timeout=max(1, min(30, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            trials += 1
            unresolved = [i['seed'] for i in pending if cache.read(i['sha256']) is None]
        recovered = len(request['instances']) - len(unresolved)
    if file_sha(generator) != request['generator_sha256']:
        raise ValueError('Generator changed during execution')
    return {'family': request['family'], 'recovered': recovered, 'unresolved_seeds': unresolved,
            'hash_seed_trials': trials}


def recover_catalog(catalog, synth, cache, *, archives=(), regenerate=False, workers=2):
    synth = Path(synth).resolve()
    records = catalog['missing'] + catalog['missing_lineage_payloads']
    needed = {r['sha256'] for r in records if cache.read(r['sha256']) is None}
    print(f'Recovery: {len(needed)} uncached byte hashes; checking existing archives', flush=True)
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if not needed:
                    break
                if info.is_dir() or not info.filename.endswith('.json') or info.file_size > 4 * 1024 * 1024:
                    continue
                raw = archive.read(info)
                digest = sha(raw)
                if digest in needed:
                    cache.save(raw, digest)
                    needed.remove(digest)
    print('Recovery: checking protected text/example copies', flush=True)
    recover_evidence(catalog['missing_lineage_payloads'], synth, cache)
    needed = {r['sha256'] for r in records if cache.read(r['sha256']) is None}
    # Saved exports/task_v1/checked-samples can be aliases of a missing path.
    print(f'Recovery: searching local copies for {len(needed)} remaining hashes', flush=True)
    suffixes = {Path(r['path']).suffix for r in records if r['sha256'] in needed}
    for directory, dirs, files in os.walk(synth / 'results', followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in
                         {'_shared', '__pycache__', 'node_modules', 'logs', 'traces', 'sessions'}
                         and not (Path(directory) / d).is_symlink())
        if Path(directory) == synth / 'results':
            dirs[:] = [d for d in dirs if d.startswith(('task-variations', 'task-composition',
                                                       'task-compsition', 'exported-datasets'))]
        if not needed:
            break
        for name in sorted(files):
            if Path(name).suffix not in suffixes:
                continue
            path = Path(directory) / name
            if path.is_symlink():
                continue
            # os.walk never follows links, and linked directory entries were
            # removed above; avoid resolving every ancestor for every file.
            if not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
                continue
            raw = path.read_bytes()
            digest = sha(raw)
            if digest in needed:
                cache.save(raw, digest)
                needed.remove(digest)
    requests = []
    if regenerate:
        for family, record in catalog['families'].items():
            instances = [i for i in record['instances'] if i['kind'] != 'original'
                         and i['sha256'] in needed]
            if instances:
                requests.append(generator_request(family, instances, synth, catalog))
    print(f'Recovery: {len(requests)} generator families need missing recorded seeds', flush=True)
    def run(request):
        with tempfile.TemporaryDirectory(dir=cache.directory, prefix='.worker-') as folder:
            path = Path(folder) / 'request.json'
            path.write_bytes(encoded({**request, 'cache': str(cache.directory)}))
            try:
                process = subprocess.run([sys.executable, '-B', '-m', WORKER_MODULE, str(path)],
                    cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                    env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0'}, timeout=300)
                if process.returncode:
                    return {'family': request['family'], 'error': process.stderr[-1500:]}
                return json.loads(process.stdout.splitlines()[-1])
            except subprocess.TimeoutExpired:
                return {'family': request['family'], 'error': 'Generator exceeded 300 seconds'}
    with (cache.directory / 'generation.jsonl').open('a') as journal:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, result in enumerate(pool.map(run, requests), 1):
                journal.write(encoded(result).decode() + '\n')
                journal.flush()
                if index % 50 == 0 or index == len(requests):
                    print(f'Recovered generator families {index}/{len(requests)}', flush=True)
    remaining = [r for r in records if cache.read(r['sha256']) is None]
    return {'requested_paths': len(records), 'remaining_paths': len(remaining),
            'generator_families_attempted': len(requests)}


if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_bytes())
    print(json.dumps(generate(request, RecoveryCache(request['cache'])), separators=(',', ':')))
