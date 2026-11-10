"""Mandatory dataset-wide pair uniqueness and successful-author participation."""
from collections import Counter
from contextlib import contextmanager
import fcntl
from itertools import combinations
from pathlib import Path

from v2_storage import read, write

POLICY = 'unique-least-used-v1'
LEDGER = '.composition-v3-pair-reservations.json'


@contextmanager
def dataset_lock(plan):
    collection = Path(plan['parents_root']) / plan['dataset']
    with (collection / '.composition-v3-pair-order.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another v3 prepare/run owns this dataset; resume it before starting another') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def ledger_path(plan):
    return Path(plan['parents_root']) / plan['dataset'] / LEDGER


def pair_key(pair):
    if len(pair) != 2 or pair[0] == pair[1]:
        raise ValueError('Each composition must have two distinct parents')
    return tuple(sorted(pair))


def load_ledger(plan):
    path = ledger_path(plan)
    ledger = read(path) if path.exists() else {
        'schema_version': 4, 'policy': POLICY,
        'schedule_fingerprint': plan['pair_schedule']['fingerprint'],
        'next_serial': 1, 'roots': {}, 'retired': []}
    if ledger.get('schema_version') != 4 or ledger.get('policy') != POLICY:
        raise ValueError('Legacy pair queues need cleanup.py before another v3 batch')
    if ledger['schedule_fingerprint'] != plan['pair_schedule']['fingerprint']:
        raise ValueError('Reservation ledger belongs to another dataset order')
    used = set()
    ids = {p['id'] for p in plan['candidates']}
    for record in [r for rows in ledger['roots'].values() for r in rows] + ledger['retired']:
        pair = pair_key(record['parents'])
        if not set(pair) <= ids:
            raise ValueError('Ledger contains parents outside the pinned dataset')
        if pair in used:
            raise ValueError(f'Duplicate unordered pair in dataset ledger: {pair}')
        used.add(pair)
    return ledger


def author_complete(root, name):
    path = Path(root) / name / 'checkpoints.json'
    return path.exists() and bool(read(path).get('author'))


def completed_counts(plan, ledger):
    counts = Counter({p['id']: 0 for p in plan['candidates']})
    for root, records in ledger['roots'].items():
        for record in records:
            if author_complete(root, record['name']):
                counts.update(record['parents'])
    return counts


def check_single_queue(ledger, root):
    root = Path(root).resolve()
    for other, records in ledger['roots'].items():
        if other != str(root) and any(not author_complete(other, r['name']) for r in records):
            raise ValueError(f'Finish or clean up the unfinished composition batch first: {other}')


def check_plan(ledger, root, plan):
    root = Path(root).resolve()
    records = {r['name']: r for r in ledger['roots'].get(str(root), [])}
    seen = set()
    for job in plan['jobs']:
        pair = pair_key(job['parents'])
        if pair in seen:
            raise ValueError('Plan repeats an unordered pair')
        seen.add(pair)
        if job['name'] not in records or pair_key(records[job['name']]['parents']) != pair:
            raise ValueError(f'Plan differs from the shared pair ledger: {job["name"]}')


def least_used_pairs(task_ids, counts, used, count, task_order):
    """Always advance the lowest level; never silently skip an exhausted task."""
    counts = Counter({t: counts[t] for t in task_ids})
    unused = [p for p in combinations(task_ids, 2) if pair_key(p) not in used]
    if len(unused) < count:
        raise ValueError('Requested batch exceeds remaining unique dataset pairs')
    positions = {task_ids[index]: position for position, index in enumerate(task_order)}
    selected, steps = [], []
    for _ in range(count):
        pair = min(unused, key=lambda p: (*sorted((counts[p[0]], counts[p[1]])),
                                         *sorted((positions[p[0]], positions[p[1]]))))
        if min(counts[p] for p in pair) != min(counts.values()):
            raise ValueError('Least-used tasks have no unused partner; cannot advance without repeating a pair')
        steps.append({'parents': list(pair), 'counts_before': [counts[p] for p in pair]})
        selected.append(list(pair))
        counts.update(pair)
        unused.remove(pair)
    return selected, counts, steps


def allocate(root, plan, count):
    """Reserve all entry points through one locked, explicit dataset ledger."""
    root = Path(root).resolve()
    with dataset_lock(plan):
        ledger = load_ledger(plan)
        check_single_queue(ledger, root)
        check_plan(ledger, root, plan)
        records = ledger['roots'].setdefault(str(root), [])
        if count < len(records):
            raise ValueError('Output root already reserves a larger batch; resume its original count')
        counts = completed_counts(plan, ledger)
        # Forecast only this queue while preparing; reported/running counts are
        # always successful author checkpoints, never these reservations.
        for record in records:
            if not author_complete(root, record['name']):
                counts.update(record['parents'])
        used = {pair_key(r['parents']) for rows in ledger['roots'].values() for r in rows}
        used.update(pair_key(r['parents']) for r in ledger['retired'])
        ids = [p['id'] for p in plan['candidates']]
        pairs, _, steps = least_used_pairs(ids, counts, used, count - len(records),
                                         plan['pair_schedule']['task_order'])
        for pair, step in zip(pairs, steps):
            serial = ledger['next_serial']
            ledger['next_serial'] += 1
            records.append({'name': f'{serial:06d}-{pair[0]}-{pair[1]}', 'parents': pair,
                            'counts_before': step['counts_before']})
        write(ledger_path(plan), ledger)
        return records
