"""Signal-valued branching and repeated majority voting at transverse gates.

Only standard-library Python and NumPy are used. The solver reads geometric
roles from the input; construction settings and seeds are never consulted.
"""
from collections import Counter
import numpy as np


def _ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        todo, cells = [p], [p]
        while todo:
            r, c = todo.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
                    cells.append(q)
        result.append(cells)
    return result


def _runs(row, neutral):
    result = []
    c = 0
    while c < len(row):
        if row[c] != neutral:
            c += 1
            continue
        start = c
        while c+1 < len(row) and row[c+1] == neutral:
            c += 1
        result.append((start, c))
        c += 1
    return result


def _interpret(a, bg, neutral, signals):
    h, w = a.shape
    panels = []
    stub_mask = np.zeros(a.shape, dtype=bool)
    mask = np.isin(a, signals)
    for cells in _components(mask):
        rr, cc = zip(*cells)
        ph, left, right = max(rr)+1, min(cc), max(cc)
        pw = right-left+1
        if (min(rr) != 0 or not 2 <= ph <= 5 or not 2 <= pw <= 5
                or len(cells) != ph*pw or ph+2 >= h):
            raise ValueError("sources must be separated boundary rectangles")
        block = a[:ph, left:right+1]
        counts = [int(np.sum(block == s)) for s in signals]
        if min(counts) == 0 or counts[0] == counts[1]:
            raise ValueError("source vote must be mixed and strict")
        columns = np.flatnonzero(a[ph, left:right+1] == neutral)+left
        if len(columns) != 1:
            raise ValueError("source requires one attached launch stub")
        c = int(columns[0])
        if a[ph+1, c] != neutral:
            raise ValueError("launch stub must have length two")
        stub_mask[ph:ph+2, c] = True
        panels.append(dict(height=ph, left=left, right=right, column=c,
                           label=signals[int(counts[1] > counts[0])],
                           counts=counts, contact=int(a[ph-1, c])))
    if not 2 <= len(panels) <= 5:
        raise ValueError("expected 2..5 source panels")
    panels.sort(key=lambda p: p['left'])
    if any(x['right']+1 >= y['left'] for x, y in zip(panels, panels[1:])):
        raise ValueError("panels need a background column between them")
    start = max(p['height'] for p in panels)+2
    if not np.array_equal(a[:start] == neutral, stub_mask[:start]):
        raise ValueError("unrecognized neutral cell in the launch band")
    gates = {}
    for r in range(start, h):
        row_runs = _runs(a[r], neutral)
        if row_runs:
            if r <= start or any((b-c+1) % 2 != 1 for c, b in row_runs):
                raise ValueError("gates need an upstream blank layer and odd length")
            if any(y[0]-x[1] < 3 for x, y in zip(row_runs, row_runs[1:])):
                raise ValueError("separate gates need two blank flank columns")
            gates[r] = row_runs
    if not gates or any(y-x < 2 for x, y in zip(gates, list(gates)[1:])):
        raise ValueError("gate layers must have a blank layer between them")
    return dict(grid=a, background=bg, neutral=neutral, signals=signals,
                panels=panels, start=start, gates=gates, panel_mask=mask,
                stub_mask=stub_mask)


def _parse(value):
    grid = _grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 4 or np.sum(counts == counts.max()) != 1:
        raise ValueError("input needs four roles and a uniquely dominant background")
    bg = int(colors[np.argmax(counts)])
    candidates = []
    for rotation in range(4):
        a = np.rot90(grid, -rotation).copy()
        for neutral in map(int, colors):
            if neutral == bg:
                continue
            signals = tuple(int(s) for s in colors if s not in (bg, neutral))
            try:
                info = _interpret(a, bg, neutral, signals)
                info['rotation'] = rotation
                candidates.append(info)
            except ValueError:
                pass
    if len(candidates) != 1:
        raise ValueError("roles and entry direction must have one interpretation")
    return candidates[0]


def _vote(labels):
    counts = Counter(labels)
    if not counts:
        return None
    winners = [s for s, n in counts.items() if n == max(counts.values())]
    return winners[0] if len(winners) == 1 else None


def _transition(active, runs, width, ignore_fanout=False):
    hit_columns = {c for c in active if any(a <= c <= b for a, b in runs)}
    following = {c: s for c, s in active.items() if c not in hit_columns}
    events, mergers = [], 0
    for a, b in runs:
        columns = sorted(c for c in active if a <= c <= b)
        if not columns:
            continue
        counts = Counter(active[c] for c in columns)
        winner = _vote(active[c] for c in columns)
        events.append(dict(left=a, right=b, columns=columns,
                           counts=dict(counts), winner=winner))
        if winner is None:
            continue
        outgoing = columns if ignore_fanout else [c for c in (a-1, b+1) if 0 <= c < width]
        for c in outgoing:
            if c in following:
                if following[c] != winner:
                    raise ValueError("opposite-colored rays cannot collide in open space")
                mergers += 1
            following[c] = winner
    return following, events, mergers


def _evaluate(info, shortcut=None):
    a = info['grid']
    h, w = a.shape
    bg, neutral = info['background'], info['neutral']
    out = a.copy()
    active = {}
    for p in info['panels']:
        label = p['contact'] if shortcut == 'contact_color' else p['label']
        active[p['column']] = label
        out[p['height']:info['start'], p['column']] = label
    reached, events = set(), []
    simultaneous = mergers = 0
    for r in range(info['start'], h):
        runs = info['gates'].get(r, [])
        following, row_events, merged = _transition(active, runs, w, shortcut == 'no_fanout')
        mergers += merged
        simultaneous += len(row_events) >= 2
        # Broadcast crosspieces intentionally replace the immediately upstream
        # beam segments: their uniform ink displays the gate's consensus.
        for e in row_events:
            left, right, label = e['left'], e['right'], e['winner']
            ink = neutral if label is None else label
            out[r-1, max(0, left-1):min(w, right+2)] = ink
            if label is not None:
                out[r, (left+right)//2] = label
            reached.add((r, left, right))
            events.append(dict(e, row=r))
        for c, label in following.items():
            if a[r, c] == bg:
                out[r, c] = label
        active = following
    all_gates = {(r, x, y) for r, runs in info['gates'].items() for x, y in runs}
    inert = all_gates-reached
    source_counts = Counter(p['label'] for p in info['panels'])
    raw_counts = {s: int(np.sum(a == s)) for s in info['signals']}
    minority_winners = [e['winner'] for e in events if e['winner'] is not None
                        and len(e['counts']) == 2
                        and source_counts[e['winner']] < len(info['panels'])/2]
    stats = dict(events=events, reached=reached, inert=inert, simultaneous=simultaneous,
                 mergers=mergers, ties=sum(e['winner'] is None for e in events),
                 splits=sum(e['winner'] is not None and e['left'] > 0 and e['right'] < w-1
                            for e in events),
                 clipped=sum(e['winner'] is not None and (e['left'] == 0 or e['right'] == w-1)
                             for e in events),
                 minority_panel_winner=bool(minority_winners),
                 raw_disagrees=any(raw_counts[s] < raw_counts[t] for s in minority_winners
                                   for t in info['signals'] if t != s),
                 reached_lengths={b-c+1 for _, c, b in reached},
                 inert_lengths={b-c+1 for _, c, b in inert})
    return out, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer entry orientation and roles, then evaluate the voting circuit."""
    info = _parse(input_grid)
    out, _ = _evaluate(info)
    return np.ascontiguousarray(np.rot90(out, info['rotation']), dtype=np.int8)


def _pattern(rng, ph, pw, label, contact, force_contact=False, small=False):
    area = ph*pw
    if small:
        minority_count = 1
    else:
        minority_count = int(rng.integers(1, (area-1)//2+1))
    style = int(rng.integers(3))
    if style == 0:
        order = list(map(int, rng.permutation(area)))
    elif style == 1:
        rr, cc = np.indices((ph, pw))
        r, c = int(rng.integers(ph)), int(rng.integers(pw))
        order = list(map(int, np.argsort((abs(rr-r)+abs(cc-c)+rng.random((ph, pw))*.1).flat)))
    else:
        order = list(range(area))
        if rng.random() < .5:
            order.reverse()
    if force_contact:
        order.remove(contact)
        order.insert(0, contact)
    flat = np.full(area, label, dtype=np.int8)
    flat[order[:minority_count]] = 5-label  # canonical signals are 2 and 3
    return flat.reshape(ph, pw)


def _source(rng, grid, band, column, label, force=False, small=False, large=False):
    lo, hi = band
    pw = int(rng.integers(2, min(5, hi-lo+1)+1))
    ph = int(rng.integers(2, 6))
    if small:
        ph, pw = 2, 2
    if large:
        ph, pw = 5, min(5, hi-lo+1)
    left = int(rng.integers(max(lo, column-pw+1), min(column, hi-pw+1)+1))
    grid[:ph, left:left+pw] = _pattern(rng, ph, pw, label,
                                      (ph-1)*pw+column-left, force, small)
    grid[ph:ph+2, column] = 1
    return dict(height=ph, column=column, label=label)


def _odd_intervals(width, containing, existing=(), inert=False):
    for left in range(width):
        for right in range(left, width, 2):
            if any(not (right+3 <= a or b+3 <= left) for a, b in existing):
                continue
            hits = any(left <= c <= right for c in containing)
            if hits != (not inert):
                continue
            yield left, right


def _choose(rng, options):
    return options[int(rng.integers(len(options)))]


def _amplify(rng, label):
    w = int(rng.integers(26, 31))
    grid = np.zeros((30, w), dtype=np.int8)
    c0, c1, c2 = int(rng.integers(6, 9)), int(rng.integers(15, 19)), int(rng.integers(w-4, w-1))
    sources = [_source(rng, grid, (1, 10), c0, label, True, small=True),
               _source(rng, grid, (13, 19), c1, 5-label, large=True),
               _source(rng, grid, (21, w-1), c2, 5-label, large=True)]
    start = max(p['height'] for p in sources)+2
    n = int(rng.choice([1, 3, 5]))
    first = (c0-n//2, c0+n//2)
    # The other gate fans out one opposing beam onto the second opposing beam.
    # Their same-colored coincidence is one occupied position, not two votes.
    end = c2-1
    second = (int(_choose(rng, [x for x in range(max(13, first[1]+3), c1+1)
                              if (end-x) % 2 == 0])), end)
    row = start+int(rng.integers(3, 5))
    gate_rows = [row, row+int(rng.integers(2, 4))]
    gate_rows.append(gate_rows[-1]+int(rng.integers(2, 4)))
    gate_rows.append(gate_rows[-1]+int(rng.integers(2, 4)))
    for a, b in (first, second):
        grid[gate_rows[0], a:b+1] = 1
    pivot = first[0]-1
    n2 = 1 if n == 1 else int(rng.choice([1, 3]))
    fork2 = (pivot-n2//2, pivot+n2//2)
    grid[gate_rows[1], fork2[0]:fork2[1]+1] = 1
    active = {p['column']: p['label'] for p in sources}
    active, _, _ = _transition(active, [first, second], w)
    active, _, _ = _transition(active, [fork2], w)
    containing = [interval for interval in _odd_intervals(w, active)
                  if interval[0] <= min(active) and interval[1] >= max(active)
                  and interval != (0, w-1)]
    third = _choose(rng, containing)
    grid[gate_rows[2], third[0]:third[1]+1] = 1
    active, _, _ = _transition(active, [third], w)
    clipped = [x for x in _odd_intervals(w, active)
               if (x[0] == 0 or x[1] == w-1) and x != (0, w-1)]
    fourth = _choose(rng, clipped)
    grid[gate_rows[3], fourth[0]:fourth[1]+1] = 1
    # An equal-length unreachable gate is placed before any branching.
    inert = [x for x in _odd_intervals(w, [c0, c1, c2], inert=True) if x[1]-x[0]+1 == n]
    a, b = _choose(rng, inert)
    grid[start+1, a:b+1] = 1
    h = int(rng.integers(gate_rows[-1]+1, min(30, gate_rows[-1]+7)+1))
    return grid[:h].copy()


def _ordinary_sources(rng, grid, count, labels=None):
    w = grid.shape[1]
    gaps = count-1
    available = w-gaps
    sizes = [available//count+(i < available % count) for i in range(count)]
    left = 0
    sources = []
    for i, size in enumerate(sizes):
        hi = left+size-1
        c = int(rng.integers(left, hi+1))
        label = int(rng.choice([2, 3])) if labels is None else labels[i]
        sources.append(_source(rng, grid, (left, hi), c, label, force=i == 0))
        left = hi+2
    return sources


def _tie(rng):
    h, w = int(rng.integers(15, 27)), int(rng.choice([13, 15, 17, 19, 21, 23]))
    grid = np.zeros((h, w), dtype=np.int8)
    sources = _ordinary_sources(rng, grid, 2, [2, 3] if rng.random() < .5 else [3, 2])
    start = max(p['height'] for p in sources)+2
    columns = sorted(p['column'] for p in sources)
    options = [x for x in _odd_intervals(w, columns)
               if x[0] <= columns[0] and x[1] >= columns[-1]]
    a, b = _choose(rng, options)
    grid[start+3, a:b+1] = 1
    if rng.random() < .7:
        a, b = _choose(rng, list(_odd_intervals(w, columns, inert=True)))
        grid[start+1, a:b+1] = 1
    return grid


def _random_circuit(rng, mode, test):
    count = int(rng.integers(4, 6)) if test and mode == 1 else int(rng.integers(2, 4 if mode == 1 else 6))
    w = int(rng.integers(max(12, count*4-1), 31))
    h = int(rng.integers(26, 31)) if test and mode == 0 else int(rng.integers(15, 31))
    grid = np.zeros((h, w), dtype=np.int8)
    sources = _ordinary_sources(rng, grid, count)
    start = max(p['height'] for p in sources)+2
    active = {p['column']: p['label'] for p in sources}
    r = start+int(rng.integers(1, 4))
    while r < h:
        runs = []
        quota = int(rng.integers(1, 4))
        for _ in range(quota):
            options = []
            for interval in _odd_intervals(w, active, runs):
                try:
                    _transition(active, sorted(runs+[interval]), w)
                    options.append(interval)
                except ValueError:
                    pass
            if not options:
                break
            # Mix local and wide gates without linking length to causal status.
            short = [x for x in options if x[1]-x[0]+1 <= 5]
            interval = _choose(rng, short if short and rng.random() < .55 else options)
            runs.append(interval)
        if rng.random() < .65:
            inert = list(_odd_intervals(w, active, runs, True))
            if inert:
                runs.append(_choose(rng, inert))
        if runs:
            for a, b in runs:
                grid[r, a:b+1] = 1
            active, _, _ = _transition(active, sorted(runs), w)
        r += int(rng.integers(2, 5))
    return grid


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))  # deeper circuits, more sources, or rotated entry
    palette = np.array(rng.choice(10, 4, replace=False), dtype=np.int8)
    varying_palette = bool(rng.random() < .5)
    rotation = int(rng.integers(4))
    minority_label = int(rng.choice([2, 3]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            for attempt in range(80):
                if split == 'train' and index == 0:
                    canonical = _amplify(rng, minority_label)
                elif split == 'train' and index == 1:
                    canonical = _tie(rng)
                else:
                    canonical = _random_circuit(rng, mode, split == 'test')
                if rng.random() < .5:
                    canonical = np.fliplr(canonical).copy()
                roles = np.array(rng.choice(10, 4, replace=False), dtype=np.int8) if varying_palette else palette
                k = (rotation+int(rng.integers(1, 4))) % 4 if split == 'test' and mode == 2 else rotation
                inp = np.ascontiguousarray(np.rot90(roles[canonical], k))
                if _key(inp) in seen:
                    continue
                try:
                    info = _parse(inp)
                    out, stats = _evaluate(info)
                    if not stats['reached']:
                        continue
                    out = np.ascontiguousarray(np.rot90(out, info['rotation']))
                    if split == 'train' and index == 0:
                        if not (stats['minority_panel_winner'] and stats['raw_disagrees']
                                and stats['mergers'] and stats['simultaneous'] and stats['clipped']
                                and stats['reached_lengths'] & stats['inert_lengths']):
                            continue
                    if split == 'train' and index == 1 and not stats['ties']:
                        continue
                except ValueError:
                    continue
                seen.add(_key(inp))
                puzzle[split].append({'input': inp, 'output': out})
                break
            else:
                raise RuntimeError("bounded voting-circuit construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _ensure(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _ensure(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _ensure(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal counts")
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _ensure(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            a, out = _grid(pair['input']), _grid(pair['output'])
            _ensure(a.shape == out.shape and not np.array_equal(a, out), "invalid shape or identity pair")
            _ensure(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            info = _parse(a)
            prediction, stats = _evaluate(info)
            expected = np.rot90(prediction, info['rotation'])
            _ensure(np.array_equal(expected, out), "incorrect voting propagation")
            canonical_out = np.rot90(out, -info['rotation'])
            original = info['grid']
            _ensure(np.array_equal(canonical_out[info['panel_mask']], original[info['panel_mask']]),
                    "panel pixels must be preserved, including minorities")
            _ensure(bool(stats['reached']), "scene lacks a causal gate")
            for r, runs in info['gates'].items():
                for x, y in runs:
                    immutable = [c for c in range(x, y+1) if c != (x+y)//2]
                    _ensure(np.all(canonical_out[r, immutable] == info['neutral']), "gate border overwritten")
                    if (r, x, y) in stats['inert']:
                        _ensure(np.all(canonical_out[r, x:y+1] == info['neutral']), "unreached gate changed")
            if split == 'train':
                training.append((info, stats, prediction))
    _ensure(any(s['minority_panel_winner'] and s['raw_disagrees'] for _, s, _ in training),
            "training must defeat source-count and raw-cell shortcuts")
    for feature in ('splits', 'ties', 'clipped', 'simultaneous', 'mergers'):
        _ensure(any(s[feature] for _, s, _ in training), f"training lacks {feature}")
    _ensure(any(s['reached_lengths'] & s['inert_lengths'] for _, s, _ in training),
            "training lacks a same-length reachability contrast")
    for shortcut in ('no_fanout', 'contact_color'):
        differs = False
        for info, _, expected in training:
            try:
                wrong, _ = _evaluate(info, shortcut)
                differs |= not np.array_equal(wrong, expected)
            except ValueError:
                differs = True
        _ensure(differs, f"training does not reject {shortcut}")


def self_test():
    """Bounded, input-derived invariant, mutation, covariance and diversity QA."""
    shapes, structures, palettes, rotations = set(), set(), set(), set()
    source_counts, gate_lengths, gate_counts, profiles = set(), set(), set(), set()
    feature_totals = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed < 3:
            repeat = generate_puzzle(seed)
            _ensure(all(_key(p[k]) == _key(q[k]) for s in puzzle
                        for p, q in zip(puzzle[s], repeat[s]) for k in p), "nondeterministic seed")
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, output = pair['input'], pair['output']
                info = _parse(inp)
                _, stats = _evaluate(info)
                _ensure(np.array_equal(solve(inp.tolist()), output), "input-only solve mismatch")
                shapes.add(inp.shape)
                rotations.add(info['rotation'])
                palettes.add((info['background'], info['neutral'], *info['signals']))
                normalized = np.zeros(info['grid'].shape, dtype=np.int8)
                for i, c in enumerate((info['background'], info['neutral'], *info['signals'])):
                    normalized[info['grid'] == c] = i
                structures.add(_key(normalized))
                source_counts.add(len(info['panels']))
                gate_counts.add(sum(map(len, info['gates'].values())))
                gate_lengths |= {b-a+1 for runs in info['gates'].values() for a, b in runs}
                for e in stats['events']:
                    profiles.add(tuple(sorted(e['counts'].values())))
                for feature in ('ties', 'splits', 'clipped', 'mergers', 'simultaneous', 'minority_panel_winner'):
                    feature_totals[feature] += bool(stats[feature])
                if seed < 3:
                    # The same circuit is covariant under rotation and arbitrary
                    # bijective recoloring. Neither operation changes its votes.
                    for k in range(4):
                        _ensure(np.array_equal(solve(np.rot90(inp, k)), np.rot90(output, k)), "rotation failure")
                    mapping = np.arange(10, dtype=np.int8)[::-1]
                    _ensure(np.array_equal(solve(mapping[inp]), mapping[output]), "palette failure")
    _ensure(len(shapes) >= 50 and len(structures) >= 180, "geometry variation collapsed")
    _ensure(len(palettes) >= 30 and rotations == set(range(4)), "role variation collapsed")
    _ensure(source_counts == {2, 3, 4, 5}, "source-count coverage missing")
    _ensure(len(gate_lengths) >= 12 and len(gate_counts) >= 6 and len(profiles) >= 8,
            "gate topology or vote diversity collapsed")
    _ensure(all(feature_totals[x] >= 42 for x in feature_totals), "family witness missing")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(900+nt*10+nq, nt, nq)
            _ensure(len(p['train']) == nt and len(p['test']) == nq, "counts not honored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p = generate_puzzle(17, 2, 1)
    q = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    info = _parse(q['train'][0]['input'])
    event = next(e for e in _evaluate(info)[1]['events'] if e['winner'] is not None)
    changed = np.rot90(q['train'][0]['output'], -info['rotation']).copy()
    changed[event['row'], (event['left']+event['right'])//2] = info['neutral']
    q['train'][0]['output'] = np.rot90(changed, info['rotation']).copy()
    try:
        validate_generated_puzzle(q)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong gate readout accepted")
    return dict(seeds=42, pairs=210, shapes=len(shapes), structures=len(structures),
                palettes=len(palettes), source_counts=sorted(source_counts),
                gate_lengths=sorted(gate_lengths), vote_profiles=sorted(profiles),
                feature_totals=dict(feature_totals))
