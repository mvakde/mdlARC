"""Topological melody: periodic error correction over cavity-count classes.

Only the standard library and NumPy are used. No external files, parent code,
examples, sampled metadata, or seed enter the input-only solve path.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _need(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _need(bool(np.all((a >= 0) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _neighbors(r, c, h, w):
    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask)
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc))
            for nr, nc in _neighbors(rr, cc, h, w):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _mask(cells):
    rc = np.asarray(cells, dtype=int)
    lo, hi = rc.min(axis=0), rc.max(axis=0) + 1
    mask = np.zeros(tuple(hi - lo), dtype=bool)
    mask[tuple((rc - lo).T)] = True
    return mask, (int(lo[0]), int(lo[1]))


def _trim(mask):
    return _mask(np.argwhere(mask))[0]


def _cavities(mask):
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    # Padding gives a real exterior even if the carrier touches the canvas.
    return [[(r - 1, c - 1) for r, c in cells]
            for cells in _components(~padded)
            if not any(r == 0 or c == 0 or r == padded.shape[0] - 1
                       or c == padded.shape[1] - 1 for r, c in cells)]


def _holes(mask):
    return len(_cavities(mask))


def _connected(mask):
    return len(_components(mask)) == 1


def _recover(observed):
    """All strict-majority periodic explanations must agree on the repair."""
    observed = tuple(int(k) for k in observed)
    n = len(observed)
    candidates = []
    for period in range(1, n // 3 + 1):
        unit = []
        for phase in range(period):
            values = observed[phase::period]
            count = Counter(values)
            note, votes = max(count.items(), key=lambda item: item[1])
            if 2 * votes <= len(values):
                break
            unit.append(note)
        else:
            repaired = tuple(unit[i % period] for i in range(n))
            errors = sum(a != b for a, b in zip(observed, repaired))
            if errors <= n // 5:
                candidates.append((period, repaired, errors))
    _need(bool(candidates), "no supported periodic topology sequence")
    period, recovered, errors = candidates[0]
    _need(all(c[1] == recovered for c in candidates), "ambiguous periodic topology repair")
    return recovered, period, errors


def _parse_canonical(a, separator_row, rail, bg):
    """Parse an orientation with the key above the separator."""
    h, w = a.shape
    header = a[:separator_row]
    work = np.zeros(a.shape, dtype=bool)
    work[separator_row + 1:] = True
    _need(not np.any(header == rail), "key must not contain rail color")
    _need(np.any(work & (a == rail)), "missing work rail")
    other = [int(c) for c in np.unique(a[work]) if c not in (bg, rail)]
    _need(len(other) == 2, "work area needs query and start roles")
    singletons = [c for c in other if np.count_nonzero(work & (a == c)) == 1]
    _need(len(singletons) == 1, "start must be the unique singleton role")
    start = singletons[0]
    query = next(c for c in other if c != start)
    _need(not np.any((header == query) | (header == start)), "key/work color conflict")

    entries, mapping = [], {}
    for color in np.unique(header):
        color = int(color)
        if color == bg:
            continue
        parts = _components(header == color)
        _need(len(parts) == 1, "each key color needs one glyph")
        mask, (r, c) = _mask(parts[0])
        cavities = _cavities(mask)
        _need(all(header[r + rr, c + cc] == bg
                  for cavity in cavities for rr, cc in cavity), "foreign key cavity contents")
        k = len(cavities)
        _need(0 <= k <= 4 and k not in mapping, "invalid or duplicate key class")
        mapping[k] = color
        entries.append((c, r, k, color))
    _need(2 <= len(entries) <= 4, "key must have 2..4 entries")
    entries.sort()

    path_mask = work & ((a == rail) | (a == start))
    cells = {tuple(int(v) for v in rc) for rc in np.argwhere(path_mask)}
    _need(len(cells) >= 2, "rail too short")
    begin = tuple(int(v) for v in np.argwhere(work & (a == start))[0])
    adjacency = {cell: [x for x in _neighbors(*cell, h, w) if x in cells] for cell in cells}
    _need(sum(len(v) == 1 for v in adjacency.values()) == 2
          and all(1 <= len(v) <= 2 for v in adjacency.values())
          and len(adjacency[begin]) == 1, "rail must be a marked simple path")
    order, visited = [], set()
    current = begin
    while current not in visited:
        visited.add(current)
        order.append(current)
        following = [v for v in adjacency[current] if v not in visited]
        if not following:
            break
        current = following[0]
    _need(visited == cells, "rail disconnected or cyclic")
    path_index = {cell: i for i, cell in enumerate(order)}

    objects = []
    for component in _components(work & (a == query)):
        mask, pos = _mask(component)
        cavities = _cavities(mask)
        _need(all(a[pos[0] + r, pos[1] + c] == bg
                  for cavity in cavities for r, c in cavity), "foreign query cavity contents")
        contacts = sorted({path_index[x] for r, c in component
                           for x in _neighbors(r, c, h, w) if x in path_index})
        _need(bool(contacts), "glyph has no rail contact")
        _need(len(cavities) <= 4, "too many observed cavities")
        objects.append(dict(cells=component, mask=mask, pos=pos,
                            count=len(cavities), contacts=contacts,
                            areas=tuple(len(cavity) for cavity in cavities)))
    objects.sort(key=lambda item: item['contacts'][0])
    _need(6 <= len(objects) <= 12, "work area needs 6..12 notes")
    _need(all(a['contacts'][-1] < b['contacts'][0]
              for a, b in zip(objects, objects[1:])), "interleaving glyph contacts")
    observed = tuple(obj['count'] for obj in objects)
    recovered, period, errors = _recover(observed)
    return dict(grid=a, bg=bg, rail=rail, query=query, start=start,
                mapping=mapping, entries=entries, objects=objects, path=order,
                observed=observed, recovered=recovered, period=period, errors=errors)


def _parse(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _need(all(int(grid[r, c]) == bg for r, c in
              ((0, 0), (0, grid.shape[1] - 1), (grid.shape[0] - 1, 0),
               (grid.shape[0] - 1, grid.shape[1] - 1))), "corners must show background")
    candidates = []
    for rotation in range(4):
        a = np.rot90(grid, rotation)
        rows = [r for r in range(a.shape[0])
                if a[r, 0] != bg and np.all(a[r] == a[r, 0])]
        if len(rows) != 1:
            continue
        r = rows[0]
        if r < 2 or r >= a.shape[0] - 2:
            continue
        try:
            parsed = _parse_canonical(a, r, int(a[r, 0]), bg)
        except ValueError:
            continue
        parsed['rotation'] = rotation
        candidates.append(parsed)
    _need(len(candidates) == 1, "input must have a unique key/rail interpretation")
    return candidates[0]


def _paint(parsed, notes=None, mapping=None):
    output = parsed['grid'].copy()
    if notes is None:
        notes = parsed['recovered']
    if mapping is None:
        mapping = parsed['mapping']
    for obj, k in zip(parsed['objects'], notes):
        output[tuple(np.asarray(obj['cells'], dtype=int).T)] = mapping.get(k, parsed['bg'])
    return np.rot90(output, -parsed['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover and render the answer using only the supplied input pixels."""
    return _paint(_parse(input_grid))


def _compact(k):
    if k == 0:
        return np.ones((2, 3), dtype=bool)
    rows = 1 if k <= 2 else 2
    cols = (k + rows - 1) // rows
    result = np.ones((2 * rows + 1, 2 * cols + 1), dtype=bool)
    for i in range(k):
        result[1 + 2 * (i // cols), 1 + 2 * (i % cols)] = False
    return result


def _make_shape(k, max_h, max_w, rng):
    """Small independent carriers, with finite topology-preserving edits."""
    if k == 0 and rng.random() < 0.4:
        mask = np.zeros((max_h, max_w), dtype=bool)
        mask[int(rng.integers(max_h)), int(rng.integers(max_w))] = True
        target = int(rng.integers(4, max(5, max_h * max_w // 2)))
        for _ in range(target - 1):
            degree = np.zeros_like(mask, dtype=int)
            degree[1:] += mask[:-1]
            degree[:-1] += mask[1:]
            degree[:, 1:] += mask[:, :-1]
            degree[:, :-1] += mask[:, 1:]
            choices = np.argwhere(~mask & (degree == 1))
            if not len(choices):
                break
            candidate = mask.copy()
            candidate[tuple(choices[int(rng.integers(len(choices)))])] = True
            if _holes(candidate) == 0:
                mask = candidate
        return _trim(mask)
    compact = _compact(k)
    if rng.random() < 0.25:
        candidates = [compact, compact.T]
        fitting = [a for a in candidates if a.shape[0] <= max_h and a.shape[1] <= max_w]
        return fitting[int(rng.integers(len(fitting)))].copy()
    ch, cw = compact.shape
    if rng.random() < 0.5 and cw <= max_h and ch <= max_w:
        ch, cw = cw, ch
    height = int(rng.integers(ch, max_h + 1))
    width = int(rng.integers(cw, max_w + 1))
    mask = np.ones((height, width), dtype=bool)
    slots = [(r, c) for r in range(1, height - 1, 2) for c in range(1, width - 1, 2)]
    if k:
        for i in rng.choice(len(slots), size=k, replace=False):
            mask[slots[int(i)]] = False
    for _ in range(12):
        candidate = mask.copy()
        if rng.random() < 0.5:
            # Try enlarging an existing cavity or opening a notch.
            r, c = int(rng.integers(height)), int(rng.integers(width))
        else:
            side = int(rng.integers(4))
            r = 0 if side == 0 else height - 1 if side == 1 else int(rng.integers(height))
            c = 0 if side == 2 else width - 1 if side == 3 else int(rng.integers(width))
        candidate[r, c] = False
        if candidate.any() and _connected(candidate) and _holes(candidate) == k:
            mask = candidate
    result = _trim(mask)
    assert _connected(result) and _holes(result) == k
    return result


def _primitive(unit):
    n = len(unit)
    return not any(n % p == 0 and all(unit[i] == unit[i % p] for i in range(n))
                   for p in range(1, n))


def _sample_notes(rng, split, index):
    if split == 'train' and index == 0:
        n, period = 8, 2
    elif split == 'train' and index == 1:
        n, period = 9, 3
    else:
        n = int(rng.integers(6, 10) if split == 'train' else rng.integers(10, 13))
        period = int(rng.integers(2, n // 3 + 1))
    special = split == 'train' and index < 2
    unit = None
    # A finite distinct-note fallback always has primitive requested period.
    for attempt in range(33):
        if special or attempt == 32:
            proposed = tuple(int(k) for k in rng.choice(5, size=period, replace=False))
        else:
            proposed = tuple(int(k) for k in rng.integers(0, 5, size=period))
        if _primitive(proposed) and len(set(proposed)) >= 2:
            unit = proposed
            break
    assert unit is not None
    clean = tuple(unit[i % period] for i in range(n))
    distinct = list(dict.fromkeys(unit))
    if special:
        keys = distinct if index == 0 else distinct[:2]
        errors = 1
    else:
        key_count = int(rng.choice([2, 2, 3, 3, 4]))
        first = int(rng.choice(distinct))
        keys = [first] + [int(k) for k in rng.choice(
            [k for k in range(5) if k != first], size=key_count - 1, replace=False)]
        errors = 2 if split == 'test' else int(rng.choice([0, 1, 1]))
    # Changing one note in a phase with >=3 observations preserves its majority;
    # distinct-note units give a deterministic fallback if a random unit admits
    # a competing repair. This is count-level rejection, not output synthesis.
    for attempt in range(64):
        observed = list(clean)
        if special:
            wanted = unit[0] if index == 0 else unit[2]
            sites = [i for i, k in enumerate(clean) if k == wanted]
            site = int(rng.choice(sites))
            choices = [k for k in range(5) if k not in keys] if index == 0 else keys
            observed[site] = int(rng.choice(choices))
        elif errors:
            phases = [int(k) for k in rng.choice(period, size=errors, replace=False)]
            for phase in phases:
                site = int(rng.choice(list(range(phase, n, period))))
                observed[site] = int(rng.choice([k for k in range(5) if k != clean[site]]))
        try:
            recovered, found_period, _ = _recover(observed)
        except ValueError:
            continue
        if recovered == clean and found_period == period:
            return clean, tuple(observed), keys
    # With pair lengths <=12, a distinct-note unit and one error per selected
    # phase gives a unique repair. Enumerate a bounded set to cover this case.
    unit = tuple(range(period))
    clean = tuple(unit[i % period] for i in range(n))
    for phase in range(period):
        observed = list(clean)
        sites = [phase] if errors == 1 else [phase, (phase + 1) % period] if errors else []
        for site in sites:
            observed[site] = 4
        recovered, found_period, _ = _recover(observed)
        if recovered == clean and found_period == period:
            return clean, tuple(observed), [0, 1]
    raise RuntimeError('finite note construction exhausted')


def _make_pair(rng, palette, serial, split, index, rotation, reflect):
    clean, observed, keys = _sample_notes(rng, split, index)
    n = len(clean)
    cols = 4 if n >= 10 or n == 8 else 3
    if n in (6, 7):
        cols = int(rng.choice([3, 4]))
    rows = (n + cols - 1) // cols
    body_w = 5 if cols == 4 else int(rng.integers(5, 8))
    body_h = 5 if rows == 3 else int(rng.integers(5, 8))
    width = cols * (body_w + 1) + 5
    # These ordered role pairs are all distinct for the maximum 14 examples.
    bg = palette[serial % 10]
    rail = palette[(serial % 10 + 1 + serial // 10) % 10]
    pool = [c for c in palette if c not in (bg, rail)]
    query, start = (int(c) for c in rng.choice(pool, size=2, replace=False))
    colors = [int(c) for c in rng.permutation([c for c in pool if c not in (query, start)])]
    if len(keys) == 4 and width < 25:
        keys = keys[:3]
    mapping = dict(zip(keys, colors))
    if split == 'train' and index == 0:
        # After reflection and solver normalization the first key is descending.
        key_order = sorted(keys, reverse=not reflect)
    else:
        key_order = [int(k) for k in rng.permutation(keys)]
    glyphs = []
    for k in key_order:
        mask = _compact(k)
        if rng.random() < 0.5:
            mask = mask.T
        glyphs.append(mask)
    header_h = max(mask.shape[0] for mask in glyphs) + 2
    key_width = sum(mask.shape[1] for mask in glyphs) + len(glyphs) - 1
    assert key_width <= width - 2
    height = header_h + 2 + rows * (body_h + 2)
    assert 1 <= height <= 30 and 1 <= width <= 30
    inp = np.full((height, width), bg, dtype=np.int8)
    col = int(rng.integers(1, width - key_width))
    for k, mask in zip(key_order, glyphs):
        top = int(rng.integers(1, header_h - mask.shape[0]))
        inp[top:top + mask.shape[0], col:col + mask.shape[1]][mask] = mapping[k]
        col += mask.shape[1] + 1
    inp[header_h] = rail
    work_top = header_h + 2
    pitch = body_h + 2
    for row in range(rows):
        rr = work_top + row * pitch
        inp[rr, 1:width - 1] = rail
        if row < rows - 1:
            cc = width - 2 if row % 2 == 0 else 1
            inp[rr:rr + pitch + 1, cc] = rail
    inp[work_top, 1] = start
    out = inp.copy()
    stations = [(r, c) for r in range(rows)
                for c in (range(cols) if r % 2 == 0 else range(cols - 1, -1, -1))]
    selected = sorted(int(i) for i in rng.choice(len(stations), size=n, replace=False))
    for i, slot in enumerate(selected):
        row, col = stations[slot]
        mask = _make_shape(observed[i], body_h, body_w, rng)
        rr = work_top + row * pitch + 1
        cc = 3 + col * (body_w + 1) + int(rng.integers(body_w - mask.shape[1] + 1))
        inp[rr:rr + mask.shape[0], cc:cc + mask.shape[1]][mask] = query
        out[rr:rr + mask.shape[0], cc:cc + mask.shape[1]][mask] = mapping.get(clean[i], bg)
    if reflect:
        inp, out = inp[:, ::-1], out[:, ::-1]
    return {'input': np.rot90(inp, rotation).copy(),
            'output': np.rot90(out, rotation).copy()}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    _need(num_train > num_test, 'num_train must exceed num_test')
    rng = np.random.default_rng(int(seed))
    palette = tuple(int(c) for c in rng.permutation(10))
    preferred = int(rng.integers(4))
    regime = int(rng.integers(5))
    puzzle = {'train': [], 'test': []}
    for serial in range(num_train + num_test):
        split = 'train' if serial < num_train else 'test'
        index = serial if split == 'train' else serial - num_train
        if regime == 0:
            rotation = preferred
        elif regime == 1:
            rotation = preferred if split == 'train' else (preferred + 1) % 4
        else:
            rotation = int(rng.integers(4))
        puzzle[split].append(_make_pair(rng, palette, serial, split, index,
                                        rotation, bool(rng.integers(2))))
    return puzzle


def _signature(grid):
    return grid.shape, grid.tobytes()


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _count(len(puzzle['train']), 2, 10, 'training count')
    _count(len(puzzle['test']), 1, 4, 'test count')
    _need(len(puzzle['train']) > len(puzzle['test']), 'counts must satisfy train > test')
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, 'output dimensions changed')
            signature = _signature(inp)
            _need(signature not in seen, 'duplicate input')
            seen.add(signature)
            p = _parse(inp)
            _need(np.array_equal(_paint(p), out), 'incorrect melody transformation')
            canonical_out = np.rot90(out, p['rotation'])
            carrier_mask = p['grid'] == p['query']
            _need(np.array_equal(canonical_out[~carrier_mask], p['grid'][~carrier_mask]),
                  'modified cell outside query carriers')
            for obj, intended in zip(p['objects'], p['recovered']):
                rc = tuple(np.asarray(obj['cells']).T)
                _need(np.all(canonical_out[rc] == p['mapping'].get(intended, p['bg'])),
                      'carrier not painted uniformly')
            if split == 'train':
                direct = _paint(p, notes=p['observed'])
                rank_map = {rank: entry[3] for rank, entry in enumerate(p['entries'])}
                training.append(dict(p=p, direct_diff=not np.array_equal(direct, out),
                                     rank_diff=not np.array_equal(_paint(p, mapping=rank_map), out)))
    _need(all(d['direct_diff'] for d in training[:2]), 'training must distinguish direct lookup')
    _need(training[0]['rank_diff'], 'training must distinguish key-rank lookup')
    first, second = training[0]['p'], training[1]['p']
    _need(any(a != b and a not in first['mapping'] and b in first['mapping']
              for a, b in zip(first['observed'], first['recovered'])),
          'missing retained unlisted-observation witness')
    _need(any(a != b and a in second['mapping'] and b not in second['mapping']
              for a, b in zip(second['observed'], second['recovered'])),
          'missing erased listed-observation witness')
    _need(len({k for d in training for k in d['p']['recovered'] if k in d['p']['mapping']}) >= 2,
          'training needs multiple listed intended classes')


def _features(pair):
    p = _parse(pair['input'])
    shapes = tuple(obj['mask'].shape for obj in p['objects'])
    same_class_shapes = {}
    for obj in p['objects']:
        same_class_shapes.setdefault(obj['count'], set()).add(_signature(obj['mask']))
    path = p['path']
    directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return dict(shape=tuple(pair['input'].shape), period=p['period'], notes=len(p['objects']),
                errors=p['errors'], rotation=p['rotation'],
                roles=(p['bg'], p['rail'], p['query'], p['start']),
                key=tuple(e[2] for e in p['entries']),
                unit=p['recovered'][:p['period']], carrier_shapes=shapes,
                cavity_areas=tuple(a for obj in p['objects'] for a in obj['areas']),
                removed=sum(k not in p['mapping'] for k in p['recovered']),
                unused=sum(k not in p['recovered'] for k in p['mapping']),
                turns=sum(a != b for a, b in zip(directions, directions[1:])),
                shape_witness=any(len(values) > 1 for values in same_class_shapes.values()),
                partial=len(p['objects']) % p['period'] != 0,
                repeated_unit_notes=len(set(p['recovered'][:p['period']])) < p['period'])


def self_test():
    """Bounded sequential sweep; no files or full-puzzle corpus are retained."""
    diagonal = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    assert _holes(diagonal) == 1
    diagonal[0, 1] = False
    assert _holes(diagonal) == 0
    assert _holes(np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool)) == 0
    assert _recover([1, 2, 4, 2, 1, 2, 1, 2])[0] == (1, 2, 1, 2, 1, 2, 1, 2)
    assert _recover([2, 0, 2, 2, 0, 1, 2, 0, 1])[0] == (2, 0, 1, 2, 0, 1, 2, 0, 1)
    assert _recover([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])[1] == 3
    for invalid in ([0, 1, 2, 3, 4, 0], [0, 1, 2, 3, 0, 1]):
        try:
            _recover(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid melody accepted')
    features, hashes, normalized_hashes = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                features.append(_features(pair))
                a = pair['input']
                digest.update(bytes(a.shape) + a.tobytes())
                roles, normalized = {}, []
                for c in a.flat:
                    normalized.append(roles.setdefault(int(c), len(roles)))
                normalized_hashes.add(hashlib.sha256(bytes(a.shape) + bytes(normalized)).digest())
                assert np.array_equal(solve(a.copy()), pair['output'])
        hashes.add(digest.digest())
    assert len(hashes) == 42 and len(normalized_hashes) >= 180
    assert len({f['shape'] for f in features}) >= 15
    assert {f['period'] for f in features} == {2, 3, 4}
    assert {f['notes'] for f in features}.issuperset({6, 7, 8, 9, 10, 11, 12})
    assert {f['errors'] for f in features} == {0, 1, 2}
    assert len({f['key'] for f in features}) >= 30
    assert len({f['unit'] for f in features}) >= 40
    assert len({f['roles'] for f in features}) >= 100
    assert len({f['carrier_shapes'] for f in features}) >= 150
    assert sum(f['shape_witness'] for f in features) >= 180
    assert any(f['repeated_unit_notes'] for f in features)
    assert any(f['unused'] for f in features) and any(f['partial'] for f in features)
    assert any(f['removed'] == 0 for f in features) and any(f['removed'] >= 3 for f in features)
    assert all(f['turns'] >= 2 for f in features)
    base = generate_puzzle(83, 2, 1)
    again = generate_puzzle(83, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in base
               for a, b in zip(base[s], again[s]) for k in a)
    # Input-only geometry and palette equivariance, including reflection.
    permutation = np.array([7, 3, 9, 1, 6, 0, 8, 4, 2, 5], dtype=np.int8)
    for pair in base['train'] + base['test']:
        for k in range(4):
            inp = permutation[np.rot90(pair['input'][:, ::-1], k)]
            out = permutation[np.rot90(pair['output'][:, ::-1], k)]
            assert np.array_equal(solve(inp), out)
    # Every explicit legal count combination, with only one puzzle alive at once.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + 10 * a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for role in ('glyph', 'other'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in base[s]] for s in base}
        pair = bad['test'][0]
        parsed = _parse(pair['input'])
        # Choose coordinates in the original orientation for deliberate damage.
        mask = np.rot90(parsed['grid'] == parsed['query'], -parsed['rotation'])
        cell = tuple(np.argwhere(mask if role == 'glyph' else ~mask)[0])
        pair['output'][cell] = (int(pair['output'][cell]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('corrupt answer accepted')
    return dict(seeds=42, pairs=len(features), distinct_puzzles=len(hashes),
                normalized_inputs=len(normalized_hashes),
                shapes=len({f['shape'] for f in features}),
                units=len({f['unit'] for f in features}),
                keys=len({f['key'] for f in features}),
                carrier_geometry=len({f['carrier_shapes'] for f in features}),
                shape_witness_pairs=sum(f['shape_witness'] for f in features),
                partial_pairs=sum(f['partial'] for f in features),
                unused_key_pairs=sum(bool(f['unused']) for f in features),
                repeated_note_units=sum(f['repeated_unit_notes'] for f in features),
                independent_solver=False)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
