"""Hidden drains: arithmetic switch lattices, rewiring, and bead conservation.

Only Python/NumPy; no I/O, examples, seed state, or generation metadata in solve.
The canonical orientation has horizontal lanes and the erased drain below them.
Canonicalization is recovered from the input, never from a puzzle parameter.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _need(condition, message="invalid hidden-drain lattice"):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_), "integer cells required")
    _need(bool(np.all((0 <= a) & (a <= 9))), "grid colors")
    return a.astype(np.int8, copy=True)


def _corner_cells(box):
    r0, r1, c0, c1 = box
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _markers(a):
    candidates = []
    for value in np.unique(a):
        cells = np.argwhere(a == value)
        if len(cells) != 6:
            continue
        r0, c0 = map(int, cells.min(axis=0))
        r1, c1 = map(int, cells.max(axis=0))
        if r1 - r0 < 1 or c1 - c0 < 1 or r1 - r0 + c1 - c0 < 4:
            continue
        box = r0, r1, c0, c1
        if set(map(tuple, cells)) == _corner_cells(box):
            candidates.append((int(value), box))
    _need(len(candidates) == 1, "one unambiguous pair of corner Ls required")
    return candidates[0]


def _neighbors(cell, cells):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in _DIRECTIONS if (r + dr, c + dc) in cells]


def _paths(structural, track):
    """Input-side solution: visit and order complete four-neighbor paths."""
    cells = set(map(tuple, np.argwhere(structural == track)))
    remaining, paths = set(cells), []
    while remaining:
        start = min(remaining)
        component, todo = set(), [start]
        while todo:
            cell = todo.pop()
            if cell in component:
                continue
            component.add(cell)
            todo.extend(p for p in _neighbors(cell, cells) if p not in component)
        remaining.difference_update(component)
        degrees = {p: len(_neighbors(p, cells)) for p in component}
        _need(all(d in (1, 2) for d in degrees.values()), "branch or isolated cell")
        ends = sorted(p for p, d in degrees.items() if d == 1)
        _need(len(ends) == 2, "closed track")
        ordered, previous = [ends[0]], None
        while ordered[-1] != ends[1]:
            choices = [p for p in _neighbors(ordered[-1], cells) if p != previous]
            _need(len(choices) == 1)
            previous = ordered[-1]
            ordered.append(choices[0])
        _need(len(ordered) == len(component))
        steps = {b[0] - a[0] for a, b in zip(ordered, ordered[1:]) if a[0] != b[0]}
        _need(len(steps) <= 1, "track must have no gravity trap")
        paths.append(ordered)
    return paths


def _deposit(structural, track, bead, paths, sources):
    answer = structural.copy()
    destinations, counts = set(), []
    for path in paths:
        occupants = sources.intersection(path)
        counts.append(len(occupants))
        if path[0][0] == path[-1][0]:
            targets = occupants
        else:
            inward = path if path[0][0] > path[-1][0] else path[::-1]
            targets = set(inward[:len(occupants)])
        destinations.update(targets)
        for p in targets:
            answer[p] = bead
    _need(sum(counts) == len(sources) == len(destinations), "beads lost")
    return answer, destinations, counts


def _render(model, turned):
    """Input-side cell construction from the fitted progression."""
    bg, track = model['background'], model['track']
    result = np.full(model['shape'], bg, dtype=np.int8)
    top, left, right, n = (model[k] for k in ('top', 'left', 'right', 'separators'))
    bottom = top + 2 * n
    result[top:bottom + 1:2, left:right + 1] = track
    rows = []
    for i in range(n):
        count = model['count'] + model['growth'] * i
        centers = [model['first'] + model['drift'] * i + model['spacing'] * j for j in range(count)]
        rows.append(centers)
    for r, centers in ((top, rows[0]), (bottom, rows[-1])):
        for a, b in zip(centers, centers[1:]):
            result[r, (a + b) // 2] = bg
    # Writes are checked, including compatible overlap of adjacent tiles.
    writes = {}
    for i, centers in enumerate(rows):
        r, color = top + 2 * i + 1, model['cycle'][i % len(model['cycle'])]
        for c in centers:
            if turned:
                for y in range(r - 1, r + 2):
                    for x in range(c - 1, c + 2):
                        value = color if x == c else track
                        _need((y, x) not in writes or writes[y, x] == value, "conflicting switch tiles")
                        writes[y, x] = value
            else:
                result[r, c - 1:c + 2] = color
    for cell, value in writes.items():
        result[cell] = value
    return result, rows


def _linear(values):
    _need(len(values) >= 3)
    delta = values[1] - values[0]
    _need(all(value == values[0] + delta * i for i, value in enumerate(values)), "nonarithmetic evidence")
    return values[0], delta


def _fit(a, mask, bg, marker, rotation):
    """Fit only fully exposed rows, then check ALL exposed pixels."""
    pos = np.argwhere(a != bg)
    _need(len(pos) > 0)
    top, left = map(int, pos.min(axis=0))
    last, right = map(int, pos.max(axis=0))
    holes = np.argwhere(mask)
    r0, c0 = map(int, holes.min(axis=0))
    r1, c1 = map(int, holes.max(axis=0))
    _need(top + 6 <= r0 and last <= r1 and (r1 - top) % 2 == 0)
    n = (r1 - top) // 2
    visible = (r0 - top) // 2
    _need(3 <= visible < n <= 13 and right - left + 1 >= 9)
    _need(c0 <= left or c1 >= right, "window must contain a lateral drain edge")
    centers, colors = [], []
    for i in range(visible):
        line = a[top + 2 * i + 1, left:right + 1]
        values = set(map(int, np.unique(line))) - {bg}
        _need(len(values) == 1, "each exposed switch row has one color")
        color = next(iter(values))
        locations = np.flatnonzero(line == color).tolist()
        groups = np.split(locations, np.flatnonzero(np.diff(locations) > 1) + 1)
        _need(all(len(group) == 3 for group in groups), "three-cell dividers required")
        centers.append([left + int(group[1]) for group in groups])
        colors.append(color)
    count, growth = _linear([len(row) for row in centers])
    first, drift = _linear([row[0] for row in centers])
    _need(growth in (-1, 0, 1) and 2 <= abs(drift) <= 6)
    distances = {b - a for row in centers for a, b in zip(row, row[1:])}
    if distances:
        _need(len(distances) == 1)
        spacing = next(iter(distances))
        _need(spacing in (4, 6, 8) and 2 <= abs(drift) <= spacing - 2)
    else:
        _need(count == 1 and growth == 0)
        spacing = 2 * ((abs(drift) + 3) // 2)  # Irrelevant for a one-divider row.
    _need(growth == 0 or growth * drift < 0, "growth must interlace")
    periods = [p for p in range(1, 4) if visible >= max(3, p + 1)
               and len(set(colors[:p])) == p and all(v == colors[i % p] for i, v in enumerate(colors))]
    _need(len(periods) == 1, "color cycle needs a repeated observation")
    cycle = tuple(colors[:periods[0]])
    rest = set(map(int, np.unique(a))) - {bg, *cycle}
    _need(len(rest) == 2, "distinct track and bead roles required")
    ranked = sorted(rest, key=lambda x: int(np.count_nonzero(a == x)))
    bead, track = ranked
    _need(np.count_nonzero(a == track) >= 2 * np.count_nonzero(a == bead), "beads must be sparse")
    model = dict(shape=a.shape, background=bg, marker=marker, track=track, bead=bead,
                 top=top, left=left, right=right, separators=n, visible=visible,
                 first=first, count=count, growth=growth, drift=drift,
                 spacing=spacing, cycle=cycle, box=(r0, r1, c0, c1), rotation=rotation)
    for i in range(n):
        k = count + i * growth
        lo = first + i * drift
        hi = lo + (k - 1) * spacing
        _need(1 <= k <= 7 and left + 2 <= lo <= hi <= right - 2)
    original, rows = _render(model, False)
    observed = a.copy()
    observed[observed == bead] = track
    _need(np.array_equal(observed[~mask], original[~mask]), "exposed pixels conflict with continuation")
    structural, _ = _render(model, True)
    sources = set(map(tuple, np.argwhere(a == bead)))
    _need(bool(sources) and all(original[p] == track and structural[p] == track for p in sources), "bead on a switch or break")
    paths = _paths(structural, track)
    answer, targets, occupancy = _deposit(structural, track, bead, paths, sources)
    model.update(original=original, structural=structural, paths=paths, sources=sources,
                 targets=targets, occupancy=occupancy, centers=rows, full_answer=answer)
    crop = answer[r0:r1 + 1, c0:c1 + 1]
    return np.rot90(crop, -rotation).copy(), model


def _analyze(input_grid):
    a = _grid(input_grid)
    marker, box = _markers(a)
    bg = int(a[0, 0])
    _need(bg != marker and all(int(a[r, c]) == bg for r, c in ((0, -1), (-1, 0), (-1, -1))), "background corners required")
    r0, r1, c0, c1 = box
    mask = np.zeros(a.shape, dtype=bool)
    mask[r0:r1 + 1, c0:c1 + 1] = True
    _need(np.all(np.isin(a[mask], (bg, marker))), "window must be erased")
    clean = a.copy()
    clean[mask] = bg
    answers = []
    for rotation in range(4):
        try:
            answers.append(_fit(np.rot90(clean, rotation), np.rot90(mask, rotation), bg, marker, rotation))
        except ValueError:
            continue
    _need(bool(answers), "no supported lattice fits the input")
    _need(all(np.array_equal(answers[0][0], answer) for answer, _ in answers), "ambiguous lattice orientation")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the entire hidden-drain rule from this input alone."""
    return _analyze(input_grid)[0]


@lru_cache(maxsize=24)
def _layouts(growth, period, test):
    """Small bounded parameter catalog; contains no rendered grids."""
    result = []
    for visible in range(max(3, period + 1), 6):
        for hidden in ((2, 3, 4) if test else (1,)):
            n = visible + hidden
            for count in range(1, 8):
                if not all(1 <= count + growth * i <= 7 for i in range(n)):
                    continue
                for spacing in (4, 6, 8):
                    if count == 1 and growth == 0 and spacing != 4:
                        continue
                    for amount in range(2, spacing - 1):
                        for sign in ((-1,) if growth > 0 else (1,) if growth < 0 else (-1, 1)):
                            drift = sign * amount
                            rows = [[drift * i + spacing * j for j in range(count + growth * i)] for i in range(n)]
                            lo = min(min(row) for row in rows)
                            hi = max(max(row) for row in rows)
                            if hi - lo + 5 <= 30:
                                result.append((visible, hidden, count, spacing, drift, lo, hi))
    _need(bool(result), "no compatible lattice parameters")
    return tuple(result)


def _latent(rng, growth, period, test, palette):
    """Ground truth is drawn from explicit gate objects, before any erasure."""
    layouts = _layouts(growth, period, test)
    visible, hidden, count, spacing, drift, lo, hi = layouts[int(rng.integers(len(layouts)))]
    n = visible + hidden
    spare = 30 - (hi - lo + 5)
    extra_left = int(rng.integers(min(2, spare) + 1))
    extra_right = int(rng.integers(min(2, spare - extra_left) + 1))
    width = hi - lo + 5 + extra_left + extra_right
    # Top/bottom padding makes all four corners background even if side padding is zero.
    top_pad, bottom_pad = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    side_room = min(4, 30 - width)
    side_total = int(rng.integers(side_room + 1))
    left_pad = int(rng.integers(side_total + 1))
    right_pad = side_total - left_pad
    bg, track, bead, marker, *gate_palette = palette
    phase = int(rng.integers(period))
    cycle = gate_palette[:period]
    cycle = cycle[phase:] + cycle[:phase]
    shape = (2 * n + 1 + top_pad + bottom_pad, width + left_pad + right_pad)
    first = left_pad + 2 + extra_left - lo
    gates = [(top_pad + 2 * i + 1, first + drift * i + spacing * j, cycle[i % period])
             for i in range(n) for j in range(count + growth * i)]
    base = np.full(shape, bg, dtype=np.int8)
    for i in range(n + 1):
        base[top_pad + 2 * i, left_pad:left_pad + width] = track
    for i in (0, n - 1):
        centers = sorted(c for r, c, color in gates if r == top_pad + 2 * i + 1)
        boundary = top_pad if i == 0 else top_pad + 2 * n
        for a, b in zip(centers, centers[1:]):
            base[boundary, (a + b) // 2] = bg
    for r, c, color in gates:
        base[r, c - 1:c + 2] = color
    structural = base.copy()
    for r, c, color in gates:
        structural[r - 1:r + 2, c - 1:c + 2] = track
        structural[r - 1:r + 2, c] = color
    # The aperture includes all switches in at least one unobserved row and a lateral edge.
    r0, r1 = top_pad + 2 * visible + int(rng.integers(2)), top_pad + 2 * n
    last_centers = [c for r, c, color in gates if r >= r0]
    kind = int(rng.integers(3))
    if kind == 0:
        c0, c1 = left_pad, left_pad + width - 1
    elif kind == 1:
        c0 = left_pad
        c1 = max(last_centers) + 1 + int(rng.integers(left_pad + width - max(last_centers) - 1))
    else:
        c1 = left_pad + width - 1
        c0 = int(rng.integers(left_pad, min(last_centers)))
    box = r0, r1, c0, c1
    mask = np.zeros(shape, dtype=bool)
    mask[r0:r1 + 1, c0:c1 + 1] = True
    # Source beads are visible on surviving lane cells. They may be inside a path.
    eligible = set(map(tuple, np.argwhere((base == track) & (structural == track) & ~mask)))
    return base, structural, gates, eligible, box


def _label_union(structural, track, bead, sources):
    """Label-side accounting uses union-find, not the solver's component walk."""
    cells = set(map(tuple, np.argwhere(structural == track)))
    parent = {p: p for p in cells}
    def root(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p
    for r, c in sorted(cells):
        for q in ((r + 1, c), (r, c + 1)):
            if q in cells:
                parent[root(q)] = root((r, c))
    components = {}
    for p in sorted(cells):
        components.setdefault(root(p), []).append(p)
    answer = structural.copy()
    records = []
    for component in components.values():
        ends = sorted(p for p in component if len(_neighbors(p, cells)) == 1)
        _need(len(ends) == 2)
        present = sources.intersection(component)
        if ends[0][0] == ends[1][0]:
            targets = present
        else:
            current, previous, targets = max(ends), None, set()
            for _ in range(len(present)):
                targets.add(current)
                following = [p for p in _neighbors(current, cells) if p != previous]
                previous, current = current, following[0] if following else current
        for p in targets:
            answer[p] = bead
        records.append((component, ends))
    return answer, records


def _choose_sources(rng, structural, eligible, box, track, bead):
    """Require a multi-bead packet whose sink is hidden, plus optional sources."""
    _, components = _label_union(structural, track, bead, set())
    r0, r1, c0, c1 = box
    inside = lambda p: r0 <= p[0] <= r1 and c0 <= p[1] <= c1
    # The required packet starts before the aperture and crosses a wholly
    # missing switch row. A source on a merely exposed side strip is insufficient.
    upstream = {p for p in eligible if p[0] < r0}
    candidates = [(component, ends) for component, ends in components
                  if ends[0][0] < ends[1][0] and inside(max(ends)) and max(ends)[0] > r0
                  and len(upstream.intersection(component)) >= 3]
    _need(bool(candidates), "no visible feeder for hidden drain")
    component, ends = candidates[int(rng.integers(len(candidates)))]
    pool = sorted(upstream.intersection(component))
    # At least two beads join at a hidden sink; three/four-bead packets also occur.
    count = int(rng.integers(2, min(4, len(pool)) + 1))
    chosen = {pool[int(j)] for j in rng.choice(len(pool), count, replace=False)}
    others = sorted(eligible - chosen)
    rng.shuffle(others)
    # Four reserved optional cells give sixteen guaranteed distinct occupancy variants.
    _need(len(others) >= 4)
    switches, others = others[:4], others[4:]
    budget = max(0, len(eligible) // 4 - len(chosen) - 4)
    extra = int(rng.integers(min(budget, 12) + 1))
    chosen.update(others[:extra])
    return chosen, switches


def _request_int(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _request_int('num_train', num_train, 2, 10)
    num_test = _request_int('num_test', num_test, 1, 4)
    _need(num_train > num_test, "num_train must exceed num_test")
    _need(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
          "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.choice(10, 7, replace=False)))
    period = int(rng.integers(1, 4))
    shared_rotation = int(rng.integers(4))
    mixed = bool(rng.integers(2))
    growths = [1, -1] + [0 if i == 0 else int(rng.integers(-1, 2)) for i in range(num_train - 2)]
    rng.shuffle(growths)
    result, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            growth = growths[index] if split == 'train' else int(rng.integers(-1, 2))
            # These retries only select a crop with an informative feeder. Disk use is zero.
            for attempt in range(48):
                base, structural, gates, eligible, box = _latent(rng, growth, period, split == 'test', palette)
                try:
                    chosen, switches = _choose_sources(rng, structural, eligible, box, palette[1], palette[2])
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("bounded feeder construction exhausted")
            rotation = int(rng.integers(4)) if mixed else shared_rotation
            reflection = bool(rng.integers(2))
            start = int(rng.integers(16))
            for variant in range(16):
                bits = (start + variant) % 16
                sources = chosen | {cell for j, cell in enumerate(switches) if bits & (1 << j)}
                inp = base.copy()
                for p in sources:
                    inp[p] = palette[2]
                full, _ = _label_union(structural, palette[1], palette[2], sources)
                r0, r1, c0, c1 = box
                inp[r0:r1 + 1, c0:c1 + 1] = palette[0]
                mask = np.zeros(inp.shape, dtype=bool)
                mask[r0:r1 + 1, c0:c1 + 1] = True
                if reflection:
                    inp, full, mask = inp[:, ::-1], full[:, ::-1], mask[:, ::-1]
                inp, full, mask = (np.rot90(a, rotation).copy() for a in (inp, full, mask))
                pos = np.argwhere(mask)
                ar0, ac0 = map(int, pos.min(axis=0))
                ar1, ac1 = map(int, pos.max(axis=0))
                for cell in _corner_cells((ar0, ar1, ac0, ac1)):
                    inp[cell] = palette[3]
                if _key(inp) not in seen:
                    seen.add(_key(inp))
                    result[split].append({'input': inp, 'output': full[ar0:ar1 + 1, ac0:ac1 + 1].copy()})
                    break
            else:
                # At most 13 earlier inputs versus 16 occupancy variants.
                raise AssertionError("distinct-occupancy construction failed")
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _analyze(inp)
            assert np.array_equal(expected, out), 'incorrect crop'
            bg, track, bead, marker = (info[k] for k in ('background', 'track', 'bead', 'marker'))
            r0, r1, c0, c1 = info['box']
            region = lambda a: a[r0:r1 + 1, c0:c1 + 1]
            assert marker not in out and set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            assert np.count_nonzero(info['full_answer'] == bead) == len(info['sources'])
            assert not np.array_equal(region(info['original']), region(info['structural'])), 'rotation is invisible'
            assert not np.array_equal(region(info['structural']), region(info['full_answer'])), 'bead transport is invisible'
            assert any(len({p for p in path if p in info['sources'] and p[0] < r0}) >= 2
                       and r0 < max(path[0], path[-1])[0] <= r1
                       and c0 <= max(path[0], path[-1])[1] <= c1 for path in info['paths']), 'no packet crosses an inferred switch'
            assert all(p[0] % 2 == info['top'] % 2 for p in info['sources'])
            # Full-component conservation and exact terminal packing, including empty paths.
            for path, count in zip(info['paths'], info['occupancy']):
                assert sum(info['full_answer'][p] == bead for p in path) == count
                if path[0][0] != path[-1][0]:
                    inward = path if path[0][0] > path[-1][0] else path[::-1]
                    assert {p for p in path if info['full_answer'][p] == bead} == set(inward[:count])
            assert all(info['full_answer'][p] == info['structural'][p]
                       for p in np.ndindex(info['shape']) if p not in info['targets'])
            infos[split].append(info)
    assert {1, -1} <= {i['growth'] for i in infos['train']}, 'need growth and shrink demonstrations'
    assert max(i['separators'] - i['visible'] for i in infos['train']) < min(i['separators'] - i['visible'] for i in infos['test'])
    assert len({(i['background'], i['track'], i['bead'], i['marker'], frozenset(i['cycle']))
                for s in infos.values() for i in s}) == 1


def self_test():
    """42 sequential seeds; retain only bounded hashes and feature sets in RAM."""
    hashes, shapes, structures, growths, spacings, drifts, directions, periods = (set() for _ in range(8))
    colors = [set() for _ in range(4)]
    totals = dict(pairs=0, multi_packet=0, partial=0, internal_sources=0, overlapping_tiles=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                digest.update(str(a.shape).encode())
                digest.update(a.tobytes())
                digest.update(b.tobytes())
                _, info = _analyze(a)
                shapes.add((a.shape, b.shape))
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in a.flat)
                structures.add((a.shape, hashlib.sha256(normalized).digest()))
                growths.add(info['growth'])
                spacings.add(info['spacing'])
                drifts.add(info['drift'])
                directions.add(info['rotation'])
                periods.add(len(info['cycle']))
                for values, role in zip(colors, ('background', 'track', 'bead', 'marker')):
                    values.add(info[role])
                r0, r1, c0, c1 = info['box']
                totals['pairs'] += 1
                totals['multi_packet'] += max(info['occupancy']) >= 3
                totals['partial'] += c0 > info['left'] or c1 < info['right']
                endpoints = {p for path in info['paths'] for p in (path[0], path[-1])}
                totals['internal_sources'] += bool(info['sources'] - endpoints)
                totals['overlapping_tiles'] += any(abs(a - b) == 2 for row, nxt in zip(info['centers'], info['centers'][1:]) for a in row for b in nxt)
        hashes.add(digest.digest())
    assert len(hashes) == 42 and len(shapes) >= 80 and len(structures) >= 200
    assert growths == {-1, 0, 1} and {4, 6, 8} <= spacings and len(drifts) >= 8
    assert directions == {0, 1, 2, 3} and periods == {1, 2, 3}
    assert all(values == set(range(10)) for values in colors)
    assert totals['multi_packet'] >= 100 and totals['partial'] >= 60
    assert totals['internal_sources'] >= 180 and totals['overlapping_tiles'] >= 80
    def signature(puzzle):
        return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])
    assert signature(generate_puzzle(765, 10, 4)) == signature(generate_puzzle(765, 10, 4))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(900 + nt * 4 + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    corrupt = generate_puzzle(33, 2, 1)
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupted label accepted')
    return dict(seeds=42, shapes=len(shapes), structures=len(structures), **totals)
