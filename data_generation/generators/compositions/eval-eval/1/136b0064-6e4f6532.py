"""Compass relay printers: input-only registration, interpretation, and handoff.

No parent module, external file, learned dictionary, or generation state is used
by solve. Coordinates below use (row, column); D4 matrices act on column vectors.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import itertools
import numpy as np


GLYPHS = (
    ((1, 1, 0), (1, 0, 1), (0, 1, 0)),  # right 3
    ((1, 0, 1), (1, 0, 1), (1, 1, 1)),  # left 2
    ((1, 0, 1), (0, 1, 0), (0, 1, 0)),  # down 2
    ((1, 1, 1), (0, 1, 0), (1, 0, 1)),  # left 4
)
STROKES = ((0, 1, 3), (0, -1, 2), (1, 0, 2), (0, -1, 4))
SIDES = ((-1, 0), (0, 1), (1, 0), (0, -1))
NEIGHBORS = tuple(p for p in itertools.product((-1, 0, 1), repeat=2)
                  if p != (0, 0))


def _matrix(t):
    m = np.diag((1, -1)) if t >= 4 else np.eye(2, dtype=int)
    for _ in range(t % 4):
        m = np.array(((0, 1), (-1, 0))) @ m
    return m.astype(int)


MATRICES = tuple(_matrix(t) for t in range(8))


def _pose(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, -(t % 4)).copy()


def _mv(m, p):
    r, c = m @ np.asarray(p, dtype=int)
    return int(r), int(c)


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        found, stack = {p}, [p]
        while stack:
            q = stack.pop()
            for d in NEIGHBORS:
                v = _add(q, d)
                if v in remaining:
                    remaining.remove(v)
                    found.add(v)
                    stack.append(v)
        result.append(found)
    return result


def _normal(points):
    r0 = min(r for r, c in points)
    c0 = min(c for r, c in points)
    return tuple(sorted((r - r0, c - c0) for r, c in points))


def _shape_class(points):
    return _cached_shape_class(_normal(points))


@lru_cache(maxsize=512)
def _cached_shape_class(points):
    return min(_normal({_mv(m, p) for p in points}) for m in MATRICES)


def _pattern_pool():
    # These are procedural registration polyominoes, not command vocabulary.
    cells = [(r, c) for r in range(-2, 1) for c in range(3) if (r, c) != (0, 0)]
    pool = {n: [] for n in range(2, 7)}
    for n in pool:
        for others in itertools.combinations(cells, n - 1):
            points = {(0, 0), *others}
            if len(_components(points)) == 1:
                pool[n].append(tuple(sorted(points)))
    return pool


PATTERNS = _pattern_pool()


def _frame_mask(rows, lanes):
    h, w = 4 * rows + 5, 4 * lanes + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    mask[0, 0] = False  # The chamfer names canonical upper-left.
    mask[4, :] = True  # Header/program crossbar removes transpose ambiguity.
    return mask


def _trace(tokens):
    endpoint = (0, 0)
    pixels = []
    for command, color in tokens:
        dr, dc, length = STROKES[command]
        r, c = endpoint
        for k in range(length):
            pixels.append(((r + 1 + k * dr, c + k * dc), int(color)))
        endpoint = pixels[-1][0]
    return pixels, endpoint


def _card_array(card, roles):
    bg, frame, reg, seed, *rest = roles
    bands = rest[:4]
    mask = _frame_mask(card['rows'], card['lanes'])
    h, w = mask.shape
    a = np.full((h + 2, w + 2), bg, dtype=np.int8)
    a[1:h + 1, 1:w + 1][mask] = frame
    for r, c in card['pattern']:
        a[r + 4, c + 2] = reg
    for i, (command, color) in enumerate(card['tokens']):
        lane, band = divmod(i, card['rows'])
        block = a[6 + 4 * band:9 + 4 * band, 2 + 4 * lane:5 + 4 * lane]
        block[np.array(GLYPHS[command], dtype=bool)] = color
    north = _mv(card['matrix'], (-1, 0))
    west = _mv(card['matrix'], (0, -1))
    a[0, 1 + w // 2] = bands[SIDES.index(north)]
    a[3, 0] = bands[SIDES.index(west)]
    return a


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid dimensions'
    assert np.issubdtype(a.dtype, np.integer), 'noninteger cells'
    assert np.all((0 <= a) & (a <= 9)), 'invalid colors'
    return a.astype(np.int8, copy=False)


def _parse_card(patch, frame, bg, band_colors):
    candidates = []
    for t in range(8):
        a = _pose(patch, t)
        h, w = a.shape[0] - 2, a.shape[1] - 2
        if h not in (9, 13) or w not in (5, 9):
            continue
        rows, lanes = (h - 5) // 4, (w - 1) // 4
        mask = _frame_mask(rows, lanes)
        expected_frame = np.zeros(a.shape, dtype=bool)
        expected_frame[1:h + 1, 1:w + 1] = mask
        if not np.array_equal(a == frame, expected_frame):
            continue
        header = a[2:5, 2:5]
        values = np.unique(header[header != bg])
        if len(values) != 1:
            continue
        reg = int(values[0])
        if reg in band_colors or reg in (bg, frame):
            continue
        pattern = tuple((int(r) - 2, int(c)) for r, c in np.argwhere(header == reg))
        if (0, 0) not in pattern or not 2 <= len(pattern) <= 6 or len(_components(pattern)) != 1:
            continue
        top, left = int(a[0, 1 + w // 2]), int(a[3, 0])
        if top not in band_colors or left not in band_colors:
            continue
        vd, vr = -np.array(band_colors[top]), -np.array(band_colors[left])
        if int(vd @ vr) != 0:
            continue
        matrix = np.column_stack((vd, vr))
        expected = np.full_like(a, bg)
        expected[expected_frame] = frame
        expected[2:5, 2:5] = header
        expected[0, 1 + w // 2], expected[3, 0] = top, left
        tokens = []
        valid = True
        for lane in range(lanes):
            for band in range(rows):
                r, c = 6 + 4 * band, 2 + 4 * lane
                glyph = a[r:r + 3, c:c + 3]
                inks = np.unique(glyph[glyph != bg])
                occupancy = tuple(tuple(int(v) for v in row) for row in glyph != bg)
                if (len(inks) != 1 or occupancy not in GLYPHS
                        or int(inks[0]) in {bg, frame, reg, *band_colors}):
                    valid = False
                    break
                tokens.append((GLYPHS.index(occupancy), int(inks[0])))
                expected[r:r + 3, c:c + 3] = glyph
            if not valid:
                break
        if valid and np.array_equal(a, expected):
            candidates.append(dict(rows=rows, lanes=lanes, pattern=pattern, reg=reg,
                                   tokens=tokens, matrix=matrix, source_matrix=MATRICES[t].T))
    assert len(candidates) == 1, 'capsule has no unique canonical reading'
    return candidates[0]


def _read_input(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    assert h >= 15 and w >= 15
    bands = [int(a[0, 0]), int(a[1, -1]), int(a[-1, 0]), int(a[1, 0])]
    assert len(set(bands)) == 4
    assert np.all(a[0] == bands[0]) and np.all(a[-1] == bands[2])
    assert np.all(a[1:-1, -1] == bands[1]) and np.all(a[1:-1, 0] == bands[3])
    interior = a[1:-1, 1:-1]
    counts = np.bincount(interior.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert bg not in bands and np.count_nonzero(counts == counts[bg]) == 1
    band_colors = dict(zip(bands, SIDES))
    possibilities = []
    for color in range(10):
        if color == bg or color in bands:
            continue
        objects = _components(map(tuple, np.argwhere(interior == color) + 1))
        big = [p for p in objects if len(p) >= 20]
        if not big:
            continue
        cards = []
        covered = np.zeros_like(a, dtype=bool)
        try:
            for points in big:
                r0, r1 = min(r for r, c in points) - 1, max(r for r, c in points) + 2
                c0, c1 = min(c for r, c in points) - 1, max(c for r, c in points) + 2
                assert 1 <= r0 < r1 <= h - 1 and 1 <= c0 < c1 <= w - 1
                assert not np.any(covered[r0:r1, c0:c1])
                card = _parse_card(a[r0:r1, c0:c1], color, bg, band_colors)
                card['window'] = (r0, r1, c0, c1)
                covered[r0:r1, c0:c1] = True
                cards.append(card)
            assert len({card['reg'] for card in cards}) == 1
            assert len(cards) in range(1, 5)
        except (AssertionError, ValueError):
            continue
        possibilities.append((color, cards, covered))
    assert len(possibilities) == 1, 'no unique capsule frame role'
    frame, cards, covered = possibilities[0]
    reg = cards[0]['reg']
    remaining = a.copy()
    remaining[covered] = bg
    remaining[[0, -1], :] = bg
    remaining[:, [0, -1]] = bg
    other = set(int(v) for v in remaining[remaining != bg]) - {reg}
    assert len(other) == 1, 'one visible start color is required'
    seed = next(iter(other))
    assert seed not in {bg, frame, reg, *bands}
    seed_points = np.argwhere(remaining == seed)
    assert len(seed_points) == 1
    start = tuple(int(v) for v in seed_points[0])
    ports = _components(map(tuple, np.argwhere(remaining != bg)))
    assert len(ports) == len(cards)
    assert len({_shape_class(card['pattern']) for card in cards}) == len(cards)
    assigned = set()
    for card in cards:
        marks = {_mv(card['matrix'], p) for p in card['pattern']}
        matches = []
        for i, port in enumerate(ports):
            if len(port) != len(marks):
                continue
            p, q = min(port), min(marks)
            anchor = p[0] - q[0], p[1] - q[1]
            if {_add(anchor, v) for v in marks} == port:
                matches.append((i, anchor))
        assert len(matches) == 1, 'registration and compass must identify one port'
        i, anchor = matches[0]
        assert i not in assigned
        assigned.add(i)
        card['anchor'], card['port'] = anchor, ports[i]
        if start in ports[i]:
            assert anchor == start, 'start pin must replace the port anchor'
    assert sum(card['anchor'] == start for card in cards) == 1
    inks = {c for card in cards for _, c in card['tokens']}
    assert inks.isdisjoint({bg, frame, reg, seed, *bands}) and 1 <= len(inks) <= 2
    base = np.full_like(a, bg)
    base[0], base[-1] = a[0], a[-1]
    base[:, 0], base[:, -1] = a[:, 0], a[:, -1]
    for p in set().union(*ports):
        base[p] = a[p]
    return dict(grid=a, background=bg, frame=frame, registration=reg, seed=seed,
                bands=bands, cards=cards, ports=ports, start=start, base=base,
                covered=covered)


def _world_trace(card, matrix=None):
    pixels, end = _trace(card['tokens'])
    m = card['matrix'] if matrix is None else matrix
    return [(_add(card['anchor'], _mv(m, p)), c) for p, c in pixels], _add(card['anchor'], _mv(m, end))


def _execute(info):
    out = info['base'].copy()
    anchors = {card['anchor']: i for i, card in enumerate(info['cards'])}
    protected = set().union(*info['ports'])
    endpoint, visited, painted = info['start'], [], {}
    crossings = recolored = 0
    while endpoint in anchors:
        i = anchors[endpoint]
        assert i not in visited, 'reachable relay cycles are outside this family'
        visited.append(i)
        trace, end = _world_trace(info['cards'][i])
        assert len({p for p, c in trace}) == len(trace)
        for j, (p, color) in enumerate(trace):
            r, c = p
            assert 1 <= r < out.shape[0] - 1 and 1 <= c < out.shape[1] - 1, 'stroke hits boundary'
            if p in protected:
                assert j == len(trace) - 1 and p == end and p in anchors, 'stroke hits nonendpoint port'
                continue
            if p in painted:
                crossings += 1
                recolored += painted[p] != color
            out[p] = color
            painted[p] = color
        endpoint = end
    return out, dict(visited=visited, painted=painted, crossings=crossings,
                     recolored=recolored, terminal=endpoint)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase capsules and execute only the start-reachable docked programs."""
    return _execute(_read_input(input_grid))[0]


def _bounds(points):
    return min(r for r, c in points), max(r for r, c in points), min(c for r, c in points), max(c for r, c in points)


def _inside(points, h, w):
    return all(1 <= r < h - 1 and 1 <= c < w - 1 for r, c in points)


def _halo(points):
    return {_add(p, d) for p in points for d in ((0, 0), *NEIGHBORS)}


def _pack_cards(rng, cards, arrays, targets, h, w, teaching):
    """Bounded rectangle packing; cards can occupy future trajectory pixels."""
    blocked = np.zeros((h, w), dtype=bool)
    blocked[[0, -1], :] = True
    blocked[:, [0, -1]] = True
    for r, c in _halo(targets):
        if 0 <= r < h and 0 <= c < w:
            blocked[r, c] = True
    order = sorted(range(len(cards)), key=lambda i: -arrays[i].size)
    choices = []
    for i, card in enumerate(cards):
        if i == 0 and teaching:
            relative = MATRICES[1:4] if teaching == 1 else MATRICES[4:8]
            allowed = [t for t, m in enumerate(MATRICES)
                       if any(np.array_equal(m, card['matrix'] @ q) for q in relative)]
        else:
            allowed = list(range(8))
        rng.shuffle(allowed)
        # Pose changes pixels, but rectangle feasibility depends only on shape.
        # Randomly retain one pose of each rectangular extent to avoid repeating
        # the exact same failed geometric packing branch four or eight times.
        extents = {}
        for t in allowed:
            patch = _pose(arrays[i], t)
            extents.setdefault(patch.shape, (t, patch))
        choices.append(list(extents.values()))
    placed, budget = {}, [150]

    def search(depth, occupied):
        if depth == len(order):
            return True
        if budget[0] <= 0:
            return False
        i = order[depth]
        integral = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        candidates = []
        for t, patch in choices[i]:
            ph, pw = patch.shape
            if ph >= h - 1 or pw >= w - 1:
                continue
            totals = (integral[ph:, pw:] - integral[:-ph, pw:]
                      - integral[ph:, :-pw] + integral[:-ph, :-pw])
            for r, c in np.argwhere(totals[1:-1, 1:-1] == 0) + 1:
                candidates.append((int(r), int(c), t, patch))
        rng.shuffle(candidates)
        for r, c, t, patch in candidates[:40]:
            budget[0] -= 1
            ph, pw = patch.shape
            following = occupied.copy()
            following[max(0, r - 1):r + ph + 1, max(0, c - 1):c + pw + 1] = True
            placed[i] = (r, c, t, patch)
            if search(depth + 1, following):
                return True
        placed.pop(i, None)
        return False

    return placed if search(0, blocked) else None


def _choose_patterns(rng, count):
    patterns, used = [], set()
    for i in range(count):
        n = 3 if i < 2 else int(rng.integers(2, 7))
        pool = [p for p in PATTERNS[n] if _shape_class(p) not in used]
        if i == 0:
            # A nonidentity symmetry fixing the anchor makes the compass
            # necessary even when the complete registration shape is known.
            pool = [p for p in pool if any({_mv(m, q) for q in p} == set(p)
                                          for m in MATRICES[1:])]
        if not pool:
            pool = [p for group in PATTERNS.values() for p in group if _shape_class(p) not in used]
        p = pool[int(rng.integers(len(pool)))]
        patterns.append(p)
        used.add(_shape_class(p))
    return patterns


def _construct_pair(rng, roles, active, count, teaching):
    bg, frame, reg, seed = roles[:4]
    ink = roles[8:10]
    patterns = _choose_patterns(rng, count)
    cards = []
    for i in range(count):
        if teaching == 1 and i < 2:
            commands = (0, 2) if i == 0 else (1, 3)
            rows, lanes, colors = 1, 2, ink
        elif teaching == 2 and i == 0:
            commands, rows, lanes = (0, 1, 2, 3), 2, 2
            colors = (ink[1], ink[1], ink[0], ink[0])
        elif teaching == 2 and i == 1:
            commands, rows, lanes, colors = (3, 0), 1, 2, (ink[1], ink[0])
        else:
            # One longer scroll at most, with thin one-glyph capsules mixed in.
            if i == 0 and not teaching and rng.random() < .35:
                rows, lanes = 2, 2
            elif i < min(2, active):
                rows, lanes = 1, 2
            else:
                rows, lanes = 1, 1
            commands = tuple(int(v) for v in rng.integers(0, 4, rows * lanes))
            colors = tuple(int(v) for v in rng.choice(ink, rows * lanes))
        matrix = MATRICES[int(rng.integers(8))].copy()
        if teaching == 2 and i == 1:
            matrix = -cards[0]['matrix']
        cards.append(dict(rows=rows, lanes=lanes, tokens=list(zip(commands, colors)),
                          pattern=patterns[i], matrix=matrix))
    # Form the relay in relative space before sampling its absolute position.
    anchor, traces, target_sets = (0, 0), [], []
    for card in cards[:active]:
        card['anchor'] = anchor
        trace, anchor = _world_trace(card)
        traces.append(trace)
        target_sets.append({_add(card['anchor'], _mv(card['matrix'], p)) for p in card['pattern']})
    anchors = {card['anchor'] for card in cards[:active]}
    if len(anchors) != active or anchor in anchors:
        return None
    targets = set().union(*target_sets)
    if any(a & _halo(b) for i, a in enumerate(target_sets) for b in target_sets[i + 1:]):
        return None
    for i, trace in enumerate(traces):
        permitted = cards[i + 1]['anchor'] if i + 1 < active else None
        if {p for p, c in trace} & targets != ({permitted} if permitted else set()):
            return None
    earlier, crossing, recolored = {}, 0, 0
    for trace in traces:
        for p, c in trace:
            if p not in targets:
                if p in earlier:
                    crossing += 1
                    recolored += earlier[p] != c
                earlier[p] = c
    if teaching == 1 and crossing:
        return None
    if teaching == 2 and not recolored:
        return None
    low = 22 if count <= 2 else 25 if count == 3 else 27
    h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
    extent = targets | {p for tr in traces for p, c in tr}
    r0, r1, c0, c1 = _bounds(extent)
    if r1 - r0 > h - 3 or c1 - c0 > w - 3:
        return None
    shift = (int(rng.integers(1 - r0, h - 1 - r1)),
             int(rng.integers(1 - c0, w - 1 - c1)))
    for card in cards[:active]:
        card['anchor'] = _add(card['anchor'], shift)
    target_sets = [{_add(p, shift) for p in ts} for ts in target_sets]
    traces = [[(_add(p, shift), c) for p, c in tr] for tr in traces]
    targets = set().union(*target_sets)
    painted = {p for tr in traces for p, c in tr} - targets
    # Inactive ports are genuine registration matches, but never lie on the relay.
    for card in cards[active:]:
        candidates = [(r, c) for r in range(2, h - 2) for c in range(2, w - 2)]
        rng.shuffle(candidates)
        for anchor in candidates:
            card['anchor'] = anchor
            marks = {_add(anchor, _mv(card['matrix'], p)) for p in card['pattern']}
            trace, end = _world_trace(card)
            pp = {p for p, c in trace}
            if (not _inside(marks | pp, h, w) or marks & _halo(targets)
                    or marks & painted or pp & (targets | marks) or not pp - painted):
                continue
            target_sets.append(marks)
            targets.update(marks)
            break
        else:
            return None
    arrays = [_card_array(card, roles) for card in cards]
    packing = _pack_cards(rng, cards, arrays, targets, h, w, teaching)
    if packing is None:
        return None
    base = np.full((h, w), bg, dtype=np.int8)
    base[0, :], base[-1, :] = roles[4], roles[6]
    base[1:-1, -1], base[1:-1, 0] = roles[5], roles[7]
    for p in targets:
        base[p] = reg
    base[cards[0]['anchor']] = seed
    out = base.copy()
    for trace in traces:
        for p, color in trace:
            if p not in targets:
                out[p] = color
    inp = base.copy()
    for r, c, t, patch in packing.values():
        ph, pw = patch.shape
        inp[r:r + ph, c:c + pw] = patch
    # Certification is input-only; the constructor's traced answer is separate.
    info = _read_input(inp)
    predicted, execution = _execute(info)
    assert np.array_equal(out, predicted)
    assert len(execution['visited']) == active
    if teaching == 1 and not _compass_alternatives(info, out):
        return None
    return {'input': inp, 'output': out}


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f'{name} is out of range')
    return value


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    puzzle, used, previous_bg = {'train': [], 'test': []}, set(), None
    for split, number in (('train', nt), ('test', nq)):
        for i in range(number):
            roles = [int(v) for v in rng.permutation(10)]
            if roles[0] == previous_bg:
                roles[0], roles[1] = roles[1], roles[0]
            previous_bg = roles[0]
            teaching = i + 1 if split == 'train' and i < 2 else 0
            active = 2 if teaching else (3 if split == 'test' else int(rng.integers(1, 3)))
            count = 3 if teaching else (4 if split == 'test' else int(rng.integers(active + 1, 5)))
            for attempt in range(240):
                pair = _construct_pair(rng, roles, active, count, teaching)
                if pair is not None and _key(pair['input']) not in used:
                    break
            else:
                raise RuntimeError('bounded construction exhausted; no altered-rule fallback')
            used.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _all_printers(info):
    out = info['base'].copy()
    protected = set().union(*info['ports'])
    for card in info['cards']:
        trace, _ = _world_trace(card)
        for p, c in trace:
            if p not in protected:
                out[p] = c
    return out


def _without_compass(info, visited):
    out = info['base'].copy()
    protected = set().union(*info['ports'])
    for i in visited:
        card = info['cards'][i]
        trace, _ = _world_trace(card, card['source_matrix'])
        for (r, c), value in trace:
            if not (1 <= r < out.shape[0] - 1 and 1 <= c < out.shape[1] - 1):
                return None
            if (r, c) not in protected:
                out[r, c] = value
    return out


def _compass_alternatives(info, correct):
    """Different valid relays with exact registration but no start-card labels.

The start pin still fixes the anchor; other capsules retain their actual frames.
This is a conservative ablation, not an independent solver.
"""
    i = next(i for i, card in enumerate(info['cards']) if card['anchor'] == info['start'])
    source = info['cards'][i]
    alternatives = set()
    for m in MATRICES:
        if np.array_equal(m, source['matrix']):
            continue
        marks = {_add(source['anchor'], _mv(m, p)) for p in source['pattern']}
        if marks != source['port']:
            continue
        changed = dict(info)
        changed['cards'] = [dict(card) for card in info['cards']]
        changed['cards'][i]['matrix'] = m
        try:
            out, _ = _execute(changed)
        except AssertionError:
            continue
        if not np.array_equal(out, correct):
            alternatives.add(_key(out))
    return alternatives


def _features(inp, out):
    info = _read_input(inp)
    expected, execution = _execute(info)
    assert np.array_equal(out, expected), 'output violates compass relay rule'
    visited = execution['visited']
    protected = set().union(*info['ports'])
    # All reachable strokes must add information beyond merely removing cards.
    assert not np.array_equal(out, info['base'])
    # Inactive cards have real, fitting programs; drawing all is a valid but wrong shortcut.
    for card in info['cards']:
        trace, _ = _world_trace(card)
        assert _inside({p for p, c in trace}, *inp.shape)
    assert len(visited) < len(info['cards'])
    assert not np.array_equal(out, _all_printers(info)), 'inactive programs are not distinguishable'
    no_compass = _without_compass(info, visited)
    moves = []
    for i in visited:
        card = info['cards'][i]
        delta = card['matrix'] @ card['source_matrix'].T
        moves.append('reflection' if round(np.linalg.det(delta)) < 0 else
                     'translation' if np.array_equal(delta, np.eye(2, dtype=int)) else 'rotation')
    return dict(shape=tuple(inp.shape), cards=len(info['cards']), active=len(visited),
                program_lengths=tuple(len(card['tokens']) for card in info['cards']),
                patterns=tuple(_shape_class(card['pattern']) for card in info['cards']),
                moves=moves, recolored=execution['recolored'], crossings=execution['crossings'],
                background=info['background'], frames=tuple(sorted(info['bands'])),
                matrices=tuple(tuple(int(v) for v in card['matrix'].flat) for card in info['cards']),
                compass_changes=no_compass is None or not np.array_equal(out, no_compass),
                old_new_overlap=sum(inp[p] != info['background'] for p in execution['painted']),
                registration_counts=tuple(len(card['pattern']) for card in info['cards']),
                info=info, execution=execution)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in features:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            features[split].append(_features(inp, out))
    first, second = features['train'][:2]
    for teacher in (first, second):
        commands = {k for i in teacher['execution']['visited'] for k, c in teacher['info']['cards'][i]['tokens']}
        assert commands == set(range(4)), 'teaching example misses a command'
        assert teacher['compass_changes'], 'compass omission has no consequence'
        assert teacher['active'] == 2 and teacher['cards'] == 3
    assert 'rotation' in first['moves'] and 'reflection' in second['moves']
    assert first['crossings'] == 0 and second['recolored'] >= 1
    assert _compass_alternatives(first['info'], np.asarray(puzzle['train'][0]['output']))
    assert any(card['rows'] == card['lanes'] == 2 for card in second['info']['cards'])
    # Explicitly disprove row-major decoding on the teaching scroll.
    altered = dict(second['info'])
    altered['cards'] = [dict(card) for card in altered['cards']]
    for card in altered['cards']:
        if card['rows'] == card['lanes'] == 2:
            card['tokens'] = [card['tokens'][i] for i in (0, 2, 1, 3)]
    try:
        row_major = _execute(altered)[0]
    except AssertionError:
        row_major = None
    assert row_major is None or not np.array_equal(row_major, puzzle['train'][1]['output'])
    assert any(Counter(first['registration_counts'])[n] >= 2 for n in first['registration_counts'])
    assert len({f['background'] for f in features['train']}) >= 2
    assert all(f['active'] <= 2 for f in features['train'])
    assert all(f['active'] == 3 and f['cards'] == 4 for f in features['test'])


def self_test():
    """42 bounded seeds, meaningful semantic checks, and mutation regressions."""
    keys, shapes, patterns, frames, matrices, lengths, card_counts = set(), set(), set(), set(), set(), set(), set()
    overlaps = crossings = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        keys.add(tuple(_key(pair['input']) for s in ('train', 'test') for pair in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair['input'], pair['output'])
                shapes.add(f['shape'])
                patterns.update(f['patterns'])
                frames.add(f['frames'])
                matrices.update(f['matrices'])
                lengths.update(f['program_lengths'])
                card_counts.add(f['cards'])
                overlaps += int(f['old_new_overlap'] > 0)
                crossings += int(f['recolored'] > 0)
    assert len(keys) == 42 and len(shapes) >= 20 and len(patterns) >= 25
    assert len(frames) >= 25 and len(matrices) == 8
    assert lengths == {1, 2, 4} and card_counts >= {2, 3, 4}
    assert overlaps >= 20 and crossings >= 42
    a, b = generate_puzzle(104729, 2, 1), generate_puzzle(104729, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt, nq in ((2, 1), (10, 4)):
        generate_puzzle(31415, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Changing a genuine painted cell must fail semantic validation.
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    f = _features(damaged['train'][0]['input'], damaged['train'][0]['output'])
    p = next(iter(f['execution']['painted']))
    damaged['train'][0]['output'][p] = f['background']
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged stroke accepted')
    return dict(seeds=42, distinct_puzzles=len(keys), grid_shapes=len(shapes),
                registration_classes=len(patterns), boundary_palettes=len(frames),
                orientations=len(matrices), old_new_overlap_pairs=overlaps,
                recolored_crossing_pairs=crossings)


if __name__ == '__main__':
    print(self_test())
