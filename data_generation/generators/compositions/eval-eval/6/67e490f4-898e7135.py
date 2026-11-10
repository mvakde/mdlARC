"""Shape elections determine an elastic blueprint's colors and line thicknesses.

Only standard-library modules and NumPy are used. The solver has no dependency
on the sampler, seed, shape pool, demonstrations, or saved artifacts.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from math import isqrt

import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _normalize(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


@lru_cache(maxsize=4096)
def _variants(shape, reflect=True):
    forms = set()
    for flip in range(2 if reflect else 1):
        current = tuple((r, -c if flip else c) for r, c in shape)
        for _ in range(4):
            forms.add(_normalize(current))
            current = tuple((c, -r) for r, c in current)
    return tuple(sorted(forms))


def _canonical(cells):
    return _variants(_normalize(cells))[0]


def _size(shape):
    return max(r for r, _ in shape)+1, max(c for _, c in shape)+1


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _N4:
                p = r+dr, c+dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        groups.append(tuple(cells))
    return groups


def _hole_free(shape):
    h, w = _size(shape)
    empty = np.ones((h+2, w+2), dtype=bool)
    for r, c in shape:
        empty[r+1, c+1] = False
    return len(_components(empty)) == 1


def _bbox(cells):
    return (min(r for r, _ in cells), max(r for r, _ in cells)+1,
            min(c for _, c in cells), max(c for _, c in cells)+1)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _decode_blocks(cells, scale):
    """Decode relative to this object's origin, never a global pixel lattice."""
    r0, r1, c0, c1 = _bbox(cells)
    if (r1-r0) % scale or (c1-c0) % scale:
        return None
    blocks = Counter(((r-r0)//scale, (c-c0)//scale) for r, c in cells)
    if any(count != scale*scale for count in blocks.values()):
        return None
    return tuple(sorted(blocks))


def _render(board, holes, elected):
    logical = board.copy()
    rows = np.ones(board.shape[0], dtype=int)
    cols = np.ones(board.shape[1], dtype=int)
    for cells in holes:
        color, scale = elected[_canonical(cells)]
        for r, c in cells:
            logical[r, c] = color
            rows[r] = max(int(rows[r]), scale)
            cols[c] = max(int(cols[c]), scale)
    return np.repeat(np.repeat(logical, rows, axis=0), cols, axis=1), rows, cols


def _recover(input_grid):
    """All rule data are reconstructed exclusively from the input array."""
    grid = _grid(input_grid)
    candidates = []
    for carrier in map(int, np.unique(grid)):
        for cells in _components(grid == carrier):
            r0, r1, c0, c1 = _bbox(cells)
            box = grid[r0:r1, c0:c1]
            if min(box.shape) < 3:
                continue
            if not (np.all(box[0] == carrier) and np.all(box[-1] == carrier)
                    and np.all(box[:, 0] == carrier) and np.all(box[:, -1] == carrier)):
                continue
            others = set(map(int, np.unique(box))) - {carrier}
            if len(others) != 1:
                continue
            outside = np.ones(grid.shape, dtype=bool)
            outside[r0:r1, c0:c1] = False
            if np.any((grid == carrier) & outside):
                continue
            bg = others.pop()
            holes = _components(box == bg)
            if not holes or any(not _hole_free(_normalize(h)) for h in holes):
                continue
            candidates.append((carrier, bg, (r0, r1, c0, c1), box.copy(), holes))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one complete-border blueprint")
    carrier, bg, bounds, board, holes = candidates[0]
    r0, r1, c0, c1 = bounds
    exterior = grid.copy()
    exterior[r0:r1, c0:c1] = bg
    # A clipped one-cell moat is sufficient at canvas edges.
    moat = exterior[max(0, r0-1):r1+1, max(0, c0-1):c1+1]
    if np.any(moat != bg):
        raise ValueError("The blueprint must be separated from ballots")
    components = [(color, cells) for color in map(int, np.unique(exterior)) if color != bg
                  for cells in _components(exterior == color)]
    occupied = exterior != bg
    for _, cells in components:
        group = set(cells)
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    y, x = r+dr, c+dc
                    if (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]
                            and occupied[y, x] and (y, x) not in group):
                        raise ValueError("Ballots must have background moats")
    classes = set(_canonical(h) for h in holes)
    votes, scales, objects = defaultdict(Counter), {}, []
    for color, cells in components:
        matches = []
        for key in sorted(classes):
            if len(cells) % len(key):
                continue
            ratio = len(cells)//len(key)
            k = isqrt(ratio)
            if k < 2 or k*k != ratio:
                continue
            raw = _decode_blocks(cells, k)
            if raw is not None and _canonical(raw) == key:
                matches.append((key, k, raw))
        if len(matches) != 1:
            raise ValueError("Every ballot needs exactly one shape/scale interpretation")
        key, k, raw = matches[0]
        if (key, color) in scales and scales[key, color] != k:
            raise ValueError("A shape-color candidate must use one scale")
        scales[key, color] = k
        votes[key][color] += 1
        objects.append(dict(key=key, color=color, scale=k, raw=raw, cells=cells))
    if set(votes) != classes:
        raise ValueError("Every cavity class needs supporting ballots")
    elected = {}
    for key, counts in votes.items():
        high = max(counts.values())
        winners = [color for color, n in counts.items() if n == high]
        if len(winners) != 1:
            raise ValueError("Shape elections cannot tie")
        color = winners[0]
        elected[key] = color, scales[key, color]
    answer, rows, cols = _render(board, holes, elected)
    if max(answer.shape) > 30:
        raise ValueError("The stretched board exceeds the grid bounds")
    return answer, dict(background=bg, carrier=carrier, bounds=bounds,
                        board=board, holes=holes, classes=classes, objects=objects,
                        votes=dict(votes), scales=scales, elected=elected,
                        rows=rows, cols=cols)


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _recover(input_grid)[0]


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _pool(rng):
    shapes = {((0, 0),)}
    for length in range(2, 8):
        shapes.add(_canonical(tuple((0, c) for c in range(length))))
    for h in range(2, 5):
        for w in range(2, 5):
            cells = {(r, 0) for r in range(h)} | {(h-1, c) for c in range(w)}
            shapes.add(_canonical(cells))
            if h*w <= 8:
                shapes.add(_canonical(tuple((r, c) for r in range(h) for c in range(w))))
            if 2*h+w-2 <= 8:
                shapes.add(_canonical(cells | {(r, w-1) for r in range(h)}))
    for base in (((0, 1), (1, 0), (1, 1), (1, 2)),
                 ((0, 1), (0, 2), (1, 0), (1, 1)),
                 ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))):
        shapes.add(_canonical(base))
    for _ in range(48):
        cells = {(0, 0)}
        area = int(rng.integers(3, 9))
        for _ in range(area-1):
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _N4} - cells)
            cells.add(_choice(rng, frontier))
        shape = _canonical(cells)
        if max(_size(shape)) <= 5 and _hole_free(shape):
            shapes.add(shape)
    return tuple(sorted(shapes))


def _mask(shape, scale):
    mask = np.zeros(_size(shape), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)


def _origins(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    integral = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    return np.argwhere(sums == 0)


def _pack(rng, board, proposals, bg):
    """Scatter bounding rectangles with moats; use shelves only as fallback."""
    items = [(None, board)] + [(color, _mask(raw, k)) for raw, color, k in proposals]
    cost = sum((a.shape[0]+1)*(a.shape[1]+1) for _, a in items)
    for attempt in range(24):
        if attempt < 10:
            low = min(29, max(16, int(np.sqrt(cost)*1.1)))
            height, width = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        else:
            height = width = 30
        canvas = np.full((height, width), bg, dtype=np.int8)
        blocked = np.zeros(canvas.shape, dtype=bool)
        order = [0] + sorted(rng.permutation(np.arange(1, len(items))).tolist(),
                             key=lambda i: -items[i][1].size)
        for index in order:
            color, data = items[index]
            h, w = data.shape
            positions = _origins(blocked, h, w)
            if not len(positions):
                break
            top, left = map(int, _choice(rng, positions))
            view = canvas[top:top+h, left:left+w]
            if color is None:
                view[:] = data
            else:
                view[data] = color
            blocked[max(0, top-1):top+h+1, max(0, left-1):left+w+1] = True
        else:
            return _trim(rng, canvas, bg)
    # Conservatively pack complete rectangles; orientation of sampled ballots
    # remains unchanged, including the reflection teaching witness.
    order = sorted(range(len(items)), key=lambda i: (-items[i][1].shape[0], -items[i][1].shape[1]))
    shelves = []
    positions = {}
    for i in order:
        h, w = items[i][1].shape
        for shelf in shelves:
            if h <= shelf[1] and shelf[2]+1+w <= 30:
                positions[i] = shelf[0], shelf[2]+1
                shelf[2] += 1+w
                break
        else:
            y = sum(s[1]+1 for s in shelves)
            if y+h > 30 or w > 30:
                return None
            shelves.append([y, h, w])
            positions[i] = y, 0
    canvas = np.full((30, 30), bg, dtype=np.int8)
    for i, (color, data) in enumerate(items):
        y, x = positions[i]
        view = canvas[y:y+data.shape[0], x:x+data.shape[1]]
        if color is None:
            view[:] = data
        else:
            view[data] = color
    return _trim(rng, canvas, bg)


def _trim(rng, canvas, bg):
    if rng.random() < .25:
        return canvas
    r0, r1, c0, c1 = _bbox(tuple(zip(*np.nonzero(canvas != bg))))
    r0, c0 = max(0, r0-int(rng.integers(4))), max(0, c0-int(rng.integers(4)))
    r1, c1 = min(canvas.shape[0], r1+int(rng.integers(4))), min(canvas.shape[1], c1+int(rng.integers(4)))
    return canvas[r0:r1, c0:c1].copy()


def _palette(rng, previous=None):
    palette = list(map(int, rng.permutation(10)))
    if previous is None:
        return palette
    bg = _choice(rng, [c for c in palette if c != previous['background']])
    carrier = _choice(rng, [c for c in palette if c not in (bg, previous['carrier'])])
    return [bg, carrier] + [c for c in palette if c not in (bg, carrier)]


def _finish(rng, board, holes, elected, proposals, bg, turns):
    output, _, _ = _render(board, holes, elected)
    if max(output.shape) > 30:
        return None
    canvas = _pack(rng, board, proposals, bg)
    if canvas is None:
        return None
    pair = {'input': np.rot90(canvas, turns).copy(), 'output': np.rot90(output, turns).copy()}
    try:
        recovered, _ = _recover(pair['input'])
    except ValueError:
        return None
    if not np.array_equal(recovered, pair['output']):
        raise AssertionError("Construction and recovery disagree")
    return pair


def _teaching_pair(rng, anchor, companion, second, turns, previous=None):
    palette = _palette(rng, previous)
    bg, carrier = palette[:2]
    available = palette[2:]
    forbidden = previous['elected'][anchor][0] if previous is not None else -1
    awin = _choice(rng, [c for c in available if c != forbidden])
    bwin = _choice(rng, [c for c in available if c != awin])
    losers = [c for c in available if c != awin]
    if not second:
        a = _choice(rng, [v for v in _variants(anchor) if _size(v) == (3, 2)])
        b = _choice(rng, [v for v in _variants(companion) if _size(v) == (2, 3)])
        top, left = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        placements = [(anchor, tuple((r+top, c+left) for r, c in a)),
                      (companion, tuple((r+top+1, c+left+4) for r, c in b))]
        h, w = top+3+int(rng.integers(1, 4)), left+7+int(rng.integers(1, 3))
        elected = {anchor: (awin, 2), companion: (bwin, 3)}
        # All winning anchor ballots are reflected relative to its cavity.
        reflected = [v for v in _variants(a) if v not in _variants(a, False)]
        proposals = [(_choice(rng, reflected), awin, 2) for _ in range(2)]
        proposals += [(a, losers[0], 3), (a, losers[1], 2)]
        proposals += [(_choice(rng, _variants(b)), bwin, 3) for _ in range(2)]
        proposals += [(b, _choice(rng, [c for c in available if c != bwin]), 2)]
    else:
        a = _choice(rng, [v for v in _variants(anchor) if _size(v) == (2, 3)])
        b = _choice(rng, [v for v in _variants(companion) if _size(v) == (3, 2)])
        top, left = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        placements = [(anchor, tuple((r+top, c+left) for r, c in a)),
                      (companion, tuple((r+top+4, c+left+2) for r, c in b)),
                      (anchor, tuple((r+top+9, c+left) for r, c in a))]
        h, w = top+11+int(rng.integers(1, 3)), left+4+int(rng.integers(1, 4))
        elected = {anchor: (awin, 3), companion: (bwin, 2)}
        proposals = [(_choice(rng, _variants(a)), awin, 3) for _ in range(3)]
        proposals += [(_choice(rng, _variants(a)), losers[0], 2) for _ in range(2)]
        proposals += [(_choice(rng, _variants(b)), bwin, 2)]
    board = np.full((h, w), carrier, dtype=np.int8)
    holes = [cells for _, cells in placements]
    for cells in holes:
        for r, c in cells:
            board[r, c] = bg
    pair = _finish(rng, board, holes, elected, proposals, bg, turns)
    if pair is None:
        raise RuntimeError("Teaching geometry could not be packed")
    return pair


def _place_cavities(rng, shapes, h, w):
    occupied, placements = set(), []
    for key in sorted(shapes, key=lambda s: -len(s)):
        found = False
        for vi in rng.permutation(len(_variants(key))):
            raw = _variants(key)[int(vi)]
            sh, sw = _size(raw)
            candidates = [(r, c) for r in range(1, h-sh) for c in range(1, w-sw)]
            for pi in rng.permutation(len(candidates)):
                r0, c0 = candidates[int(pi)]
                cells = tuple((r+r0, c+c0) for r, c in raw)
                if any(p in occupied or any((p[0]+dr, p[1]+dc) in occupied for dr, dc in _N4)
                       for p in cells):
                    continue
                placements.append(cells)
                occupied.update(cells)
                found = True
                break
            if found:
                break
        if not found:
            return None
    return placements


def _random_pair(rng, pool, novel=None):
    profiles = ((1,), (2, 1), (3, 1), (3, 2), (2, 1, 1), (3, 2, 1), (2, 1, 1, 1), (4, 2), (5, 1))
    for _ in range(100):
        count = int(rng.choice([1, 2, 3], p=[.25, .55, .20]))
        eligible = [s for s in pool if len(s) <= (8 if count == 1 else 6)]
        if novel is not None:
            unseen = [s for s in eligible if s not in novel]
            if not unseen:
                continue
            chosen = [_choice(rng, unseen)]
        else:
            chosen = [_choice(rng, eligible)]
        remaining = [s for s in eligible if s not in chosen]
        chosen += [remaining[int(j)] for j in rng.choice(len(remaining), count-1, replace=False)]
        palette = _palette(rng)
        bg, carrier, available = palette[0], palette[1], palette[2:]
        elected, proposals = {}, []
        for key in chosen:
            votes = _choice(rng, profiles)
            colors = list(map(int, rng.choice(available, len(votes), replace=False)))
            k = int(rng.choice([2, 3, 4], p=[.60, .32, .08]))
            elected[key] = colors[0], k
            for index, (color, copies) in enumerate(zip(colors, votes)):
                scale = k if index == 0 else int(rng.choice([2, 3, 4], p=[.55, .37, .08]))
                proposals += [(_choice(rng, _variants(key)), color, scale) for _ in range(copies)]
        # Physical constraints limit combinatorial load, never change elections.
        cost = sum((_size(s)[0]*k+1)*(_size(s)[1]*k+1) for s, _, k in proposals)
        if cost > 570 or len(proposals) > 13 or any(max(_size(s))*k > 20 for s, _, k in proposals):
            continue
        repeats = int(rng.integers(0, min(4, 7-count)))
        occurrences = chosen + [_choice(rng, chosen) for _ in range(repeats)]
        for _ in range(12):
            h, w = int(rng.integers(5, 14)), int(rng.integers(5, 14))
            if rng.random() < .15:
                w = h
            holes = _place_cavities(rng, occurrences, h, w)
            if holes is None:
                continue
            board = np.full((h, w), carrier, dtype=np.int8)
            for cells in holes:
                for r, c in cells:
                    board[r, c] = bg
            output, _, _ = _render(board, holes, elected)
            if max(output.shape) > 30:
                continue
            pair = _finish(rng, board, holes, elected, proposals, bg, int(rng.integers(4)))
            if pair is not None:
                return pair
    # Preserve the novel-shape shift even in the bounded conservative fallback.
    candidates = [s for s in pool if novel is None or s not in novel]
    key = _choice(rng, [s for s in candidates if max(_size(s)) <= 4])
    palette = _palette(rng)
    bg, carrier, color = palette[:3]
    sh, sw = _size(key)
    board = np.full((sh+2, sw+2), carrier, dtype=np.int8)
    holes = [tuple((r+1, c+1) for r, c in key)]
    for r, c in holes[0]:
        board[r, c] = bg
    pair = _finish(rng, board, holes, {key: (color, 2)}, [(key, color, 2)], bg, int(rng.integers(4)))
    if pair is None:
        raise RuntimeError("Conservative construction failed")
    return pair


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    pool = _pool(rng)
    teachers = [s for s in pool if len(s) == 4 and sorted(_size(s)) == [2, 3]]
    chiral = [s for s in teachers if len(_variants(s)) > len(_variants(s, False))]
    anchor = _choice(rng, chiral)
    companion = _choice(rng, [s for s in teachers if s != anchor])
    turns = int(rng.integers(4))
    first = _teaching_pair(rng, anchor, companion, False, turns)
    _, info = _recover(first['input'])
    second = _teaching_pair(rng, anchor, companion, True, turns, info)
    puzzle = {'train': [first, second], 'test': []}
    seen = {_key(first['input']), _key(second['input'])}
    for split, count in (('train', num_train-2), ('test', num_test)):
        for _ in range(count):
            known = set().union(*(_recover(p['input'])[1]['classes'] for p in puzzle['train']))
            for _ in range(20):
                pair = _random_pair(rng, pool, known if split == 'test' else None)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("Distinct-input retries exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _unique_max(counter):
    if not counter:
        return None
    high = max(counter.values())
    winners = [c for c, n in counter.items() if n == high]
    return winners[0] if len(winners) == 1 else None


def _witnesses(info):
    objects, elected = info['objects'], info['elected']
    pixels, largest = False, False
    for key, (winner, scale) in elected.items():
        area_votes = Counter()
        for obj in objects:
            if obj['key'] == key:
                area_votes[obj['color']] += len(obj['cells'])
                largest |= obj['color'] != winner and obj['scale'] > scale
        pixels |= _unique_max(area_votes) != winner
    exact = any(len(a) == len(b) and sorted(_size(a)) == sorted(_size(b))
                and elected[a] != elected[b] for a in elected for b in elected if a != b)
    reflection = False
    for cells in info['holes']:
        key = _canonical(cells)
        counts = Counter(obj['color'] for obj in objects if obj['key'] == key
                         and obj['raw'] in _variants(_normalize(cells), False))
        reflection |= _unique_max(counts) != elected[key][0]
    row_keys, col_keys = defaultdict(set), defaultdict(set)
    for cells in info['holes']:
        key = _canonical(cells)
        for r, c in cells:
            row_keys[r].add(key)
            col_keys[c].add(key)
    row_overlap = any(len({elected[k][1] for k in ks}) > 1 for ks in row_keys.values())
    col_overlap = any(len({elected[k][1] for k in ks}) > 1 for ks in col_keys.values())
    return dict(pixel_ablation=pixels, losing_scale=largest, exact_shape=exact,
                reflection=reflection, row_overlap=row_overlap, col_overlap=col_overlap,
                repeated=len(info['holes']) > len(info['classes']),
                plurality=any(len(v) > 1 and max(v.values())*2 <= sum(v.values())
                              for v in info['votes'].values()))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, "Duplicate input"
            seen.add(_key(inp))
            expected, info = _recover(inp)
            assert np.array_equal(expected, out), "Incorrect fill, stretch, or crop"
            assert not np.array_equal(inp, out)
            # Each structural logical cell expands unchanged into its rectangle.
            y, x = np.r_[0, np.cumsum(info['rows'])], np.r_[0, np.cumsum(info['cols'])]
            for r, c in zip(*np.nonzero(info['board'] == info['carrier'])):
                assert np.all(out[y[r]:y[r+1], x[c]:x[c+1]] == info['carrier'])
            assert 1 in info['rows'] and 1 in info['cols']  # intact border lines
            infos[split].append(info)
    teaching = [_witnesses(i) for i in infos['train']]
    for feature in ('pixel_ablation', 'losing_scale', 'exact_shape', 'reflection',
                    'row_overlap', 'col_overlap', 'repeated', 'plurality'):
        assert any(w[feature] for w in teaching), f"Missing training evidence: {feature}"
    history = defaultdict(set)
    for i in infos['train']:
        for key, winner in i['elected'].items():
            history[key].add(winner)
    assert any(len({c for c, k in entries}) > 1 and len({k for c, k in entries}) > 1
               for entries in history.values()), "Need changing local color/scale winners"
    for role in ('background', 'carrier'):
        assert len({i[role] for i in infos['train']}) >= 2
    known = set().union(*(i['classes'] for i in infos['train']))
    assert all(i['classes'] - known for i in infos['test']), "Need a supported novel test shape"


def _puzzle_key(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def self_test():
    puzzles, infos = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(seed))
        puzzles.append(p)
        infos.extend(_recover(pair['input'])[1] for split in ('train', 'test') for pair in p[split])
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({i['board'].shape for i in infos}) >= 30
    assert len({(tuple(i['rows']), tuple(i['cols'])) for i in infos}) >= 90
    assert len({k for i in infos for k in i['classes']}) >= 35
    assert {1, 2, 3, 4} <= {k for i in infos for k in (*i['rows'], *i['cols'])}
    assert len({len(i['holes']) for i in infos}) >= 5
    assert len({len(i['objects']) for i in infos}) >= 7
    assert len({(i['background'], i['carrier']) for i in infos}) >= 40
    assert len({tuple(sorted(v.values(), reverse=True)) for i in infos for v in i['votes'].values()}) >= 8
    assert any(len({c for c, _ in i['elected'].values()}) < len(i['classes']) for i in infos)
    # Count validation is cheap; generation covers boundary and middle requests.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            assert _counts(a, b) == (a, b)
    for a, b in ((2, 1), (10, 4), (7, 3)):
        p = generate_puzzle(963+a, a, b)
        assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    base = puzzles[0]['train'][0]
    answer, i = _recover(base['input'])
    target = next(o for o in i['objects'] if o['color'] != i['elected'][o['key']][0]
                  and o['scale'] > i['elected'][o['key']][1])
    # Enlarge all copies of just one losing candidate: output must be invariant.
    proposals = [(o['raw'], o['color'], 4 if (o['key'], o['color']) == (target['key'], target['color']) else o['scale'])
                 for o in i['objects']]
    changed = _pack(np.random.default_rng(1729), i['board'], proposals, i['background'])
    assert changed is not None and np.array_equal(solve(changed), answer)
    # Add two votes to the first demo's one-vote loser, making it win 3 versus 2.
    proposals = [(o['raw'], o['color'], o['scale']) for o in i['objects']]
    proposals += [(target['raw'], target['color'], target['scale'])]*2
    changed = _pack(np.random.default_rng(1730), i['board'], proposals, i['background'])
    assert changed is not None
    different, j = _recover(changed)
    assert j['elected'][target['key']] == (target['color'], target['scale'])
    assert not np.array_equal(different, answer) and different.shape != answer.shape
    # The solver accepts scales outside the sampler's routine 2..4 range.
    board = np.full((5, 5), 2, dtype=np.int8)
    board[2, 2] = 0
    large = _pack(np.random.default_rng(1731), board, [(((0, 0),), 7, 5)], 0)
    result = solve(large)
    assert result.shape == (9, 9) and np.all(result[2:7, 2:7] == 7)
    # Wrong labels, broken macro-pixels, ties, inconsistent scales, ambiguity.
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Wrong output accepted")
    broken = base['input'].copy()
    broken[i['objects'][0]['cells'][0]] = i['background']
    _must_reject(broken)
    for proposals in ([(((0, 0),), 7, 2), (((0, 0),), 8, 3)],
                      [(((0, 0),), 7, 2), (((0, 0),), 7, 3)]):
        _must_reject(_pack(np.random.default_rng(1732), board, proposals, 0))
    ambiguous = np.full((7, 8), 2, dtype=np.int8)
    ambiguous[1, 1] = 0
    ambiguous[3:5, 4:6] = 0
    _must_reject(_pack(np.random.default_rng(1733), ambiguous, [(((0, 0),), 7, 4)], 0))


def _must_reject(grid):
    assert grid is not None
    try:
        solve(grid)
    except ValueError:
        return
    raise AssertionError("Malformed or ambiguous input accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds)')
