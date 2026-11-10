"""Sealed chambers steer a thread. Only NumPy and the standard library are used.

The four colors are decoded afresh from every input. A boundary seed supplies
the thread color; a single marked endpoint on each component supplies command
ink. Sealed cavities, rather than colored ticks, are the unary clearance code.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(((0 <= a) & (a <= 9)).all()), "invalid color")
    return a.astype(int, copy=True)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _components(mask, neighbors=N4):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack, cells = [p], [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
                    stack.append(q)
        result.append(cells)
    return result


def _holes(mask, neighbors=N4):
    """Object-local topology with a padded, explicitly reachable exterior."""
    p = np.pad(mask, 1)
    seen = np.zeros(p.shape, bool)
    seen[0, 0] = True
    stack = [(0, 0)]
    h, w = p.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not p[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                stack.append((rr, cc))
    return (~p & ~seen)[1:-1, 1:-1]


def _dilate(mask):
    p = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([p[1 + r:1 + r + h, 1 + c:1 + c + w] for r, c in N8])


def _clear_topology(mask):
    a, b, c, d = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
    return not np.any((a == d) & (b == c) & (a != b))


def _from_forward(a, direction):
    # 0: down, 1: right, 2: up, 3: left. Lateral low/high is intrinsic.
    if direction == 0:
        return a.copy()
    if direction == 1:
        return a.T.copy()
    if direction == 2:
        return a[::-1].copy()
    return a[::-1].T.copy()


def _to_forward(a, direction):
    if direction in (0, 2):
        return _from_forward(a, direction)
    return a.T.copy() if direction == 1 else a.T[::-1].copy()


def _analyze(input_grid):
    """Decode colors, original components, chambers, and routing commands."""
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    _require(len(values) == 4, "input must contain four role colors")
    _require(np.count_nonzero(counts == counts.max()) == 1, "ambiguous background")
    bg = int(values[counts.argmax()])
    _require(np.count_nonzero(grid == bg) > grid.size / 2, "background must exceed half the canvas")
    h, w = grid.shape
    candidates = []
    for color, count in zip(values, counts):
        if color == bg or count != 1:
            continue
        r, c = map(int, np.argwhere(grid == color)[0])
        edges = [r == 0, c == 0, r == h - 1, c == w - 1]
        if sum(edges) == 1:
            candidates.append((int(color), edges.index(True)))
    _require(len(candidates) == 1, "seed must be a unique noncorner boundary singleton")
    route, direction = candidates[0]
    remaining = [(int(v), int(n)) for v, n in zip(values, counts) if v not in (bg, route)]
    _require(remaining[0][1] != remaining[1][1], "ambiguous body/command ink")
    body = max(remaining, key=lambda x: x[1])[0]
    ink = min(remaining, key=lambda x: x[1])[0]
    a = _to_forward(grid, direction)
    structure = (a == body) | (a == ink)
    _require(_clear_topology(structure), "diagonal-only topology is outside the family")
    cells = _components(structure)
    _require(len(cells) == len(_components(structure, N8)), "diagonal component contact")
    objects, all_holes = [], np.zeros(a.shape, bool)
    for ccells in cells:
        rr, cc = np.array(ccells).T
        r0, r1, c0, c1 = map(int, (rr.min(), rr.max(), cc.min(), cc.max()))
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), bool)
        local[rr - r0, cc - c0] = True
        holes = _holes(local)
        _require(np.array_equal(holes, _holes(local, N8)), "ambiguous cavity leak")
        chambers = _components(holes)
        _require(len(chambers) == len(_components(holes, N8)), "diagonally adjacent cavities")
        _require(0 <= len(chambers) <= 3, "family supports zero to three cavities per object")
        flags = [(r, c) for r, c in ccells if a[r, c] == ink]
        _require(len(flags) == 1, "each object must contain one marked endpoint cell")
        fr, fc = flags[0]
        _require(c0 < c1 and fc in (c0, c1), "mark must identify one transverse endpoint")
        _require(0 < r0 <= r1 < a.shape[0] - 1 and 0 < c0 <= c1 < a.shape[1] - 1,
                 "original objects must be wholly interior")
        full_body = np.zeros(a.shape, bool)
        full_body[rr, cc] = True
        full_holes = np.zeros(a.shape, bool)
        full_holes[r0:r1 + 1, c0:c1 + 1] = holes
        _require(not np.any(full_holes & structure) and not np.any(full_holes & all_holes),
                 "nested objects or shared cavities")
        all_holes |= full_holes
        objects.append(dict(bounds=(r0, r1, c0, c1), mask=local, holes=holes,
                            n=len(chambers), sign=1 if fc == c0 else -1,
                            full_body=full_body, full_holes=full_holes, flag=(fr, fc)))
    objects.sort(key=lambda o: o['bounds'][0])
    _require(objects and len(objects) <= 4, "family supports one to four objects")
    _require(np.array_equal(_holes(structure), all_holes), "collective enclosure")
    previous = -1
    for o in objects:
        _require(o['bounds'][0] > previous, "objects must occupy disjoint forward slabs")
        previous = o['bounds'][1]
    initial = int(np.argwhere(a == route)[0, 1])
    path, bends = _trace(a.shape, objects, initial)
    _require(not np.any(path & (structure | all_holes)), "thread collides with an object")
    outline = _dilate(structure) & ~structure & ~all_holes
    return dict(grid=grid, a=a, bg=bg, body=body, ink=ink, route=route,
                direction=direction, initial=initial, objects=objects,
                structure=structure, holes=all_holes, outline=outline, path=path, bends=bends)


def _trace(shape, objects, initial, *, ignore_cavities=False, fixed_clearance=None,
           invert_sides=False):
    """Forward-monotone thread; optional ablations are used only by self_test."""
    height, width = shape
    path = np.zeros(shape, bool)
    current, start, previous = initial, 0, -1
    bends = []
    for o in objects:
        r0, r1, c0, c1 = o['bounds']
        n = 0 if ignore_cavities else o['n']
        distance = n + 1 if fixed_clearance is None else fixed_clearance
        sign = -o['sign'] if invert_sides else o['sign']
        target = c1 + distance if sign == 1 else c0 - distance
        turn = r0 - distance
        _require(1 <= turn < height and turn > previous and turn >= start,
                 "insufficient forward space for an encoded turn")
        _require(0 <= target < width, "encoded bypass is out of bounds")
        _require(c0 <= current <= c1 and o['mask'][:, current - c0].any(),
                 "every object must intercept the incoming forward ray")
        path[start:turn + 1, current] = True
        path[turn, min(current, target):max(current, target) + 1] = True
        bends.append((turn, current, target, distance))
        current, start, previous = target, turn, r1
    path[start:, current] = True
    return path, bends


def _render(z):
    out = z['a'].copy()
    out[z['outline']] = z['ink']
    for o in z['objects']:
        out[o['full_body']] = z['ink'] if o['n'] else z['body']
        out[o['full_holes']] = z['route']
    out[z['path']] = z['route']
    return _from_forward(out, z['direction'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the answer using only this input, without learned or latent state."""
    return _render(_analyze(input_grid))


def _partition(rng, length, number):
    """Positive lengths with one wall cell between each adjacent slot."""
    free = length - (2 * number - 1)
    _require(free >= 0, "insufficient pocket space")
    sizes = 1 + rng.multinomial(free, np.full(number, 1 / number))
    result, start = [], 1
    for size in sizes:
        result.append((start, start + int(size)))
        start += int(size) + 1
    return result


def _carve(rng, shape):
    h, w = shape
    mask = np.zeros(shape, bool)
    r0 = int(rng.integers(h))
    r1 = int(rng.integers(r0 + 1, h + 1))
    c0 = int(rng.integers(w))
    c1 = int(rng.integers(c0 + 1, w + 1))
    mask[r0:r1, c0:c1] = True
    if r1 - r0 >= 2 and c1 - c0 >= 2 and rng.random() < .6:
        # One L-shaped cavity, not two cavities or a pixel-area instruction.
        mask[r0 + 1:r1, c0 + 1:c1] = False
    return mask


def _sculpt(rng, mask):
    holes = _holes(mask)
    protected = holes | _dilate(holes)
    h, w = mask.shape
    for _ in range(5):
        candidate = mask.copy()
        side = int(rng.integers(4))
        if side < 2:
            c0 = int(rng.integers(w))
            c1 = int(rng.integers(c0 + 1, w + 1))
            depth = int(rng.integers(1, max(2, h // 2 + 1)))
            region = (slice(0, depth) if side == 0 else slice(h - depth, h), slice(c0, c1))
        else:
            r0 = int(rng.integers(h))
            r1 = int(rng.integers(r0 + 1, h + 1))
            depth = int(rng.integers(1, max(2, w // 3 + 1)))
            region = (slice(r0, r1), slice(0, depth) if side == 2 else slice(w - depth, w))
        candidate[region] &= protected[region]
        if (candidate.any() and candidate.any(axis=0).all() and candidate.any(axis=1).all()
                and _clear_topology(candidate) and len(_components(candidate)) == 1
                and np.array_equal(_holes(candidate), holes)):
            mask = candidate
    return mask


def _make_shape(rng, n, height, width_max=14):
    if n == 0:
        width = int(rng.integers(4, width_max + 1))
        mask = np.ones((height, width), bool)
        style = int(rng.integers(4))
        if height >= 3 and style in (0, 1):
            # Deep bay, sometimes with a one-cell mouth that its outline seals.
            mask[1:-1, 1:-1] = False
            col = int(rng.integers(1, width - 1))
            mask[0, col] = False
            if style == 1 and width >= 5:
                mask[0, col:min(width - 1, col + 2)] = False
        elif height >= 2 and style == 2:
            for r in range(height - 1):
                mask[r, :min(width - 1, r + 1)] = False
        if rng.random() < .55:
            mask = _sculpt(rng, mask)
    else:
        _require(height >= 3, "sealed cavities need height at least three")
        layouts = ['horizontal']
        if height >= 2 * n + 1:
            layouts.append('vertical')
        if n == 3 and height >= 5:
            layouts.append('grid')
        layout = str(rng.choice(layouts))
        min_width = 2 * n + 1 if layout == 'horizontal' else 5 if layout == 'grid' else 3
        width = int(rng.integers(min_width, width_max + 1))
        mask = np.ones((height, width), bool)
        if layout == 'horizontal':
            slots = [(1, height - 1, c0, c1) for c0, c1 in _partition(rng, width - 2, n)]
        elif layout == 'vertical':
            slots = [(r0, r1, 1, width - 1) for r0, r1 in _partition(rng, height - 2, n)]
        else:
            rs, cs = _partition(rng, height - 2, 2), _partition(rng, width - 2, 2)
            options = [(r0, r1, c0, c1) for r0, r1 in rs for c0, c1 in cs]
            slots = [options[int(i)] for i in rng.permutation(4)[:3]]
        for r0, r1, c0, c1 in slots:
            mask[r0:r1, c0:c1] &= ~_carve(rng, (r1 - r0, c1 - c0))
        if rng.random() < .75:
            mask = _sculpt(rng, mask)
    if rng.random() < .5:
        mask = mask[::-1].copy()
    if rng.random() < .5:
        mask = mask[:, ::-1].copy()
    _require(len(_components(mask)) == 1 and _clear_topology(mask), "bad sampled body")
    _require(len(_components(_holes(mask))) == n, "sampled cavity count changed")
    return mask


def _contrast(rng, width_max):
    h, w = int(rng.integers(6, 8)), int(rng.integers(7, width_max + 1))
    closed = np.ones((h, w), bool)
    # A large, nonrectangular single cavity distinguishes region count from
    # pixel count, and leaves background in the bay after its mouth is outlined.
    hole = np.ones((h - 2, w - 2), bool)
    corner = int(rng.integers(4))
    hole[0 if corner < 2 else -1, 0 if corner % 2 == 0 else -1] = False
    closed[1:-1, 1:-1] &= ~hole
    opened = closed.copy()
    rr, cc = np.argwhere(hole)[int(rng.integers(hole.sum()))] + 1
    opened[:rr + 1, cc] = False
    if rng.random() < .5:
        closed, opened = closed[::-1].copy(), opened[::-1].copy()
    _require(not _holes(opened).any() and len(_components(opened)) == 1, "bad contrast door")
    return [opened, closed]


def _span_plan(rng, width, masks, ns, prescribed):
    orders = {}

    @lru_cache(None)
    def search(index, current):
        if index == len(masks):
            return ()
        length, d = masks[index].shape[1], ns[index] + 1
        signs = prescribed[index] if prescribed is not None else (-1, 1)
        state = (index, current)
        if state not in orders:
            options = []
            for c0 in range(1, width - length):
                c1 = c0 + length - 1
                if not c0 <= current <= c1:
                    continue
                if not masks[index][:, current - c0].any():
                    continue
                for sign in signs:
                    target = c1 + d if sign == 1 else c0 - d
                    if 0 <= target < width:
                        options.append((c0, target, sign))
            orders[state] = [options[int(i)] for i in rng.permutation(len(options))]
        for c0, target, sign in orders[state]:
            rest = search(index + 1, target)
            if rest is not None:
                return ((c0, target, sign),) + rest
        return None

    for initial in rng.permutation(np.arange(1, width - 1)):
        result = search(0, int(initial))
        if result is not None:
            return int(initial), result
    return None


def _make_input(rng, palette, direction, masks, prescribed=None):
    bg, body, ink, route = palette
    ns = [len(_components(_holes(m))) for m in masks]
    minimum = sum(m.shape[0] + n + 1 for m, n in zip(masks, ns)) + 2
    _require(minimum <= 30, "shapes cannot fit in the forward dimension")
    height = int(rng.integers(minimum, 31))
    min_width = max(m.shape[1] + n + 3 for m, n in zip(masks, ns))
    width0 = int(rng.integers(min_width, 31))
    for width in dict.fromkeys((width0, min(30, width0 + 4), 30)):
        plan = _span_plan(rng, width, masks, ns, prescribed)
        if plan is not None:
            break
    else:
        raise AssertionError("no feasible lateral plan")
    initial, spans = plan
    allocation = rng.multinomial(height - minimum, np.full(len(masks) + 1, 1 / (len(masks) + 1)))
    a = np.full((height, width), bg, int)
    a[0, initial] = route
    previous = 0
    for i, (mask, n, (c0, target, sign)) in enumerate(zip(masks, ns, spans)):
        turn = previous + 1 + int(allocation[i])
        r0 = turn + n + 1
        mh, mw = mask.shape
        a[r0:r0 + mh, c0:c0 + mw][mask] = body
        end = 0 if sign == 1 else mw - 1
        row = int(rng.choice(np.flatnonzero(mask[:, end])))
        a[r0 + row, c0 + end] = ink
        previous = r0 + mh - 1
    _require(previous + 1 + int(allocation[-1]) == height - 1, "forward plan arithmetic")
    _require(np.count_nonzero(a == bg) > a.size / 2, "scene too dense")
    return _from_forward(a, direction)


def _heights(rng, ns, cap=7, lower=1):
    heights = [max(lower, 3 if n else 1) for n in ns]
    budget = 28 - sum(n + 1 for n in ns) - sum(heights)
    _require(budget >= 0, "minimum obstacle sizes exceed canvas")
    desires = [int(rng.integers(h, cap + 1)) - h for h in heights]
    for i in rng.permutation(len(ns)):
        extra = min(budget, desires[int(i)])
        heights[int(i)] += extra
        budget -= extra
    return heights


def _checked_counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + " must be an integer")
        if not low <= value <= high:
            raise ValueError(name + " outside supported range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Reproducible geometry, with two complementary teaching scenes."""
    num_train, num_test = _checked_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    variable_palette = bool(rng.random() < .7)
    shift = int(rng.integers(3))  # held-out heading, longer sequence, wider bodies
    directions = list(map(int, rng.permutation(4)))
    puzzle, seen = {'train': [], 'test': []}, set()

    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for attempt in range(80):
                pcolors = tuple(map(int, rng.choice(10, 4, replace=False))) if variable_palette else palette
                width_max = 10 if shift == 2 and split == 'train' else 16 if shift == 2 else 14
                prescribed = None
                if split == 'train' and index == 0:
                    masks = _contrast(rng, width_max)
                    masks = [masks[int(i)] for i in rng.permutation(2)]
                    sign = int(rng.choice((-1, 1)))
                    prescribed = [(sign,), (sign,)]
                elif split == 'train' and index == 1:
                    ns = list(map(int, rng.permutation((2, 3))))
                    masks = [_make_shape(rng, n, h, width_max)
                             for n, h in zip(ns, _heights(rng, ns))]
                    sign = int(rng.choice((-1, 1)))
                    prescribed = [(sign,), (-sign,)]
                else:
                    number = int(rng.integers(3, 5)) if split == 'test' and shift == 1 else int(rng.integers(1, 4))
                    if split == 'train' and shift == 1:
                        number = int(rng.integers(1, 3))
                    ns = list(map(int, rng.integers(0, 4, size=number)))
                    # Wider tests change the object's span, not just its canvas.
                    lower = 3 if split == 'test' and shift == 2 else 1
                    heights = _heights(rng, ns, lower=lower)
                    masks = [_make_shape(rng, n, h, width_max) for n, h in zip(ns, heights)]
                    if split == 'test' and shift == 2 and max(m.shape[1] for m in masks) <= 10:
                        continue
                if shift == 0:
                    direction = directions[index % 2] if split == 'train' else int(rng.choice(directions[2:]))
                else:
                    direction = int(rng.integers(4))
                try:
                    inp = _make_input(rng, pcolors, direction, masks, prescribed)
                except AssertionError:
                    continue
                if _key(inp) in seen:
                    continue
                seen.add(_key(inp))
                puzzle[split].append({'input': inp, 'output': solve(inp)})
                break
            else:
                raise RuntimeError("bounded construction could not find a distinct feasible example")
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def _separated_runs(mask):
    return any(len(v := np.flatnonzero(row)) > 1 and np.any(np.diff(v) > 1)
               for rows in (mask, mask.T) for row in rows)


def validate_generated_puzzle(puzzle):
    """Pixel-derived semantics and teaching coverage; no generation metadata."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _checked_counts(len(puzzle['train']), len(puzzle['test']))
    seen, inspections = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape and _key(a) not in seen, "shape change or duplicate input")
            seen.add(_key(a))
            z = _analyze(a)
            canonical = _to_forward(b, z['direction'])
            _require(np.all(canonical[z['path']] == z['route']), "incomplete thread")
            _require(np.all(canonical[z['outline'] & ~z['path']] == z['ink']), "incorrect exterior collar")
            _require(np.all(canonical[z['holes']] == z['route']), "cavities not filled")
            for o in z['objects']:
                _require(np.all(canonical[o['full_body']] == (z['ink'] if o['n'] else z['body'])),
                         "incorrect whole-body classification or flag consumption")
                # Clearance is exact in BOTH axes and opposite the marked end.
            for o, (turn, incoming, target, distance) in zip(z['objects'], z['bends']):
                r0, r1, c0, c1 = o['bounds']
                _require(distance == o['n'] + 1 and r0 - turn == distance, "wrong approach clearance")
                _require((target - c1 if o['sign'] == 1 else c0 - target) == distance, "wrong endpoint clearance")
            far = ~z['structure'] & ~z['holes'] & ~z['outline'] & ~z['path']
            _require(far.any() and np.all(canonical[far] == z['bg']), "distant background changed")
            _require(np.array_equal(b, _render(z)), "unexpected changed cells")
            inspections[split].append(z)
    objects = [o for z in inspections['train'] for o in z['objects']]
    _require({o['n'] for o in objects} == {0, 1, 2, 3}, "training must teach all cavity counts")
    _require({o['sign'] for o in objects} == {-1, 1}, "training must teach both marked ends")
    _require(any(o['n'] == 0 and _separated_runs(o['mask']) for o in objects), "no open-bay counterexample")
    transitions = {(a['sign'], b['sign']) for z in inspections['train']
                   for a, b in zip(z['objects'], z['objects'][1:])}
    _require(any(a == b for a, b in transitions) and any(a != b for a, b in transitions),
             "training must distinguish command following from automatic alternation")
    _require(any({o['n'] for o in z['objects']} == {0, 1}
                 and len({o['mask'].shape for o in z['objects']}) == 1 for z in inspections['train']),
             "missing comparable open/sealed teaching shapes")
    train_dirs = {z['direction'] for z in inspections['train']}
    max_count = max(len(z['objects']) for z in inspections['train'])
    max_width = max(o['mask'].shape[1] for o in objects)
    _require(all(z['direction'] not in train_dirs or len(z['objects']) > max_count
                 or max(o['mask'].shape[1] for o in z['objects']) > max_width
                 for z in inspections['test']), "no visible train/test shift")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def _literal_regression():
    # Literal geometry and expected bends; no random plan supplies this answer.
    a = np.zeros((20, 18), int)
    a[0, 6] = 7
    # Open 3x5 and sealed 3x6 chambers. Both command a right bypass.
    # On the second, the incoming ray is closer to the forbidden marked end.
    a[3:6, 3:8] = 2
    a[4, 4:7] = 0
    a[3, 5] = 0
    a[4, 3] = 4
    a[11:14, 7:13] = 2
    a[12, 8:12] = 0
    a[12, 7] = 4
    expected_path = np.zeros(a.shape, bool)
    expected_path[:3, 6] = True
    expected_path[2, 6:9] = True
    expected_path[2:10, 8] = True
    expected_path[9, 8:15] = True
    expected_path[9:, 14] = True
    expected = a.copy()
    structure = (a == 2) | (a == 4)
    holes = np.zeros(a.shape, bool)
    holes[12, 8:12] = True
    expected[_dilate(structure) & ~structure & ~holes] = 4
    expected[4, 3] = 2
    expected[11:14, 7:13][structure[11:14, 7:13]] = 4
    expected[holes] = 7
    expected[expected_path] = 7
    z = _analyze(a)
    _require([o['n'] for o in z['objects']] == [0, 1], "open/sealed regression")
    _require(z['bends'] == [(2, 6, 8, 1), (9, 8, 14, 2)], "literal bend regression")
    # Shortcut comparisons use explicit alternative bends, not the decoder.
    _require((8 - 7) < (12 - 8), "nearest-end shortcut must disagree")
    _require(np.array_equal(z['path'], expected_path), "literal thread regression")
    _require(expected[4, 5] == 4, "original open bay receives collar, not cavity fill")
    for direction in range(4):
        # Also test arbitrary recoloring, including nonzero background.
        colors = np.array([9, 1, 0, 8, 5, 6, 2, 3, 4, 7])
        _require(np.array_equal(solve(colors[_from_forward(a, direction)]),
                               colors[_from_forward(expected, direction)]), "rotation/palette regression")


def self_test():
    """Bounded 42-seed audit plus literal regressions and corruption checks."""
    _literal_regression()
    shapes, palettes, bodies, pockets, layouts = set(), set(), set(), set(), set()
    joint, transitions, counts, areas, dimensions = set(), set(), set(), set(), set()
    irregular, halo_seals_open, mixed_area, nontemplate = 0, 0, 0, 0
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(_puzzle_key(puzzle))
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                z = _analyze(pair['input'])
                _require(np.array_equal(solve(pair['input'].copy()), pair['output']), "input-only solve disagrees")
                shapes.add(pair['input'].shape)
                palettes.add((z['bg'], z['body'], z['ink'], z['route']))
                semantic = np.zeros(z['a'].shape, np.uint8)
                for i, color in enumerate((z['bg'], z['body'], z['ink'], z['route'])):
                    semantic[z['a'] == color] = i
                layouts.add(_key(semantic))
                counts.add(len(z['objects']))
                joint.add((z['direction'], split))
                transitions.update((a['sign'], b['sign']) for a, b in zip(z['objects'], z['objects'][1:]))
                for o in z['objects']:
                    bodies.add(_key(o['mask']))
                    dimensions.add(o['mask'].shape)
                    for cavity in _components(o['holes']):
                        rr, cc = np.array(cavity).T
                        cmask = o['holes'][rr.min():rr.max() + 1, cc.min():cc.max() + 1]
                        pockets.add(_key(cmask))
                        irregular += len(cavity) < cmask.size
                        areas.add(len(cavity))
                    if not o['n']:
                        halo_seals_open += _holes(o['mask'] | _dilate(o['mask'])).any()
                    nontemplate += not (o['mask'][0].all() and o['mask'][-1].all()
                                         and o['mask'][:, 0].all() and o['mask'][:, -1].all())
                local_hollows = [o for o in z['objects'] if o['n']]
                mixed_area += any(a['n'] == b['n'] and a['holes'].sum() != b['holes'].sum()
                                  for i, a in enumerate(local_hollows) for b in local_hollows[i + 1:])
        # Removing topology must change actual routing, not just recolored cells.
        witness = next(_analyze(p['input']) for p in puzzle['train']
                       if {o['n'] for o in _analyze(p['input'])['objects']} == {0, 1})
        try:
            no_a, _ = _trace(witness['a'].shape, witness['objects'], witness['initial'], ignore_cavities=True)
        except AssertionError:
            # A wrong first bend may also miss the next gate entirely.
            no_a = None
        _require(no_a is None or not np.array_equal(no_a, witness['path']),
                 "topology ablation did not change route")
        # Without B's route drawing, at least one exterior thread pixel vanishes.
        _require(np.any(witness['path'] & ~witness['outline'] & (witness['a'] == witness['bg'])),
                 "routing ablation did not change output")
    _require(len(set(puzzles)) == 42, "cross-seed puzzle duplication")
    _require(len(shapes) >= 65 and len(layouts) >= 180 and len(bodies) >= 120, "insufficient structural variation")
    _require(len(palettes) >= 70 and len(dimensions) >= 35 and len(pockets) >= 12, "insufficient palette/shape variation")
    _require(counts == {1, 2, 3, 4} and len(joint) == 8, "missing count or direction/split diversity")
    _require(transitions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}, "compulsory alternation shortcut")
    _require(irregular >= 8 and halo_seals_open >= 20 and nontemplate >= 40 and len(areas) >= 6,
             "missing cavity/silhouette contrasts")
    _require(mixed_area >= 3, "cavity count is confounded with cavity area")
    _require(_puzzle_key(generate_puzzle(7)) == puzzles[7], "nondeterminism")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + 10 * train + test, train, test))
    for request in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                    (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *request)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    puzzle = generate_puzzle(8)
    z = _analyze(puzzle['train'][0]['input'])
    canonical_output = _to_forward(puzzle['train'][0]['output'], z['direction'])
    zones = [z['path'] & (z['a'] == z['bg']), z['outline'] & ~z['path'], z['holes'], z['structure']]
    for zone in zones:
        if not zone.any():
            continue
        damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzle[s]] for s in puzzle}
        bad = canonical_output.copy()
        r, c = np.argwhere(zone)[0]
        bad[r, c] = (int(bad[r, c]) + 1) % 10
        damaged['train'][0]['output'] = _from_forward(bad, z['direction'])
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted zone corruption")
    return dict(seeds=42, pairs=210, canvas_shapes=len(shapes), normalized_inputs=len(layouts),
                palettes=len(palettes), body_masks=len(bodies), cavity_masks=len(pockets),
                irregular_cavities=int(irregular), outline_sealed_open_bays=int(halo_seals_open),
                nonrectangular_bodies=int(nontemplate), equal_count_different_area_pairs=int(mixed_area),
                directions_by_split=len(joint), transitions=sorted(transitions), object_counts=sorted(counts))


if __name__ == '__main__':
    report = self_test()
    Path('scratch').mkdir(exist_ok=True)
    Path('scratch/self_test.json').write_text(json.dumps(report, separators=(',', ':')) + '\n')
    print(json.dumps(report, separators=(',', ':')))
