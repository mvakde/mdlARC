"""Counted tile dictionary: color components are exact replication quotas.

Only NumPy and the standard library are used. No parent code is imported.
See task_spec.md for the visible conventions and the admissible input family.
"""
from __future__ import annotations

from collections import Counter
import json

import numpy as np


class _SearchLimit(Exception):
    pass


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer 1..30 by 1..30 grid, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    h, w = mask.shape
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _box(cells):
    ys, xs = zip(*cells)
    return min(ys), min(xs), max(ys)+1, max(xs)+1


def _rim(shape, thickness):
    h, w = shape
    m = np.ones(shape, bool)
    m[thickness:h-thickness, thickness:w-thickness] = False
    return m


def _inspect(input_grid):
    """Recover the board, dictionary, and vouchers from the input alone."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError("The canvas needs a complete background outer margin")
    boards = []
    for color in np.unique(a):
        color = int(color)
        if color == bg:
            continue
        cells = np.argwhere(a == color)
        y, x, y1, x1 = _box(cells)
        panel = a[y:y1, x:x1]
        if min(panel.shape) < 5:
            continue
        rim = _rim(panel.shape, 2)
        if (np.all(panel[rim] == color) and np.all((panel == bg) | (panel == color))
                and np.any(panel[2:-2, 2:-2] == bg)):
            boards.append((color, (y, x, y1, x1), panel.copy()))
    if len(boards) != 1:
        raise ValueError("There must be one board with a two-cell solid rim")
    wall, board_box, board = boards[0]
    excluded = np.zeros(a.shape, bool)
    y, x, y1, x1 = board_box
    excluded[y:y1, x:x1] = True
    templates = []
    for color in np.unique(a):
        color = int(color)
        if color in (bg, wall):
            continue
        for comp in _components((a == color) & ~excluded):
            r, c, r1, c1 = _box(comp)
            shape = r1-r, c1-c
            if min(shape) < 5 or len(comp) != 2*sum(shape)-4:
                continue
            rim = _rim(shape, 1)
            local = a[r:r1, c:c1]
            if not np.all(local[rim] == color):
                continue
            inside = local[1:-1, 1:-1]
            occupied = inside != bg
            if not np.any(occupied):
                continue
            py, px, py1, px1 = _box(np.argwhere(occupied))
            if py == 0 or px == 0 or py1 == inside.shape[0] or px1 == inside.shape[1]:
                continue
            if int(occupied.sum()) != (py1-py)*(px1-px):
                continue
            pattern = inside[py:py1, px:px1].copy()
            templates.append(dict(color=color, box=(r, c, r1, c1), pattern=pattern))
    if not 2 <= len(templates) <= 4 or len({t['color'] for t in templates}) != len(templates):
        raise ValueError("Expected two through four distinct dictionary frame colors")
    for t in templates:
        r, c, r1, c1 = t['box']
        if np.any(excluded[r:r1, c:c1]):
            raise ValueError("Panels overlap")
        excluded[r:r1, c:c1] = True
    # Structural rectangles and free objects have a background moat, including
    # diagonally. This also prevents a voucher merging into a template border.
    labels = np.full(a.shape, -1, int)
    labels[y:y1, x:x1] = 0
    for i, t in enumerate(templates, 1):
        r, c, r1, c1 = t['box']
        labels[r:r1, c:c1] = i
    objects = []
    for color in np.unique(a[~excluded]):
        color = int(color)
        if color == bg:
            continue
        for comp in _components((a == color) & ~excluded):
            if not 1 <= len(comp) <= 12:
                raise ValueError("Voucher objects have area 1..12")
            objects.append(dict(color=color, cells=comp))
            i = len(templates)+len(objects)
            for r, c in comp:
                labels[r, c] = i
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        r0, r1 = max(0, dr), a.shape[0]+min(0, dr)
        c0, c1 = max(0, dc), a.shape[1]+min(0, dc)
        lhs = labels[r0:r1, c0:c1]
        rhs = labels[r0-dr:r1-dr, c0-dc:c1-dc]
        if np.any((lhs >= 0) & (rhs >= 0) & (lhs != rhs)):
            raise ValueError("Separate panels and objects must not touch")
    counts = Counter(o['color'] for o in objects)
    areas = Counter()
    for o in objects:
        areas[o['color']] += len(o['cells'])
    quotas = tuple(counts[t['color']] for t in templates)
    if not 2 <= sum(quotas) <= 8:
        raise ValueError("The counted inventory contains two through eight copies")
    return dict(grid=a, background=bg, wall=wall, box=board_box, board=board,
                mask=board == bg, templates=templates, quotas=quotas,
                objects=objects, areas=areas)


def _covers(mask, patterns, quotas, *, exact=True, limit=None):
    """Enumerate up to two distinct pixel completions, respecting finite quotas.

    This is not just a construction-placement check: every legal rectangle
    translation is searched. Multiple decompositions with the same pixels are
    allowed. With exact=False, counts are capacities and unused copies are legal.
    """
    cells = [tuple(map(int, p)) for p in np.argwhere(mask)]
    n = len(cells)
    if exact and sum(p.size*q for p, q in zip(patterns, quotas)) != n:
        return [], dict(nodes=0, failures=1)
    if sum(p.size*q for p, q in zip(patterns, quotas)) < n:
        return [], dict(nodes=0, failures=1)
    indices = {p: i for i, p in enumerate(cells)}
    by_cell = [[] for _ in cells]
    placements = []
    for k, pattern in enumerate(patterns):
        if quotas[k] == 0:
            continue
        h, w = pattern.shape
        for y in range(mask.shape[0]-h+1):
            for x in range(mask.shape[1]-w+1):
                if not np.all(mask[y:y+h, x:x+w]):
                    continue
                assignment = tuple((indices[y+dy, x+dx], int(pattern[dy, dx]))
                                   for dy in range(h) for dx in range(w))
                bits = sum(1 << i for i, _ in assignment)
                p = len(placements)
                placements.append((k, bits, assignment, (y, x)))
                for i, _ in assignment:
                    by_cell[i].append(p)
    remaining = list(quotas)
    full = (1 << n)-1
    rendered = [-1]*n
    outcomes, dead, path = {}, set(), []
    nodes = failures = 0

    def visit(covered):
        nonlocal nodes, failures
        if len(outcomes) == 2:
            return True
        nodes += 1
        if limit is not None and nodes > limit:
            raise _SearchLimit
        if covered == full:
            if not exact or not any(remaining):
                key = tuple(rendered)
                outcomes.setdefault(key, tuple(path))
                return True
            return False
        state = covered, tuple(remaining)
        if state in dead:
            return False
        best = None
        for i in range(n):
            if covered & (1 << i):
                continue
            options = [j for j in by_cell[i] if remaining[placements[j][0]]
                       and not covered & placements[j][1]]
            if not options:
                dead.add(state)
                failures += 1
                return False
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        success = False
        for j in best:
            k, bits, assignment, pos = placements[j]
            remaining[k] -= 1
            for i, color in assignment:
                rendered[i] = color
            path.append((k, pos))
            success |= visit(covered | bits)
            path.pop()
            for i, _ in assignment:
                rendered[i] = -1
            remaining[k] += 1
            if len(outcomes) == 2:
                break
        if not success:
            dead.add(state)
            failures += 1
        return success

    visit(0)
    return list(outcomes.items()), dict(nodes=nodes, failures=failures)


def _answer(detail, *, limit=None):
    patterns = [t['pattern'] for t in detail['templates']]
    solutions, stats = _covers(detail['mask'], patterns, detail['quotas'], limit=limit)
    if len(solutions) != 1:
        raise ValueError("Counted tiles must give exactly one rendered completion")
    output = detail['board'].copy()
    output[detail['mask']] = solutions[0][0]
    return output, solutions[0][1], stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver: count objects, exact-cover, copy patterns, crop board."""
    return _answer(_inspect(input_grid))[0]


def _shape(rng, area):
    kind = int(rng.integers(4))
    if kind == 0:
        cells = {(0, i) for i in range(area)}
    elif kind == 1:
        w = int(rng.integers(1, min(4, area)+1))
        cells = {(i//w, i % w) for i in range(area)}
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr, dc in ((0, 1), (1, 0), (-1, 0), (0, -1))} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-y, c-x) for r, c in cells))


def _pattern(rng, shape, palette, asymmetric=False):
    h, w = shape
    a, b = map(int, rng.choice(palette, 2, replace=False))
    p = np.full(shape, a, np.int8)
    yy, xx = np.indices(shape)
    kind = int(rng.integers(6))
    if kind == 1:
        p[yy % 2 == 1] = b
    elif kind == 2:
        p[xx % 2 == 1] = b
    elif kind == 3:
        p[(yy+xx) % 2 == 1] = b
    elif kind == 4:
        p = rng.choice(palette, size=shape).astype(np.int8)
    elif kind == 5:
        p[int(rng.integers(h)), int(rng.integers(w))] = b
    if asymmetric:
        p.fill(a)
        p[0, 0] = b
    return p


def _grow(rng, dimensions, quotas, size):
    mask = np.zeros((size, size), bool)
    inventory = [k for k, n in enumerate(quotas) for _ in range(n)]
    rng.shuffle(inventory)
    rects = []
    for k in inventory:
        h, w = dimensions[k]
        proposals = []
        for _ in range(80):
            if not rects or rng.random() < .08:
                y, x = int(rng.integers(size-h+1)), int(rng.integers(size-w+1))
            else:
                _, r, c, rh, rw = rects[int(rng.integers(len(rects)))]
                side = int(rng.integers(4))
                if side < 2:
                    y = r-h if side == 0 else r+rh
                    x = c+int(rng.integers(1-w, rw))
                else:
                    x = c-w if side == 2 else c+rw
                    y = r+int(rng.integers(1-h, rh))
            if y < 0 or x < 0 or y+h > size or x+w > size or mask[y:y+h, x:x+w].any():
                continue
            proposals.append((y, x))
        if not proposals:
            return None
        y, x = proposals[int(rng.integers(len(proposals)))]
        mask[y:y+h, x:x+w] = True
        rects.append((k, y, x, h, w))
    y, x, y1, x1 = _box(np.argwhere(mask))
    margin = int(rng.integers(0, 3))
    cropped = np.pad(mask[y:y1, x:x1], margin)
    return cropped, [(k, r-y+margin, c-x+margin, h, w) for k, r, c, h, w in rects]


def _frame_tile(rng, color, pattern, bg):
    h, w = pattern.shape
    eh, ew = int(rng.integers(3)), int(rng.integers(3))
    panel = np.full((h+4+eh, w+4+ew), bg, np.int8)
    panel[_rim(panel.shape, 1)] = color
    y, x = 2+int(rng.integers(eh+1)), 2+int(rng.integers(ew+1))
    panel[y:y+h, x:x+w] = pattern
    return panel


def _scatter(rng, panels, objects, bg, training):
    """Scatter structural rectangles and actual polyomino cells, without a bank."""
    for attempt in range(20):
        low = 21 if training else 23
        h, w = map(int, rng.integers(low, 31, size=2)) if attempt < 12 else (30, 30)
        a = np.full((h, w), bg, np.int8)
        busy = np.zeros((h, w), bool)
        busy[0] = busy[-1] = True
        busy[:, 0] = busy[:, -1] = True
        positions = {}
        order = sorted(range(len(panels)), key=lambda i: -panels[i].size)
        ok = True
        for k in order:
            panel = panels[k]
            ph, pw = panel.shape
            legal = [(y, x) for y in range(1, h-ph) for x in range(1, w-pw)
                     if not busy[y:y+ph, x:x+pw].any()]
            if not legal:
                ok = False
                break
            y, x = legal[int(rng.integers(len(legal)))]
            a[y:y+ph, x:x+pw] = panel
            busy[y-1:y+ph+1, x-1:x+pw+1] = True
            positions[k] = (y, x)
        if not ok:
            continue
        for color, cells in sorted(objects, key=lambda o: -len(o[1])):
            ph, pw = 1+max(y for y, x in cells), 1+max(x for y, x in cells)
            legal = np.ones((max(0, h-ph-1), max(0, w-pw-1)), bool)
            for dy, dx in cells:
                legal &= ~busy[1+dy:h-ph+dy, 1+dx:w-pw+dx]
            ys, xs = np.nonzero(legal)
            if not len(ys):
                ok = False
                break
            j = int(rng.integers(len(ys)))
            y, x = int(ys[j])+1, int(xs[j])+1
            for dy, dx in cells:
                a[y+dy, x+dx] = color
                busy[y+dy-1:y+dy+2, x+dx-1:x+dx+2] = True
        if ok:
            return a, positions[0]
    return None


def _make_pair(rng, colors, quotas, bg, wall, distractors, forced_areas,
               training, rotation, asymmetric):
    # The first two dimensions deliberately have a geometric substitution:
    # two small tiles can replace one long tile if the quotas are ignored.
    for attempt in range(220):
        base = [int(rng.integers(1, 4)), int(rng.integers(1, 4))]
        if asymmetric and base == [1, 1]:
            base[int(rng.integers(2))] = 3
        axis = int(rng.integers(2))
        double = base.copy()
        double[axis] *= 2
        dimensions = [tuple(base), tuple(double)]
        while len(dimensions) < len(colors):
            d = tuple(map(int, rng.integers(1, 5, size=2)))
            if d not in dimensions:
                dimensions.append(d)
        grown = _grow(rng, dimensions, quotas, 12 if training else 14)
        if grown is None:
            continue
        mask, rects = grown
        if max(mask.shape) > 16:
            continue
        # Distinct artificial type labels make a conservative geometry screen.
        markers = [np.full(d, 10+k, int) for k, d in enumerate(dimensions)]
        try:
            geometric, _ = _covers(mask, markers, quotas, limit=3500)
        except _SearchLimit:
            continue
        if len(geometric) != 1:
            continue
        palette = [c for c in range(10) if c not in (bg, wall)]
        rng.shuffle(palette)
        palette = palette[:int(rng.integers(2, 7))]
        patterns = [_pattern(rng, d, palette, asymmetric and k == 0)
                    for k, d in enumerate(dimensions)]
        try:
            # Every pair has an observable counterexample to unlimited copying.
            unlimited, _ = _covers(mask, patterns, [int(mask.sum())//p.size for p in patterns],
                                   exact=False, limit=3500)
        except _SearchLimit:
            continue
        if len(unlimited) != 2:
            continue
        board = np.full((mask.shape[0]+4, mask.shape[1]+4), wall, np.int8)
        view = board[2:-2, 2:-2]
        view[mask] = bg
        output = board.copy()
        for k, y, x, h, w in rects:
            output[y+2:y+2+h, x+2:x+2+w] = patterns[k]
        panels = [board] + [_frame_tile(rng, c, p, bg) for c, p in zip(colors, patterns)]
        objects = []
        for k, (color, n) in enumerate(zip(colors, quotas)):
            areas = forced_areas.get(k)
            for j in range(n):
                area = areas[j] if areas is not None else int(rng.integers(1, 10))
                objects.append((color, _shape(rng, area)))
        for color in distractors:
            objects.append((color, _shape(rng, int(rng.integers(1, 10)))))
        arranged = _scatter(rng, panels, objects, bg, training)
        if arranged is None:
            continue
        inp, _ = arranged
        inp, out = np.rot90(inp, rotation).copy(), np.rot90(output, rotation).copy()
        try:
            actual = _answer(_inspect(inp), limit=6000)[0]
        except (ValueError, _SearchLimit):
            continue
        if np.array_equal(actual, out):
            return {'input': inp, 'output': out}
    raise RuntimeError("Bounded geometry/scatter sampling exhausted")


def _integer(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer('num_train', num_train, 2, 10)
    nq = _integer('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    order = list(map(int, rng.permutation(10)))
    anchor, novel = order[:2]
    collision = int(rng.integers(4, 9))
    cut = int(rng.integers(1, collision))
    rotation = int(rng.integers(4))
    extrapolate = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    first_second = None
    for i in range(nt+nq):
        training = i < nt
        if i == 0:
            n, quotas = 2, [1, 1]
        elif i == 1:
            n, quotas = 3, [2, 1, int(rng.integers(2))]
        else:
            n = int(rng.integers(2, 4)) if training else int(rng.integers(3, 5))
            quotas = [int(rng.integers(1, 3 if training or not extrapolate else 4)),
                      int(rng.integers(1, 3))]
            quotas += [int(rng.integers(3)) for _ in range(n-2)]
            while sum(quotas) > 8:
                quotas[-1] -= 1
        exclude = {anchor, novel} if i < 2 else {novel} if training else {novel}
        if i == 1:
            exclude.add(first_second)
        bg, wall = map(int, rng.choice([c for c in range(10) if c not in exclude], 2, replace=False))
        pool = [c for c in range(10) if c not in (bg, wall, novel) and not (i == 1 and c == first_second)]
        if i < 2:
            pool.remove(anchor)
            colors = [anchor] + list(map(int, rng.choice(pool, n-1, replace=False)))
        else:
            colors = list(map(int, rng.choice(pool, n, replace=False)))
            if not training:
                colors[0] = novel
        if i == 0:
            first_second = colors[1]
        dpool = [c for c in range(10) if c not in (bg, wall) and c not in colors]
        distractors = list(map(int, rng.choice(dpool, int(rng.integers(4)), replace=True)))
        if i == 0:
            distractors = [novel] + distractors[:2]
        if i == 1:
            distractors = [first_second] + distractors[:2]
        forced = {0: [collision], 1: [int(rng.choice([a for a in range(1, 10) if a != collision]))]} if i == 0 else {
            0: [cut, collision-cut]} if i == 1 else {}
        turn = rotation if i == 0 else (rotation+1) % 4 if i == 1 else int(rng.integers(4))
        for _ in range(8):
            pair = _make_pair(rng, colors, quotas, bg, wall, distractors, forced,
                              training, turn, asymmetric=i < 2)
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError("Distinct-input sampling exhausted")
        seen.add(_key(pair['input']))
        puzzle['train' if training else 'test'].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, details = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            d = _inspect(inp)
            expected, path, stats = _answer(d)
            assert np.array_equal(expected, out), "Incorrect exact cover or pattern transfer"
            assert Counter(k for k, pos in path) == Counter({k: n for k, n in enumerate(d['quotas']) if n})
            assert out.shape == d['board'].shape
            assert np.array_equal(out[~d['mask']], d['board'][~d['mask']]), "Board cells changed"
            assert not np.any(out[d['mask']] == d['background']), "Unfilled vacancy"
            patterns = [t['pattern'] for t in d['templates']]
            alternatives, _ = _covers(d['mask'], patterns,
                                       [int(d['mask'].sum())//p.size for p in patterns], exact=False)
            assert len(alternatives) == 2, "Counts did not resolve a geometric alternative"
            d.update(path=path, stats=stats)
            details[split].append(d)
    examples = [(t['color'], n, d['areas'][t['color']]) for d in details['train']
                for t, n in zip(d['templates'], d['quotas'])]
    assert any(c == cc and n != nn and a == aa for c, n, a in examples for cc, nn, aa in examples), "Area/count collision missing"
    assert any(n == nn and a != aa for c, n, a in examples for cc, nn, aa in examples), "Area independence missing"
    # Presence/absence alone is also insufficient: positive categories may
    # admit additional copies, giving a different pixel completion.
    presence_witness = False
    for d in details['train']:
        patterns = [t['pattern'] for t in d['templates']]
        capacities = [int(d['mask'].sum())//p.size if n else 0
                      for p, n in zip(patterns, d['quotas'])]
        alternatives, _ = _covers(d['mask'], patterns, capacities, exact=False)
        presence_witness |= len(alternatives) == 2
    assert presence_witness, "Training only teaches category presence, not multiplicity"
    counted = {t['color'] for d in details['train'] for t in d['templates']}
    discarded = {o['color'] for d in details['train'] for o in d['objects']
                 if o['color'] not in {t['color'] for t in d['templates']}}
    assert counted & discarded, "Local category/distractor role switch missing"
    assert any(t['color'] not in counted for d in details['test'] for t in d['templates']), "New test category missing"
    assert any(not np.array_equal(t['pattern'], t['pattern'][::-1, ::-1]) and n
               for d in details['train'] for t, n in zip(d['templates'], d['quotas'])), "Asymmetric transfer evidence missing"
    return None


def _metrics(puzzle):
    result = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            d = _inspect(pair['input'])
            out, path, stats = _answer(d)
            masks = []
            for obj in d['objects']:
                y, x, _, _ = _box(obj['cells'])
                masks.append(tuple(sorted((r-y, c-x) for r, c in obj['cells'])))
            comps = _components(d['mask'])
            irregular = sum(len(c) != (_box(c)[2]-_box(c)[0])*(_box(c)[3]-_box(c)[1]) for c in comps)
            result.append(dict(split=split, shape=list(d['grid'].shape), output_shape=list(out.shape),
                               background=d['background'], wall=d['wall'], categories=len(d['templates']),
                               quotas=list(d['quotas']), dimensions=[list(t['pattern'].shape) for t in d['templates']],
                               copies=sum(d['quotas']), components=len(comps), irregular=irregular,
                               areas=[len(o['cells']) for o in d['objects']], object_masks=masks,
                               nodes=stats['nodes'], failures=stats['failures']))
    return result


def self_test():
    """Bounded 42-seed semantic/diversity audit, plus small adversarial cases."""
    signatures, records = set(), []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_key(pair['input']) for s in ('train', 'test') for pair in p[s]))
        records.extend(_metrics(p))
    assert len(signatures) == 42
    assert len({tuple(r['shape']) for r in records}) >= 20
    assert len({tuple(d) for r in records for d in r['dimensions']}) >= 15
    assert {r['background'] for r in records} == set(range(10))
    assert {r['categories'] for r in records} == {2, 3, 4}
    assert {q for r in records for q in r['quotas']} >= {0, 1, 2, 3}
    assert len({m for r in records for m in r['object_masks']}) >= 60
    assert any(r['copies'] >= 6 for r in records)
    assert any(r['components'] > 1 for r in records)
    assert sum(r['irregular'] > 0 for r in records) >= 100
    # Quotas alter the answer with stencil and prototypes held fixed.
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], bool)
    patterns = [np.array([[2, 3]]), np.array([[4, 4, 5, 4]])]
    a, _ = _covers(mask, patterns, [3, 0])
    b, _ = _covers(mask, patterns, [1, 1])
    assert len(a) == len(b) == 1 and a[0][0] != b[0][0]
    assert not _covers(mask, patterns, [2, 0])[0]
    assert len(_covers(mask, patterns, [3, 1], exact=False)[0]) == 2
    c, _ = _covers(mask[::-1], patterns, [1, 1])
    assert len(c) == 1 and b[0][0] != c[0][0]  # Geometry matters at fixed counts.
    # Complete pattern copying rather than monochrome category painting.
    p = generate_puzzle(107, 2, 1)
    repeat = generate_puzzle(107, 2, 1)
    assert tuple((_key(x['input']), _key(x['output'])) for x in p['train']+p['test']) == tuple(
        (_key(x['input']), _key(x['output'])) for x in repeat['train']+repeat['test'])
    bad = {'train': p['train'], 'test': [dict(p['test'][0])]}
    bad['test'][0]['output'] = np.full_like(bad['test'][0]['output'], 0)
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Corrupted completion accepted")
    for nt, nq in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    return dict(seeds=42, pairs=len(records), distinct_puzzles=42,
                input_shapes=len({tuple(r['shape']) for r in records}),
                tile_dimensions=len({tuple(d) for r in records for d in r['dimensions']}),
                voucher_shapes=len({m for r in records for m in r['object_masks']}),
                irregular_pairs=sum(r['irregular'] > 0 for r in records))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
