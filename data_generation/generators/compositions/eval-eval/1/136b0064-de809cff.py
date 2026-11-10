"""Palette-filtered hole seeking: damaged rectangles are a turtle's address space.

No file access, parent imports, stored examples, or training-dependent solver.
"""
from itertools import permutations

import numpy as np


# Filled triangular arrowheads. Geometry, not ink color, gives direction.
_ARROWS = {
    (-1, 0): np.array([[0, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=bool),
    (1, 0): np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    (0, -1): np.array([[0, 1, 1], [1, 1, 1], [0, 1, 1]], dtype=bool),
    (0, 1): np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool),
}


def _integer(name, value, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _runs(mask):
    answer = []
    for i in np.flatnonzero(mask):
        i = int(i)
        if not answer or answer[-1][1] != i:
            answer.append([i, i + 1])
        else:
            answer[-1][1] += 1
    return answer


def _components(a, color):
    cells = set(map(tuple, np.argwhere(a == color).tolist()))
    answer = []
    while cells:
        p = min(cells)
        cells.remove(p)
        stack, group = [p], [p]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in cells:
                    cells.remove(q)
                    stack.append(q)
                    group.append(q)
        answer.append(group)
    return answer


def _boxes(a, fills):
    boxes = []
    for color in fills:
        for group in _components(a, color):
            if len(group) < 18:
                continue
            rr, cc = zip(*group)
            box = (min(rr), min(cc), max(rr) + 1, max(cc) + 1, color)
            assert min(box[2] - box[0], box[3] - box[1]) >= 5
            boxes.append(box)
    assert 2 <= len(boxes) <= 3
    assert set(b[4] for b in boxes) == set(fills)
    return tuple(boxes)


def _paint(shape, bg, boxes):
    a = np.full(shape, bg, dtype=np.int8)
    for r0, c0, r1, c1, color in boxes:
        a[r0:r1, c0:c1] = color
    return a


def _recover(pane, bg, fills, start):
    """Exhaust all covering orders of input-derived dense component boxes.

    Family constraints preserve every visible rectangle's bounding extent;
    small wrong-color components and detached exterior speckles are ignored.
    A different best visible scene or a one-error runner-up is rejected.
    """
    boxes = _boxes(pane, fills)
    edges = [(i, j) for i in range(len(boxes)) for j in range(i) if _overlap(boxes[i], boxes[j])]
    degrees = [sum(i in edge for edge in edges) for i in range(len(boxes))]
    assert len(edges) == len(boxes) - 1 and sorted(degrees) == [1, 1] + [2] * (len(boxes) - 2)
    assert all(boxes[i][4] != boxes[j][4] for i, j in edges), 'Fill-alternating spatial chain required'
    supported = {}
    observed = pane.copy()
    observed[start] = bg
    for order in permutations(boxes):
        clean = _paint(pane.shape, bg, order)
        good = True
        for r0, c0, r1, c1, color in order:
            visible = clean[r0:r1, c0:c1] == color
            correct = pane[r0:r1, c0:c1] == color
            sides = (visible[0], visible[-1], visible[:, 0], visible[:, -1])
            if visible.sum() < 25 or min(s.sum() for s in sides) < 2:
                good = False
            if np.count_nonzero(correct & visible) < .78 * visible.sum():
                good = False
        if not good:
            continue
        compare = clean != observed
        compare[start] = False
        loss = int(compare.sum())
        supported[clean.tobytes()] = (loss, clean, order)
    assert supported, "Insufficient rectangle evidence"
    ranked = sorted(supported.values(), key=lambda item: (item[0], item[1].tobytes()))
    loss, clean, order = ranked[0]
    assert loss <= max(10, .16 * pane.size), "Damage too dense"
    assert all(x[0] >= loss + 2 for x in ranked[1:]), "Ambiguous covering order"
    holes = set(map(tuple, np.argwhere((pane == bg) & (clean != bg)).tolist()))
    assert holes, "No erased addresses"
    return clean, holes, order


def _read_input(input_grid):
    a = _grid(input_grid)
    candidates = []
    for sep in range(3, a.shape[1] - 8):
        sc = int(a[0, sep])
        if not np.all(a[:, sep] == sc) or np.count_nonzero(a == sc) != a.shape[0]:
            continue
        panel, pane = a[:, :sep], a[:, sep + 1:]
        values, counts = np.unique(panel, return_counts=True)
        bg = int(values[np.argmax(counts)])
        if bg == sc:
            continue
        try:
            rr = _runs(np.any(panel != bg, axis=1))
            cc = _runs(np.any(panel != bg, axis=0))
            assert 1 <= len(rr) <= 6 and 1 <= len(cc) <= 3
            assert rr[0][0] == cc[0][0] == 0 and cc[-1][1] == sep
            assert all(y - x == 3 for x, y in rr + cc)
            assert all(1 <= b[0] - x[1] <= 2 for x, b in zip(rr, rr[1:]))
            assert all(1 <= b[0] - x[1] <= 2 for x, b in zip(cc, cc[1:]))
            pv, pc = np.unique(pane, return_counts=True)
            marks = [int(v) for v, n in zip(pv, pc) if v != bg and n == 1]
            assert len(marks) == 1
            highlight = marks[0]
            fills = tuple(int(v) for v in pv if v not in (bg, highlight))
            assert len(fills) == 2 and sc not in pv
            start = tuple(map(int, np.argwhere(pane == highlight)[0]))
            tokens, expected_panel = [], np.full(panel.shape, bg, dtype=np.int8)
            for left, right in cc:
                for top, bottom in rr:
                    block = panel[top:bottom, left:right]
                    ink = np.unique(block[block != bg])
                    assert len(ink) == 1 and int(ink[0]) in fills
                    ds = [d for d, m in _ARROWS.items() if np.array_equal(block != bg, m)]
                    assert len(ds) == 1
                    tokens.append((ds[0], int(ink[0])))
                    target = expected_panel[top:bottom, left:right]
                    target[_ARROWS[ds[0]]] = int(ink[0])
            assert np.array_equal(panel, expected_panel)
            clean, holes, boxes = _recover(pane, bg, fills, start)
            candidates.append(dict(pane=pane.copy(), clean=clean, holes=holes,
                                   boxes=boxes, tokens=tokens, rows=len(rr), lanes=len(cc),
                                   bg=bg, fills=fills, highlight=highlight,
                                   separator=sc, start=start))
        except (AssertionError, IndexError):
            continue
    assert len(candidates) == 1, "Absent or ambiguous program/terrain structure"
    return candidates[0]


def _ray(start, direction, shape):
    r, c = start
    dr, dc = direction
    r, c = r + dr, c + dc
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        yield r, c
        r, c = r + dr, c + dc


def _destination(clean, holes, start, direction, color, ignore_color=False):
    for p in _ray(start, direction, clean.shape):
        if p in holes and (ignore_color or int(clean[p]) == color):
            return p
    raise ValueError("Instruction has no matching erased destination")


def _execute(info, tokens=None, ignore_color=False, complement=True, rings=True):
    clean, holes = info['clean'], info['holes']
    out = clean.copy()
    cursor, visited, trace = info['start'], {info['start']}, []
    for direction, color in info['tokens'] if tokens is None else tokens:
        dest = _destination(clean, holes, cursor, direction, color, ignore_color)
        used_color = int(clean[dest])
        paint = next(c for c in info['fills'] if c != used_color) if complement else used_color
        path = []
        for p in _ray(cursor, direction, clean.shape):
            out[p] = paint
            path.append(p)
            if p == dest:
                break
        if rings:
            r, c = dest
            out[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = paint
        trace.append((cursor, direction, color, dest, tuple(path)))
        visited.add(dest)
        cursor = dest
    # All reached centers, including revisited ones, override every stroke/ring.
    for p in visited:
        out[p] = info['highlight']
    return out, trace, visited


def solve(input_grid: np.ndarray) -> np.ndarray:
    """The fixed family convention and visibly encoded palette suffice."""
    return _execute(_read_input(input_grid))[0]


def _overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _scene(rng, shape, count):
    h, w = shape
    for _ in range(150):
        boxes = []
        for index in range(count):
            for _ in range(80):
                rh = int(rng.integers(6, min(h - 1, max(8, h * 4 // 5)) + 1))
                rw = int(rng.integers(6, min(w - 1, max(8, w * 4 // 5)) + 1))
                r = int(rng.integers(h - rh + 1))
                c = int(rng.integers(w - rw + 1))
                box = (r, c, r + rh, c + rw, 1 + index % 2)
                if boxes:
                    overlap = _overlap(box, boxes[-1])
                    if not 4 <= overlap <= .45 * min(rh * rw, (boxes[-1][2] - boxes[-1][0]) * (boxes[-1][3] - boxes[-1][1])):
                        continue
                    if any(_overlap(box, b) for b in boxes[:-1]):
                        continue
                boxes.append(box)
                break
            else:
                break
        if len(boxes) != count:
            continue
        ordered = tuple(boxes[int(i)] for i in rng.permutation(count))
        clean = _paint(shape, 0, ordered)
        try:
            observed_boxes = _boxes(clean, (1, 2))
            if set(observed_boxes) != set(boxes):
                continue
            # The same input-only evidence conditions used after corruption.
            trial = clean.copy()
            start = tuple(map(int, np.argwhere(clean == 0)[0]))
            trial[start] = 3
            p = tuple(map(int, np.argwhere(clean == 1)[len(np.argwhere(clean == 1)) // 2]))
            trial[p] = 0
            recovered, _, _ = _recover(trial, 0, (1, 2), start)
            if np.array_equal(recovered, clean):
                return clean, boxes
        except (AssertionError, IndexError):
            pass
    return None


def _spaced_coordinates(rng, size, count):
    # Irregular spacings, with a two-cell minimum; allows shared marker rings.
    result = []
    for i in rng.permutation(size):
        i = int(i)
        if all(abs(i - j) >= 2 for j in result):
            result.append(i)
            if len(result) == count:
                break
    return sorted(result)


def _hole_field(rng, clean, length):
    rr = _spaced_coordinates(rng, clean.shape[0], int(rng.integers(4, 7)))
    cc = _spaced_coordinates(rng, clean.shape[1], int(rng.integers(4, 7)))
    available = [(r, c) for r in rr for c in cc if clean[r, c] != 0]
    if len(available) < max(8, length + 2):
        return None
    rng.shuffle(available)
    count = min(len(available), int(rng.integers(max(8, length + 2), 20)))
    holes = set(available[:count])
    if len({int(clean[p]) for p in holes}) != 2:
        return None
    return holes


def _plan(rng, clean, holes, length, teach):
    starts = [(r, c) for r in range(clean.shape[0]) for c in range(clean.shape[1])
              if (r, c) not in holes and any(p[0] == r or p[1] == c for p in holes)]
    for _ in range(100):
        start = starts[int(rng.integers(len(starts)))]
        cursor, tokens, targets, seen_dirs = start, [], set(), set()
        for i in range(length):
            options = []
            for d in _ARROWS:
                for color in (1, 2):
                    try:
                        p = _destination(clean, holes, cursor, d, color)
                    except ValueError:
                        continue
                    if p not in targets and max(abs(p[0] - cursor[0]), abs(p[1] - cursor[1])) >= 2:
                        options.append((d, color, p))
            if not options:
                break
            if teach:
                missing = [option for option in options if option[0] not in seen_dirs]
                if missing:
                    options = missing
            d, color, p = options[int(rng.integers(len(options)))]
            tokens.append((d, color))
            targets.add(p)
            seen_dirs.add(d)
            cursor = p
        if len(tokens) == length and (not teach or len(seen_dirs) == 4):
            return start, tokens
    return None


def _row_major(info):
    return [info['tokens'][c * info['rows'] + r]
            for r in range(info['rows']) for c in range(info['lanes'])]


def _alternative_differs(info, **kwargs):
    try:
        return not np.array_equal(_execute(info, **kwargs)[0], _execute(info)[0])
    except ValueError:
        return True


def _filter_counterexample(info):
    # Require a complete wrong answer, not merely a failing alternative route.
    try:
        return not np.array_equal(_execute(info, ignore_color=True)[0], _execute(info)[0])
    except ValueError:
        return False


def _features(info):
    out, trace, visited = _execute(info)
    skipped, farther = False, False
    for start, direction, color, dest, path in trace:
        skipped |= any(p in info['holes'] and info['clean'][p] != color for p in path[:-1])
        beyond = False
        for p in _ray(dest, direction, info['clean'].shape):
            beyond |= p in info['holes'] and info['clean'][p] == color
        farther |= beyond
    # Detect a contested non-center cell, demonstrating temporal overpainting.
    demands, conflicts = {}, 0
    for _, _, color, dest, path in trace:
        paint = next(c for c in info['fills'] if c != color)
        r, c = dest
        footprint = set(path) | {(y, x) for y in range(max(0, r - 1), min(out.shape[0], r + 2))
                                 for x in range(max(0, c - 1), min(out.shape[1], c + 2))}
        for p in footprint - visited:
            conflicts += p in demands and demands[p] != paint
            demands[p] = paint
    return dict(skipped=skipped, farther=farther, conflicts=conflicts,
                visited=len(visited) - 1, unused=len(info['holes'] - visited),
                colors={color for d, color in info['tokens']},
                directions={d for d, color in info['tokens']})


def _first_paint_wins(info):
    """A deliberately wrong temporal rule for demonstration adequacy checks."""
    _, trace, visited = _execute(info)
    out = info['clean'].copy()
    touched = set()
    for _, _, color, dest, path in trace:
        paint = next(c for c in info['fills'] if c != color)
        r, c = dest
        footprint = set(path) | {(y, x) for y in range(max(0, r - 1), min(out.shape[0], r + 2))
                                 for x in range(max(0, c - 1), min(out.shape[1], c + 2))}
        for p in footprint - touched:
            out[p] = paint
        touched.update(footprint)
    for p in visited:
        out[p] = info['highlight']
    return out


def _compose(clean, holes, start, tokens, rows, lanes, gap, palette, sep_color, highlight):
    bg, f1, f2 = palette
    h, w = clean.shape
    panel_w = lanes * 3 + (lanes - 1) * gap
    a = np.full((h, panel_w + 1 + w), bg, dtype=np.int8)
    a[:, panel_w] = sep_color
    pane = np.array(palette, dtype=np.int8)[clean]
    for p in holes:
        pane[p] = bg
    pane[start] = highlight
    a[:, panel_w + 1:] = pane
    for i, (direction, color) in enumerate(tokens):
        lane, band = divmod(i, rows)
        block = a[band * (3 + gap):band * (3 + gap) + 3,
                  lane * (3 + gap):lane * (3 + gap) + 3]
        block[_ARROWS[direction]] = palette[color]
    return a


def _pair(rng, length, layout, palette, separator, highlight, teach):
    rows, lanes, gap = layout
    panel_w = lanes * 3 + (lanes - 1) * gap
    for _ in range(240):
        h = int(rng.integers(max(18, rows * (3 + gap) - gap), 31))
        w = int(rng.integers(14, 30 - panel_w))
        count = int(rng.choice([2, 3], p=[.7, .3]))
        scene = _scene(rng, (h, w), count)
        if scene is None:
            continue
        clean, boxes = scene
        for _ in range(12):
            holes = _hole_field(rng, clean, length)
            if holes is None:
                continue
            plan = _plan(rng, clean, holes, length, teach)
            if plan is None:
                continue
            start, tokens = plan
            a = _compose(clean, holes, start, tokens, rows, lanes, gap,
                         palette, separator, highlight)
            try:
                info = _read_input(a)
                if not np.array_equal(info['clean'], np.array(palette)[clean]):
                    continue
                f = _features(info)
                if f['unused'] < 2 or len(f['colors']) != 2:
                    continue
                if teach and (not f['skipped'] or not f['farther'] or f['conflicts'] == 0):
                    continue
                if teach and not _alternative_differs(info, tokens=_row_major(info)):
                    continue
                if teach and np.array_equal(_first_paint_wins(info), _execute(info)[0]):
                    continue
                if teach and not _filter_counterexample(info):
                    continue
            except AssertionError:
                continue
            # Wrong-fill interior errors and detached exterior speckles are
            # ordinary damage, never ray targets. Accept only exact recovery.
            pane_offset = panel_w + 1
            points = [(r, c) for r in range(h) for c in range(w) if (r, c) not in holes and (r, c) != start]
            rng.shuffle(points)
            wrong, speckles = 0, 0
            wrong_goal, speckle_goal = int(rng.integers(0, 5)), int(rng.integers(1, 6))
            for p in points:
                r, c = p
                neighborhood = clean[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
                if clean[p] != 0 and wrong < wrong_goal and np.all(neighborhood == clean[p]):
                    replacement = palette[3 - int(clean[p])]
                    kind = 'wrong'
                elif clean[p] == 0 and speckles < speckle_goal and np.all(neighborhood == 0):
                    replacement = palette[int(rng.integers(1, 3))]
                    kind = 'speckle'
                else:
                    continue
                trial = a.copy()
                trial[r, pane_offset + c] = replacement
                try:
                    ti = _read_input(trial)
                    if not np.array_equal(ti['clean'], info['clean']) or ti['holes'] != info['holes']:
                        continue
                except AssertionError:
                    continue
                a = trial
                wrong += kind == 'wrong'
                speckles += kind == 'speckle'
                if wrong == wrong_goal and speckles == speckle_goal:
                    break
            if teach and speckles == 0:
                continue
            return {'input': a, 'output': solve(a)}
    raise RuntimeError("Bounded construction exhausted; no ambiguous fallback is emitted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer('seed', seed, 0)
    nt = _integer('num_train', num_train, 2, 10)
    nq = _integer('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    separator, highlight = map(int, rng.choice(10, 2, replace=False))
    available = [c for c in range(10) if c not in (separator, highlight)]
    base = tuple(map(int, rng.choice(available, 3, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, total in (('train', nt), ('test', nq)):
        for i in range(total):
            teach = split == 'train' and i < 2
            length = 4 if teach else int(rng.integers(2, 5)) if split == 'train' else int(rng.choice([6, 8, 9]))
            layouts = [(length // lanes, lanes, gap) for lanes in (1, 2, 3) for gap in (1, 2)
                       if length % lanes == 0 and length // lanes <= 6
                       and (length // lanes) * (3 + gap) - gap <= 30]
            if teach:
                layouts = [(2, 2, 1), (2, 2, 2)]
            layout = layouts[int(rng.integers(len(layouts)))]
            palette = base if rng.random() < .35 else tuple(map(int, rng.choice(available, 3, replace=False)))
            for _ in range(6):
                pair = _pair(rng, length, layout, palette, separator, highlight, teach)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('Could not construct distinct examples')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), {'train': [], 'test': []}
    for split in analyses:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _read_input(inp)
            expected, trace, visited = _execute(info)
            assert out.shape == info['pane'].shape and np.array_equal(expected, out)
            assert all(out[p] == info['highlight'] for p in visited)
            assert info['holes'] - visited, 'Need unvisited holes to show selection'
            assert all(dest in info['holes'] and info['clean'][dest] == color
                       for start, d, color, dest, path in trace)
            analyses[split].append(info)
    all_info = analyses['train'] + analyses['test']
    assert len({(i['separator'], i['highlight']) for i in all_info}) == 1
    assert min(len(i['tokens']) for i in analyses['test']) > max(len(i['tokens']) for i in analyses['train'])
    for info in analyses['train'][:2]:
        f = _features(info)
        assert f['directions'] == set(_ARROWS) and f['colors'] == set(info['fills'])
        assert f['skipped'] and f['farther'] and f['conflicts'] > 0
        assert _alternative_differs(info, ignore_color=True), 'Fill filter not observable'
        assert _filter_counterexample(info), 'Need a complete contrasting unfiltered answer'
        assert _alternative_differs(info, complement=False), 'Complement not observable'
        assert _alternative_differs(info, rings=False), 'Rings not observable'
        assert _alternative_differs(info, tokens=_row_major(info)), 'Traversal not observable'
        assert not np.array_equal(_first_paint_wins(info), _execute(info)[0]), 'Temporal priority not observable'


def self_test():
    """Bounded diversity audit plus mutations and explicit local semantics."""
    shapes, geometries, programs, palettes, holes, lanes, box_counts = set(), set(), set(), set(), set(), set(), set()
    puzzles, semantic_programs = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        for pair in puzzle['train'] + puzzle['test']:
            info = _read_input(pair['input'])
            shapes.add(pair['input'].shape)
            geometries.add((info['pane'].shape, tuple(sorted(b[:4] for b in info['boxes']))))
            programs.add(tuple(info['tokens']))
            semantic_programs.add(tuple((d, info['fills'].index(c)) for d, c in info['tokens']))
            palettes.add((info['bg'], info['fills'], info['highlight']))
            holes.add(len(info['holes']))
            lanes.add(info['lanes'])
            box_counts.add(len(info['boxes']))
    assert len(shapes) >= 35 and len(geometries) >= 150 and len(programs) >= 130
    assert len(palettes) >= 60 and len(holes) >= 8
    assert len(semantic_programs) >= 90, 'Programs must vary beyond literal ink colors'
    assert lanes == {1, 2, 3} and box_counts == {2, 3}
    p0, p1 = puzzles[0], generate_puzzle(0)
    assert all(np.array_equal(p['input'], q['input']) and np.array_equal(p['output'], q['output'])
               for split in p0 for p, q in zip(p0[split], p1[split]))
    # Analytic ray/paint fixture: a wrong-fill nearer hole must be skipped.
    clean = np.ones((7, 9), dtype=np.int8)
    clean[:, 4:] = 2
    info = dict(clean=clean, holes={(3, 2), (3, 5), (3, 7)}, start=(3, 0),
                fills=(1, 2), highlight=8, tokens=[((0, 1), 2)])
    out, trace, visited = _execute(info)
    assert trace[0][3] == (3, 5) and visited == {(3, 0), (3, 5)}
    assert out[3, 0] == out[3, 5] == 8 and out[2, 4] == 1 and out[3, 7] == 2
    # Clipping, repeat visits, and later-command priority are explicit rules.
    info.update(holes={(0, 2), (0, 5)}, start=(0, 0),
                tokens=[((0, 1), 2), ((0, -1), 1), ((0, 1), 2)])
    out, trace, visited = _execute(info)
    assert out.shape == clean.shape and out[0, 2] == out[0, 5] == 8
    assert out[1, 3] == 2 and out[0, 3] == 1
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(42 + nt * 10 + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal count accepted')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in p0[s]] for s in p0}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Mutated label accepted')
