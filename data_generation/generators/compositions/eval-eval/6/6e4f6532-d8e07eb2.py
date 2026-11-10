"""Collective docking: a compass resolves token identity; complete lines move.

Only standard-library modules and NumPy are used. No parent modules are imported.
The parser and solver consume pixels only, including all palette conventions.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _check_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, component = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    todo.append(p)
        result.append(component)
    return result


def _transform(cells, t):
    transformed = []
    for (r, c), value in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        transformed.append((r, c, value))
    r0 = min(r for r, _, _ in transformed)
    c0 = min(c for _, c, _ in transformed)
    return {(r - r0, c - c0): v for r, c, v in transformed}


def _shift(cells, dr, dc):
    return {(r + dr, c + dc): v for (r, c), v in cells.items()}


def _token_key(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0, int(v)) for (r, c), v in cells.items()))


def _facing(cells, compass):
    core = [p for p, v in cells.items() if v not in compass]
    assert core
    r0, r1 = min(r for r, _ in core), max(r for r, _ in core)
    c0, c1 = min(c for _, c in core), max(c for _, c in core)
    faces = {}
    for (r, c), v in cells.items():
        if v in compass:
            dirs = [d for d, ok in enumerate((r < r0, c > c1, r > r1, c < c0)) if ok]
            assert len(dirs) == 1, "each compass label must project on one side"
            faces[(r, c)] = dirs[0]
    assert len(set(faces.values())) >= 2
    return faces


def _line_members(positions, rows, cols):
    selected = set(positions)
    lines = []
    for r in range(rows):
        line = {(r, c) for c in range(cols)}
        if line <= selected:
            lines.append(line)
    for c in range(cols):
        line = {(r, c) for r in range(rows)}
        if line <= selected:
            lines.append(line)
    return set().union(*lines) if lines else set(), lines


def _parse(input_grid):
    grid = _check_grid(input_grid)
    h, w = grid.shape
    bg = int(grid[0, 0])
    assert min(h, w) >= 8
    assert all(int(grid[p]) == bg for p in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    bands = (grid[0, 1:-1], grid[1:-1, -1], grid[-1, 1:-1], grid[1:-1, 0])
    compass = tuple(int(b[0]) for b in bands)
    assert len(set(compass) | {bg}) == 5
    assert all(np.all(b == color) for b, color in zip(bands, compass))
    # Detect the one monochrome, connected rectangular lattice. It has at
    # least two cells per axis and 5..7-cell square interiors.
    candidates = []
    for color in sorted(set(map(int, grid[1:-1, 1:-1].ravel())) - set(compass) - {bg}):
        points = {(int(r + 1), int(c + 1)) for r, c in np.argwhere(grid[1:-1, 1:-1] == color)}
        for comp in _components(points):
            r0, r1 = min(r for r, _ in comp), max(r for r, _ in comp)
            c0, c1 = min(c for _, c in comp), max(c for _, c in comp)
            block = grid[r0:r1 + 1, c0:c1 + 1]
            rr = [i for i in range(len(block)) if np.all(block[i] == color)]
            cc = [j for j in range(block.shape[1]) if np.all(block[:, j] == color)]
            if not (3 <= len(rr) <= 5 and 3 <= len(cc) <= 5
                    and rr[0] == cc[0] == 0 and rr[-1] == len(block) - 1
                    and cc[-1] == block.shape[1] - 1):
                continue
            steps = set(np.diff(rr)) | set(np.diff(cc))
            if len(steps) != 1 or next(iter(steps)) not in (6, 7, 8):
                continue
            s = int(next(iter(steps)) - 1)
            expected = {(r0 + r, c0 + c) for r in range(len(block)) for c in range(block.shape[1])
                        if r in rr or c in cc}
            if expected == comp:
                candidates.append((color, r0, c0, len(rr) - 1, len(cc) - 1, s))
    assert len(candidates) == 1, "board lattice must be uniquely visible"
    divider, r0, c0, rows, cols, s = candidates[0]
    board_h, board_w = rows * (s + 1) + 1, cols * (s + 1) + 1
    board_mask = np.zeros(grid.shape, bool)
    board_mask[r0:r0 + board_h, c0:c0 + board_w] = True
    tokens, slots, lookup = {}, {}, {}
    for r, c in itertools.product(range(rows), range(cols)):
        sr, sc = r0 + 1 + r * (s + 1), c0 + 1 + c * (s + 1)
        block = grid[sr:sr + s, sc:sc + s]
        assert np.all(block[[0, -1]] == bg) and np.all(block[:, [0, -1]] == bg)
        marks = {(sr + int(i), sc + int(j)): int(block[i, j]) for i, j in np.argwhere(block != bg)}
        assert marks and len(set(marks.values())) == 1
        assert not (set(marks.values()) & (set(compass) | {divider}))
        key = _token_key(marks)
        assert key not in lookup, "colored registration tokens must be unique"
        lookup[key] = (r, c)
        tokens[(r, c)], slots[(r, c)] = marks, (sr, sc)
    inks = set(v for marks in tokens.values() for v in marks.values())
    assert 1 <= len(inks) <= 3
    outside = {(int(r), int(c)) for r, c in np.argwhere((grid != bg) & ~board_mask)
               if 0 < r < h - 1 and 0 < c < w - 1}
    sources = [{p: int(grid[p]) for p in comp} for comp in _components(outside)]
    assert sources
    bodies = set(v for source in sources for v in source.values()) - inks - set(compass)
    assert len(bodies) == 1 and divider not in bodies
    body = next(iter(bodies))
    placements, assignments, poses, canonicals = [], [], [], []
    for source in sources:
        assert body in source.values()
        assert len(set(source.values()) & inks) == 1
        label_colors = set(source.values()) & set(compass)
        assert len(label_colors) >= 2
        assert any({compass[d], compass[(d + 1) % 4]} <= label_colors for d in range(4))
        oriented = {}
        for t in range(8):
            cells = _transform(source, t)
            faces = _facing(cells, compass)
            if all(cells[p] == compass[d] for p, d in faces.items()):
                oriented[tuple(sorted(cells.items()))] = (cells, t)
        assert len(oriented) == 1, "compass must determine one complete colored piece"
        cells, t = next(iter(oriented.values()))
        marks = {p: v for p, v in cells.items() if v in inks}
        key = _token_key(marks)
        assert key in lookup, "oriented registration must have an exact board copy"
        target = lookup[key]
        tr, tc = min(tokens[target])
        mr, mc = min(marks)
        placed = _shift(cells, tr - mr, tc - mc)
        assert {p: v for p, v in placed.items() if v in inks} == tokens[target]
        sr, sc = slots[target]
        assert all(sr <= r < sr + s and sc <= c < sc + s for r, c in placed), "docked body must fit its slot"
        placements.append(placed)
        assignments.append(target)
        poses.append(t)
        canonicals.append(cells)
    assert len(set(assignments)) == len(assignments), "query registrations cannot repeat"
    eligible, lines = _line_members(assignments, rows, cols)
    return dict(grid=grid, bg=bg, compass=compass, divider=divider, body=body, inks=inks,
                rows=rows, cols=cols, s=s, board_mask=board_mask, tokens=tokens, slots=slots,
                sources=sources, placements=placements, assignments=assignments,
                poses=poses, canonicals=canonicals, eligible=eligible, lines=lines)


def _render_solution(info, ignore_completion=False):
    out = info["grid"].copy()
    movers = set(info["assignments"]) if ignore_completion else info["eligible"]
    for source, target in zip(info["sources"], info["assignments"]):
        if target in movers:
            for p in source:
                out[p] = info["bg"]
    for placed, target in zip(info["placements"], info["assignments"]):
        if target in movers:
            for p, v in placed.items():
                out[p] = v
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles, orientations, matches, and completed lines from input."""
    return _render_solution(_parse(input_grid))


def _trim(mask):
    rr, cc = np.where(mask)
    return mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()


def _mask_key(mask):
    return _key(_trim(mask))


def _new_mask(rng, q):
    gh, gw = map(int, rng.integers(1, min(q, 3) + 1, size=2))
    mode = int(rng.integers(5))
    if mode == 0:
        mask = np.ones((gh, gw), bool)
    elif mode == 1:
        mask = np.indices((gh, gw)).sum(axis=0) % 2 == 0
    elif mode == 2 and min(gh, gw) >= 3:
        mask = np.ones((gh, gw), bool)
        mask[1:-1, 1:-1] = False
    else:
        mask = rng.random((gh, gw)) < rng.uniform(.25, .85)
        mask[int(rng.integers(gh)), int(rng.integers(gw))] = True
    return _trim(mask)


def _board_inventory(rng, rows, cols, q, inks, selected, eligible, adversarial):
    positions = list(itertools.product(range(rows), range(cols)))
    inventory, used = {}, set()

    def add(pos, mask, ink):
        key = (_mask_key(mask), ink)
        assert key not in used
        used.add(key)
        inventory[pos] = (mask.copy(), ink)

    if adversarial:
        p = sorted(eligible)[int(rng.integers(len(eligible)))]
        spare = [pos for pos in positions if pos not in selected]
        rng.shuffle(spare)
        # A chiral 3x3 registration has both rotated/mirrored same-ink
        # distractors and a same-shape/different-ink distractor.
        mask = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]], bool)
        mask = np.rot90(np.fliplr(mask) if rng.integers(2) else mask, int(rng.integers(4))).copy()
        ink = int(rng.choice(inks))
        add(p, mask, ink)
        add(spare[0], np.rot90(mask).copy(), ink)
        add(spare[1], mask, next(v for v in inks if v != ink))
        if len(spare) >= 3:
            add(spare[2], np.fliplr(mask).copy(), ink)
    for pos in positions:
        if pos in inventory:
            continue
        for _ in range(200):
            mask, ink = _new_mask(rng, q), int(rng.choice(inks))
            if (_mask_key(mask), ink) not in used and not (pos in selected and int(mask.sum()) == q * q):
                add(pos, mask, ink)
                break
        else:
            # Finite exhaustive mask fallback (there are many more keys than
            # the largest supported board needs).
            for code in range(1, 512):
                mask = _trim(np.array([(code >> i) & 1 for i in range(9)], bool).reshape(3, 3))
                if (_mask_key(mask), inks[0]) not in used and not (pos in selected and int(mask.sum()) == q * q):
                    add(pos, mask, inks[0])
                    break
    return inventory


def _path(rng, start, end):
    r, c = start
    result = {(r, c)}
    while (r, c) != end:
        if r != end[0] and (c == end[1] or rng.integers(2)):
            r += 1 if end[0] > r else -1
        else:
            c += 1 if end[1] > c else -1
        result.add((r, c))
    return result


def _piece(rng, mask, ink, q, body, compass):
    gh, gw = mask.shape
    ro, co = 1 + (q - gh) // 2, 1 + (q - gw) // 2
    marks = {(ro + int(r), co + int(c)): ink for r, c in np.argwhere(mask)}
    first = int(rng.integers(4))
    sides = {first, (first + 1) % 4}
    if rng.random() < .35:
        sides.add((first + 2) % 4)
    if rng.random() < .12:
        sides = set(range(4))
    labels, supports = {}, []
    for d in sorted(sides):
        length = int(rng.integers(1, min(3, q) + 1))
        start = int(rng.integers(1, q - length + 2))
        for k in range(start, start + length):
            p, support = (((0, k), (1, k)), ((k, q + 1), (k, q)),
                          ((q + 1, k), (q, k)), ((k, 0), (k, 1)))[d]
            labels[p] = compass[d]
            supports.append(support)
    ends = sorted(marks) + supports + [(1, int(rng.integers(1, q + 1))),
             (q, int(rng.integers(1, q + 1))), (int(rng.integers(1, q + 1)), 1),
             (int(rng.integers(1, q + 1)), q)]
    rng.shuffle(ends)
    core = {ends[0]}
    for endpoint in ends[1:]:
        closest = min(core, key=lambda p: (abs(p[0] - endpoint[0]) + abs(p[1] - endpoint[1]), p))
        core.update(_path(rng, closest, endpoint))
    density = float(rng.uniform(.35, .95))
    for r, c in itertools.product(range(1, q + 1), repeat=2):
        if rng.random() < density and any((r + dr, c + dc) in core for dr, dc in DIRS):
            core.add((r, c))
    if not core - set(marks):
        empty = [(r, c) for r, c in itertools.product(range(1, q + 1), repeat=2) if (r, c) not in marks]
        assert empty
        endpoint = empty[int(rng.integers(len(empty)))]
        closest = min(core, key=lambda p: (abs(p[0] - endpoint[0]) + abs(p[1] - endpoint[1]), p))
        core.update(_path(rng, closest, endpoint))
    return {**{p: body for p in core}, **marks, **labels}


def _layouts(rows, cols, s, count):
    bh, bw = rows * (s + 1) + 1, cols * (s + 1) + 1
    options = []
    for columns in range(1, count + 1):
        nr = (count + columns - 1) // columns
        for gap, separation in itertools.product((1, 2), (1, 2, 3)):
            sh, sw = nr * s + (nr - 1) * gap, columns * s + (columns - 1) * gap
            for side in ("below", "right"):
                height, width = ((bh + separation + sh + 2, max(bw, sw) + 2) if side == "below"
                                 else (max(bh, sh) + 2, bw + separation + sw + 2))
                if height <= 28 and width <= 28:
                    options.append((height, width, columns, nr, gap, separation, side))
    return options


def _selection(rng, rows, cols, kind, axis=None, count=None):
    universe = list(itertools.product(range(rows), range(cols)))
    if kind == "cross":
        r, c = int(rng.integers(rows)), int(rng.integers(cols))
        return {(r, j) for j in range(cols)} | {(i, c) for i in range(rows)}
    if kind == "diagonal":
        assert rows == cols == 3
        selected = {(i, i) for i in range(3)} if rng.integers(2) else {(i, 2 - i) for i in range(3)}
        extra = [p for p in universe if p not in selected]
        return selected | {extra[int(rng.integers(len(extra)))]}
    if kind == "line":
        if axis is None:
            axis = int(rng.integers(2))
        i = int(rng.integers(rows if axis == 0 else cols))
        selected = {(i, c) for c in range(cols)} if axis == 0 else {(r, i) for r in range(rows)}
        if count is None:
            count = len(selected) + 1
        extras = [p for p in universe if p not in selected]
        if count > len(selected):
            selected.update(extras[int(j)] for j in rng.choice(len(extras), count - len(selected), replace=False))
        return selected
    assert kind == "scatter"
    count = count or min(rows, cols)
    choices = [set(p) for p in itertools.combinations(universe, count) if not _line_members(p, rows, cols)[0]]
    assert choices
    return choices[int(rng.integers(len(choices)))]


def _make_pair(rng, rows, cols, s, selected, palette, adversarial=False, forced_motion=False):
    bg, body, divider = palette[:3]
    compass, inks = tuple(palette[3:7]), palette[7:]
    eligible, _ = _line_members(selected, rows, cols)
    q = s - 2
    inventory = _board_inventory(rng, rows, cols, q, inks, selected, eligible, adversarial)
    options = _layouts(rows, cols, s, len(selected))
    assert options, "requested board and query packing must fit"
    height, width, columns, nr, gap, separation, side = options[int(rng.integers(len(options)))]
    height += int(rng.integers(min(2, 28 - height) + 1))
    width += int(rng.integers(min(2, 28 - width) + 1))
    h, w = height + 2, width + 2
    inp = np.full((h, w), bg, np.int8)
    inp[0, 1:-1], inp[-1, 1:-1] = compass[0], compass[2]
    inp[1:-1, 0], inp[1:-1, -1] = compass[3], compass[1]
    bh, bw = rows * (s + 1) + 1, cols * (s + 1) + 1
    sh, sw = nr * s + (nr - 1) * gap, columns * s + (columns - 1) * gap
    if side == "below":
        br = int(rng.integers(2, h - 1 - bh - separation - sh))
        bc = int(rng.integers(2, w - bw))
        ar, ac = br + bh + separation, int(rng.integers(2, w - sw))
    else:
        br = int(rng.integers(2, h - bh))
        bc = int(rng.integers(2, w - 1 - bw - separation - sw))
        ar, ac = int(rng.integers(2, h - sh)), bc + bw + separation
    board_mask = np.zeros(inp.shape, bool)
    board_mask[br:br + bh, bc:bc + bw] = True
    for r in range(rows + 1):
        inp[br + r * (s + 1), bc:bc + bw] = divider
    for c in range(cols + 1):
        inp[br:br + bh, bc + c * (s + 1)] = divider
    for (r, c), (mask, ink) in inventory.items():
        gh, gw = mask.shape
        sr = br + 2 + r * (s + 1) + (q - gh) // 2
        sc = bc + 2 + c * (s + 1) + (q - gw) // 2
        block = inp[sr:sr + gh, sc:sc + gw]
        block[mask] = ink
    out = inp.copy()
    order = sorted(selected)
    rng.shuffle(order)
    movers = [p for p in order if p in eligible]
    slot_order = list(map(int, rng.permutation(columns * nr)[:len(order)]))
    for pos, slot in zip(order, slot_order):
        mask, ink = inventory[pos]
        canonical = _piece(rng, mask, ink, q, body, compass)
        t = int(rng.integers(8))
        if forced_motion and pos in movers[:2]:
            t = int(rng.integers(1, 4) if pos == movers[0] else rng.integers(4, 8))
        cells = _transform(canonical, t)
        ph, pw = 1 + max(r for r, _ in cells), 1 + max(c for _, c in cells)
        sr = ar + (slot // columns) * (s + gap) + int(rng.integers(s - ph + 1))
        sc = ac + (slot % columns) * (s + gap) + int(rng.integers(s - pw + 1))
        source = _shift(cells, sr, sc)
        assert all(inp[p] == bg for p in source)
        for p, value in source.items():
            inp[p] = value
        if pos not in eligible:
            for p, value in source.items():
                out[p] = value
        else:
            # Registration is centered identically in the core and target
            # slot. Construction renders directly, rather than calling solve.
            fr, fc = br + 1 + pos[0] * (s + 1), bc + 1 + pos[1] * (s + 1)
            for p, value in _shift(canonical, fr, fc).items():
                assert out[p] in (bg, ink)
                out[p] = value
    # A whole-scene D4 transform changes storage location and the displayed
    # compass, without introducing any absolute color-to-direction dictionary.
    scene_t = int(rng.integers(8))
    def orient(a):
        return np.rot90(np.fliplr(a) if scene_t >= 4 else a, scene_t % 4).copy()
    return {"input": orient(inp), "output": orient(out)}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    ink_count = int(rng.integers(2, 4))
    palette = palette[:7 + ink_count]
    first_axis = int(rng.integers(2))
    large_slot = int(rng.integers(5, 8))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", nt), ("test", nq)):
        for index in range(number):
            adversarial = forced = False
            if split == "train" and index == 0:
                rows = cols = 3
                s, kind, axis, count = 5, "line", first_axis, 4
                adversarial = forced = True
            elif split == "train" and index == 1:
                rows = cols = 3
                s, kind, axis, count = 5, "diagonal", None, 4
            elif split == "train" and index == 2:
                rows = cols = 3
                s, kind, axis, count = 5, "line", 1 - first_axis, int(rng.integers(3, 5))
            elif split == "train" and index == 3:
                rows = cols = 2
                s, kind, axis, count = large_slot, "cross", None, None
            elif split == "train" and index == 4:
                rows, cols = (2, 4) if first_axis == 0 else (4, 2)
                s, kind, axis, count = 5, "line", first_axis, 5
            else:
                if nt < 4:
                    rows = cols = 3
                    s = 5
                else:
                    dimensions = [(2, 3), (3, 2), (2, 2), (3, 3)]
                    if nt >= 5:
                        dimensions += [(2, 4), (4, 2)]
                    if split == "test" and index == 0:
                        # Combine the two taught square-board line lengths in
                        # a novel rectangular board, rather than relying on
                        # chance to create the main structural test shift.
                        rows, cols = ((2, 3), (3, 2))[int(rng.integers(2))]
                    else:
                        rows, cols = dimensions[int(rng.integers(len(dimensions)))]
                    s = (5 if (rows == cols == 3 or max(rows, cols) == 4)
                         else int(rng.integers(5, large_slot + 1)))
                axis = first_axis if nt == 2 else int(rng.integers(2))
                kinds = ["line", "scatter"] + (["cross"] if nt >= 4 else [])
                kind = kinds[int(rng.integers(len(kinds)))]
                if kind == "line":
                    length = cols if axis == 0 else rows
                    count = length + int(rng.integers(2))
                elif kind == "scatter":
                    count = min(max(rows, cols), 4)
                    if rows == cols == 2:
                        count = 2  # opposite corners
                else:
                    count = None
                if nt == 2 and split == "test":
                    # Both demonstrations have four queries. Three new keys
                    # now either form the full line or scatter, so cardinality
                    # alone still cannot decide the test across this regime.
                    count = 3
                # Choose only combinations for which an actual packing exists.
                wanted = rows + cols - 1 if kind == "cross" else count
                if not _layouts(rows, cols, s, wanted):
                    s = 5
                if not _layouts(rows, cols, s, wanted):
                    kind, count = "line", cols if axis == 0 else rows
            # Optional local palettes keep every role visible in the pixels.
            local_palette = palette
            if nt >= 4 and rng.random() < .45:
                local_palette = list(map(int, rng.permutation(10)))[:7 + ink_count]
            for _ in range(30):
                selected = _selection(rng, rows, cols, kind, axis, count)
                pair = _make_pair(rng, rows, cols, s, selected, local_palette, adversarial, forced)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("distinct-input sampling exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    # Teaching responsibilities belong to the set, never to the returned index.
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    puzzle["test"] = [puzzle["test"][int(i)] for i in rng.permutation(nq)]
    return puzzle


def _ablation_evidence(info):
    """Visible counterfactuals: wrong orientation/ink identity breaks a line."""
    selected = set(info["assignments"])
    orientation_needed = color_needed = False
    for index, target in enumerate(info["assignments"]):
        if target not in info["eligible"]:
            continue
        cells = info["canonicals"][index]
        marks = {p: v for p, v in cells.items() if v in info["inks"]}
        shape = tuple((r, c) for r, c, _ in _token_key(marks))
        for other, token in info["tokens"].items():
            if other in selected:
                continue
            other_key = _token_key(token)
            changed, _ = _line_members((selected - {target}) | {other}, info["rows"], info["cols"])
            if changed == info["eligible"]:
                continue
            if tuple((r, c) for r, c, _ in other_key) == shape:
                color_needed = True
            if any(_token_key(_transform(marks, t)) == other_key for t in range(1, 8)):
                orientation_needed = True
    return orientation_needed, color_needed


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, train_info = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _check_grid(pair["input"]), _check_grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            assert np.array_equal(_render_solution(info), out), "incorrect collective docking output"
            assert np.array_equal(inp[[0, -1]], out[[0, -1]])
            assert np.array_equal(inp[:, [0, -1]], out[:, [0, -1]])
            assert np.array_equal(inp[inp == info["divider"]], out[inp == info["divider"]])
            for marks in info["tokens"].values():
                assert all(out[p] == v for p, v in marks.items()), "board registration ink must be preserved"
            counts = Counter(inp.ravel())
            expected_delta = Counter()
            for source, placed, target in zip(info["sources"], info["placements"], info["assignments"]):
                if target in info["eligible"]:
                    for p, v in source.items():
                        expected_delta[v] -= 1
                        expected_delta[info["bg"]] += 1
                    for p, v in placed.items():
                        expected_delta[int(inp[p])] -= 1
                        expected_delta[v] += 1
                else:
                    assert all(out[p] == v for p, v in source.items()), "stray sources must remain unchanged"
            output_counts = Counter(out.ravel())
            assert all(output_counts[c] - counts[c] == expected_delta[c] for c in range(10))
            if split == "train":
                train_info.append(info)
    assert any(i["eligible"] and len(i["eligible"]) < len(i["sources"]) for i in train_info), "teach movers alongside strays"
    assert any(not i["eligible"] and len(i["sources"]) == 4 for i in train_info), "teach same-count noncompletion"
    assert any(all(_ablation_evidence(i)) for i in train_info), "teach compass and shape-and-ink identity"
    assert any(not np.array_equal(_render_solution(i, True), _render_solution(i)) for i in train_info), "completion must matter"
    assert any(0 < pose < 4 and target in i["eligible"] for i in train_info
               for pose, target in zip(i["poses"], i["assignments"])), "teach an actual rotating mover"
    assert any(pose >= 4 and target in i["eligible"] for i in train_info
               for pose, target in zip(i["poses"], i["assignments"])), "teach an actual reflecting mover"
    if nt >= 4:
        assert any(len(i["lines"]) >= 2 for i in train_info), "teach the union of intersecting lines"


def self_test():
    records, signatures, all_shapes = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        repeat = generate_puzzle(seed)
        signature = tuple(_key(p[k]) for split in ("train", "test") for p in puzzle[split] for k in ("input", "output"))
        assert signature == tuple(_key(p[k]) for split in ("train", "test") for p in repeat[split] for k in ("input", "output"))
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                all_shapes.add(pair["input"].shape)
                records.append((split, info))
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                assert set(map(int, pair["output"].ravel())) <= set(map(int, pair["input"].ravel()))
    assert len(signatures) == 42 and len(all_shapes) >= 15
    infos = [i for _, i in records]
    assert len({(i["rows"], i["cols"], i["s"]) for i in infos}) >= 8
    assert {i["s"] for i in infos} == {5, 6, 7}
    assert len({i["bg"] for i in infos}) >= 8
    assert len({tuple(i["assignments"]) for i in infos}) >= 50
    assert {len(i["sources"]) for i in infos} >= {2, 3, 4}
    assert any(len(i["lines"]) >= 2 for i in infos)
    tests = [i for split, i in records if split == "test"]
    assert any(i["eligible"] for i in tests) and any(not i["eligible"] for i in tests)
    bodies = {_token_key({p: 1 for p, v in cells.items() if v == i["body"]})
              for i in infos for cells in i["canonicals"]}
    assert len(bodies) >= 80
    # Metamorphic checks exercise all compass/scene symmetries, including a
    # rectangular board, without consulting the seed while solving.
    baseline = generate_puzzle(73)
    for t in range(8):
        rotated = {split: [{k: np.rot90(np.fliplr(v) if t >= 4 else v, t % 4).copy()
                           for k, v in pair.items()} for pair in baseline[split]] for split in baseline}
        validate_generated_puzzle(rotated)
    # Meaningful semantic damage: dock all strays, alter registration ink,
    # destroy structure, and duplicate an input. Each must be rejected.
    for defect in ("strays", "ink", "frame", "duplicate"):
        damaged = {split: [{k: v.copy() for k, v in p.items()} for p in baseline[split]] for split in baseline}
        if defect == "duplicate":
            damaged["train"][1] = {k: v.copy() for k, v in damaged["train"][0].items()}
        else:
            pair = next(p for p in damaged["train"] if not _parse(p["input"])["eligible"])
            info = _parse(pair["input"])
            if defect == "strays":
                pair["output"] = _render_solution(info, True)
            elif defect == "ink":
                p = min(next(iter(info["tokens"].values())))
                pair["output"][p] = info["body"]
            else:
                pair["output"][0, 1] = info["bg"]
        try:
            validate_generated_puzzle(damaged)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("validator accepted " + defect)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(42, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "grid_shapes": len(all_shapes),
            "board_geometries": len({(i["rows"], i["cols"], i["s"]) for i in infos}),
            "body_geometries": len(bodies), "pairs": len(records)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
