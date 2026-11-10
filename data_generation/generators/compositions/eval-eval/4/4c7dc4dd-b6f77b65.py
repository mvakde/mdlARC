"""Mechanical analogy: infer a junction law from a collapse, then transfer it.

Only standard-library modules and NumPy are required. Generation does not write
artifacts; self_test streams 42 puzzles and retains only small feature sets.
"""
from collections import Counter
import hashlib

import numpy as np


# A stick is (orientation, top, left, length, material).
def _cells(p):
    o, r, c, n, _ = p
    return frozenset((r + (k if o == "V" else 0), c + (k if o == "H" else 0))
                     for k in range(n))


def _shift(p, d):
    return p[0], p[1] + d, p[2], p[3], p[4]


def _joints(pieces):
    cells = [_cells(p) for p in pieces]
    return {(j, i) for i in range(len(pieces)) for j in range(i) if cells[i] & cells[j]}


def _groups(pieces, welded):
    parent = list(range(len(pieces)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if welded:
        for i, j in _joints(pieces):
            parent[root(i)] = root(j)
    groups = {}
    for i in range(len(pieces)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def _settle(pieces, height, welded):
    """Greatest simultaneously movable set; new contacts are solid in both laws.

    Welding is computed on the surviving rods, so erasing an articulation rod
    can split an assembly. Sliding permissions belong to initial crossing pairs.
    """
    result = list(pieces)
    groups = _groups(result, welded)
    joints = _joints(result) if not welded else set()
    for _ in range(height * (len(groups) + 1)):
        footprints = [_cells(p) for p in result]
        bodies = [frozenset().union(*(footprints[i] for i in g)) for g in groups]
        targets = [frozenset((r + 1, c) for r, c in body) for body in bodies]
        moving = {i for i, target in enumerate(targets) if all(r < height for r, _ in target)}
        while True:
            blocked = set()
            for i in moving:
                for j in range(len(groups)):
                    if j in moving or i == j:
                        continue
                    if not welded and tuple(sorted((groups[i][0], groups[j][0]))) in joints:
                        continue
                    if targets[i] & bodies[j]:
                        blocked.add(i)
                        break
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return result
        ids = {k for i in moving for k in groups[i]}
        result = [_shift(p, 1) if i in ids else p for i, p in enumerate(result)]
    raise AssertionError("gravity did not converge")


def _render(shape, bg, band, pieces):
    grid = np.full(shape, bg, dtype=np.int8)
    grid[0] = band
    # The visible horizontal line is uninterrupted through every crossing.
    for p in sorted(pieces, key=lambda p: p[0] == "H"):
        for cell in _cells(p):
            grid[cell] = p[4]
    return grid


def _controls(grid, bg):
    assert grid.shape[0] >= 6 and np.all(grid[1] == bg)
    controls = [int(x) for x in grid[0] if x != bg]
    assert 1 <= len(controls) <= 3 and len(set(controls)) == len(controls)
    return controls


def _parse_rods(grid, bg):
    """Maximal horizontal runs, then visibly bracketed vertical continuations.

    No concealed endpoint is inferred. A vertical rod has at least two visible
    cells outside horizontal runs and both endpoints are visible. A covered
    interval can join them only if every covered cell belongs to a horizontal.
    """
    _controls(grid, bg)
    h, w = grid.shape
    horizontal = np.zeros(grid.shape, dtype=bool)
    pieces = []
    for r in range(2, h):
        c = 0
        while c < w:
            end = c + 1
            while end < w and grid[r, end] == grid[r, c]:
                end += 1
            if grid[r, c] != bg and end - c >= 2:
                pieces.append(("H", r, c, end - c, int(grid[r, c])))
                horizontal[r, c:end] = True
            c = end
    for c in range(w):
        for color in sorted(set(map(int, grid[2:, c])) - {bg}):
            visible = [r for r in range(2, h) if grid[r, c] == color and not horizontal[r, c]]
            runs = []
            for r in visible:
                if runs and all(horizontal[y, c] or grid[y, c] == color
                                for y in range(runs[-1][-1] + 1, r)):
                    runs[-1].append(r)
                else:
                    runs.append([r])
            for run in runs:
                if len(run) >= 2:
                    pieces.append(("V", run[0], c, run[-1] - run[0] + 1, color))
    assert pieces
    assert all(pieces[i][0] != pieces[j][0] for i, j in _joints(pieces))
    assert np.array_equal(_render(grid.shape, bg, grid[0], pieces), grid), "unresolved sticks"
    return pieces


def _predict_scene(scene, bg, welded):
    controls = set(_controls(scene, bg))
    rods = _parse_rods(scene, bg)
    survivors = [p for p in rods if p[4] not in controls]
    return _render(scene.shape, bg, scene[0], _settle(survivors, scene.shape[0], welded))


def _components(grid, bg):
    unseen = set(map(tuple, np.argwhere(grid != bg)))
    while unseen:
        start = unseen.pop()
        color = int(grid[start])
        cells, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in unseen and grid[p] == color:
                    unseen.remove(p)
                    cells.add(p)
                    todo.append(p)
        yield color, cells


def _extract(input_grid):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= n <= 30 for n in grid.shape) and np.all((0 <= grid) & (grid <= 9))
    bg = int(grid[0, 0])
    assert all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    panels = []
    for color, cells in _components(grid, bg):
        top = min(r for r, _ in cells)
        bottom = max(r for r, _ in cells)
        left = min(c for _, c in cells)
        right = max(c for _, c in cells)
        if bottom - top < 7 or right - left < 6:
            continue
        perimeter = {(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)
                     if r in (top, bottom) or c in (left, right)}
        if cells == perimeter:
            interior = grid[top + 1:bottom, left + 1:right].copy()
            _controls(interior, bg)
            panels.append((color, (top, left, bottom + 1, right + 1), interior))
    assert len(panels) == 3, "expected three isolated panel outlines"
    counts = Counter(p[0] for p in panels)
    assert sorted(counts.values()) == [1, 2]
    ref_color = next(c for c, n in counts.items() if n == 2)
    target = next(p for p in panels if p[0] != ref_color)
    reference = [p for p in panels if p[0] == ref_color]
    assert reference[0][2].shape == reference[1][2].shape
    # Frame materials are reserved and no outside drawing is allowed.
    covered = np.zeros(grid.shape, bool)
    for _, (r, c, b, d), scene in panels:
        assert not covered[r:b, c:d].any()
        covered[r:b, c:d] = True
        assert not any(np.any(scene == color) for color in counts)
    assert np.all(grid[~covered] == bg)
    return bg, reference, target


def _infer_law(reference, bg):
    a, b = [p[2] for p in reference]
    assert np.array_equal(a[:2], b[:2])
    candidates = []
    for before, after in ((a, b), (b, a)):
        controls = _controls(before, bg)
        if not any(np.any(before[2:] == c) for c in controls):
            continue
        if any(np.any(after[2:] == c) for c in controls):
            continue
        rods = _parse_rods(before, bg)
        for welded in (False, True):
            if (rods == _settle(rods, before.shape[0], welded)
                    and np.array_equal(_predict_scene(before, bg, welded), after)):
                candidates.append(welded)
    assert len(candidates) == 1, "reference must distinguish sliding from welding"
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only the supplied input; return the unframed target companion."""
    bg, reference, target = _extract(input_grid)
    return _predict_scene(target[2], bg, _infer_law(reference, bg))


def _band(rng, width, bg, controls):
    band = np.full(width, bg, dtype=np.int8)
    for col, color in zip(rng.choice(width, len(controls), replace=False), rng.permutation(controls)):
        band[int(col)] = color
    return band


def _reference(rng, bg, materials):
    h, w = int(rng.integers(8, 10)), int(rng.integers(7, 11))
    row = int(rng.integers(4, h - 2))
    x = int(rng.integers(2, w - 2))
    support = int(rng.choice([0, w - 1]))
    beam, erased, rail = map(int, rng.permutation(materials)[:3])
    if rng.random() < .45:
        rail = beam
    rods = [("V", 2, x, h - 2, rail), ("V", row + 1, support, h - row - 1, erased),
            ("H", row, 0, w, beam)]
    before = _render((h, w), bg, _band(rng, w, bg, [erased]), rods)
    assert set(_parse_rods(before, bg)) == set(rods)
    return before, [_predict_scene(before, bg, welded) for welded in (False, True)]


def _drop(pieces, o, c, n, h):
    p = (o, 2, c, n, -1)
    occupied = frozenset().union(*(_cells(q) for q in pieces))
    if any(r >= h for r, _ in _cells(p)) or _cells(p) & occupied:
        return None
    while all(r + 1 < h for r, _ in _cells(p)) and not (_cells(_shift(p, 1)) & occupied):
        p = _shift(p, 1)
    return p


def _decorate(rng, rods, materials):
    """Resolve optional colors without same-color accidental straight mergers."""
    result = []
    for p in rods:
        if p[4] >= 0:
            result.append(p)
            continue
        cells = _cells(p)
        neighbors = cells | {(r + dr, c + dc) for r, c in cells
                             for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))}
        forbidden = {q[4] for q in result if neighbors & _cells(q) and not cells & _cells(q)}
        choices = [int(c) for c in materials if c not in forbidden]
        if not choices:
            return None
        result.append((*p[:4], int(rng.choice(choices))))
    return result


def _target(rng, bg, materials, large=False, core=False):
    for attempt in range(160):
        h = int(rng.integers(12, 15) if large else 11 if core else rng.integers(10, 12))
        w = int(rng.integers(11, 16) if large else rng.integers(11 if core else 9, 14))
        left, right = int(rng.integers(0, 2)), w - 1 - int(rng.integers(0, 2))
        floating = core or (h >= 11 and rng.random() < .45)
        row = int(rng.integers(5, h - 5 if floating else h - 2))
        values = list(map(int, rng.permutation(materials)))
        beam, cut, stem_cut, rail, landing = values[:5]
        if not core and rng.random() < .25:
            stem_cut = cut
        controls = list(dict.fromkeys([cut, stem_cut]))
        positions = list(range(left + 3, right - 1))
        rng.shuffle(positions)
        rail_x = [positions[0]]
        if len(positions) >= 3 and (core or rng.random() < (.75 if large else .45)):
            options = [x for x in positions if abs(x - rail_x[0]) >= 2]
            if options:
                rail_x.append(options[0])
        rods = [("V", row + 1, left, h - row - 1, cut),
                ("H", row, left, right - left + 1, beam),
                ("V", row - 2, left + 1, 2, stem_cut),
                ("H", row - 3, left, 2, int(rng.choice([beam, rail, landing])))]
        # A second tall leg can make the lower release genuinely joint.
        if rng.random() < .60:
            rods.append(("V", row + 1, right, h - row - 1, int(rng.choice(controls))))
        pedestal_span = None
        if floating:
            # Unlike a floor rail, a welded rail/beam assembly is free to move
            # when this visibly separate pedestal and its post are erased.
            lo, hi = min(rail_x) - 1, max(rail_x) + 1
            pedestal_span = (lo, hi)
            rods.extend([("H", h - 3, lo, hi - lo + 1, cut),
                         ("V", h - 2, int(rng.choice(rail_x)), 2, cut)])
        used_columns = {p[2] for p in rods if p[0] == "V"} | set(rail_x)
        landings = [x for x in range(left + 2, right) if x not in used_columns
                    and (pedestal_span is None or not pedestal_span[0] <= x <= pedestal_span[1])]
        rng.shuffle(landings)
        for x in landings[:int(rng.integers(0, min(3, len(landings)) + 1))]:
            n = int(rng.integers(2, h - row - 1)) if h - row - 1 > 2 else 2
            if n < h - row - 1:
                rods.append(("V", h - n, x, n, landing))
        # Drop extra sticks into the actual existing support network.
        extras = int(rng.integers(2, 6) if large else rng.integers(0, 3)) if attempt < 100 else 0
        for _ in range(extras * 3):
            if extras == 0:
                break
            o = str(rng.choice(["H", "V"]))
            n = int(rng.integers(2, min(w - 1, 7) + 1)) if o == "H" else int(rng.integers(2, 5))
            c = int(rng.integers(0, w - n + 1)) if o == "H" else int(rng.integers(w))
            if o == "V" and c in rail_x:
                continue
            p = _drop(rods, o, c, n, h)
            if p is None or (o == "H" and p[1] in (2, h - 1) and
                             any(p[2] <= x < p[2] + p[3] for x in rail_x)):
                continue
            rods.append(p)
            extras -= 1
        # A deleted rail tests that deleted joints do not keep bodies glued.
        remove_rail = not core and len(rail_x) > 1 and len(values) >= 6 and rng.random() < .50
        for i, x in enumerate(rail_x):
            color = beam if core and i == 0 else values[5] if remove_rail and i == 0 else rail
            rods.append(("V", 2, x, h - 5 if floating else h - 2, color))
        if remove_rail:
            controls.append(values[5])
        rods = _decorate(rng, rods, materials)
        if rods is None:
            continue
        if len(controls) < 3 and rng.random() < .15:
            absent = [c for c in materials if c not in {p[4] for p in rods}]
            if absent:
                controls.append(int(rng.choice(absent)))
        scene = _render((h, w), bg, _band(rng, w, bg, controls), rods)
        try:
            parsed = _parse_rods(scene, bg)
            assert set(parsed) == set(rods)
            if core:
                pairs = _joints(parsed)
                assert any(parsed[i][4] == parsed[j][4] for i, j in pairs)
                assert any(parsed[i][4] != parsed[j][4] for i, j in pairs)
            assert all(parsed == _settle(parsed, h, welded) for welded in (False, True))
            outputs = [_predict_scene(scene, bg, welded) for welded in (False, True)]
            assert not np.array_equal(*outputs)
            for welded in (False, True):
                survivors = [p for p in parsed if p[4] not in controls]
                assert any(p[1] != q[1] for p, q in zip(survivors, _settle(survivors, h, welded)))
            if core:
                survivors = [p for p in parsed if p[4] not in controls]
                settled = _settle(survivors, h, True)
                assert any(len(group) >= 2 and settled[group[0]][1] > survivors[group[0]][1]
                           for group in _groups(survivors, True))
        except AssertionError:
            continue
        return scene, outputs
    raise AssertionError("bounded target construction exhausted")


def _outline(grid, box, color, interior):
    r, c, h, w = box
    grid[r:r + h, c] = grid[r:r + h, c + w - 1] = color
    grid[r, c:c + w] = grid[r + h - 1, c:c + w] = color
    grid[r + 1:r + h - 1, c + 1:c + w - 1] = interior


def _layout(rng, ref_shape, target_shape):
    rh, rw = (x + 2 for x in ref_shape)
    th, tw = (x + 2 for x in target_shape)
    options = [(False, rh + th + 3, max(2 * rw + 1, tw) + 2)]
    if 2 * rh + 3 <= 30 and rw + tw + 3 <= 30:
        options.append((True, max(2 * rh + 1, th) + 2, rw + tw + 3))
    vertical, mh, mw = options[int(rng.integers(len(options)))]
    assert mh <= 30 and mw <= 30
    h, w = int(rng.integers(mh, 31)), int(rng.integers(mw, 31))
    if not vertical:
        gap = int(rng.integers(1, w - 2 - 2 * rw + 1))
        start = int(rng.integers(1, w - (2 * rw + gap)))
        r0 = int(rng.integers(1, h - rh - th - 1))
        rt = int(rng.integers(r0 + rh + 1, h - th))
        boxes = [(r0, start, rh, rw), (r0, start + rw + gap, rh, rw),
                 (rt, int(rng.integers(1, w - tw)), th, tw)]
        if rng.integers(2):
            boxes = [(h - r - bh, c, bh, bw) for r, c, bh, bw in boxes]
    else:
        gap = int(rng.integers(1, h - 2 - 2 * rh + 1))
        start = int(rng.integers(1, h - (2 * rh + gap)))
        c0 = int(rng.integers(1, w - rw - tw - 1))
        ct = int(rng.integers(c0 + rw + 1, w - tw))
        boxes = [(start, c0, rh, rw), (start + rh + gap, c0, rh, rw),
                 (int(rng.integers(1, h - th)), ct, th, tw)]
        if rng.integers(2):
            boxes = [(r, w - c - bw, bh, bw) for r, c, bh, bw in boxes]
    if rng.integers(2):
        boxes[0], boxes[1] = boxes[1], boxes[0]
    return (h, w), boxes


def _make_pair_set(rng, large=False, twins=False):
    # Retry layout artifacts only; no change of semantics or cached answer.
    for _ in range(24):
        colors = list(map(int, rng.permutation(10)))
        bg, ref_frame, target_frame = colors[:3]
        materials = colors[3:3 + int(rng.integers(5, 8))]
        before, reference_outputs = _reference(rng, bg, materials)
        target, target_outputs = _target(rng, bg, materials, large=large, core=twins)
        shape, boxes = _layout(rng, before.shape, target.shape)
        modes = [False, True] if twins else [bool(rng.integers(2))]
        pairs = []
        try:
            for welded in modes:
                grid = np.full(shape, bg, dtype=np.int8)
                for box, frame, content in zip(boxes, [ref_frame, ref_frame, target_frame],
                                               [before, reference_outputs[int(welded)], target]):
                    _outline(grid, box, frame, content)
                out = target_outputs[int(welded)]
                assert np.array_equal(solve(grid), out)
                pairs.append({"input": grid, "output": out.copy()})
        except AssertionError:
            continue
        return pairs
    raise AssertionError("bounded panel construction exhausted")


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " outside legal range")
    return int(value)


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    a = _integer(num_train, 2, 10, "num_train")
    b = _integer(num_test, 1, 4, "num_test")
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle = {"train": _make_pair_set(rng, twins=True), "test": []}
    seen = {_key(p["input"]) for p in puzzle["train"]}
    for split, n in (("train", a - 2), ("test", b)):
        for _ in range(n):
            for _ in range(24):
                pair = _make_pair_set(rng, large=split == "test")[0]
                key = _key(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise AssertionError("distinct-input construction exhausted")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    return puzzle


def _features(pair):
    bg, reference, target = _extract(pair["input"])
    welded = _infer_law(reference, bg)
    scene = target[2]
    rods = _parse_rods(scene, bg)
    controls = set(_controls(scene, bg))
    survivors = [p for p in rods if p[4] not in controls]
    settled = _settle(survivors, scene.shape[0], welded)
    distances = [q[1] - p[1] for p, q in zip(survivors, settled)]
    return {"mode": "weld" if welded else "slide", "rods": len(rods),
            "joints": len(_joints(rods)), "surviving_joints": len(_joints(survivors)),
            "commands": len(controls), "shape": scene.shape,
            "background": bg, "materials": len({p[4] for p in rods}),
            "falls": tuple(sorted(set(distances) - {0})),
            "moved": sum(d > 0 for d in distances),
            "moving_welded_group": welded and any(
                len(group) >= 2 and distances[group[0]] > 0 for group in _groups(survivors, True)),
            "erased_joint": any(rods[i][4] in controls or rods[j][4] in controls
                                for i, j in _joints(rods)),
            "same_color_joint": any(rods[i][4] == rods[j][4] for i, j in _joints(rods)),
            "different_color_joint": any(rods[i][4] != rods[j][4] for i, j in _joints(rods))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, teaching = set(), {}
    modes = set()
    moving_weld_taught = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = (np.asarray(pair[k]) for k in ("input", "output"))
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            bg, reference, target = _extract(inp)
            welded = _infer_law(reference, bg)
            scene = target[2]
            rods = _parse_rods(scene, bg)
            assert rods == _settle(rods, scene.shape[0], welded), "input must be stable"
            assert out.shape == scene.shape and np.array_equal(out[:2], scene[:2])
            assert np.array_equal(solve(inp), out)
            alternate = _predict_scene(scene, bg, not welded)
            assert not np.array_equal(alternate, out), "junction analogy must affect every answer"
            controls = set(_controls(scene, bg))
            assert not any(np.any(out[2:] == c) for c in controls)
            survivors = [p for p in rods if p[4] not in controls]
            settled = _settle(survivors, scene.shape[0], welded)
            assert any(q[1] > p[1] for p, q in zip(survivors, settled)), "gravity must matter"
            for p, q in zip(survivors, settled):
                assert (p[0], p[2], p[3], p[4]) == (q[0], q[2], q[3], q[4]) and q[1] >= p[1]
            assert settled == _settle(settled, scene.shape[0], welded)
            if split == "train":
                modes.add(welded)
                moving_weld_taught |= welded and any(
                    len(group) >= 2 and settled[group[0]][1] > survivors[group[0]][1]
                    for group in _groups(survivors, True))
                teaching.setdefault(_key(scene), []).append((welded, _key(out)))
    assert modes == {False, True}, "demonstrate both locally identified laws"
    assert moving_weld_taught, "welds must be shown moving as rigid assemblies"
    assert any(len({m for m, _ in rows}) == 2 and len({o for _, o in rows}) == 2
               for rows in teaching.values()), "missing same-target reference counterfactual"


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(grid.shape))
                h.update(grid.tobytes())
    return h.hexdigest()


def _mechanics_test():
    # The exact same crossing falls five cells as a slider and zero as a weld.
    rods = [("V", 2, 4, 8, 1), ("H", 4, 1, 7, 2)]
    assert _settle(rods, 10, False) == [rods[0], _shift(rods[1], 5)]
    assert _settle(rods, 10, True) == rods
    # Deleting the sole connecting rail must split two welded shelves.
    rods = [("V", 2, 4, 8, 1), ("H", 4, 1, 7, 2), ("H", 7, 2, 5, 3)]
    assert _settle(rods, 10, True) == rods
    assert _settle(rods[1:], 10, True) == [_shift(rods[1], 4), _shift(rods[2], 2)]
    # Redundant legs: one removal does not release a shelf, two do.
    rods = [("V", 5, 1, 5, 1), ("V", 5, 7, 5, 2), ("H", 4, 0, 9, 3)]
    assert _settle(rods[1:], 10, False) == rods[1:]
    assert _settle([rods[-1]], 10, False) == [_shift(rods[-1], 5)]


def self_test():
    """Bounded 42-seed checks; no saved payloads or execution traces."""
    _mechanics_test()
    signatures, shapes, counts, backgrounds, falls, structures = set(), set(), set(), set(), set(), set()
    flags = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_fingerprint(puzzle))
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        for pair in puzzle["train"] + puzzle["test"]:
            f = _features(pair)
            shapes.add(f["shape"])
            counts.add(f["rods"])
            backgrounds.add(f["background"])
            falls.update(f["falls"])
            bg, _, target = _extract(pair["input"])
            structures.add(_key((target[2][2:] != bg).astype(np.int8)))
            for name in ("erased_joint", "same_color_joint", "different_color_joint", "moving_welded_group"):
                flags[name] += bool(f[name])
            flags["unequal_falls"] += len(f["falls"]) >= 2
            flags[f["mode"]] += 1
    assert len(signatures) == 42 and len(shapes) >= 15 and len(counts) >= 5
    assert len(backgrounds) == 10 and len(falls) >= 5 and len(structures) >= 100
    assert all(flags[name] >= 3 for name in ("erased_joint", "same_color_joint", "different_color_joint",
                                           "unequal_falls", "moving_welded_group", "weld", "slide"))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(444, a, b)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (a, b)
            validate_generated_puzzle(puzzle)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Validator must reject an incorrect answer, not just check formatting.
    damaged = generate_puzzle(17, 2, 1)
    damaged["test"][0]["output"][2, 0] = (int(damaged["test"][0]["output"][2, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong answer accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal count combinations)")
