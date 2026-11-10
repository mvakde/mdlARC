"""Assemble compass-tagged mosaic tiles; use mirror agreement to assign twins.

Only standard-library modules and NumPy are used. No parent code is imported.
The solver searches visible docking possibilities before accepting mirror repair.
Generation keeps at most one puzzle and one candidate pair in memory, saves no
traces, and has bounded retries. self_test retains aggregate sets, not puzzles.
"""
from __future__ import annotations

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
KEYS = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),  # O
    ((0, 0), (1, 0), (2, 0), (2, 1)),  # L
    ((0, 0), (0, 1), (0, 2), (1, 1)),  # T
    ((0, 1), (0, 2), (1, 0), (1, 1)),  # S
    ((0, 0), (1, 0), (2, 0), (3, 0)),  # I: a centered teaching key
)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("invalid ARC grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _transform(cells, t):
    result = []
    for (r, c), v in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        result.append((r, c, v))
    r0, c0 = min(p[0] for p in result), min(p[1] for p in result)
    return {(r - r0, c - c0): v for r, c, v in result}


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        component, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _frame(grid, bg):
    """Detect the sole four-colored rectangular frame, including its corners."""
    runs = {}
    h, w = grid.shape
    for r in range(h):
        c = 0
        while c < w:
            end = c + 1
            while end < w and grid[r, end] == grid[r, c]:
                end += 1
            if grid[r, c] != bg and end - c >= 8:
                runs.setdefault((c, end - 1), []).append(r)
            c = end
    found = []
    for (left, right), rows in runs.items():
        for i, top in enumerate(rows):
            for bottom in rows[i + 1:]:
                if bottom - top < 7:
                    continue
                n, s = int(grid[top, left]), int(grid[bottom, left])
                west, east = grid[top + 1:bottom, left], grid[top + 1:bottom, right]
                if not (np.all(west == west[0]) and np.all(east == east[0])):
                    continue
                colors = (n, int(east[0]), s, int(west[0]))
                if len(set(colors) | {bg}) != 5:
                    continue
                inside = grid[top + 1:bottom, left + 1:right]
                marks = set(int(v) for v in np.unique(inside)) - {bg}
                if len(marks) != 1 or next(iter(marks)) in colors:
                    continue
                if np.count_nonzero(inside != bg) < 8:
                    continue
                found.append((top, bottom, left, right, colors, marks.pop()))
    if len(found) != 1:
        raise ValueError("expected one visible canvas frame")
    return found[0]


def _parse(grid):
    bg = int(grid[0, 0])
    if any(grid[p] != bg for p in ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("workspace corners must be background")
    top, bottom, left, right, bands, reg = _frame(grid, bg)
    outside = grid != bg
    outside[top:bottom + 1, left:right + 1] = False
    sources = [{p: int(grid[p]) for p in comp} for comp in _components(outside)]
    inner = grid[top + 1:bottom, left + 1:right]
    targets = _components(inner == reg)
    if not (2 <= len(sources) == len(targets) <= 4):
        raise ValueError("source/socket counts disagree")
    if any(len(t) != 4 for t in targets):
        raise ValueError("registration stencils must have four cells")
    return {"grid": grid, "bg": bg, "reg": reg, "bands": bands,
            "frame": (top, bottom, left, right), "shape": inner.shape,
            "sources": sources, "targets": targets}


def _placements(source, info, use_compass=True):
    choices, seen = [], set()
    bg, reg, bands = info["bg"], info["reg"], info["bands"]
    ah, aw = info["shape"]
    for t in range(8):
        cells = _transform(source, t)
        core = {p: v for p, v in cells.items() if v not in bands}
        if not core:
            continue
        r0, c0 = min(r for r, c in core), min(c for r, c in core)
        r1, c1 = max(r for r, c in core), max(c for r, c in core)
        labels = [(p, v) for p, v in cells.items() if v in bands]
        facing, good = set(), True
        for (r, c), v in labels:
            ds = [d for d, ok in enumerate((r < r0, c > c1, r > r1, c < c0)) if ok]
            if len(ds) != 1 or (use_compass and bands[ds[0]] != v):
                good = False
                break
            facing.add(ds[0])
        if not good or len(facing) != 2 or (max(facing) - min(facing)) == 2:
            continue
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if min(h, w) < 4:
            continue
        body = np.full((h, w), bg, dtype=np.int8)
        for (r, c), v in core.items():
            body[r - r0, c - c0] = v
        # No missing boundary cell: the packet rectangle is explicit in input.
        if any(np.any(edge == bg) for edge in (body[0], body[-1], body[:, 0], body[:, -1])):
            continue
        marks = set(zip(*np.nonzero(body == reg)))
        if len(marks) != 4:
            continue
        mr, mc = min(marks)
        for j, target in enumerate(info["targets"]):
            tr, tc = min(target)
            r, c = tr - mr, tc - mc
            if {(y + r, x + c) for y, x in marks} != target:
                continue
            if not (0 <= r <= ah - h and 0 <= c <= aw - w):
                continue
            signature = (j, r, c, _key(body))
            if signature in seen:
                continue
            seen.add(signature)
            data = body.astype(np.int16)
            data[(body == bg) | (body == reg)] = -1
            choices.append((j, r, c, data, t))
    return choices


def _assemblies(info, use_compass=True):
    """All exact-cover dockings, WITHOUT imposing symmetry or filling holes."""
    lists = [_placements(s, info, use_compass) for s in info["sources"]]
    order = sorted(range(len(lists)), key=lambda i: len(lists[i]))
    canvas = np.full(info["shape"], -1, dtype=np.int16)
    owner = np.full(info["shape"], -1, dtype=np.int8)

    def search(depth, used, poses):
        if depth == len(order):
            if np.all(owner >= 0):
                yield canvas.copy(), owner.copy(), tuple(poses)
            return
        source_index = order[depth]
        for j, r, c, data, t in lists[source_index]:
            h, w = data.shape
            region = np.s_[r:r + h, c:c + w]
            if j in used or np.any(owner[region] >= 0):
                continue
            owner[region], canvas[region] = source_index, data
            yield from search(depth + 1, used | {j}, poses + [t])
            owner[region], canvas[region] = -1, -1
    yield from search(0, set(), [])


def _orbits(h, w):
    for r in range((h + 1) // 2):
        for c in range((w + 1) // 2):
            yield tuple(sorted({(r, c), (h - 1 - r, c),
                                (r, w - 1 - c), (h - 1 - r, w - 1 - c)}))


def _complete(raw):
    result = raw.copy()
    for orbit in _orbits(*raw.shape):
        visible = {int(raw[p]) for p in orbit if raw[p] >= 0}
        if len(visible) != 1:
            return None
        value = visible.pop()
        for p in orbit:
            result[p] = value
    return result.astype(np.int8)


def _compass_needed(info):
    outputs = set()
    for raw, _, _ in _assemblies(info, use_compass=False):
        completed = _complete(raw)
        if completed is not None:
            outputs.add(_key(completed))
            if len(outputs) > 1:
                return True
    return False


def _analyze(grid):
    info = _parse(grid)
    results, assembly_count = {}, 0
    for raw, owner, poses in _assemblies(info):
        assembly_count += 1
        completed = _complete(raw)
        if completed is not None:
            results.setdefault(_key(completed), (completed, raw, owner, poses))
    if len(results) != 1:
        raise ValueError("docking and mirror evidence do not determine one output")
    completed, raw, owner, poses = next(iter(results.values()))
    top, bottom, left, right = info["frame"]
    output = np.full_like(grid, info["bg"])
    output[top:bottom + 1, left:right + 1] = grid[top:bottom + 1, left:right + 1]
    output[top + 1:bottom, left + 1:right] = completed
    h, w = raw.shape
    diagonal = cross = 0
    for r, c in zip(*np.nonzero(raw < 0)):
        rr, cc = h - 1 - r, w - 1 - c
        diagonal += int(raw[rr, c] < 0 and raw[r, cc] < 0 and raw[rr, cc] >= 0)
        witnesses = [(y, x) for y, x in {(r, c), (rr, c), (r, cc), (rr, cc)}
                     if raw[y, x] >= 0]
        cross += int(all(owner[q] != owner[r, c] for q in witnesses))
    info.update(raw=raw, owner=owner, poses=poses, assemblies=assembly_count,
                diagonal=diagonal, cross=cross, completed=completed)
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use this input's frame, keys, tabs and surviving payload only."""
    return _analyze(_grid(input_grid))[0]


def _rectangles(rng, count, transpose):
    # At least two equally shaped sockets coexist with an anchored reference.
    h = int(rng.choice((10, 12))) if count >= 3 else int(rng.integers(7, 13))
    a, b = int(rng.integers(5, 7)), int(rng.integers(5, 7))
    w = a + b
    if count == 2:
        rects = [(0, 0, h, a), (0, a, h, b)]
    elif count == 3:
        rects = [(0, 0, h, a), (0, a, h // 2, b), (h // 2, a, h // 2, b)]
    else:
        rects = [(0, 0, h // 2, a), (h // 2, 0, h // 2, a),
                 (0, a, h // 2, b), (h // 2, a, h // 2, b)]
    if transpose:
        h, w = w, h
        rects = [(c, r, cw, rh) for r, c, rh, cw in rects]
    return (h, w), rects


def _field(rng, shape, payload):
    out = np.empty(shape, dtype=np.int8)
    style = int(rng.integers(4))
    k, offset = len(payload), int(rng.integers(len(payload)))
    for orbit in _orbits(*shape):
        r, c = orbit[0]
        if style == 0:
            index = int(rng.integers(k))
        elif style == 1:
            index = (r // 2 + c // 2 + offset) % k
        elif style == 2:
            index = (max(r, c) + offset) % k
        else:
            index = (r + 2 * c // 3 + offset) % k
        if rng.random() < .18:
            index = int(rng.integers(k))
        for p in orbit:
            out[p] = payload[index]
    return out


def _construct(rng, count, transpose, harder, teach):
    colors = [int(x) for x in rng.permutation(10)]
    bg, reg, bands = colors[0], colors[1], tuple(colors[2:6])
    payload = colors[6:6 + int(rng.integers(2, 5))]
    shape, rects = _rectangles(rng, count, transpose)
    if teach:
        # Both keys are invariant under a local left/right flip. Flipping
        # BOTH equal strips preserves global mirror agreement but changes the
        # picture. The tabs, rather than an incidental key offset, resolve it.
        length = int(rng.integers(8, 13))
        shape = (length, 10)
        rects = [(0, 0, length, 5), (0, 5, length, 5)]
        if transpose:
            shape = shape[::-1]
            rects = [(c, r, cw, rh) for r, c, rh, cw in rects]
    h, w = shape
    field = _field(rng, shape, payload)
    if len(np.unique(field)) < 2:
        return None
    stencil = np.zeros(shape, dtype=bool)
    kinds = [int(x) for x in rng.permutation(4)]
    repeated = None
    for i, (r, c, rh, cw) in enumerate(rects):
        if teach:
            points = {p: reg for p in KEYS[4 if i == 0 else 2]}
            ph = 1 + max(y for y, x in points)
            pw = 1 + max(x for y, x in points)
            length = cw if transpose else rh
            dr, dc = int(rng.integers(1, length - ph)), (5 - pw) // 2
            if transpose:
                points = {(x, y): v for (y, x), v in points.items()}
                dr, dc = dc, dr
        elif count >= 3 and i >= count - 2:
            if repeated is not None:
                points, dr, dc = repeated
            else:
                points = _transform({p: reg for p in KEYS[kinds[-1]]}, int(rng.integers(8)))
                ph, pw = 1 + max(y for y, x in points), 1 + max(x for y, x in points)
                dr, dc = int(rng.integers(1, rh - ph)), int(rng.integers(1, cw - pw))
                repeated = points, dr, dc
        else:
            points = _transform({p: reg for p in KEYS[kinds[i]]}, int(rng.integers(8)))
            ph, pw = 1 + max(y for y, x in points), 1 + max(x for y, x in points)
            dr, dc = int(rng.integers(1, rh - ph)), int(rng.integers(1, cw - pw))
        for y, x in points:
            stencil[r + dr + y, c + dc + x] = True
    # Internal missing cells keep each packet's outline explicitly readable.
    eligible = np.zeros(shape, dtype=bool)
    for r, c, rh, cw in rects:
        eligible[r + 1:r + rh - 1, c + 1:c + cw - 1] = True
    eligible &= ~stencil
    mask = (rng.random(shape) < (rng.uniform(.10, .20) if harder else rng.uniform(.025, .09))) & eligible
    available = [o for o in _orbits(h, w) if len(o) == 4 and all(eligible[p] for p in o)]
    if not available:
        return None
    rng.shuffle(available)
    for orbit in available[:int(rng.integers(1, 4 if harder else 3))]:
        survivor = orbit[int(rng.integers(4))]
        for p in orbit:
            mask[p] = p != survivor
    for orbit in _orbits(h, w):
        if all(stencil[p] or mask[p] for p in orbit):
            can_reveal = [p for p in orbit if not stencil[p]]
            if not can_reveal:
                return None
            mask[can_reveal[int(rng.integers(len(can_reveal)))]] = False
    data = field.copy()
    data[stencil], data[mask] = reg, bg
    gh, gw = int(rng.integers(27, 31)), int(rng.integers(27, 31))
    top = int(rng.integers(1, gh - h - 2))
    left = int(rng.integers(1, gw - w - 2))
    bottom, right = top + h + 1, left + w + 1
    inp = np.full((gh, gw), bg, dtype=np.int8)
    inp[top, left:right + 1], inp[bottom, left:right + 1] = bands[0], bands[2]
    inp[top + 1:bottom, left], inp[top + 1:bottom, right] = bands[3], bands[1]
    inp[top + 1:bottom, left + 1:right][stencil] = reg
    blocked = np.zeros((gh, gw), dtype=bool)
    blocked[max(0, top - 1):bottom + 2, max(0, left - 1):right + 2] = True
    for i, (r, c, rh, cw) in enumerate(rects):
        body = data[r:r + rh, c:c + cw]
        cells = {(y, x): int(body[y, x]) for y in range(rh) for x in range(cw) if body[y, x] != bg}
        d = int(rng.integers(4))
        for side in (d, (d + 1) % 4):
            length = cw if side % 2 == 0 else rh
            along = int(rng.integers(1, length - 1))
            p = ((-1, along), (along, cw), (rh, along), (along, -1))[side]
            cells[p] = bands[side]
        t = int(rng.integers(8))
        if teach and i == 0:
            t = int(rng.integers(1, 4))
        elif teach and i == 1:
            t = int(rng.integers(4, 8))
        cells = _transform(cells, t)
        bh, bw = 1 + max(y for y, x in cells), 1 + max(x for y, x in cells)
        origins = [(y, x) for y in range(1, gh - bh) for x in range(1, gw - bw)]
        rng.shuffle(origins)
        for y, x in origins:
            if not blocked[y:y + bh, x:x + bw].any():
                for (dy, dx), v in cells.items():
                    inp[y + dy, x + dx] = v
                blocked[y - 1:y + bh + 1, x - 1:x + bw + 1] = True
                break
        else:
            return None
    expected = np.full_like(inp, bg)
    expected[top:bottom + 1, left:right + 1] = inp[top:bottom + 1, left:right + 1]
    expected[top + 1:bottom, left + 1:right] = field
    try:
        predicted, info = _analyze(inp)
    except ValueError:
        return None
    if not np.array_equal(predicted, expected) or not info["diagonal"] or not info["cross"]:
        return None
    if count >= 3 and info["assemblies"] < 2:
        return None
    if teach and not _compass_needed(info):
        return None
    return {"input": inp, "output": expected}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer(num_train, "num_train", 2, 10)
    ns = _integer(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    orientation = bool(rng.integers(2))
    count_shift = bool(rng.integers(2))
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            if split == "train":
                pieces = 2 if i == 0 else 3 if i == 1 else int(rng.integers(2, 4 if count_shift else 5))
            else:
                pieces = 4 if count_shift else 3
            for _ in range(120):
                pair = _construct(rng, pieces, orientation if split == "train" else not orientation,
                                  split == "test", split == "train" and i == 0)
                if pair is not None and _key(pair["input"]) not in used:
                    break
            else:
                raise RuntimeError("bounded mosaic construction exhausted")
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, ambiguous_training, compass_training, nontrivial_poses = set(), False, False, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            predicted, info = _analyze(inp)
            assert np.array_equal(predicted, out), "wrong docking or mirror repair"
            assert info["diagonal"] > 0 and info["cross"] > 0
            art = info["completed"]
            assert np.array_equal(art, art[::-1]) and np.array_equal(art, art[:, ::-1])
            assert not set(np.unique(art)) & {info["bg"], info["reg"], *info["bands"]}
            assert 2 <= len(np.unique(art)) <= 4
            assert not np.array_equal(inp, out)
            if split == "train":
                ambiguous_training |= info["assemblies"] > 1
                if not compass_training:
                    compass_training = _compass_needed(info)
                nontrivial_poses.update(info["poses"])
            else:
                assert info["assemblies"] > 1, "tests must couple docking and symmetry"
    assert ambiguous_training, "training must expose symmetry-dependent assignment"
    assert compass_training, "training must expose compass-dependent orientation"
    assert any(0 < t < 4 for t in nontrivial_poses)
    assert any(t >= 4 for t in nontrivial_poses)


def self_test():
    """42 seeds, semantic ablations, determinism, count/error boundaries; no files."""
    shapes, art_shapes, backgrounds, palettes, stencils, poses, signatures = set(), set(), set(), set(), set(), set(), set()
    sizes, ambiguities, diagonals, cross_tiles = set(), [], 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        sig = tuple(_key(q[k]) for s in ("train", "test") for q in p[s] for k in ("input", "output"))
        assert sig not in signatures
        signatures.add(sig)
        if seed in (0, 21, 41):
            repeat = generate_puzzle(seed)
            assert sig == tuple(_key(q[k]) for s in ("train", "test") for q in repeat[s] for k in ("input", "output"))
        for split in ("train", "test"):
            for q in p[split]:
                before = q["input"].copy()
                predicted, info = _analyze(before)
                assert np.array_equal(before, q["input"]) and np.array_equal(predicted, q["output"])
                shapes.add(before.shape)
                art_shapes.add(info["shape"])
                backgrounds.add(info["bg"])
                palettes.add(tuple(int(c) for c in np.unique(info["completed"])))
                poses.update(info["poses"])
                sizes.add(len(info["sources"]))
                for target in info["targets"]:
                    norm = _transform({x: 1 for x in target}, 0)
                    stencils.add(min(tuple(sorted(_transform(norm, t))) for t in range(8)))
                ambiguities.append(info["assemblies"])
                diagonals += info["diagonal"]
                cross_tiles += info["cross"]
                # B-only repair preserves the scrambled exterior. A-only docking
                # leaves unknowns and, for twins, has competing exact covers.
                assert all(predicted[pt] == info["bg"] for s in info["sources"] for pt in s)
                assert np.any(info["raw"] < 0)
    assert sizes == {2, 3, 4} and poses == set(range(8))
    assert len(shapes) >= 12 and len(art_shapes) >= 16
    assert backgrounds == set(range(10)) and len(palettes) >= 30 and len(stencils) == 5
    assert sum(x > 1 for x in ambiguities) >= 100
    for nt in range(2, 11):
        for ns in range(1, min(nt - 1, 4) + 1):
            validate_generated_puzzle(generate_puzzle(8000 + 10 * nt + ns, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(123, 2, 1)
    _, info = _analyze(bad["test"][0]["input"])
    top, _, left, _ = info["frame"]
    bad["test"][0]["output"][top + 1, left + 1] = info["reg"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": len(ambiguities), "workspace_shapes": len(shapes),
            "artwork_shapes": len(art_shapes), "backgrounds": len(backgrounds),
            "payload_palettes": len(palettes), "key_geometries": len(stencils),
            "piece_counts": sorted(sizes), "poses": sorted(poses),
            "ambiguous_without_symmetry": sum(x > 1 for x in ambiguities),
            "diagonal_only_cells": diagonals, "cross_tile_only_cells": cross_tiles}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
