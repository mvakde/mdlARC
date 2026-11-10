"""Restore first-seen columns, then propagate terminal paint into closed loops.

Only NumPy and the standard library are used. No parent generator is imported.
Generation constructs owned interiors; solve independently recovers visible
systems and holes. Diagnostics are bounded and self_test writes no files.
"""
from collections import deque
import hashlib
import json

import numpy as np


D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
D8 = D4 + ((-1, -1), (-1, 1), (1, 1), (1, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)
            and 2 <= num_train <= 10 and 1 <= num_test <= 4
            and num_train > num_test):
        raise ValueError("require integer counts 2..10 train, 1..4 test, train > test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if not (a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
            and np.issubdtype(a.dtype, np.integer)
            and np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _neighbors(cells, shape, steps=D8):
    h, w = shape
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps
            if 0 <= r + dr < h and 0 <= c + dc < w} - set(cells)


def _components(cells, steps=D8):
    left = set(cells)
    result = []
    while left:
        root = min(left)
        left.remove(root)
        group, todo = {root}, [root]
        for r, c in todo:
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in left:
                    left.remove(q)
                    group.add(q)
                    todo.append(q)
        result.append(group)
    return result


def _column_indices(a):
    seen, indices = set(), []
    for c in range(a.shape[1]):
        key = tuple(int(v) for v in a[:, c])
        if key not in seen:
            seen.add(key)
            indices.append(c)
    return indices


def _compact(a):
    return a[:, _column_indices(a)].copy()


def _cups(a, bg):
    """A terminal occupies one corner of a 2x2, with three wire neighbors.

    Check the complete eight-neighborhood, so an incidental colored corner
    with another foreground neighbor is not a terminal.
    """
    found = {}
    h, w = a.shape
    for r in range(1, h - 1):
        for c in range(w):
            if a[r, c] == bg:
                continue
            edge = _neighbors({(r, c)}, a.shape)
            occupied = {q for q in edge if a[q] != bg}
            if len(occupied) != 3:
                continue
            colors = {int(a[q]) for q in occupied}
            if len(colors) != 1 or int(a[r, c]) in colors:
                continue
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                if occupied == {(r + dr, c), (r, c + dc), (r + dr, c + dc)}:
                    found[(r, c)] = int(a[r, c])
                    break
    return found


def _paint(a, *, steps=D8, strict=True):
    """Recover all ownership from a compact canvas, without generation data."""
    bg = int(a[0, 0])
    _require(np.all(a[0] == bg) and np.all(a[-1] == bg), "missing background rows")
    systems = _components(zip(*np.where(a != bg)), steps)
    owner = {p: i for i, cells in enumerate(systems) for p in cells}
    cups = _cups(a, bg)
    supplies = [set() for _ in systems]
    for p, color in cups.items():
        supplies[owner[p]].add(color)
    if strict:
        _require(all(len(s) <= 1 for s in supplies), "conflicting terminal colors")
    holes = []
    out = a.copy()
    for cells in _components(zip(*np.where(a == bg)), D4):
        if any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1) for r, c in cells):
            continue
        adjacent = _neighbors(cells, a.shape, D4)
        ids = {owner[p] for p in adjacent if p in owner}
        if strict:
            _require(len(ids) == 1, "ambiguous cavity ownership")
        if len(ids) != 1:
            continue
        index = next(iter(ids))
        paint = next(iter(supplies[index])) if len(supplies[index]) == 1 else None
        holes.append((cells, index, paint))
        if paint is not None:
            for p in cells:
                out[p] = paint
    return out, {"bg": bg, "systems": systems, "cups": cups,
                 "supplies": supplies, "holes": holes}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """The fixed family rule; no examples, seed, caches, or hidden parameters."""
    return _paint(_compact(_grid(input_grid)))[0]


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _translate(cells, dr, dc):
    return {(r + dr, c + dc) for r, c in cells}


def _shape(rng):
    """Small connected interiors, including drifting, waisted and bent profiles."""
    h, w = _ri(rng, 1, 4), _ri(rng, 1, 4)
    kind = _ri(rng, 0, 3)
    cells, left = set(), 0
    for r in range(h):
        if r and kind in (1, 3):
            left += _ri(rng, -1, 1)
        width = max(1, w - (kind == 2 and 0 < r < h - 1))
        # Width >= 2 for drifting profiles guarantees adjacent row overlap.
        if kind in (1, 3):
            width = max(2, width)
        cells.update((r, c) for c in range(left, left + width))
    if _ri(rng, 0, 1):
        cells = {(c, r) for r, c in cells}
    shell = {(r + dr, c + dc) for r, c in cells for dr, dc in D4} - cells
    r0, c0 = min(r for r, c in shell), min(c for r, c in shell)
    return _translate(cells, -r0, -c0), _translate(shell, -r0, -c0)


def _route(rng, starts, goals, allowed):
    order = [D8[int(i)] for i in rng.permutation(8)]
    starts = sorted(starts)
    rng.shuffle(starts)
    todo = deque(starts)
    prev = {p: None for p in starts}
    end = None
    while todo:
        p = todo.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in order:
            q = (p[0] + dr, p[1] + dc)
            if q in allowed and q not in prev:
                prev[q] = p
                todo.append(q)
    if end is None:
        return None
    path = set()
    while end is not None:
        path.add(end)
        end = prev[end]
    return path


def _scene(rng, lesson):
    """Construct sparse geometry with independently known cavity ownership."""
    # Both mandatory lessons have unsupplied loops; lesson 1 also has two paints.
    if lesson == 0:
        counts, active = [2, 1], 1
    elif lesson == 1:
        counts, active = [1, 1, 1], 2
    else:
        active = _ri(rng, 1, 2)
        inactive = _ri(rng, 0, 1)
        counts = [_ri(rng, 1, 2) for _ in range(active + inactive)]
        while sum(counts) > 4:
            counts[counts.index(max(counts))] -= 1
    h, w = _ri(rng, 18, 30), _ri(rng, 9, 16)
    palette = [int(v) for v in rng.permutation(10)]
    bg, boundary = palette[:2]
    wires = palette[2:2 + len(counts)]
    paints = palette[2 + len(counts):2 + len(counts) + active]
    # Some later scenes reuse a paint on disconnected marked systems.
    if lesson not in (0, 1) and active == 2 and rng.random() < .2:
        paints[1] = paints[0]
    a = np.full((h, w), bg, dtype=np.int8)
    regions, occupied = [], set()
    owners = [i for i, n in enumerate(counts) for _ in range(n)]
    rng.shuffle(owners)
    for owner in owners:
        inside, shell = _shape(rng)
        hh, ww = max(r for r, c in shell) + 1, max(c for r, c in shell) + 1
        for _ in range(60):
            dr, dc = _ri(rng, 1, h - hh - 1), _ri(rng, 0, w - ww)
            ins, edge = _translate(inside, dr, dc), _translate(shell, dr, dc)
            block = ins | edge
            if not (block | _neighbors(block, a.shape)) & occupied:
                regions.append((ins, edge, owner))
                occupied.update(block)
                break
        else:
            return None
    wire_cells = [set() for _ in counts]
    sources = []
    for owner in range(active):
        for _ in range(60):
            r, c = _ri(rng, 1, h - 3), _ri(rng, 0, w - 2)
            square = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
            if (square | _neighbors(square, a.shape)) & occupied:
                continue
            p = sorted(square)[_ri(rng, 0, 3)]
            wire_cells[owner] = square - {p}
            sources.append((p, paints[owner]))
            occupied.update(square)
            break
        else:
            return None
    all_regions = set().union(*(ins | edge for ins, edge, _ in regions))
    source_cells = {p for p, _ in sources}
    canvas = {(r, c) for r in range(1, h - 1) for c in range(w)}
    # Unmarked contours can be detached, or carry an ordinary open tail.
    for owner in range(active, len(counts)):
        edge = next(edge for _, edge, i in regions if i == owner)
        candidates = sorted(_neighbors(edge, a.shape) & canvas - all_regions
                            - _neighbors(all_regions - edge, a.shape)
                            - source_cells - _neighbors(source_cells, a.shape))
        if candidates:
            wire_cells[owner] = {candidates[_ri(rng, 0, len(candidates) - 1)]}
    for owner in rng.permutation(len(counts)):
        owner = int(owner)
        own = wire_cells[owner]
        if not own:
            continue
        others = set().union(*(s for i, s in enumerate(wire_cells) if i != owner))
        other_edges = set().union(*(edge for _, edge, i in regions if i != owner))
        foreign = others | other_edges
        chain = rng.random() < .5
        starts = set(own)
        for index in rng.permutation(len(regions)):
            ins, edge, i = regions[int(index)]
            if i != owner:
                continue
            other_regions = all_regions - ins - edge
            if chain:
                other_regions -= starts
            blocked = (all_regions | _neighbors(other_regions, a.shape)
                       | foreign | _neighbors(foreign, a.shape)
                       | source_cells | _neighbors(source_cells, a.shape))
            allowed = canvas - blocked
            goals = _neighbors(edge, a.shape) & allowed
            if not goals:
                return None
            # Choose an attachment side independently of source color/location.
            axis, extreme = _ri(rng, 0, 1), _ri(rng, 0, 1)
            value = (max if extreme else min)(p[axis] for p in goals)
            preferred = {p for p in goals if p[axis] == value}
            path = _route(rng, starts, preferred, allowed | starts)
            if path is None:
                path = _route(rng, starts, goals, allowed | starts)
            if path is None:
                return None
            own.update(path - all_regions)
            # Some routes use a contour as a relay; others branch from the wire.
            starts = edge if chain else set(own)
        wire_cells[owner] = own
    for ins, edge, owner in regions:
        for p in edge:
            a[p] = boundary
    for owner, cells in enumerate(wire_cells):
        for p in cells:
            a[p] = wires[owner]
    for p, paint in sources:
        a[p] = paint
    expected = a.copy()
    for ins, edge, owner in regions:
        if owner < active:
            for p in ins:
                expected[p] = paints[owner]
    # Empty padding carries no rule evidence. Retain only 1..3 blank rows at
    # either end, subject to the original canvas boundaries.
    rows = np.where(np.any(a != bg, axis=1))[0]
    top = max(0, int(rows[0]) - _ri(rng, 1, 3))
    bottom = min(h, int(rows[-1]) + 1 + _ri(rng, 1, 3))
    a, expected = a[top:bottom].copy(), expected[top:bottom].copy()
    # Naturally redundant rectangle columns are removed during construction.
    # Accept only when this preserves every intended hole and its ownership.
    keep = _column_indices(a)
    a, expected = a[:, keep].copy(), expected[:, keep].copy()
    if not 6 <= a.shape[1] <= 14:
        return None
    try:
        predicted, info = _paint(a)
    except AssertionError:
        return None
    if (len(info["cups"]) != active or len(info["holes"]) != len(regions)
            or not np.array_equal(predicted, expected)):
        return None
    if len([s for s in info["supplies"] if s]) != active:
        return None
    if lesson == 0 and not any(sum(i == j for _, j, p in info["holes"]) >= 2
                               for i, s in enumerate(info["supplies"]) if s):
        return None
    if lesson == 1 and len({p for _, _, p in info["holes"] if p is not None}) != 2:
        return None
    if lesson in (0, 1) and not any(p is None for _, _, p in info["holes"]):
        return None
    if rng.random() < .5:
        a, expected = a[::-1].copy(), expected[::-1].copy()
    if rng.random() < .5:
        a, expected = a[:, ::-1].copy(), expected[:, ::-1].copy()
    return a, expected


def _expand(rng, a, *, test=False):
    """First occurrences retain the scene order; early copies defeat cropping."""
    w = a.shape[1]
    order = list(range(w))
    # A separated duplicate BEFORE the final new vector is mandatory.
    source = _ri(rng, 0, w - 3)
    position = _ri(rng, source + 2, w - 1)
    order.insert(position, source)
    if test and 2 * w + 2 <= 30:
        source2 = _ri(rng, 0, w - 2)
        position2 = _ri(rng, source2 + 1, len(order) - 1)
        order.insert(position2, source2)
    replay = list(range(w))
    style = _ri(rng, 0, 2)
    if style == 1 or test:
        replay.reverse()
    elif style == 2:
        rng.shuffle(replay)
    order.extend(replay)
    budget = 30 - len(order)
    if budget:
        extra = _ri(rng, 0, min(budget, 4 if test else 2))
        order.extend(_ri(rng, 0, w - 1) for _ in range(extra))
    return a[:, order].copy()


def _witnesses(pair):
    a, out = np.asarray(pair["input"]), np.asarray(pair["output"])
    keep = _column_indices(a)
    compact = a[:, keep]
    adjacent = [0] + [c for c in range(1, a.shape[1])
                      if not np.array_equal(a[:, c], a[:, c - 1])]
    raw_painted, raw_info = _paint(a, strict=False)
    compact_info = _paint(compact)[1]
    four = _paint(compact, steps=D4, strict=False)[0]
    return {"crop": not np.array_equal(a[:, :len(keep)], compact),
            "nonadjacent": len(adjacent) > len(keep),
            "order": not np.array_equal(raw_painted[:, keep], out),
            "eight": not np.array_equal(four, out),
            "extra_raw_cavities": (sum(p is not None for _, _, p in raw_info["holes"])
                                   > sum(p is not None for _, _, p in compact_info["holes"]))}


def _new_pair(rng, lesson, test, seen):
    for _ in range(320):
        scene = _scene(rng, lesson)
        if scene is None:
            continue
        compact, expected = scene
        for _ in range(12):
            a = _expand(rng, compact, test=test)
            key = (a.shape, a.tobytes())
            if key in seen:
                continue
            pair = {"input": a, "output": expected.copy()}
            if lesson in (0, 1):
                witnesses = _witnesses(pair)
                if not (witnesses["order"] and witnesses["eight"]):
                    break
                if lesson == 0 and not witnesses["extra_raw_cavities"]:
                    continue
            seen.add(key)
            return pair
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    seen = set()
    train = [_new_pair(rng, i if i < 2 else 2, False, seen) for i in range(nt)]
    rng.shuffle(train)
    test = [_new_pair(rng, 2, True, seen) for _ in range(ne)]
    return {"train": train, "test": test}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, lessons, fill_colors = set(), [], set()
    multi, inactive, relay = False, False, False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            key = (a.shape, a.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            compact = _compact(a)
            _require(compact.shape[1] < a.shape[1], "A is vacuous")
            _require(compact.shape == b.shape, "output must preserve compacted dimensions")
            # Every column has a full later copy, not merely isolated run repeats.
            keys = [tuple(int(v) for v in a[:, c]) for c in range(a.shape[1])]
            _require(all(keys.count(k) >= 2 for k in set(keys)), "missing exact column copies")
            prediction, info = _paint(compact)
            _require(np.array_equal(prediction, b), "input-derived solution disagrees")
            changed = compact != b
            _require(np.any(changed) and np.all(compact[changed] == info["bg"]), "B is vacuous or overwrites structure")
            _require(len(info["cups"]) >= 1, "no visible terminal")
            _require(set(np.unique(a)) == set(np.unique(b)), "colors must be copied")
            witness = _witnesses(pair)
            _require(witness["crop"] and witness["nonadjacent"], "missing global-dedup witness")
            if split == "train":
                lessons.append(witness)
                paints = {p for _, _, p in info["holes"] if p is not None}
                fill_colors.update(paints)
                multi |= len(paints) >= 2
                inactive |= any(p is None for _, _, p in info["holes"])
                relay |= any(sum(j == i for _, j, p in info["holes"]) >= 2
                             for i, s in enumerate(info["supplies"]) if s)
    _require(multi and inactive and relay and len(fill_colors) >= 2, "insufficient source/loop lessons")
    _require(any(w["order"] for w in lessons), "reversed causal order is not distinguished")
    _require(any(w["eight"] for w in lessons), "diagonal connectivity is not taught")
    _require(any(w["extra_raw_cavities"] for w in lessons), "omitting A must retain extra painted cavities")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 puzzles streamed through checks; no saved grids or execution traces."""
    fingerprints, shapes, palettes, structures = set(), set(), set(), set()
    hole_counts, areas, backgrounds, signatures = set(), set(), set(), set()
    order_witnesses = diagonal_witnesses = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        if seed < 2:
            _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterminism")
        for pair in puzzle["train"] + puzzle["test"]:
            a, b = pair["input"], pair["output"]
            compact = _compact(a)
            prediction, info = _paint(compact)
            _require(np.array_equal(solve(a.tolist()), prediction), "list/array solver discrepancy")
            _require(np.array_equal(solve(a), b), "solution mismatch")
            shapes.add((a.shape, b.shape))
            palettes.add(tuple(int(v) for v in np.unique(a)))
            roles = {}
            normalized = bytes(roles.setdefault(int(v), len(roles)) for v in compact.flat)
            structures.add((compact.shape, hashlib.sha256(normalized).hexdigest()))
            backgrounds.add(info["bg"])
            hole_counts.add(len(info["holes"]))
            areas.update(len(cells) for cells, _, _ in info["holes"])
            signatures.add((len(info["holes"]), len(info["cups"]),
                            sum(p is None for _, _, p in info["holes"])))
            witness = _witnesses(pair)
            order_witnesses += witness["order"]
            diagonal_witnesses += witness["eight"]
    _require(len(fingerprints) == 42, "duplicate seeded puzzles")
    _require(len(shapes) >= 60 and len(structures) >= 150, "insufficient geometric variation")
    _require(len(palettes) >= 30 and len(backgrounds) >= 8, "insufficient palette variation")
    _require(len(hole_counts) >= 3 and len(areas) >= 6 and len(signatures) >= 5, "insufficient topology variation")
    _require(order_witnesses >= 84 and diagonal_witnesses >= 84, "causal witnesses missing")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(0, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A partial fill and a damaged structural cell must both be rejected.
    original = generate_puzzle(0, 2, 1)
    for damage in ("partial", "structure"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in original[s]]
               for s in ("train", "test")}
        pair = bad["test"][0]
        compact = _compact(pair["input"])
        cells = np.argwhere(compact != pair["output"] if damage == "partial"
                            else compact != compact[0, 0])
        r, c = cells[0]
        pair["output"][r, c] = compact[r, c] if damage == "partial" else compact[0, 0]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("damaged answer accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(fingerprints),
            "shape_pairs": len(shapes), "normalized_compact_scenes": len(structures),
            "palettes": len(palettes), "backgrounds": sorted(backgrounds),
            "cavity_counts": sorted(hole_counts), "cavity_areas": sorted(areas),
            "topology_signatures": len(signatures), "order_witnesses": order_witnesses,
            "diagonal_witnesses": diagonal_witnesses,
            "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
