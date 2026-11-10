"""Porous reservoirs: local plurality determines the capacity of inward layers.

Only NumPy and the standard library are used. Generation constructs labels from
its placement plan; solve reconstructs frames, supplies, votes and capacities
from the input. No palette or seed is needed by solve.
"""

from __future__ import annotations

import numpy as np


EDGES = ("top", "bottom", "left", "right")


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        result.append(cells)
    return result


def _mode(a):
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) < 2 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("a reservoir needs at least two colors and one unique mode")
    return int(colors[np.argmax(counts)])


def _front(shape, edge, depth):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if edge == "top":
        mask[:depth, :] = True
    elif edge == "bottom":
        mask[h - depth:, :] = True
    elif edge == "left":
        mask[:, :depth] = True
    else:
        mask[:, w - depth:] = True
    return mask


def _capacities(core, mode, edge):
    eligible = core != mode
    q = eligible.sum(axis=1 if edge in EDGES[:2] else 0)
    return q[::-1] if edge in ("bottom", "right") else q


def _ray(shape, rect, edge, side):
    """Continue an interior edge, starting just beyond its one-cell frame."""
    r, c, h, w = rect
    H, W = shape
    if edge in EDGES[:2]:
        y = r if edge == "top" else r + h - 1
        xs = range(c - 2, -1, -1) if side == 0 else range(c + w + 1, W)
        return [(y, x) for x in xs]
    x = c if edge == "left" else c + w - 1
    ys = range(r - 2, -1, -1) if side == 0 else range(r + h + 1, H)
    return [(y, x) for y in ys]


def _decode_component(a, cells):
    cellset = set(cells)
    candidates = []
    for marker in sorted({int(a[p]) for p in cells}):
        core_cells = [p for p in cells if a[p] != marker]
        if not core_cells:
            continue
        rs, cs = zip(*core_cells)
        r, c = min(rs), min(cs)
        h, w = max(rs) - r + 1, max(cs) - c + 1
        if h < 2 or w < 2 or len(core_cells) != h * w:
            continue
        if r == 0 or c == 0 or r + h >= a.shape[0] or c + w >= a.shape[1]:
            continue
        core = a[r:r + h, c:c + w]
        if np.any((core == 0) | (core == marker)):
            continue
        frame = {(y, x) for y in range(r - 1, r + h + 1)
                 for x in range(c - 1, c + w + 1)
                 if y in (r - 1, r + h) or x in (c - 1, c + w)}
        if not frame <= cellset or any(a[p] != marker for p in frame):
            continue
        extra = cellset - frame - set(core_cells)
        if not extra or any(a[p] != marker for p in extra):
            continue
        matches = []
        for edge in EDGES:
            sides = []
            for side in (0, 1):
                run = []
                for p in _ray(a.shape, (r, c, h, w), edge, side):
                    if p not in extra:
                        break
                    run.append(p)
                sides.append(run)
            if set(sides[0] + sides[1]) == extra:
                matches.append((edge, sides))
        if len(matches) != 1:
            continue
        edge, sides = matches[0]
        try:
            mode = _mode(core)
        except ValueError:
            continue
        capacity = _capacities(core, mode, edge)
        if np.any(capacity == 0):
            continue
        quota = len(extra)
        depths = np.flatnonzero(np.cumsum(capacity) == quota)
        if len(depths) != 1:
            continue
        depth = int(depths[0]) + 1
        candidates.append({"rect": (r, c, h, w), "marker": marker,
                           "mode": mode, "edge": edge, "depth": depth,
                           "capacity": tuple(map(int, capacity)), "quota": quota,
                           "bars": tuple(sorted(extra)), "frame": frame,
                           "side_lengths": tuple(map(len, sides))})
    if len(candidates) != 1:
        raise ValueError("each component must have one unambiguous framed reservoir and supply")
    return candidates[0]


def _decode(a):
    components = _components(a != 0)
    if not components:
        raise ValueError("no reservoirs")
    labels = np.zeros(a.shape, dtype=int)
    for index, cells in enumerate(components, 1):
        for p in cells:
            labels[p] = index
    for first, second in ((labels[:-1, :-1], labels[1:, 1:]),
                          (labels[:-1, 1:], labels[1:, :-1])):
        if np.any((first > 0) & (second > 0) & (first != second)):
            raise ValueError("different reservoirs must not touch diagonally")
    return [_decode_component(a, cells) for cells in components]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every local role and fill frontier from the input alone."""
    a = _grid(input_grid)
    out = a.copy()
    for info in _decode(a):
        r, c, h, w = info["rect"]
        for p in info["bars"]:
            out[p] = 0
        mask = _front((h, w), info["edge"], info["depth"])
        mask &= a[r:r + h, c:c + w] != info["mode"]
        out[r:r + h, c:c + w][mask] = info["marker"]
    return out


def _counts(rng, area, k, plurality):
    low = (area + 2 * k - 2) // k
    high = area // 2 if plurality else area - k + 1
    if low > high:
        return None
    if plurality:
        top = int(rng.integers(low, high + 1))
    elif max(low, area // 2 + 1) <= high and rng.random() < 0.4:
        top = int(rng.integers(max(low, area // 2 + 1), high + 1))
    else:
        top = int(rng.integers(low, high + 1))
    other = np.ones(k - 1, dtype=int)
    for _ in range(area - top - k + 1):
        choices = np.flatnonzero(other < top - 1)
        other[int(rng.choice(choices))] += 1
    rng.shuffle(other)
    return [top] + other.tolist()


def _arrange(rng, h, w, colors, counts, style):
    positions = [(r, c) for r in range(h) for c in range(w)]
    if style == 0:
        rng.shuffle(positions)
    elif style == 1:
        if rng.integers(2):
            positions.reverse()
    elif style == 2:
        positions.sort(key=lambda p: (p[1], p[0]))
        if rng.integers(2):
            positions.reverse()
    else:
        # A compact minority island competes with a spatially fragmented mode.
        j = 1 + int(np.argmax(counts[1:]))
        anchor = (int(rng.integers(h)), int(rng.integers(w)))
        rng.shuffle(positions)
        positions.sort(key=lambda p: abs(p[0] - anchor[0]) + abs(p[1] - anchor[1]))
        island, rest = positions[:counts[j]], positions[counts[j]:]
        a = np.full((h, w), -1, dtype=np.int8)
        for p in island:
            a[p] = colors[j]
        rng.shuffle(rest)
        phase = int(rng.integers(2))
        rest.sort(key=lambda p: (p[0] + p[1] + phase) % 2)
        for p in rest[:counts[0]]:
            a[p] = colors[0]
        tail = [colors[i] for i in range(1, len(colors)) if i != j for _ in range(counts[i])]
        rng.shuffle(tail)
        for p, value in zip(rest[counts[0]:], tail):
            a[p] = value
        return a
    a = np.empty((h, w), dtype=np.int8)
    values = [color for color, count in zip(colors, counts) for _ in range(count)]
    for p, value in zip(positions, values):
        a[p] = value
    return a


def _fragment_counterexample(core, mode):
    modal = _components(core == mode)
    biggest_other = max(len(g) for color in np.unique(core) if color != mode
                        for g in _components(core == color))
    return len(modal) > 1 and max(map(len, modal)) < biggest_other


def _reservoir(rng, marker, mode, pool, edge, *, anchor=0, test=False,
               split=None, full=None, all_colors=False):
    for _ in range(700):
        if anchor:
            h, w = map(int, rng.integers(3, 6, size=2))
        else:
            h, w = map(int, rng.integers(2, 8 if test else 7, size=2))
        others = [x for x in pool if x != mode and x != marker]
        rng.shuffle(others)
        k = len(others) + 1 if all_colors else int(rng.integers(2, len(others) + 2))
        colors = [mode] + others[:k - 1]
        counts = _counts(rng, h * w, k, plurality=anchor == 1)
        if counts is None:
            continue
        style = 3 if anchor == 1 else int(rng.integers(4))
        core = _arrange(rng, h, w, colors, counts, style)
        if anchor == 1 and not _fragment_counterexample(core, mode):
            continue
        capacity = _capacities(core, mode, edge)
        if np.any(capacity == 0):
            continue
        max_depth = len(capacity)
        if full is True:
            depth = max_depth
        elif full is False:
            depth = int(rng.integers(1, max_depth))
        else:
            depth = int(rng.integers(1, max_depth + 1))
        front = _front((h, w), edge, depth)
        if not np.any(front & (core == mode)):
            continue
        quota = int(capacity[:depth].sum())
        edge_length = w if edge in EDGES[:2] else h
        if anchor == 1 and quota % edge_length == 0:
            continue
        horizontal = edge in EDGES[:2]
        if (w + 2 + quota if horizontal else h + 2 + quota) > 30:
            continue
        use_split = bool(rng.integers(2)) if split is None else split
        if use_split and quota < 2:
            continue
        before = int(rng.integers(1, quota)) if use_split else (quota if rng.integers(2) else 0)
        if horizontal:
            patch = np.zeros((h + 2, w + 2 + quota), dtype=np.int8)
            r, c = 1, before + 1
        else:
            patch = np.zeros((h + 2 + quota, w + 2), dtype=np.int8)
            r, c = before + 1, 1
        patch[r - 1:r + h + 1, c - 1:c + w + 1] = marker
        patch[r:r + h, c:c + w] = core
        # Unary supply runs terminate at the outer face of the frame.
        bars = (_ray(patch.shape, (r, c, h, w), edge, 0)[:before]
                + _ray(patch.shape, (r, c, h, w), edge, 1)[:quota - before])
        for p in bars:
            patch[p] = marker
        expected = patch.copy()
        for p in bars:
            expected[p] = 0
        expected[r:r + h, c:c + w][front & (core != mode)] = marker
        return patch, expected
    raise RuntimeError("could not sample a reservoir with the requested evidence")


def _pack(rng, patches):
    max_h = max(p[0].shape[0] for p in patches)
    max_w = max(p[0].shape[1] for p in patches)
    for _ in range(100):
        if len(patches) == 1:
            H = min(30, max_h + int(rng.integers(7)))
            W = min(30, max_w + int(rng.integers(7)))
        else:
            H = int(rng.integers(max(12, max_h), 31))
            W = int(rng.integers(max(12, max_w), 31))
        inp, out = np.zeros((H, W), dtype=np.int8), np.zeros((H, W), dtype=np.int8)
        blocked = np.zeros((H, W), dtype=bool)
        order = list(rng.permutation(len(patches)))
        order.sort(key=lambda i: -np.count_nonzero(patches[i][0]))
        for i in order:
            p, q = patches[i]
            h, w = p.shape
            mask = p != 0
            placed = False
            for _ in range(180):
                r = int(rng.integers(H - h + 1))
                c = int(rng.integers(W - w + 1))
                if np.any(blocked[r:r + h, c:c + w] & mask):
                    continue
                inp[r:r + h, c:c + w][mask] = p[mask]
                out[r:r + h, c:c + w][mask] = q[mask]
                occupied = inp != 0
                blocked = occupied.copy()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ys, xs = max(0, dy), max(0, dx)
                        ye, xe = min(H, H + dy), min(W, W + dx)
                        blocked[ys:ye, xs:xe] |= occupied[ys - dy:ye - dy, xs - dx:xe - dx]
                placed = True
                break
            if not placed:
                break
        else:
            return {"input": inp, "output": out}
    return None


def _integer(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must lie in {low}..{high}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _integer("num_train", num_train, 2, 10)
    num_test = _integer("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(1, 10))
    pool = [c for c in range(1, 10) if c != marker]
    rng.shuffle(pool)
    pool = pool[:int(rng.integers(3, 6))]
    a, b = pool[:2]
    mixed = bool(rng.integers(2))
    alternate_marker = int(rng.choice([c for c in range(1, 10) if c != marker and c not in pool]))
    train_limit = int(rng.integers(2, 4))
    anchor_counts = [int(rng.integers(1, 3)), int(rng.integers(1, 3))]
    if anchor_counts == [1, 1]:
        anchor_counts[int(rng.integers(2))] = 2
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(80):
                anchor = index + 1 if split == "train" and index < 2 else 0
                n = anchor_counts[index] if anchor else (int(rng.integers(1, train_limit + 1))
                    if split == "train" else int(rng.integers(train_limit + 1, 5)))
                if anchor == 1:
                    edges = [EDGES[int(rng.integers(2))]]
                elif anchor == 2:
                    edges = [EDGES[2 + int(rng.integers(2))]]
                elif split == "test":
                    edges = [EDGES[int(rng.integers(2))], EDGES[2 + int(rng.integers(2))]]
                else:
                    edges = []
                edges += [EDGES[int(rng.integers(4))] for _ in range(n - len(edges))]
                patches = []
                for j, edge in enumerate(edges):
                    local_marker = marker
                    local_pool = pool
                    mode = a if anchor == 1 else b if anchor == 2 else int(rng.choice(pool))
                    if anchor and j > 0:
                        mode = b if anchor == 1 else a
                        if mixed:
                            local_marker = alternate_marker
                    elif not anchor and mixed and rng.random() < 0.65:
                        local_marker = int(rng.integers(1, 10))
                        available = [c for c in range(1, 10) if c != local_marker]
                        rng.shuffle(available)
                        local_pool = available[:int(rng.integers(2, 6))]
                        mode = int(rng.choice(local_pool))
                    patches.append(_reservoir(rng, local_marker, mode, local_pool, edge,
                        anchor=anchor if j == 0 else 0, test=split == "test",
                        split=(anchor == 2) if anchor and j == 0 else None,
                        full=(anchor == 2) if anchor and j == 0 else None,
                        all_colors=bool(anchor)))
                pair = _pack(rng, patches)
                if pair is None:
                    continue
                key = (pair["input"].shape, pair["input"].tobytes())
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not pack a distinct scene")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_facts(inp, out):
    a, b = _grid(inp), _grid(out)
    _require(a.shape == b.shape, "shape changed")
    infos = _decode(a)
    _require(np.array_equal(solve(a), b), "incorrect output")
    for info in infos:
        r, c, h, w = info["rect"]
        core, filled = a[r:r + h, c:c + w], b[r:r + h, c:c + w]
        mode, marker = info["mode"], info["marker"]
        _require(np.all(filled[core == mode] == mode), "protected material moved")
        _require(all(b[p] == marker for p in info["frame"]), "frame changed")
        _require(all(b[p] == 0 for p in info["bars"]), "supply not erased")
        newly_filled = int(np.count_nonzero(filled == marker))
        _require(newly_filled == info["quota"], "supply area not conserved")
        _require(np.any(_front((h, w), info["edge"], info["depth"]) & (core == mode)),
                 "filled frontier does not demonstrate protection")
        colors, counts = np.unique(core, return_counts=True)
        info["plurality"] = int(counts.max()) * 2 <= core.size
        info["minorities"] = set(map(int, colors)) - {mode}
        info["fragmented"] = _fragment_counterexample(core, mode)
        info["full"] = info["depth"] == len(info["capacity"])
    return infos


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= tr <= 10 and 1 <= te <= 4 and tr > te, "illegal pair counts")
    seen, facts = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a = _grid(pair["input"])
            key = (a.shape, a.astype(np.int8).tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            facts[split].append(_pair_facts(a, pair["output"]))
    train = [info for scene in facts["train"] for info in scene]
    _require(any(i["edge"] in EDGES[:2] for i in train)
             and any(i["edge"] in EDGES[2:] for i in train), "training needs both axes")
    _require({i["full"] for i in train} == {False, True}, "training needs partial and full fills")
    _require(any(i["plurality"] and i["fragmented"] for i in train), "missing frequency evidence")
    _require(any(i["quota"] % (i["rect"][3] if i["edge"] in EDGES[:2] else i["rect"][2]) for i in train),
             "training never distinguishes porous from solid capacity")
    _require({all(i["side_lengths"]) for i in train} == {False, True}, "missing split/single supplies")
    _require(any(len({i["mode"] for i in scene}) >= 2 for scene in facts["train"]),
             "no scene demonstrates local votes")
    _require(any(i["mode"] in j["minorities"] and j["mode"] in i["minorities"]
                 for i in train for j in train), "missing color role reversal")
    max_train = max(map(len, facts["train"]))
    train_markers = {i["marker"] for i in train}
    all_markers = train_markers | {i["marker"] for scene in facts["test"] for i in scene}
    _require(len(all_markers) == 1 or len(train_markers) >= 2,
             "changing frame colors must be demonstrated in training")
    for scene in facts["test"]:
        _require(len(scene) > max_train, "test must combine more reservoirs than training")
        _require(any(i["edge"] in EDGES[:2] for i in scene)
                 and any(i["edge"] in EDGES[2:] for i in scene), "test must combine axes")
    return None


def _signature(puzzle):
    return tuple((s, tuple((p[k].shape, p[k].tobytes())
                          for p in puzzle[s] for k in ("input", "output")))
                 for s in ("train", "test"))


def self_test():
    # Hand-written specification witness, including all reflections/rotations.
    source = np.array([[int(c) for c in row] for row in
                       ("00088888", "88881128", "00082318", "00081238", "00088888")])
    target = np.array([[int(c) for c in row] for row in
                       ("00088888", "00081188", "00088818", "00081238", "00088888")])
    palette = np.array([0, 9, 4, 7, 2, 3, 5, 6, 1, 8])
    for recolor in (False, True):
        src, dst = (palette[source], palette[target]) if recolor else (source, target)
        for reflect in (False, True):
            for turns in range(4):
                a = np.rot90(np.fliplr(src) if reflect else src, turns).copy()
                b = np.rot90(np.fliplr(dst) if reflect else dst, turns).copy()
                original = a.copy()
                _require(np.array_equal(solve(a), b), "hand-written witness failed")
                _require(np.array_equal(a, original), "solve mutated its input")
    shapes, structures, sizes, markers, modes, edges = set(), set(), set(), set(), set(), set()
    quotas, capacities, counts, splits, regimes = set(), set(), set(), set(), set()
    mixed_scenes = boundary_scenes = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                facts = _pair_facts(a, b)
                shapes.add(a.shape)
                names = {}
                structures.add((a.shape, tuple(names.setdefault(int(x), len(names)) for x in a.flat)))
                counts.add((split, len(facts)))
                mixed_scenes += len({i["marker"] for i in facts}) > 1
                boundary_scenes += bool(np.any(a[0]) or np.any(a[-1]) or np.any(a[:, 0]) or np.any(a[:, -1]))
                for i in facts:
                    _, _, h, w = i["rect"]
                    sizes.add((h, w))
                    markers.add(i["marker"])
                    modes.add(i["mode"])
                    edges.add(i["edge"])
                    quotas.add(i["quota"])
                    capacities.add(i["capacity"])
                    splits.add(i["side_lengths"])
                    regimes.add(i["full"])
    _require(len(signatures) == 42 and len(structures) >= 180, "structural variation collapsed")
    _require(len(shapes) >= 50 and len(sizes) >= 20, "geometry variation collapsed")
    _require(len(markers) >= 8 and len(modes) >= 8 and edges == set(EDGES), "role variation collapsed")
    _require(len(quotas) >= 15 and len(capacities) >= 80 and len(splits) >= 30, "capacity variation collapsed")
    _require(regimes == {False, True} and mixed_scenes > 5 and boundary_scenes > 10, "scene variation collapsed")
    _require({("train", 1), ("train", 2), ("train", 3), ("test", 3), ("test", 4)} <= counts,
             "object-count variation collapsed")
    _require(_signature(generate_puzzle(19, 7, 3)) == _signature(generate_puzzle(19, 7, 3)),
             "determinism failed")
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            p = generate_puzzle(900 + tr * 10 + te, tr, te)
            _require((len(p["train"]), len(p["test"])) == (tr, te), "explicit counts ignored")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            continue
        raise AssertionError("illegal pair counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "structures": len(structures),
            "shapes": len(shapes), "interior_shapes": len(sizes), "quotas": len(quotas),
            "capacity_vectors": len(capacities), "mixed_marker_scenes": mixed_scenes}


if __name__ == "__main__":
    print(self_test())
