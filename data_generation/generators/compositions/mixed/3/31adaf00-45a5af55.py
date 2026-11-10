"""Maximal blank squares filled by an edge-anchored row palette.

No files, examples, seeds, or generator metadata enter solve().  All persistent
artifacts are the caller's responsibility; this module keeps no execution logs.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _arc_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return grid


def _decode(input_grid):
    """The bottom-left padding visibly identifies the wall/blocker color."""
    grid = _arc_grid(input_grid)
    blocker = int(grid[-1, 0])
    if blocker == 0:
        raise ValueError("the stack's lower padding must be nonblack")
    columns = np.flatnonzero(grid[0] == blocker)
    rows = np.flatnonzero(grid[:, 0] == blocker)
    if not len(columns) or not len(rows):
        raise ValueError("missing palette padding or separator")
    width, height = int(columns[0]), int(rows[0])
    if not (width >= 1 and width + 1 < grid.shape[1]
            and 2 <= height <= 16 and height < grid.shape[0]):
        raise ValueError("invalid palette panel dimensions")
    code = grid[:height, 0].copy()
    if (np.any(code == 0) or np.any(code == blocker)
            or code[-1] != code[-2]
            or not np.all(grid[:height, :width] == code[:, None])
            or not np.all(grid[height:, :width] == blocker)
            or not np.all(grid[:, width] == blocker)):
        raise ValueError("invalid row-color stack")
    field = grid[:, width + 1:]
    if np.any((field != 0) & (field != blocker)):
        raise ValueError("the field must contain only black and its blocker")
    return grid, blocker, width, code, field


def _candidates(mask):
    """All solid squares >=2 with NO complete one-row/column extension.

    Integral images make the complete-window and complete-side tests explicit.
    Grid edges forbid an extension, just as a blocker would.
    """
    h, w = mask.shape
    horizontal = np.pad(mask.astype(np.int32).cumsum(1), ((0, 0), (1, 0)))
    vertical = np.pad(mask.astype(np.int32).cumsum(0), ((1, 0), (0, 0)))
    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    result = []
    for side in range(2, min(h, w) + 1):
        total = (integral[side:, side:] - integral[:-side, side:]
                 - integral[side:, :-side] + integral[:-side, :-side])
        good = total == side * side
        row = horizontal[:, side:] - horizontal[:, :-side] == side
        col = vertical[side:, :] - vertical[:-side, :] == side
        good[1:, :] &= ~row[:h - side, :]
        good[:-1, :] &= ~row[side:, :]
        good[:, 1:] &= ~col[:, :w - side]
        good[:, :-1] &= ~col[:, side:]
        result.extend((int(r), int(c), side) for r, c in np.argwhere(good))
    return sorted(result)


def _union(shape, squares):
    used = np.zeros(shape, dtype=bool)
    for r, c, s in squares:
        if used[r:r + s, c:c + s].any():
            raise ValueError("overlapping qualifying squares are outside this family")
        used[r:r + s, c:c + s] = True
    return used


def _tile(side, layers, mode="cap"):
    edge = np.minimum(np.arange(side), np.arange(side)[::-1])
    depth = np.minimum(edge[:, None], edge[None, :])
    index = depth % len(layers) if mode == "cycle" else np.minimum(depth, len(layers) - 1)
    return np.asarray(layers, dtype=np.int8)[index]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete output solely from the visible input."""
    grid, _, width, code, field = _decode(input_grid)
    squares = _candidates(field == 0)
    _union(field.shape, squares)  # The domain has no conflicting paint orders.
    output = grid.copy()
    for r, c, s in squares:
        output[r:r + s, width + 1 + c:width + 1 + c + s] = _tile(s, code[:-1])
    return output


def _pack(mask, blocker, width, code):
    height = mask.shape[0]
    if height <= len(code):
        raise ValueError("the stack needs visible lower padding")
    grid = np.full((height, width + 1 + mask.shape[1]), blocker, dtype=np.int8)
    grid[:len(code), :width] = np.asarray(code, dtype=np.int8)[:, None]
    grid[:, width + 1:][mask] = 0
    return grid


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _halo(mask):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    result = np.zeros_like(mask)
    for dr in range(3):
        for dc in range(3):
            result |= padded[dr:dr + h, dc:dc + w]
    return result


def _touch(a, b):
    r, c, s = a
    t, l, z = b
    return ((max(r, t) < min(r + s, t + z) and (c + s == l or l + z == c))
            or (max(c, l) < min(c + s, l + z) and (r + s == t or t + z == r)))


def _partial_attachment(mask, square):
    r, c, s = square
    strips = []
    if r:
        strips.append(mask[r - 1, c:c + s])
    if r + s < mask.shape[0]:
        strips.append(mask[r + s, c:c + s])
    if c:
        strips.append(mask[r:r + s, c - 1])
    if c + s < mask.shape[1]:
        strips.append(mask[r:r + s, c + s])
    return any(a.any() and not a.all() for a in strips)


def _group(sides, touching, rng):
    if not touching:
        return [(0, 0, sides[0])], sides[1:]
    a, b = sides[:2]
    local = [(0, 0, a), (a - 1, a, b)]
    h, w = a + b - 1, a + b
    turns, flip = int(rng.integers(4)), bool(rng.integers(2))
    transformed = []
    for r, c, s in local:
        indicator = np.zeros((h, w), dtype=bool)
        indicator[r:r + s, c:c + s] = True
        if flip:
            indicator = indicator[:, ::-1]
        indicator = np.rot90(indicator, turns)
        points = np.argwhere(indicator)
        transformed.append((int(points[:, 0].min()), int(points[:, 1].min()), s))
    return transformed, sides[2:]


def _place_squares(rng, shape, sides, touching):
    intended, reserved = [], np.zeros(shape, dtype=bool)
    first, remaining = _group(sides, touching, rng)
    groups = [first] + [[(0, 0, s)] for s in remaining]
    for group in groups:
        gh = max(r + s for r, _, s in group)
        gw = max(c + s for _, c, s in group)
        if gh > shape[0] or gw > shape[1]:
            return None
        for _ in range(80):
            # Border targets coexist with interior targets; no mandatory frame.
            r0 = int(rng.integers(shape[0] - gh + 1))
            c0 = int(rng.integers(shape[1] - gw + 1))
            proposed = np.zeros(shape, dtype=bool)
            for r, c, s in group:
                proposed[r0 + r:r0 + r + s, c0 + c:c0 + c + s] = True
            if not (proposed & reserved).any():
                reserved |= _halo(proposed)
                intended.extend((r0 + r, c0 + c, s) for r, c, s in group)
                break
        else:
            return None
    return sorted(intended), reserved


def _path(rng, mask, square, union, forbidden):
    r, c, s = square
    direction = int(rng.integers(4))
    k = int(rng.integers(s))
    y, x, dy, dx = ((r - 1, c + k, -1, 0), (r + s, c + k, 1, 0),
                     (r + k, c - 1, 0, -1), (r + k, c + s, 0, 1))[direction]
    for step in range(int(rng.integers(1, 8))):
        if (not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1])
                or union[y, x] or forbidden[y, x]):
            break
        mask[y, x] = True
        if step and rng.random() < 0.30:
            dy, dx = (-dx, dy) if rng.random() < 0.5 else (dx, -dy)
        y, x = y + dy, x + dx


def _make_mask(rng, sides, touching, max_width, dense):
    largest = max(sides)
    # A fresh orientation is used for actual placement; reserve enough for both.
    minimum = max(12, largest + 5, sides[0] + sides[1] if touching else 0)
    if minimum > max_width:
        return None
    h = int(rng.integers(minimum, 31))
    w = int(rng.integers(minimum, max_width + 1))
    placed = _place_squares(rng, (h, w), sides, touching)
    if placed is None:
        return None
    squares, reserved = placed
    union = _union((h, w), squares)
    density = float(rng.uniform(0.24, 0.49) if dense else rng.uniform(0.08, 0.31))
    mask = rng.random((h, w)) < density
    mask[reserved] = False
    mask[union] = True

    # One substantial non-square blank rectangle defeats 'all blank 2x2s'.
    rh = int(rng.integers(2, 4))
    rw = rh + int(rng.integers(1, 5))
    if rng.random() < 0.5:
        rh, rw = rw, rh
    forbidden = np.zeros_like(mask)
    for _ in range(100):
        top, left = int(rng.integers(h - rh + 1)), int(rng.integers(w - rw + 1))
        rectangle = np.zeros_like(mask)
        rectangle[top:top + rh, left:left + rw] = True
        if not (rectangle & reserved).any():
            forbidden = _halo(rectangle)
            # A halo may meet a square's halo, but cannot erase a square.
            if (forbidden & union).any():
                continue
            mask[forbidden] = False
            mask[rectangle] = True
            break
    else:
        return None
    for square in squares:
        for _ in range(int(rng.integers(1, 4))):
            _path(rng, mask, square, union, forbidden)

    # Remove incidental square islands, not intended structures.  All cleanup
    # is in memory, bounded, and followed by a complete geometric recheck.
    intended = set(squares)
    for _ in range(16):
        actual = _candidates(mask)
        if not intended <= set(actual):
            return None
        extras = [q for q in actual if q not in intended]
        if not extras:
            if not any(_partial_attachment(mask, q) for q in squares):
                return None
            return mask, squares
        for r, c, s in extras:
            options = np.argwhere(~union[r:r + s, c:c + s]
                                  & ~forbidden[r:r + s, c:c + s])
            if not len(options):
                return None
            y, x = options[int(rng.integers(len(options)))]
            mask[r + y, c + x] = False
    return None


def _code(rng, length, colors, mode):
    # length counts effective layers; a duplicated terminal row is appended.
    if mode == "unique":
        layers = np.asarray(rng.choice(colors, size=length, replace=False), dtype=np.int8)
    elif mode == "repeat":
        palette = rng.choice(colors, size=length - 1, replace=False)
        layers = np.insert(palette, 1, palette[0]).astype(np.int8)
    else:
        k = int(rng.integers(2, min(len(colors), length) + 1))
        palette = rng.choice(colors, size=k, replace=False)
        layers = rng.choice(palette, size=length).astype(np.int8)
        layers[0], layers[-1] = palette[0], palette[-1]
        if length >= 4 and rng.random() < 0.5:
            layers[-1] = layers[0]  # nonadjacent recurrence, when possible
            layers[1] = palette[1]
    return np.append(layers, layers[-1]).astype(np.int8)


def _make_pair(rng, blocker, colors, length, profile, seen, strip_width=None):
    for _ in range(120):
        width = int(rng.integers(1, 5)) if strip_width is None else strip_width
        natural = 2 * length
        if profile == "natural":
            sides = [natural, int(rng.choice(range(3, natural - 1, 2)))]
            touching, code_mode = False, "unique"
        elif profile == "extended":
            sides = [natural + int(rng.integers(2, 5)), 2]
            touching, code_mode = True, "repeat"
        elif profile == "test":
            sides = [natural + int(rng.integers(2, 5)), natural, int(rng.integers(2, 5))]
            if length == 3 and rng.random() < 0.4:
                sides.append(int(rng.integers(2, 5)))
            touching, code_mode = True, "free"
        else:
            # More demonstrations vary count, size, adjacency and code style.
            sides = [int(rng.integers(4, natural + 5)), int(rng.integers(2, 6))]
            if rng.random() < 0.4:
                sides.append(int(rng.integers(2, 5)))
            touching, code_mode = bool(rng.integers(2)), "free"
        made = _make_mask(rng, sides, touching, 29 - width, profile != "natural")
        if made is None:
            continue
        mask, _ = made
        code = _code(rng, length, colors, code_mode)
        grid = _pack(mask, blocker, width, code)
        key = _key(grid)
        if key in seen:
            continue
        seen.add(key)
        return {"input": grid, "output": solve(grid)}
    raise RuntimeError("bounded pair construction exhausted")


def _check_counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    blocker = int(rng.integers(1, 10))
    colors = [c for c in range(1, 10) if c != blocker]
    lengths = [int(x) for x in rng.permutation([3, 4, 5])[:2]]
    widths = [int(x) for x in rng.choice([1, 2, 3, 4], size=2, replace=False)]
    seen = set()
    train = [_make_pair(rng, blocker, colors, lengths[0], "natural", seen, widths[0]),
             _make_pair(rng, blocker, colors, lengths[1], "extended", seen, widths[1])]
    for _ in range(int(num_train) - 2):
        train.append(_make_pair(rng, blocker, colors, int(rng.choice(lengths)), "extra", seen))
    rng.shuffle(train)
    test = [_make_pair(rng, blocker, colors, int(rng.choice(lengths)), "test", seen)
            for _ in range(int(num_test))]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _wrapped_tile(side, layers, cycle=False):
    """Validation construction: successively overwrite smaller filled squares."""
    result = np.zeros((side, side), dtype=np.int8)
    for depth in range((side + 1) // 2):
        index = depth % len(layers) if cycle else min(depth, len(layers) - 1)
        result[depth:side - depth, depth:side - depth] = layers[index]
    return result


def _features(pair):
    grid, b, width, code, field = _decode(pair["input"])
    squares = _candidates(field == 0)
    sizes = [s for _, _, s in squares]
    layers = code[:-1]
    union = _union(field.shape, squares)
    blank = field == 0
    # The ablation that colors every all-blank square includes every 2x2 window.
    windows = blank[:-1, :-1] & blank[1:, :-1] & blank[:-1, 1:] & blank[1:, 1:]
    all_windows = np.zeros_like(blank)
    for dy in (0, 1):
        for dx in (0, 1):
            all_windows[dy:dy + windows.shape[0], dx:dx + windows.shape[1]] |= windows
    p = np.pad(blank, 1)
    singletons = blank & ~p[:-2, 1:-1] & ~p[2:, 1:-1] & ~p[1:-1, :-2] & ~p[1:-1, 2:]
    return {"shape": grid.shape, "blocker": b, "width": width,
            "length": len(layers), "sizes": sizes, "count": len(squares),
            "short": any(s < 2 * len(layers) for s in sizes),
            "natural": 2 * len(layers) in sizes,
            "extended": any(s >= 2 * len(layers) + 2 for s in sizes),
            "repeat": bool(np.any(layers[1:] == layers[:-1])),
            "touch": any(_touch(a, z) for i, a in enumerate(squares) for z in squares[i + 1:]),
            "edge": any(r == 0 or c == 0 or r + s == field.shape[0]
                        or c + s == field.shape[1] for r, c, s in squares),
            "partial": any(_partial_attachment(blank, q) for q in squares),
            "a_witness": bool((all_windows & ~union).any()),
            "singleton": bool(singletons.any()),
            "density": float(blank.mean()), "colors": len(np.unique(code)),
            "return": len(np.unique(layers)) < 1 + np.count_nonzero(layers[1:] != layers[:-1])}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, blockers, training, testing = set(), set(), [], []
    rejects = {name: False for name in ("constant", "reverse", "collapse", "cycle", "largest", "scale", "width")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, blocker, width, code, field = _decode(pair["input"])
            output = _arc_grid(pair["output"])
            assert output.shape == grid.shape
            assert _key(grid) not in seen, "duplicate input"
            seen.add(_key(grid))
            blockers.add(blocker)
            squares = _candidates(field == 0)
            union = _union(field.shape, squares)
            assert squares, "generated pairs should demonstrate recoloring"
            assert np.array_equal(output[:, :width + 1], grid[:, :width + 1])
            out_field = output[:, width + 1:]
            assert np.array_equal(out_field[~union], field[~union]), "distractor or blocker changed"
            assert np.array_equal(out_field != field, union), "changed-cell support is incorrect"
            for r, c, s in squares:
                assert np.array_equal(out_field[r:r + s, c:c + s], _wrapped_tile(s, code[:-1]))
                # Direct local checks also guard the integral-image detector.
                assert np.all(field[r:r + s, c:c + s] == 0)
                if r:
                    assert not np.all(field[r - 1, c:c + s] == 0)
                if r + s < field.shape[0]:
                    assert not np.all(field[r + s, c:c + s] == 0)
                if c:
                    assert not np.all(field[r:r + s, c - 1] == 0)
                if c + s < field.shape[1]:
                    assert not np.all(field[r:r + s, c + s] == 0)
            f = _features(pair)
            assert f["a_witness"], "missing preserved extendable 2x2 window"
            assert f["partial"], "missing partial-extension geometry"
            (training if split == "train" else testing).append(f)
            if split == "train":
                layers = code[:-1]
                alternatives = {"constant": layers[:1], "reverse": layers[::-1],
                                "collapse": layers[np.r_[True, layers[1:] != layers[:-1]]],
                                "cycle": layers}
                for name, alt in alternatives.items():
                    rejects[name] |= any(not np.array_equal(_wrapped_tile(s, alt, name == "cycle"),
                                                           _wrapped_tile(s, layers)) for _, _, s in squares)
                rejects["largest"] |= len(squares) > 1
                for _, _, s in squares:
                    edge = np.minimum(np.arange(s), np.arange(s)[::-1])
                    depth = np.minimum(edge[:, None], edge[None, :])
                    scaled = layers[depth * (len(layers) - 1) // max(1, (s - 1) // 2)]
                    by_width = layers[np.minimum(depth // width, len(layers) - 1)]
                    rejects["scale"] |= not np.array_equal(scaled, _wrapped_tile(s, layers))
                    rejects["width"] |= not np.array_equal(by_width, _wrapped_tile(s, layers))
    assert len(blockers) == 1, "blocker convention changed within puzzle"
    assert len({f["length"] for f in training}) >= 2
    assert len({f["width"] for f in training}) >= 2
    assert 2 in {s for f in training for s in f["sizes"]}
    assert any(s % 2 for f in training for s in f["sizes"])
    assert any(f["singleton"] for f in training), "training must show ignored singletons"
    assert any(f["natural"] and f["short"] for f in training)
    assert any(f["extended"] and f["repeat"] and f["touch"] for f in training)
    assert all(rejects.values()), f"unexcluded training shortcut: {rejects}"
    assert all(f["short"] and f["natural"] and f["extended"] and f["touch"]
               and f["count"] >= 3 for f in testing), "missing recombination shift"


def _brute_candidates(mask):
    result = []
    h, w = mask.shape
    for s in range(2, min(h, w) + 1):
        for r in range(h - s + 1):
            for c in range(w - s + 1):
                if not mask[r:r + s, c:c + s].all():
                    continue
                if r and mask[r - 1, c:c + s].all():
                    continue
                if r + s < h and mask[r + s, c:c + s].all():
                    continue
                if c and mask[r:r + s, c - 1].all():
                    continue
                if c + s < w and mask[r:r + s, c + s].all():
                    continue
                result.append((r, c, s))
    return sorted(result)


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(grid.shape))
                h.update(grid.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded, streaming checks: 42 seeds, tiny geometry oracles, all counts."""
    rng = np.random.default_rng(760281)
    for _ in range(60):
        mask = rng.random((int(rng.integers(2, 9)), int(rng.integers(2, 9)))) < 0.65
        assert _candidates(mask) == _brute_candidates(mask)
    for layers in ([1], [1, 2], [1, 1, 3], [1, 2, 1], [1, 2, 3, 4, 5]):
        for side in range(2, 20):
            assert np.array_equal(_tile(side, layers), _wrapped_tile(side, layers))
    # Exercise the solver beyond the sampler's short codes, including width-30
    # canvases, empty fields, odd centers and every supported stack height.
    for height in range(2, 17):
        layers = [(i % 8) + 1 for i in range(height - 1)]
        code = layers + [layers[-1]]
        for side in (2, 3, 6, 13, 25, 28):
            mask = np.zeros((max(side, height + 1), side), dtype=bool)
            mask[:side, :] = True
            grid = _pack(mask, 9, 1, code)
            output = solve(grid)
            assert np.array_equal(output[:side, 2:], _wrapped_tile(side, layers))
            assert np.array_equal(output[:, :2], grid[:, :2])
        empty = _pack(np.zeros((height + 1, 1), dtype=bool), 9, 1, code)
        assert np.array_equal(solve(empty), empty)
    conflict = np.zeros((5, 3), dtype=bool)
    conflict[:2, :2] = True
    conflict[1:3, 1:3] = True
    try:
        solve(_pack(conflict, 9, 1, [1, 2, 2]))
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping candidates were silently resolved")
    signatures, shapes, sizes, widths, colors, counts, blockers = set(), set(), set(), set(), set(), set(), set()
    normalized = set()
    metrics = {name: 0 for name in ("pairs", "touch", "edge", "partial", "short", "natural", "extended", "repeat", "return")}
    densities = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            f = _features(pair)
            metrics["pairs"] += 1
            for name in metrics.keys() - {"pairs"}:
                metrics[name] += int(f[name])
            shapes.add(f["shape"])
            sizes.update(f["sizes"])
            widths.add(f["width"])
            counts.add(f["count"])
            blockers.add(f["blocker"])
            densities.append(f["density"])
            colors.update(map(int, np.unique(pair["input"])))
            roles = {}
            raw = bytes(roles.setdefault(int(c), len(roles)) for c in pair["input"].flat)
            normalized.add((f["shape"], hashlib.sha256(raw).hexdigest()))
            assert np.array_equal(solve(pair["input"].copy()), pair["output"])
    assert len(signatures) == 42 and len(shapes) >= 35 and len(normalized) >= 180
    assert {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} <= sizes
    assert widths == {1, 2, 3, 4} and colors == set(range(10)) and len(blockers) >= 8
    assert {2, 3, 4} <= counts
    assert metrics["edge"] >= 12 and metrics["return"] >= 8
    assert max(densities) - min(densities) >= 0.20
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(85000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    assert _digest(generate_puzzle(654321, 6, 3)) == _digest(generate_puzzle(654321, 6, 3))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A deliberately changed frame must fail validation, even if schema is valid.
    bad = generate_puzzle(99, 2, 1)
    g, _, width, _, f = _decode(bad["test"][0]["input"])
    r, c, _ = _candidates(f == 0)[0]
    bad["test"][0]["output"][r, width + 1 + c] = 0
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output passed validation")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "normalized_inputs": len(normalized), "sizes": sorted(sizes),
            "stack_widths": sorted(widths), "target_counts": sorted(counts),
            "blocker_colors": sorted(blockers), "density_range": [min(densities), max(densities)],
            **metrics, "independent_semantic_review": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
