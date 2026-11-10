"""Scan-ordered mirror overprinting. Only NumPy and the standard library.

No files, examples, generation metadata, or random state are used by solve.
Generation and self_test keep one puzzle in memory at a time and save no traces.
"""

from __future__ import annotations

import hashlib
import numpy as np


def _integer(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _orient(grid, configuration):
    k, flip = configuration
    return np.rot90(grid[:, ::-1] if flip else grid, k).copy()


def _unorient(grid, configuration):
    k, flip = configuration
    a = np.rot90(grid, -k)
    return (a[:, ::-1] if flip else a).copy()


def _key(grid):
    """The broken guide is fast; the solid guide is slow, in every palette."""
    matches = []
    assert min(grid.shape) >= 7
    for k in range(4):
        for flip in (False, True):
            a = _unorient(grid, (k, flip))
            origin, fast, slow, bg = map(int, (a[0, 0], a[0, 1], a[1, 0], a[-1, -1]))
            if len({origin, fast, slow, bg}) != 4:
                continue
            if (a[0, 2] == bg and np.all(a[0, 3:] == fast)
                    and np.all(a[1:, 0] == slow)
                    and np.all(a[-1, 1:] == bg) and np.all(a[1:, -1] == bg)
                    and np.all(a[1, 1:] == bg) and np.all(a[1:, 1] == bg)):
                matches.append((a, (k, flip), bg, origin, fast, slow))
    assert len(matches) == 1, "Expected one visibly directed border key"
    return matches[0]


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        result.append(sorted(cells))
    return result


def _bbox(cells):
    return (min(r for r, _ in cells), max(r for r, _ in cells),
            min(c for _, c in cells), max(c for _, c in cells))


def _kind(marker):
    r0, r1, c0, c1 = _bbox(marker)
    if len(marker) == 1:
        return "P"
    if r0 == r1 and len(marker) == c1 - c0 + 1:
        return "H"
    if c0 == c1 and len(marker) == r1 - r0 + 1:
        return "V"
    raise AssertionError("Markers must be straight segments or singletons")


def _destination(source, marker, kind):
    ar, ac = marker[0]
    if kind == "H":
        return [(2 * ar - r, c) for r, c in source]
    if kind == "V":
        return [(r, 2 * ac - c) for r, c in source]
    return [(2 * ar - r, 2 * ac - c) for r, c in source]


def _eligible(source, marker, kind, grid, bg):
    if min(max(abs(r-a), abs(c-b)) for r, c in source for a, b in marker) != 1:
        return None
    r0, r1, c0, c1 = _bbox(source)
    a0, a1, b0, b1 = _bbox(marker)
    if kind == "H":
        if not (r1 < a0 or r0 > a0) or c1 < b0 or c0 > b1:
            return None
    elif kind == "V":
        if not (c1 < b0 or c0 > b0) or r1 < a0 or r0 > a1:
            return None
    else:
        if not ((r1 < a0 or r0 > a0) and (c1 < b0 or c0 > b0)):
            return None
    dest = _destination(source, marker, kind)
    h, w = grid.shape
    if any(not (2 <= r < h-1 and 2 <= c < w-1) or grid[r, c] != bg for r, c in dest):
        return None
    return dest


def _read(input_grid):
    """Parse input alone. Source bands determine lanes; no output selects a parse."""
    a, config, bg, ink, fast, slow = _key(_grid(input_grid))
    interior = np.ones(a.shape, dtype=bool)
    interior[0, :] = interior[:, 0] = False
    sources = _components(interior & (a != bg) & (a != ink))
    markers = _components(interior & (a == ink), diagonal=True)
    assert 2 <= len(sources) <= 7 and len(markers) == len(sources)
    boxes = []
    for source in sources:
        r0, r1, c0, c1 = _bbox(source)
        assert r1-r0 == 1 and 2 <= c1-c0+1 <= 12
        assert len(source) == 2 * (c1-c0+1), "Source strips must be full rectangles"
        assert 2 <= r0 and r1 < a.shape[0]-1 and 2 <= c0 and c1 < a.shape[1]-1
        boxes.append((r0, r1, c0, c1))
    lanes = sorted({b[0] for b in boxes})
    assert all(y-x >= 3 for x, y in zip(lanes, lanes[1:])), "Lane bands overlap"
    groups, used = [], set()
    for marker in markers:
        kind = _kind(marker)
        options = []
        for i, source in enumerate(sources):
            dest = _eligible(source, marker, kind, a, bg)
            if dest is not None:
                options.append((i, dest))
        assert len(options) == 1, "Ambiguous marker ownership"
        i, dest = options[0]
        assert i not in used, "A source has multiple markers"
        used.add(i)
        groups.append({"source": sources[i], "marker": marker, "kind": kind,
                       "dest": dest, "box": boxes[i]})
    assert len(used) == len(sources)
    # Each visible printer is a separate local group, even with reused colors.
    for i, g in enumerate(groups):
        for other in groups[:i]:
            assert min(max(abs(r-a0), abs(c-b0))
                       for r, c in g["source"] + g["marker"]
                       for a0, b0 in other["source"] + other["marker"]) >= 2
    groups.sort(key=lambda g: (g["box"][0], g["box"][2]))
    return {"grid": a, "config": config, "bg": bg, "ink": ink,
            "fast": fast, "slow": slow, "groups": groups}


def _paint(d, order=None, operation="symmetry"):
    a, groups = d["grid"], d["groups"]
    result = a.copy()
    for i in range(len(groups)) if order is None else order:
        g = groups[i]
        source = g["source"]
        if operation == "translation":
            # Competing rule: move the unturned rectangle to its destination bbox.
            r0, _, c0, _ = g["box"]
            t0, _, u0, _ = _bbox(g["dest"])
            dest = [(r+t0-r0, c+u0-c0) for r, c in source]
        else:
            dest = g["dest"]
        for p, q in zip(source, dest):
            result[q] = a[p]  # Always read the original; never relay painted copies.
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Copy marker-relative symmetries in key-directed raster order, last wins."""
    d = _read(input_grid)
    return _orient(_paint(d), d["config"])


def _overlap(d):
    cover = np.zeros(d["grid"].shape, dtype=np.int8)
    for g in d["groups"]:
        for q in g["dest"]:
            cover[q] += 1
    return cover


def _alternative(d, name):
    groups = d["groups"]
    n = len(groups)
    if name == "reverse":
        order = list(range(n))[::-1]
    elif name == "fast_reverse":
        order = sorted(range(n), key=lambda i: (groups[i]["box"][0], -groups[i]["box"][2]))
    elif name == "slow_reverse":
        order = sorted(range(n), key=lambda i: (-groups[i]["box"][0], groups[i]["box"][2]))
    elif name == "fast_major":
        order = sorted(range(n), key=lambda i: (groups[i]["box"][2], groups[i]["box"][0]))
    elif name == "physical":
        # Ignore the key and visit sources by their physical top/left corners.
        def physical_box(i):
            mask = np.zeros(d["grid"].shape, dtype=np.int8)
            for q in groups[i]["source"]:
                mask[q] = 1
            r, _, c, _ = _bbox(list(map(tuple, np.argwhere(_orient(mask, d["config"])).tolist())))
            return r, c
        order = sorted(range(n), key=physical_box)
    else:
        raise ValueError(name)
    return _paint(d, order)


def _witnesses(descriptors):
    for name in ("reverse", "fast_reverse", "slow_reverse", "fast_major", "physical"):
        assert any(not np.array_equal(_paint(d), _alternative(d, name)) for d in descriptors), name
    for kind in ("H", "V", "P"):
        assert any(g["kind"] == kind for d in descriptors for g in d["groups"])
        visible_witness = False
        for d in descriptors:
            expected = _paint(d)
            for i, g in enumerate(d["groups"]):
                if g["kind"] != kind:
                    continue
                # Mutate just this operation, preserving the actual overprint order.
                alternative = d["grid"].copy()
                for j, other in enumerate(d["groups"]):
                    r0, _, c0, _ = other["box"]
                    t0, _, u0, _ = _bbox(other["dest"])
                    dest = ([(r+t0-r0, c+u0-c0) for r, c in other["source"]]
                            if i == j else other["dest"])
                    for p, q in zip(other["source"], dest):
                        alternative[q] = d["grid"][p]
                visible_witness |= not np.array_equal(expected, alternative)
        assert visible_witness, f"No visible orientation witness for {kind}"


def _make_group(rng, kind, target_row, target_col, length, sy=None, sx=None):
    sy = int(rng.choice([-1, 1])) if sy is None else sy
    sx = int(rng.choice([-1, 1])) if sx is None else sx
    r = target_row + (3 * sy if kind in ("H", "P") else 0)
    c = target_col + ((length+1) * sx if kind in ("V", "P") else 0)
    source = [(rr, cc) for rr in (r, r+1) for cc in range(c, c+length)]
    if kind == "P":
        marker = [(r-1 if sy > 0 else r+2, c-1 if sx > 0 else c+length)]
    elif kind == "H":
        axis = r-1 if sy > 0 else r+2
        # Finite lines may end inside the strip or overhang its endpoints.
        lo = int(rng.integers(c-1, c+length-1))
        hi = int(rng.integers(max(lo+1, c), c+length+1))
        marker = [(axis, cc) for cc in range(lo, hi+1)]
    else:
        axis = c-1 if sx > 0 else c+length
        lo = r - int(rng.integers(2))
        hi = r + 1 + int(rng.integers(2))
        marker = [(rr, axis) for rr in range(lo, hi+1)]
    return {"source": source, "marker": marker, "kind": kind,
            "dest": _destination(source, marker, kind), "box": _bbox(source)}


def _fits(g, chosen, shape, cover_limit):
    h, w = shape
    visible = set(g["source"] + g["marker"])
    dest = set(g["dest"])
    if any(not (2 <= r < h-1 and 2 <= c < w-1) for r, c in visible | dest):
        return False
    if visible & dest:
        return False
    coverage = {}
    for old in chosen:
        old_visible = set(old["source"] + old["marker"])
        if visible & set(old["dest"]) or dest & old_visible:
            return False
        if any(max(abs(r-a), abs(c-b)) < 2 for r, c in visible for a, b in old_visible):
            return False
        for q in old["dest"]:
            coverage[q] = coverage.get(q, 0) + 1
    return all(coverage.get(q, 0) < cover_limit for q in dest)


def _pattern(rng, length, palette, witness):
    mode = int(rng.integers(5))
    if witness:
        for _ in range(16):
            a = rng.choice(palette, size=(2, length)).astype(np.int8)
            if all(not np.array_equal(a, b) for b in (a[::-1], a[:, ::-1], a[::-1, ::-1])):
                return a
        a = np.full((2, length), palette[0], dtype=np.int8)
        a[0, 0] = palette[1]
        return a
    if mode == 0:
        return np.full((2, length), rng.choice(palette), dtype=np.int8)
    if mode == 1:
        return np.repeat(rng.choice(palette, size=(1, length)), 2, axis=0).astype(np.int8)
    if mode == 2:
        return np.repeat(rng.choice(palette, size=(2, 1)), length, axis=1).astype(np.int8)
    if mode == 3:
        motif = rng.choice(palette, size=(2, int(rng.integers(1, min(4, length)+1))))
        return np.tile(motif, (1, length))[:, :length].astype(np.int8)
    return rng.choice(palette, size=(2, length)).astype(np.int8)


def _scene(rng, roles, config, regime):
    fast, slow = roles
    bg, ink = map(int, rng.choice([c for c in range(10) if c not in roles], 2, replace=False))
    h, w = int(rng.integers(17, 31)), int(rng.integers(19, 31))
    # Every stage is implicit empty space, never a drawn panel or fixed slot.
    base = int(rng.integers(2, 5))
    bands = list(range(base, h-2, 3))
    t = int(rng.choice([r for r in bands if 5 <= r <= h-6]))
    pivot = int(rng.integers(7, w-7))
    if regime == "lanes":
        kinds = ["H", "P"]
    elif regime == "within":
        kinds = ["V", "V"]
    elif regime == "triple":
        kinds = list(rng.permutation(["H", "V", "P"]))
    else:
        kinds = list(rng.choice(["H", "V", "P"], size=2))
    core_count = len(kinds)
    extra = int(rng.integers(0, 4 if regime != "triple" else 5))
    kinds += list(rng.choice(["H", "V", "P"], size=extra))
    chosen = []
    side = int(rng.choice([-1, 1]))
    for i, kind in enumerate(kinds):
        accepted = None
        for attempt in range(180):
            core = i < core_count
            row = t if core else int(rng.choice(bands))
            length = int(rng.integers(2, min(10, (w-4)//2)+1))
            col = pivot-int(rng.integers(length)) if core else int(rng.integers(2, w-length-1))
            sy = side if regime == "lanes" and i == 0 else (-side if regime == "lanes" and i == 1 else None)
            sx = side if regime == "lanes" and i == 1 else None
            if regime == "within" and i < 2:
                sx = side if i == 0 else -side
            g = _make_group(rng, str(kind), row, col, length, sy, sx)
            if _fits(g, chosen, (h, w), 7 if regime == "triple" else 2):
                accepted = g
                break
        if accepted is None:
            if i < core_count:
                return None
            break  # Optional extra printers are limited by their actual footprints.
        chosen.append(accepted)
    a = np.full((h, w), bg, dtype=np.int8)
    a[0, 0], a[0, 1:], a[1:, 0] = ink, fast, slow
    a[0, 2] = bg
    pool = [c for c in range(10) if c not in (bg, ink)]
    palette = rng.choice(pool, size=int(rng.integers(2, len(pool)+1)), replace=False)
    for i, g in enumerate(chosen):
        r0, r1, c0, c1 = g["box"]
        a[r0:r1+1, c0:c1+1] = _pattern(rng, c1-c0+1, palette, i < core_count)
        for q in g["marker"]:
            a[q] = ink
    oriented = _orient(a, config)
    try:
        d = _read(oriented)
    except AssertionError:
        return None
    expected = _paint(d)
    if np.array_equal(expected, _paint(d, list(range(len(chosen)))[::-1])):
        return None  # Color differences must make the order visible.
    if regime == "lanes" and any(np.array_equal(expected, _alternative(d, name))
                                 for name in ("slow_reverse", "fast_major")):
        return None
    if regime == "within" and any(np.array_equal(expected, _alternative(d, name))
                                  for name in ("fast_reverse", "physical")):
        return None
    return {"input": oriented, "output": _orient(expected, config)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _integer(num_train, 2, 10, "num_train")
    nq = _integer(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    first = (int(rng.integers(4)), bool(rng.integers(2)))
    configurations = [(k, flip) for k in range(4) for flip in (False, True)]
    # Teach the other physical axis, with the within-lane direction reversed
    # relative to physical top-left reading. This makes the key indispensable
    # even with only two demonstrations and no optional extra printers.
    physical_reverse = [(0, True), (1, False), (2, False), (3, True)]
    second_pool = [c for c in physical_reverse if c[0] % 2 != first[0] % 2]
    second = second_pool[int(rng.integers(len(second_pool)))]
    # All training stays at <=2 copies per cell. Tests combine >=3 at a cell.
    # The two coverage scenes teach both within-lane and between-lane priority.
    for puzzle_attempt in range(32):
        puzzle = {"train": [], "test": []}
        seen = set()
        for split, count in (("train", nt), ("test", nq)):
            for i in range(count):
                regime = ("lanes" if i == 0 else "within" if i == 1 else "ordinary") if split == "train" else "triple"
                config = first if i == 0 and split == "train" else second if i == 1 and split == "train" else configurations[int(rng.integers(8))]
                for _ in range(64):
                    pair = _scene(rng, roles, config, regime)
                    if pair is not None and _fingerprint(pair["input"]) not in seen:
                        break
                else:
                    raise RuntimeError("Bounded construction exhausted its scene budget")
                puzzle[split].append(pair)
                seen.add(_fingerprint(pair["input"]))
        try:
            _witnesses([_read(p["input"]) for p in puzzle["train"]])
        except AssertionError:
            continue
        rng.shuffle(puzzle["train"])
        validate_generated_puzzle(puzzle)
        return puzzle
    raise RuntimeError("Bounded construction could not expose all training distinctions")


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, roles, training = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen, "Duplicate inputs"
            seen.add(key)
            d = _read(inp)
            roles.add((d["fast"], d["slow"]))
            cover = _overlap(d)
            expected = _orient(_paint(d), d["config"])
            assert np.array_equal(out, expected), "Wrong symmetry, traversal, color, or overprint"
            assert np.array_equal(out[inp != d["bg"]], inp[inp != d["bg"]])
            assert np.count_nonzero(inp != out) == np.count_nonzero(cover)
            assert cover.max() >= 2
            assert not np.array_equal(_paint(d), _alternative(d, "reverse")), "Invisible ordering"
            if split == "train":
                assert cover.max() == 2
                training.append(d)
            else:
                assert cover.max() >= 3, "Missing three-way overprint shift"
    assert len(roles) == 1
    _witnesses(training)


def _signature(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for name in ("input", "output"):
                a = np.asarray(p[name], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded 42-seed variation suite; retains hashes and small sets, no grids."""
    # Hand-composed patches exercise the collision rule independently of _paint.
    def fixture(shape):
        a = np.zeros(shape, dtype=np.int8)
        a[0, 0], a[0, 1:], a[1:, 0] = 5, 1, 9
        a[0, 2] = 0
        return a

    a = fixture((12, 17))
    a[5:7, 3:5] = [[2, 3], [4, 2]]
    a[5:7, 11:14] = [[8, 2, 3], [4, 8, 2]]
    a[5:7, 5] = a[5:7, 10] = 5
    expected = a.copy()
    expected[5:7, 6:10] = [[3, 3, 2, 8], [2, 2, 8, 4]]
    assert np.array_equal(solve(a), expected)
    assert not np.array_equal(_paint(_read(a), [1, 0]), expected)
    a = fixture((16, 23))
    a[4:6, 10:14] = [[2, 3, 4, 6], [7, 8, 2, 3]]
    a[7:9, 5:9] = [[6, 4, 3, 2], [3, 2, 8, 7]]
    a[10:12, 15:19] = [[2, 8, 6, 3], [4, 3, 7, 2]]
    a[6, 12:14] = a[7:9, 9] = a[9, 14] = 5
    expected = a.copy()
    expected[7:9, 10:14] = [[2, 7, 3, 4], [3, 6, 8, 2]]
    for k in range(4):
        for flip in (False, True):
            assert np.array_equal(solve(_orient(a, (k, flip))), _orient(expected, (k, flip)))
    signatures, shapes, configs, layouts, lengths, counts, inks, backgrounds = (set() for _ in range(8))
    observed_kinds, overhangs, palettes = set(), 0, set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _read(pair["input"])
                shapes.add(pair["input"].shape)
                configs.add(d["config"])
                counts.add(len(d["groups"]))
                inks.add(d["ink"])
                backgrounds.add(d["bg"])
                palettes.add(len({int(d["grid"][q]) for g in d["groups"] for q in g["source"]}))
                layouts.add((d["grid"].shape, tuple((g["kind"], g["box"], tuple(g["marker"])) for g in d["groups"])))
                for g in d["groups"]:
                    observed_kinds.add(g["kind"])
                    lengths.add(g["box"][3]-g["box"][2]+1)
                    r0, r1, c0, c1 = g["box"]
                    a0, a1, b0, b1 = _bbox(g["marker"])
                    overhangs += int(g["kind"] == "H" and (b0 < c0 or b1 > c1)
                                     or g["kind"] == "V" and (a0 < r0 or a1 > r1))
                    block = d["grid"][r0:r1+1, c0:c1+1]
                    flipped = (block[::-1] if g["kind"] == "H" else block[:, ::-1]
                               if g["kind"] == "V" else block[::-1, ::-1])
                    t0, t1, u0, u1 = _bbox(g["dest"])
                    mapped = np.full((t1-t0+1, u1-u0+1), -1, dtype=int)
                    for p, (rr, cc) in zip(g["source"], g["dest"]):
                        mapped[rr-t0, cc-u0] = d["grid"][p]
                    assert np.array_equal(mapped, flipped)
                # Rotation/reflection equivariance tests solve's coordinate handling.
                for config in ((1, False), (0, True)):
                    transformed = _orient(pair["input"], config)
                    assert np.array_equal(solve(transformed), _orient(pair["output"], config))
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
    assert len(signatures) == 42 and len(layouts) >= 180
    assert len(shapes) >= 45 and len(configs) == 8
    assert len(lengths) >= 7 and len(counts) >= 4
    assert len(inks) >= 9 and len(backgrounds) >= 9
    assert observed_kinds == {"H", "V", "P"} and overhangs >= 20 and len(palettes) >= 4
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(10000+10*a+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    p = generate_puzzle(90001, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("Damaged label accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds; all legal count combinations)")
