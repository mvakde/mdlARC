"""Symmetry-generated ribbons: two local involutions determine an infinite repeat.

The sampler builds array blocks; solve uses affine cell orbits reconstructed from
the input. No sampled description, seed, or training pair is used by solve.
"""

from __future__ import annotations

import itertools
import numpy as np


HOST_SEEDS = (
    0, 455599968, 686425538, 545518136, 238959821, 159837281,
    334338800, 523568914, 210002887, 327888416, 142414627, 393547813,
    950258590, 111553645, 233164876, 126063646, 405712148, 825034593,
    996720889, 736323998, 342116375, 19096541, 239016206, 218261506,
    817069145, 817738135, 681118509, 727944583, 196446208, 144172911,
    111206897, 290050828, 52595808, 263469655, 783630145, 796544544,
    677125629, 676234099, 560821450, 107289856, 832005133, 452384326,
)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("Expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return g.astype(np.int8, copy=True)


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _components(mask, diagonal=False):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while todo:
        first = min(todo)
        todo.remove(first)
        stack, cells = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr), min(cc), max(cc)


def _marker_kind(cells):
    r0, r1, c0, c1 = _bbox(cells)
    if len(cells) == 1:
        return "P"
    if c0 == c1 and len(cells) == r1 - r0 + 1:
        return "V"
    if r0 == r1 and len(cells) == c1 - c0 + 1:
        return "H"
    raise ValueError("A marker is a straight segment or singleton")


def _parse_color(g, bg, marker):
    """Find two opposed end markers per complete, eight-connected seed tile."""
    marks = _components(g == marker)
    kinds = [_marker_kind(m) for m in marks]
    payloads = _components((g != bg) & (g != marker), diagonal=True)
    if not payloads or len(marks) != 2 * len(payloads):
        raise ValueError("Each seed needs exactly two markers")
    groups, used = [], set()
    for cells in payloads:
        r0, r1, c0, c1 = _bbox(cells)
        choices = []
        for vertical in (False, True):
            # (transverse coordinate, longitudinal coordinate) in a local frame.
            t0, t1, x0, x1 = ((c0, c1, r0, r1) if vertical
                               else (r0, r1, c0, c1))
            ends = [[], []]
            for j, (m, kind) in enumerate(zip(marks, kinds)):
                local = [(c, r) if vertical else (r, c) for r, c in m]
                ts = [t for t, _ in local]
                xs = {x for _, x in local}
                for side, x in enumerate((x0 - 1, x1 + 1)):
                    if xs != {x}:
                        continue
                    if kind == "P":
                        if t0 + t1 != 2 * ts[0]:
                            continue
                        contact = (x0 if side == 0 else x1, ts[0]) if vertical else (
                            ts[0], x0 if side == 0 else x1)
                        if contact not in cells:
                            continue
                    else:
                        if kind != ("H" if vertical else "V"):
                            continue
                        if min(ts) > t0 or max(ts) < t1 or min(ts) + max(ts) != t0 + t1:
                            continue
                    ends[side].append(j)
            if len(ends[0]) == len(ends[1]) == 1:
                left, right = ends[0][0], ends[1][0]
                choices.append(dict(vertical=vertical, t0=t0, t1=t1,
                                    a=x0 - 1, b=x1 + 1, cells=cells,
                                    left=left, right=right,
                                    p0=kinds[left] == "P", p1=kinds[right] == "P"))
        if len(choices) != 1:
            raise ValueError("Ambiguous or missing opposed-marker geometry")
        group = choices[0]
        indices = {group["left"], group["right"]}
        if used & indices:
            raise ValueError("A marker belongs to more than one seed")
        used.update(indices)
        group["markers"] = marks[group["left"]] | marks[group["right"]]
        groups.append(group)
    if len(used) != len(marks):
        raise ValueError("Unassociated marker")
    # Entire ribbons are parallel and their transverse footprints are separated.
    if len({s["vertical"] for s in groups}) != 1:
        raise ValueError("Crossing ribbons are outside this family")
    footprints = []
    for s in groups:
        points = s["cells"] | s["markers"]
        ts = [c if s["vertical"] else r for r, c in points]
        footprints.append((min(ts), max(ts)))
    footprints.sort()
    if any(b + 1 >= c for (_, b), (c, _) in zip(footprints, footprints[1:])):
        raise ValueError("Ribbons require an empty transverse separation")
    return groups


def _read_input(g):
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if np.count_nonzero(counts == counts[bg]) != 1:
        raise ValueError("Background must be the unique most frequent input color")
    interpretations = []
    for marker in map(int, np.unique(g)):
        if marker == bg:
            continue
        try:
            groups = _parse_color(g, bg, marker)
        except ValueError:
            continue
        interpretations.append((marker, groups))
    if len(interpretations) != 1:
        raise ValueError("Input must expose exactly one geometric marker color")
    marker, groups = interpretations[0]
    return bg, marker, groups


def _orbit_writes(g, groups, *, single_copy=False, ignore_symmetry=False,
                  points_as_lines=False, mixed_period_two=False):
    """Apply translations and a reflected coset to each source cell.

    The two original transformations generate a shift by 2D. For unlike
    markers that shift also flips the transverse coordinate. Its powers and
    one reflection are the complete orbit, including negative powers.
    The switches implement explicit competing rules for validation only.
    """
    writes = {}
    for s in groups:
        vertical = s["vertical"]
        length = g.shape[0 if vertical else 1]
        a, b = s["a"], s["b"]
        d = b - a
        p0 = s["p0"] and not points_as_lines
        p1 = s["p1"] and not points_as_lines
        mixed = p0 != p1
        for r, c in sorted(s["cells"]):
            t, x = (c, r) if vertical else (r, c)
            for k in range(-length // d - 2, length // d + 3):
                if single_copy and k not in (-1, 0, 1):
                    continue
                if ignore_symmetry:
                    xx, flip = x + k * d, False
                elif k % 2 == 0:
                    n = k // 2
                    xx = x + 2 * n * d
                    flip = mixed and bool(n % 2)
                else:
                    n = (k - 1) // 2
                    xx = 2 * b - x + 2 * n * d
                    flip = p1 != (mixed and bool(n % 2))
                if mixed_period_two and not ignore_symmetry:
                    flip = p1 if k % 2 else False
                tt = s["t0"] + s["t1"] - t if flip else t
                if not 0 <= xx < length:
                    continue
                q = (xx, tt) if vertical else (tt, xx)
                if q in writes:
                    raise ValueError("Two seed orbits overlap")
                writes[q] = int(g[r, c])
    return writes


def _with_writes(g, writes):
    out = g.copy()
    for q, value in writes.items():
        out[q] = value
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete all symmetry orbits, then clip the infinite ribbons to the grid."""
    g = _grid(input_grid)
    bg, _, groups = _read_input(g)
    writes = _orbit_writes(g, groups)
    for q, value in writes.items():
        if g[q] != bg and int(g[q]) != value:
            raise ValueError("Orbit collides with original material")
    return _with_writes(g, writes)


def _pattern(s):
    return ("P" if s["p0"] else "L") + ("P" if s["p1"] else "L")


def _tile_from_input(g, bg, s):
    if s["vertical"]:
        return g[s["a"] + 1:s["b"], s["t0"]:s["t1"] + 1].T.copy()
    return g[s["t0"]:s["t1"] + 1, s["a"] + 1:s["b"]].copy()


def _make_tile(rng, width, length, bg, palette, witness):
    mask = np.ones((width, length), dtype=bool)
    style = int(rng.integers(4))
    if style == 1 and width >= 3 and length >= 3:
        mask[1:-1, 1:-1] = False
    elif style == 2 and width >= 2 and length >= 2:
        h = int(rng.integers(1, width))
        w = int(rng.integers(1, length))
        mask[:h, :w] = False
        if rng.random() < .5:
            mask = mask[::-1]
        if rng.random() < .5:
            mask = mask[:, ::-1]
    elif style == 3 and width >= 3 and length >= 2:
        mask[:] = False
        mask[width // 2, :] = True
        col = int(rng.integers(length))
        mask[:, col] = True
        for r in range(width):
            if rng.random() < .55:
                mask[r, (col + r) % length] = True
    # The central spine makes both singleton contacts visible and keeps sparse
    # shapes connected. This is geometric evidence, not a hidden tile dimension.
    mask[width // 2, :] = True
    if len(_components(mask, diagonal=True)) != 1:
        mask[:] = True
    colors = rng.choice(palette, size=(width, length)).astype(np.int8)
    color_style = int(rng.integers(4))
    if color_style == 0:
        colors[:] = rng.choice(palette)
    elif color_style == 1:
        colors[:] = rng.choice(palette, size=(width, 1))
    elif color_style == 2:
        colors[:] = rng.choice(palette, size=(1, length))
    tile = np.where(mask, colors, bg).astype(np.int8)
    if witness:
        # A deliberately noncentral exceptional color witnesses all three flips.
        # Most witnesses keep the sampled silhouette and random color field.
        a, b = map(int, palette[:2])
        tile[0, 0], tile[-1, 0] = a, b
        tile[width // 2, 0], tile[width // 2, -1] = a, b
        transforms = (tile[:, ::-1], tile[::-1], tile[::-1, ::-1])
        if any(np.array_equal(tile, t) for t in transforms):
            tile[:] = a
            tile[0, 0] = b
    return tile


def _block(tile, pattern, bg):
    # Array construction intentionally differs from solve's affine-cell orbits.
    if pattern == "LL":
        tiles = [tile, tile[:, ::-1]]
    elif pattern == "PP":
        tiles = [tile, tile[::-1, ::-1]]
    elif pattern == "LP":
        tiles = [tile, tile[::-1, ::-1], tile[::-1], tile[:, ::-1]]
    else:
        tiles = [tile, tile[:, ::-1], tile[::-1], tile[::-1, ::-1]]
    seam = np.full((len(tile), 1), bg, dtype=np.int8)
    return np.concatenate([a for t in tiles for a in (t, seam)], axis=1)


def _sample_specs(rng, plans, regime, test, compact=False):
    specs = []
    for i, (pattern, witness) in enumerate(plans):
        length = int(rng.integers(2, 4) if witness else rng.integers(1, 4))
        if test and regime == "longer" and i == 0:
            length = int(rng.integers(4, 7))
        elif test:
            length = int(rng.integers(1, 7))
        if compact:
            length = 2 if witness else length
        if witness:
            width = 3 if regime == "wider" or compact else int(rng.choice([3, 3, 5]))
        elif test and regime == "wider" and i == 0:
            width = int(rng.choice([5, 7]))
        else:
            cap = 3 if not test and regime == "wider" else 7
            values = list(range(1, cap + 1)) if pattern == "LL" else list(range(1, cap + 1, 2))
            width = int(rng.choice(values))
        if compact and not witness and not (test and regime == "wider" and i == 0):
            width = 3 if pattern != "LL" else 2
        overhang = []
        for p in pattern:
            if p == "P":
                overhang.append(0)
            elif width == 1:
                overhang.append(1)
            else:
                overhang.append(0 if compact else int(rng.choice([0, 0, 0, 1, 2])))
        specs.append(dict(pattern=pattern, witness=witness, width=width,
                          length=length, e0=overhang[0], e1=overhang[1]))
    return specs


def _draw_pair(rng, plans, regime, test, marker, vertical):
    for attempt in range(40):
        specs = _sample_specs(rng, plans, regime, test, compact=attempt >= 20)
        heights = [s["width"] + 2 * max(s["e0"], s["e1"]) for s in specs]
        need_h = sum(heights) + len(specs) - 1
        if need_h > 30:
            continue
        need_w = max(4 * (s["length"] + 1) + 1 if s["witness"]
                     else s["length"] + 3 for s in specs)
        # Long-word tests must show enough continuation to use the new word.
        if test and regime == "longer":
            need_w = max(need_w, 3 * (specs[0]["length"] + 1))
        height = int(rng.integers(max(5, need_h), 31))
        length = int(rng.integers(max(7, need_w), 31))
        colors = [c for c in map(int, rng.permutation(10)) if c != marker]
        bg = colors[0]
        palette = colors[1:1 + int(rng.integers(2, 8))]
        g = np.full((height, length), bg, dtype=np.int8)
        out = g.copy()
        free = height - need_h
        gaps = rng.multinomial(free, np.ones(len(specs) + 1) / (len(specs) + 1))
        order = list(rng.permutation(len(specs)))
        top = int(gaps[0])
        for slot, index in enumerate(order):
            s = specs[index]
            width, word_len = s["width"], s["length"]
            e = max(s["e0"], s["e1"])
            row = top + e
            if s["witness"]:
                d = word_len + 1
                if rng.random() < .5:
                    start = int(rng.integers(1, length - 3 * d - word_len + 1))
                else:
                    start = int(rng.integers(3 * d, length - word_len))
            else:
                start = int(rng.integers(1, length - word_len))
            tile = _make_tile(rng, width, word_len, bg, palette, s["witness"])
            g[row:row + width, start:start + word_len] = tile
            block = _block(tile, s["pattern"], bg)
            phase = (np.arange(length) - start) % block.shape[1]
            out[row:row + width] = block[:, phase]
            for side, x in enumerate((start - 1, start + word_len)):
                p = s["pattern"][side]
                if p == "P":
                    g[row + width // 2, x] = marker
                    out[row + width // 2, x] = marker
                else:
                    extra = s["e0" if side == 0 else "e1"]
                    g[row - extra:row + width + extra, x] = marker
                    out[row - extra:row + width + extra, x] = marker
            top += heights[index] + 1 + int(gaps[slot + 1])
        if vertical:
            g, out = g.T.copy(), out.T.copy()
        try:
            actual_bg, actual_marker, groups = _read_input(g)
            if actual_bg != bg or actual_marker != marker:
                continue
            if not np.array_equal(solve(g), out):
                continue
            return {"input": g, "output": out}
        except ValueError:
            continue
    raise RuntimeError("Bounded construction did not find a readable ribbon scene")


def _unique_pair(pair, used, marker):
    if _key(pair["input"]) not in used:
        return pair
    # Recoloring fixes a rare duplicate without changing geometry or marker role.
    colors = [int(c) for c in np.unique(pair["input"]) if c != marker]
    available = [c for c in range(10) if c != marker]
    for values in itertools.permutations(available, len(colors)):
        lut = np.arange(10, dtype=np.int8)
        lut[colors] = values
        candidate = {k: lut[v] for k, v in pair.items()}
        if _key(candidate["input"]) not in used:
            return candidate
    raise RuntimeError("Finite duplicate repair exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    marker = int(rng.integers(10))
    regime = str(rng.choice(["longer", "wider", "recombine"]))
    plans = [[] for _ in range(num_train)]
    slots = rng.permutation(num_train)
    patterns = list(rng.permutation(["LL", "PP", str(rng.choice(["LP", "PL"]))]))
    for i, pattern in enumerate(patterns):
        plans[int(slots[i % num_train])].append((str(pattern), True))
    puzzle, used = {"train": [], "test": []}, set()
    # Opposite orientations occur in the two minimum demonstrations. This is a
    # curriculum decision; it never changes the rule or how solve reads a grid.
    first_vertical = bool(rng.integers(2))
    for i in range(num_train):
        desired = max(len(plans[i]), int(rng.integers(1, 3)))
        while len(plans[i]) < desired:
            plans[i].append((str(rng.choice(["LL", "PP", "LP", "PL"])), False))
        rng.shuffle(plans[i])
        pair = _draw_pair(rng, plans[i], regime, False, marker,
                          first_vertical if i == 0 else not first_vertical if i == 1
                          else bool(rng.integers(2)))
        pair = _unique_pair(pair, used, marker)
        used.add(_key(pair["input"]))
        puzzle["train"].append(pair)
    rng.shuffle(puzzle["train"])
    for _ in range(num_test):
        count = int(rng.integers(3, 5) if regime == "recombine" else rng.integers(1, 5))
        test_plans = [(str(rng.choice(["LL", "PP", "LP", "PL"])), False) for _ in range(count)]
        pair = _draw_pair(rng, test_plans, regime, True, marker, bool(rng.integers(2)))
        pair = _unique_pair(pair, used, marker)
        used.add(_key(pair["input"]))
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative_diff(g, out, groups, **switches):
    return not np.array_equal(_with_writes(g, _orbit_writes(g, groups, **switches)), out)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, marker_roles, patterns, witnesses = set(), set(), set(), set()
    rejected = {"single_copy": False, "ignore_symmetry": False,
                "points_as_lines": False, "mixed_period_two": False}
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert g.shape == out.shape
            assert _key(g) not in seen, "Duplicate input"
            seen.add(_key(g))
            bg, marker, groups = _read_input(g)
            marker_roles.add(marker)
            assert np.array_equal(solve(g), out), "Wrong symmetry orbit or boundary phase"
            assert np.array_equal(out[g != bg], g[g != bg]), "Original material changed"
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(g)))
            assert np.any(g != out), "Every sampled scene must require completion"
            writes = _orbit_writes(g, groups)
            assert set(zip(*np.nonzero(out != g))) <= set(writes)
            for s in groups:
                tile = _tile_from_input(g, bg, s)
                w, l = tile.shape
                assert 1 <= w <= 7 and 1 <= l <= 6
                assert len(_components(tile != bg, diagonal=True)) == 1
                kind = _pattern(s)
                records[split].append((w, l, len(groups)))
                if split == "train":
                    patterns.add("mixed" if kind in ("LP", "PL") else kind)
                    flip = tile[:, ::-1] if kind == "LL" else tile[::-1, ::-1] if kind == "PP" else tile[::-1]
                    if not np.array_equal(tile, flip):
                        witnesses.add("mixed" if kind in ("LP", "PL") else kind)
            if split == "train":
                for name in rejected:
                    rejected[name] |= _alternative_diff(g, out, groups, **{name: True})
    assert len(marker_roles) == 1, "Marker convention changes inside a puzzle"
    assert patterns == witnesses == {"LL", "PP", "mixed"}, "Training must reveal all operators"
    assert all(rejected.values()), "Training leaves a specified shortcut unrefuted"
    train, test = records["train"], records["test"]
    assert (max(x[0] for x in test) > max(x[0] for x in train)
            or max(x[1] for x in test) > max(x[1] for x in train)
            or max(x[2] for x in test) > max(x[2] for x in train)), "Missing structural train/test shift"


def measure_puzzle(puzzle):
    """Compact, pixel-derived family features for bounded diversity checks."""
    result = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            g, out = _grid(pair["input"]), _grid(pair["output"])
            bg, marker, groups = _read_input(g)
            details = []
            for s in groups:
                tile = _tile_from_input(g, bg, s)
                details.append(dict(pattern=_pattern(s), width=tile.shape[0],
                                    length=tile.shape[1], origin=s["a"] + 1,
                                    gap=s["b"] - s["a"],
                                    holes=int(np.count_nonzero(tile == bg)),
                                    palette=len(np.unique(tile[tile != bg])),
                                    marker_cells=len(s["markers"]),
                                    tile=tuple(map(tuple, tile.tolist()))))
            result.append(dict(split=split, shape=g.shape, bg=bg, marker=marker,
                               vertical=groups[0]["vertical"], count=len(groups),
                               additions=int(np.count_nonzero(g != out)), groups=details))
    return result


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    records = [m for p in puzzles for m in measure_puzzle(p)]
    groups = [s for m in records for s in m["groups"]]
    assert len({m["shape"] for m in records}) >= 60
    assert {m["count"] for m in records} == {1, 2, 3, 4}
    assert {m["vertical"] for m in records} == {False, True}
    assert len({m["marker"] for m in records}) >= 8
    assert len({m["bg"] for m in records}) >= 8
    assert {s["pattern"] for s in groups} == {"LL", "PP", "LP", "PL"}
    assert {s["width"] for s in groups} == set(range(1, 8))
    assert {s["length"] for s in groups} == set(range(1, 7))
    assert len({(s["width"], s["length"], s["pattern"], s["origin"]) for s in groups}) >= 120
    assert sum(s["holes"] > 0 for s in groups) >= 30
    assert any(s["palette"] == 1 for s in groups)
    assert any(s["palette"] >= 4 for s in groups)
    assert any(s["marker_cells"] > 2 * s["width"] for s in groups if s["pattern"] == "LL")
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g, out = pair["input"], pair["output"]
                before = g.copy()
                assert np.array_equal(solve(g), out)
                assert np.array_equal(before, g), "solve mutated its input"
                # Geometric and palette equivariance cover all four outward views.
                assert np.array_equal(solve(np.rot90(g)), np.rot90(out))
                assert np.array_equal(solve(np.fliplr(g)), np.fliplr(out))
                lut = np.arange(9, -1, -1, dtype=np.int8)
                assert np.array_equal(solve(lut[g]), lut[out])
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(19331, train, test)
            assert (len(p["train"]), len(p["test"])) == (train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
           for s in ("train", "test")}
    q = bad["test"][0]["output"]
    q[0, 0] = (int(q[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Incorrect label accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed across 42 host seeds")
