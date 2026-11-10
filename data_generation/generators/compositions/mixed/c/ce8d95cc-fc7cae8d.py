"""Guide-oriented elastic swatches, normalized by exact vector-run collapse.

No parent module, saved examples, seed, or generation metadata is used by solve.
All diagnostics in self_test are bounded and in memory; this module writes no files.
"""
from collections import Counter

import numpy as np


SIDES = ("T", "R", "B", "L")
NORMALS = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
SIDE_MAPS = {
    "identity": (0, 1, 2, 3),
    "rot90": (3, 0, 1, 2),
    "rot180": (2, 3, 0, 1),
    "rot270": (1, 2, 3, 0),
    "flip_lr": (0, 3, 2, 1),
    "flip_ud": (2, 1, 0, 3),
    "transpose": (3, 2, 1, 0),
    "anti_transpose": (1, 0, 3, 2),
}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((a >= 0) & (a <= 9))), "invalid ARC color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _adjacent(corner):
    return SIDES[corner], SIDES[(corner + 1) % 4]


def _orient(a, operation):
    if operation == "identity":
        result = a
    elif operation.startswith("rot"):
        result = np.rot90(a, int(operation[3:]) // 90)
    elif operation == "flip_lr":
        result = np.fliplr(a)
    elif operation == "flip_ud":
        result = np.flipud(a)
    elif operation == "transpose":
        result = a.T
    elif operation == "anti_transpose":
        result = np.rot90(a.T, 2)
    else:
        raise ValueError("unknown symmetry")
    return result.copy()


def _operation(guides, controls):
    matches = [name for name, mapping in SIDE_MAPS.items()
               if all(controls.get(SIDES[mapping[SIDES.index(side)]]) == color
                      for side, color in guides.items())]
    _need(len(matches) == 1, "side identities must determine exactly one symmetry")
    return matches[0]


def _collapse(a):
    # Both equivalence relations are measured on complete vectors. Removing
    # redundant rows cannot alter column equality, or conversely.
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    return a[np.ix_(rows, cols)].copy()


def _runs(a, axis):
    vectors = a if axis == 0 else a.T
    starts = [0]
    for i in range(1, len(vectors)):
        if not np.array_equal(vectors[i], vectors[i - 1]):
            starts.append(i)
    return starts, tuple(b - a for a, b in zip(starts, starts[1:] + [len(vectors)]))


def _edge(a, side):
    return {"T": a[0], "R": a[:, -1], "B": a[-1], "L": a[:, 0]}[side]


def _segment(a, side, bounds):
    r, c, h, w = bounds
    return {"T": a[r - 1, c:c + w], "R": a[r:r + h, c + w],
            "B": a[r + h, c:c + w], "L": a[r:r + h, c - 1]}[side]


def _components(a, color):
    seen = np.zeros(a.shape, dtype=bool)
    h, w = a.shape
    for sr, sc in zip(*np.where(a == color)):
        start = int(sr), int(sc)
        if seen[start]:
            continue
        seen[start] = True
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= rr < h and 0 <= cc < w and not seen[rr, cc]
                        and a[rr, cc] == color):
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        yield component


def _read_input(value):
    """Read exclusive marker roles, exact crop, and identity correspondence."""
    a = _grid(value)
    H, W = a.shape
    _need(min(H, W) >= 8, "canvas too small for this family's key and moat")
    bg = int(a[0, 0])
    _need(all(int(a[r, c]) == bg for r, c in ((0, -1), (-1, 0), (-1, -1))),
          "canvas corners must identify background")
    controls = {}
    for side in SIDES:
        core = _edge(a, side)[1:-1]
        if np.all(core == core[0]) and int(core[0]) != bg:
            controls[side] = int(core[0])
    _need(len(controls) == 2 and len(set(controls.values())) == 2,
          "need two distinct complete controls")
    _need(any(set(controls) == set(_adjacent(i)) for i in range(4)),
          "controls must be adjacent")
    descriptions = []
    for color in sorted(controls.values()):
        candidates = []
        for component in _components(a, color):
            if any(r in (0, H - 1) or c in (0, W - 1) for r, c in component):
                continue
            rows = sorted({r for r, _ in component})
            cols = sorted({c for _, c in component})
            if len(rows) == 1 and len(cols) >= 3 and cols == list(range(cols[0], cols[-1] + 1)):
                candidates.append(("H", rows[0], cols[0], cols[-1], color))
            elif len(cols) == 1 and len(rows) >= 3 and rows == list(range(rows[0], rows[-1] + 1)):
                candidates.append(("V", cols[0], rows[0], rows[-1], color))
        _need(len(candidates) == 1, "each identity needs exactly one isolated straight guide")
        descriptions.extend(candidates)
    _need({item[0] for item in descriptions} == {"H", "V"}, "guides must be perpendicular")
    horizontal = next(d for d in descriptions if d[0] == "H")
    vertical = next(d for d in descriptions if d[0] == "V")
    _, hr, c, ce, hc = horizontal
    _, vc, r, re, vc_color = vertical
    h, w = re - r + 1, ce - c + 1
    _need(hr in (r - 1, re + 1) and vc in (c - 1, ce + 1),
          "guide endpoints must define the same rectangle")
    guides = {"T" if hr == r - 1 else "B": hc,
              "L" if vc == c - 1 else "R": vc_color}
    bounds = r, c, h, w
    r0, r1 = r - int("T" in guides), r + h + int("B" in guides)
    c0, c1 = c - int("L" in guides), c + w + int("R" in guides)
    _need(r0 >= 2 and c0 >= 2 and r1 <= H - 2 and c1 <= W - 2,
          "payload and guides need a clear moat inside the frame")
    ring = np.r_[a[r0 - 1, c0 - 1:c1 + 1], a[r1, c0 - 1:c1 + 1],
                 a[r0:r1, c0 - 1], a[r0:r1, c1]]
    _need(bool(np.all(ring == bg)), "guide rectangle must have a blank moat")
    payload = a[r:r + h, c:c + w].copy()
    _need(not (set(np.unique(payload)) & set(controls.values())), "markers cannot be payload content")
    for side in SIDES:
        _need(bool(np.all(_segment(a, side, bounds) == guides.get(side, bg))),
              "exactly two complete external guides are required")
    # The little diagonal corner between the guides is not payload data.
    corner_mask = np.ones((r1 - r0, c1 - c0), dtype=bool)
    corner_mask[r - r0:r + h - r0, c - c0:c + w - c0] = False
    for side, color in guides.items():
        _need(np.count_nonzero(a[r0:r1, c0:c1] == color)
              == (w if side in ("T", "B") else h), "extra marker ink beside payload")
    _need(bool(np.all(np.isin(a[r0:r1, c0:c1][corner_mask], [bg, *guides.values()]))),
          "unexpected ink beside the guides")
    outside = np.ones(a.shape, dtype=bool)
    outside[r0:r1, c0:c1] = False
    rr, cc = np.where(outside & (a != bg))
    _need(bool(np.all(np.minimum.reduce((rr, cc, H - 1 - rr, W - 1 - cc)) <= 1)),
          "decoration must remain at the frame")
    _need(not ((set(np.unique(a[outside])) - {bg, *controls.values()})
               & (set(np.unique(payload)) - {bg})), "frame noise and data colors must differ")
    return dict(background=bg, controls=controls, guides=guides, bounds=bounds,
                payload=payload, operation=_operation(guides, controls))


def _coordinate_orient(info):
    # This implementation is used by solve/validation. Generation instead uses
    # named NumPy symmetries, so an incorrect side-map table is detectable.
    matrix = np.zeros((2, 2), dtype=int)
    for side, color in info["guides"].items():
        target = next(s for s, identity in info["controls"].items() if identity == color)
        matrix += np.outer(NORMALS[target], NORMALS[side])
    payload = info["payload"]
    positions = matrix @ np.indices(payload.shape).reshape(2, -1)
    positions -= positions.min(axis=1, keepdims=True)
    result = np.empty(tuple(positions.max(axis=1) + 1), dtype=np.int8)
    result[positions[0], positions[1]] = payload.ravel()
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Extract by guides, orient by local identities, collapse exact vector runs."""
    return _collapse(_coordinate_orient(_read_input(input_grid)))


def _palette(rng):
    p = tuple(int(x) for x in rng.permutation(10))
    return dict(background=p[0], markers=p[1:3], noise=p[3], data=p[4:])


def _word(rng, length, types):
    word = [int(x) for x in rng.permutation(types)]
    while len(word) < length:
        word.append(int(_pick(rng, [x for x in range(types) if x != word[-1]])))
    # Exchange positions without creating adjacent duplicates. This moves the
    # recurrent vectors away from a fixed prefix/suffix template.
    for _ in range(16):
        i, j = (int(x) for x in rng.integers(0, length, size=2))
        candidate = word.copy()
        candidate[i], candidate[j] = candidate[j], candidate[i]
        if all(a != b for a, b in zip(candidate, candidate[1:])):
            word = candidate
    return word


def _base_mosaic(rng, nr, nc, palette, style):
    data, bg = palette["data"], palette["background"]
    for _ in range(48):
        colors = data[:int(rng.integers(2, 7))]
        if style == 0:  # Woven profiles: crossing values have no overwrite rule.
            rows = rng.choice(colors, size=nr)
            cols = rng.choice(colors, size=nc)
            a = np.where(rng.integers(0, 2, size=(nr, nc)), rows[:, None], cols[None, :])
        elif style == 1:  # Small overlapping color patches.
            a = np.full((nr, nc), colors[0], dtype=np.int8)
            for _ in range(nr + nc + 2):
                r0, r1 = sorted(int(x) for x in rng.integers(0, nr + 1, size=2))
                c0, c1 = sorted(int(x) for x in rng.integers(0, nc + 1, size=2))
                a[r0:r1, c0:c1] = _pick(rng, colors)
        elif style == 2:  # Offset stair/band patterns, sometimes with a hole.
            offsets = rng.integers(0, len(colors), size=nr)
            a = np.array([[colors[(int(offsets[r]) + c) % len(colors)]
                           for c in range(nc)] for r in range(nr)], dtype=np.int8)
        else:  # Irregular tile mosaics, the same equality rule still applies.
            a = rng.choice(colors, size=(nr, nc)).astype(np.int8)
        if rng.random() < .45:
            a[rng.random((nr, nc)) < .12] = bg
        if (len({_key(row[None, :]) for row in a}) == nr
                and len({_key(col[None, :]) for col in a.T}) == nc
                and np.all(np.any(a != bg, axis=0)) and np.all(np.any(a != bg, axis=1))):
            return a.astype(np.int8)
    # Guaranteed finite fallback: a Latin fragment with distinct complete
    # profiles on each axis. It uses ordinary data colors, no secret markers.
    return np.array([[data[(r + c) % max(nr, nc)] for c in range(nc)]
                     for r in range(nr)], dtype=np.int8)


def _canonical(rng, palette, anchor=False):
    h = int(rng.integers(3, 7 if anchor else 11))
    w = int(rng.integers(3, 8 if anchor else 11))
    if anchor and h == w:
        w = 7 if h != 7 else 3
    nr, nc = int(rng.integers(2, min(4, h - 1) + 1)), int(rng.integers(2, min(4, w - 1) + 1))
    row_word, col_word = _word(rng, h, nr), _word(rng, w, nc)
    base = _base_mosaic(rng, nr, nc, palette, 0 if anchor else int(rng.integers(4)))
    core = base[np.ix_(row_word, col_word)]
    blank_sides = _adjacent(int(rng.integers(4))) if anchor else tuple(
        str(x) for x in rng.choice(SIDES, size=int(rng.integers(3)), replace=False))
    pads = ((int("T" in blank_sides), int("B" in blank_sides)),
            (int("L" in blank_sides), int("R" in blank_sides)))
    result = np.pad(core, pads, constant_values=palette["background"])
    assert np.array_equal(_collapse(result), result)
    return result


def _lengths(rng, a, axis, bg, repeat):
    n = a.shape[axis]
    lengths = np.ones(n, dtype=int)
    if not repeat:
        return lengths
    vectors = a if axis == 0 else a.T
    nonblank = np.where(np.any(vectors != bg, axis=1))[0]
    # Force a repeated data-bearing vector, rather than only blank margins.
    chosen = int(rng.choice(nonblank))
    extra = int(rng.integers(2, 26 - n))
    lengths[chosen] += 1
    mode = int(rng.integers(3))
    targets = ([0, n - 1] if mode == 0 else list(range(1, n - 1)) if mode == 1
               else [int(x) for x in rng.choice(n, size=int(rng.integers(1, n)), replace=False)])
    # Keep one run at thickness one so equal-step subsampling is contradicted.
    excluded = int(_pick(rng, [i for i in range(n) if i != chosen]))
    targets = [i for i in targets if i != excluded] or [chosen]
    for index in rng.choice(targets, size=extra - 1):
        lengths[int(index)] += 1
    assert lengths.sum() <= 25 and len(set(lengths)) >= 2
    return lengths


def _edge_position(shape, side, p, depth=0):
    h, w = shape
    return {"T": (depth, p), "R": (p, w - 1 - depth),
            "B": (h - 1 - depth, p), "L": (p, depth)}[side]


def _frame(rng, a, controls, palette, protected):
    for side, color in controls.items():
        _edge(a, side)[1:-1] = color
    if rng.random() < .25:
        return  # Clean cases coexist with decorated ones.
    for side in SIDES:
        n = len(_edge(a, side))
        if side not in controls:
            run = 0
            for p in range(1, n - 1):
                mark = rng.random() < .35 and run < 2
                run = run + 1 if mark else 0
                if mark:
                    a[_edge_position(a.shape, side, p)] = palette["noise"]
        else:
            for p in rng.permutation(np.arange(2, n - 2))[:int(rng.integers(5))]:
                pos = _edge_position(a.shape, side, int(p), 1)
                if not protected[pos]:
                    a[pos] = controls[side]
    for _ in range(int(rng.integers(5))):
        side = str(_pick(rng, SIDES))
        pos = _edge_position(a.shape, side, int(rng.integers(2, len(_edge(a, side)) - 2)), 1)
        if not protected[pos] and a[pos] == palette["background"]:
            a[pos] = palette["noise"]


def _make_pair(rng, config, palette, anchor=False):
    outer_corner, inner_corner, reverse = config
    inner, outer = _adjacent(inner_corner), _adjacent(outer_corner)
    canonical = _canonical(rng, palette, anchor)
    no_op = None if anchor or rng.random() >= .18 else int(rng.integers(2))
    row_lengths = _lengths(rng, canonical, 0, palette["background"], no_op != 0)
    col_lengths = _lengths(rng, canonical, 1, palette["background"], no_op != 1)
    payload = np.repeat(np.repeat(canonical, row_lengths, axis=0), col_lengths, axis=1)
    h, w = payload.shape
    H, W = int(rng.integers(h + 5, 31)), int(rng.integers(w + 5, 31))
    if rng.random() < .12:
        H = W = 30
    r = int(rng.integers(2 + int("T" in inner), H - h - 1 - int("B" in inner)))
    c = int(rng.integers(2 + int("L" in inner), W - w - 1 - int("R" in inner)))
    guides = dict(zip(inner, palette["markers"]))
    controls = dict(zip(outer[::-1] if reverse else outer, palette["markers"]))
    operation = _operation(guides, controls)
    a = np.full((H, W), palette["background"], dtype=np.int8)
    protected = np.zeros(a.shape, dtype=bool)
    r0, r1 = r - int("T" in inner), r + h + int("B" in inner)
    c0, c1 = c - int("L" in inner), c + w + int("R" in inner)
    protected[r0 - 1:r1 + 1, c0 - 1:c1 + 1] = True
    _frame(rng, a, controls, palette, protected)
    a[r:r + h, c:c + w] = payload
    for side, color in guides.items():
        _segment(a, side, (r, c, h, w))[:] = color
    return {"input": a, "output": _orient(canonical, operation)}


def _swap_controls(pair):
    a = pair["input"].copy()
    info = _read_input(a)
    colors = list(info["controls"].values())
    replacements = []
    H, W = a.shape
    for color, target in zip(colors, colors[::-1]):
        for component in _components(a, color):
            if any(r in (0, H - 1) or c in (0, W - 1) for r, c in component):
                replacements.extend((pos, target) for pos in component)
    for pos, color in replacements:
        a[pos] = color
    controls = {side: colors[1 - colors.index(color)] for side, color in info["controls"].items()}
    operation = _operation(info["guides"], controls)
    return {"input": a, "output": _orient(_collapse(info["payload"]), operation)}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    _need(low <= value <= high, f"{name} outside legal range")
    return value


def _unique_pair(pair, used):
    # Forty affine bijections of the ARC palette give forty different inputs:
    # three distinct visible colors already preclude coincident affine maps.
    # There are at most 13 previously accepted inputs. No stochastic retry or
    # unbounded retention is needed. The first candidate is the unchanged pair.
    for multiplier in (1, 3, 7, 9):
        for shift in range(10):
            candidate = {k: ((v.astype(int) * multiplier + shift) % 10).astype(np.int8)
                         for k, v in pair.items()}
            if _key(candidate["input"]) not in used:
                return candidate
    raise AssertionError("uniqueness bound violated")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ne = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    _need(nt > ne, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    held_corner = int(rng.integers(4))
    arrangements = [(o, i, rev) for o in range(4) if o != held_corner
                    for i in range(4) for rev in (False, True)]
    rotations = []
    for config in arrangements:
        o, i, rev = config
        guides = dict(zip(_adjacent(i), (1, 2)))
        controls = dict(zip(_adjacent(o)[::-1] if rev else _adjacent(o), (1, 2)))
        if _operation(guides, controls) in ("rot90", "rot270"):
            rotations.append(config)
    first_config = _pick(rng, rotations)
    palette = _palette(rng)
    stable_palette = rng.random() < .25
    first = _make_pair(rng, first_config, palette, anchor=True)
    puzzle = {"train": [first, _swap_controls(first)], "test": []}
    used = {_key(pair["input"]) for pair in puzzle["train"]}
    test_configs = [(held_corner, i, rev) for i in range(4) for rev in (False, True)]
    test_order = [int(i) for i in rng.permutation(len(test_configs))]
    for split, count in (("train", nt - 2), ("test", ne)):
        for index in range(count):
            config = _pick(rng, arrangements) if split == "train" else test_configs[test_order[index]]
            pair = _make_pair(rng, config, palette if stable_palette else _palette(rng))
            pair = _unique_pair(pair, used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _separated_recurrence(a, axis):
    vectors = a if axis == 0 else a.T
    return len({vector.tobytes() for vector in vectors}) < len(vectors)


def _asymmetric(a):
    return len({_key(_orient(a, operation)) for operation in SIDE_MAPS}) == 8


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, output = _grid(pair["input"]), _grid(pair["output"])
    info = _read_input(a)
    oriented = _coordinate_orient(info)
    # Loop-derived representative indices provide a second contraction check.
    rows, row_runs = _runs(oriented, 0)
    cols, col_runs = _runs(oriented, 1)
    expected = oriented[np.ix_(rows, cols)]
    assert np.array_equal(output, expected), "wrong crop, orientation, contraction, or retained value"
    assert np.array_equal(solve(a), output)
    assert oriented.shape != output.shape, "at least one axis must visibly contract"
    assert np.array_equal(np.repeat(np.repeat(output, row_runs, axis=0), col_runs, axis=1), oriented)
    assert len(_runs(output, 0)[0]) == output.shape[0]
    assert len(_runs(output, 1)[0]) == output.shape[1]
    assert _separated_recurrence(output, 0) and _separated_recurrence(output, 1)
    assert not (set(np.unique(output)) & set(info["controls"].values()))
    info.update(canonical=_collapse(info["payload"]), oriented=oriented,
                row_runs=row_runs, col_runs=col_runs)
    return info


def validate_generated_puzzle(puzzle):
    """Check actual grids, exact reconstruction, teaching contrasts, and shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, information = set(), {}
    for split in ("train", "test"):
        information[split] = []
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            key = _key(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            information[split].append(info)
    train, test = information["train"], information["test"]
    contrasts = [(a, b) for i, a in enumerate(train) for b in train[i + 1:]
                 if a["bounds"] == b["bounds"] and a["guides"] == b["guides"]
                 and set(a["controls"]) == set(b["controls"])
                 and a["controls"] != b["controls"]
                 and np.array_equal(a["payload"], b["payload"])
                 and _asymmetric(a["canonical"])]
    assert contrasts, "need copied-payload correspondence reversal in training"
    assert any(a["operation"] in ("rot90", "rot270") for a in train)
    assert any(a["operation"] in ("flip_lr", "flip_ud", "transpose", "anti_transpose") for a in train)
    assert any(max(a["row_runs"]) > 1 and max(a["col_runs"]) > 1
               and len(set(a["row_runs"])) > 1 and len(set(a["col_runs"])) > 1 for a in train)
    has_blank = lambda a: any(np.all(_edge(a["payload"], side) == a["background"]) for side in SIDES)
    assert any(has_blank(a) for a in train), "guide crop needs a demonstrated blank-edge witness"
    train_corners = {frozenset(a["controls"]) for a in train}
    assert all(frozenset(a["controls"]) not in train_corners for a in test), "test must hold out an outer corner"
    # Explicit necessity checks on the asymmetric teaching contrast.
    a, b = contrasts[0]
    assert not np.array_equal(_collapse(a["payload"]), _collapse(a["oriented"]))
    assert not np.array_equal(_collapse(a["oriented"]), _collapse(b["oriented"]))


def _puzzle_key(puzzle):
    return tuple(tuple((_key(_grid(p["input"])), _key(_grid(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """42 seeds, all legal counts, exact algebra, mutations; no artifact writes."""
    metrics = {"seeds": 42, "pairs": 0, "symmetries": {"train": Counter(), "test": Counter()}}
    shapes, canonical_shapes, layouts, palettes, puzzles = set(), set(), set(), set(), set()
    blank = no_op = expanded_data = 0
    for seed in range(42):
        nt = 2 + seed % 9
        ne = 1 + (seed // 3) % min(4, nt - 1)
        puzzle = generate_puzzle(seed, nt, ne)
        puzzles.add(_puzzle_key(puzzle))
        if seed < 3:
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed, nt, ne))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze_pair(pair)
                metrics["pairs"] += 1
                metrics["symmetries"][split][a["operation"]] += 1
                shapes.add(pair["input"].shape)
                canonical_shapes.add(a["canonical"].shape)
                palettes.add((a["background"], tuple(sorted(a["controls"].values()))))
                layouts.add((a["bounds"], tuple(a["guides"]), tuple(a["controls"]),
                             a["row_runs"], a["col_runs"]))
                blank += any(np.all(_edge(a["payload"], s) == a["background"]) for s in SIDES)
                no_op += (max(a["row_runs"]) == 1) != (max(a["col_runs"]) == 1)
                for axis, runs in ((0, a["row_runs"]), (1, a["col_runs"])):
                    vectors = pair["output"] if axis == 0 else pair["output"].T
                    expanded_data += any(n > 1 and np.any(v != a["background"]) for n, v in zip(runs, vectors))
                # Both axis orders and all D4 operations commute with exact
                # contraction once the correct rectangle has been extracted.
                for operation in SIDE_MAPS:
                    assert np.array_equal(_collapse(_orient(a["payload"], operation)),
                                          _orient(a["canonical"], operation))
                assert not np.array_equal(_collapse(pair["input"]), pair["output"])
    assert len(puzzles) == 42 and len(shapes) >= 50 and len(canonical_shapes) >= 25
    assert len(layouts) >= 120 and len(palettes) >= 30
    assert all(set(metrics["symmetries"][s]) == set(SIDE_MAPS) for s in ("train", "test"))
    assert blank >= 50 and no_op >= 5 and expanded_data >= 150
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    original = generate_puzzle(0)
    prototype = original["train"][0]
    affine_keys = [_key(((prototype["input"].astype(int) * m + b) % 10).astype(np.int8))
                   for m in (1, 3, 7, 9) for b in range(10)]
    assert len(set(affine_keys)) == 40
    assert _key(_unique_pair(prototype, set(affine_keys[:13]))["input"]) not in set(affine_keys[:13])
    for mutation in ("cell", "crop", "guide", "uncollapsed", "global_unique"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
        pair = bad["test"][0]
        info = _read_input(pair["input"])
        if mutation == "cell":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif mutation == "crop":
            pair["output"] = pair["output"][1:].copy()
        elif mutation == "guide":
            side = next(iter(info["guides"]))
            segment = _segment(pair["input"], side, info["bounds"])
            segment[len(segment) // 2] = info["background"]
        elif mutation == "uncollapsed":
            pair["output"] = _coordinate_orient(info)
        else:
            oriented = _coordinate_orient(info)
            row_keep, col_keep = [], []
            for i, row in enumerate(oriented):
                if not any(np.array_equal(row, oriented[j]) for j in row_keep):
                    row_keep.append(i)
            for i, col in enumerate(oriented.T):
                if not any(np.array_equal(col, oriented[:, j]) for j in col_keep):
                    col_keep.append(i)
            pair["output"] = oriented[np.ix_(row_keep, col_keep)]
        try:
            validate_generated_puzzle(bad)
        except (ValueError, AssertionError):
            pass
        else:
            raise AssertionError("invalid puzzle accepted: " + mutation)
    metrics.update(input_shapes=len(shapes), canonical_shapes=len(canonical_shapes),
                   layouts=len(layouts), local_palettes=len(palettes), blank_edge_pairs=blank,
                   one_axis_noop_pairs=no_op, axes_with_repeated_data=expanded_data)
    metrics["symmetries"] = {s: dict(c) for s, c in metrics["symmetries"].items()}
    return metrics
