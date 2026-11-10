"""Phase-printing arrows: directional writes into a periodic transparent stencil.

All conventions are visible in one input. No parent generator, file access,
training labels, seed state, or generation metadata is used by solve().
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np


DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0), (1, -1), (0, -1), (-1, -1))
DISTANCE = 5


def _cells(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return tuple(sorted((r + a, c + b) for a, b in offsets))


def _arrow(tip, direction):
    return (tuple(tip), tuple(direction), _cells(tip, direction))


def _endpoint(arrow, distance=DISTANCE):
    return tuple(arrow[0][a] + distance * arrow[1][a] for a in (0, 1))


def _phase(point, native):
    return (point[0] % native[0], point[1] % native[1])


def _compatible(arrow, selected):
    return all(max(abs(r - x), abs(c - y)) > 1
               for r, c in arrow[2] for other in selected for x, y in other[2])


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= d <= 30 for d in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or np.any((array < 0) | (array > 9))):
        raise ValueError("grids must be integer 2D arrays of colors 0..9, size 1..30")
    return array.astype(np.int64, copy=False)


def _decode(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    lines = [r for r in range(grid.shape[0])
             if grid[r, 0] != bg and np.all(grid[r] == grid[r, 0])]
    if len(lines) != 1:
        raise ValueError("expected one solid non-background separator row")
    h = lines[0]
    field, strip = grid[:h], grid[h + 1:]
    marker = int(grid[h, 0])
    if h < 8 or field.shape[1] < 8 or not 2 <= strip.shape[0] <= 7:
        raise ValueError("invalid field or carrier height")
    used_columns = np.flatnonzero(np.any(strip != bg, axis=0))
    if len(used_columns) == 0:
        raise ValueError("missing carrier")
    m = int(used_columns[-1]) + 1
    n, w = strip.shape[0], field.shape[1]
    carrier = strip[:, :m]
    if (not 2 <= m <= 9 or m >= w or np.any(carrier == bg)
            or np.any(strip[:, m:] != bg) or np.any(carrier == marker)
            or np.any(field == marker) or h % n or w % m
            or h // n < 2 or w // m < 2):
        raise ValueError("invalid carrier, padding, palette, or replication factors")
    remaining = set(map(tuple, np.argwhere(field != bg).tolist()))
    arrows = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            neighbors = {(r + a, c + b) for a in (-1, 0, 1)
                         for b in (-1, 0, 1)} & remaining
            remaining -= neighbors
            component |= neighbors
            todo.extend(sorted(neighbors))
        if len(component) != 3 or len({int(field[p]) for p in component}) != 1:
            raise ValueError("arrows must be separated monochrome triominoes")
        matches = [_arrow(t, d) for t in sorted(component) for d in DIRECTIONS
                   if set(_cells(t, d)) == component]
        if len(matches) != 1:
            raise ValueError("invalid or ambiguous arrow geometry")
        arrow = matches[0]
        point = _endpoint(arrow)
        if not (0 <= point[0] < h and 0 <= point[1] < w):
            raise ValueError("arrow endpoint outside field")
        arrows.append(arrow)
    if not arrows:
        raise ValueError("at least one arrow is required")
    colors = [int(field[a[0]]) for a in arrows]
    return bg, marker, field, carrier, arrows, colors


def _writes(arrows, colors, native, distance=DISTANCE, use_paths=False):
    groups = defaultdict(list)
    for arrow, color in zip(arrows, colors):
        steps = range(1, distance + 1) if use_paths else (distance,)
        for step in steps:
            groups[_phase(_endpoint(arrow, step), native)].append(int(color))
    return groups


def _render(carrier, shape, groups, mode="normal", base_mode="replicate",
            phase_shift=(0, 0), background=0):
    n, m = carrier.shape
    h, w = shape
    rr, cc = np.indices(shape)
    sr, sc = rr // (h // n), cc // (w // m)
    if base_mode == "tile_rows":
        sr = rr % n
    if base_mode == "tile_columns":
        sc = cc % m
    base = carrier[sr, sc].copy()
    for (r, c), inks in groups.items():
        active = len(inks) == 1
        if mode == "no_cancel":
            active = True
        elif mode == "parity":
            active = len(inks) % 2 == 1
        if active:
            mask = ((rr + phase_shift[0]) % n == r) & ((cc + phase_shift[1]) % m == c)
            base[mask] = inks[0]
    if mode == "opaque":
        transparent = np.ones(shape, dtype=bool)
        for (r, c), inks in groups.items():
            if len(inks) == 1:
                transparent[(rr % n == r) & (cc % m == c)] = False
        base[transparent] = background
    return base


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read just the input, expand its carrier, and print uncrowded shot phases."""
    _, _, field, carrier, arrows, colors = _decode(input_grid)
    groups = _writes(arrows, colors, carrier.shape)
    return _render(carrier, field.shape, groups)


def _key(grid):
    grid = np.asarray(grid)
    return (grid.shape, grid.astype(np.uint8).tobytes())


def _normalized(grid):
    names = {}
    return (grid.shape, bytes(names.setdefault(int(v), len(names)) for v in grid.flat))


def _candidates(shape):
    h, w = shape
    result = []
    for r in range(h):
        for c in range(w):
            for d in DIRECTIONS:
                a = _arrow((r, c), d)
                p = _endpoint(a)
                if ((0, 0) not in a[2] and 0 <= p[0] < h and 0 <= p[1] < w
                        and all(0 <= x < h and 0 <= y < w for x, y in a[2])):
                    result.append(a)
    return result


def _pick_group(rng, pools, count, selected, used_phases, native, kind):
    """Select an interaction geometrically, with a finite search per proposal."""
    keys = list(pools)
    for index in rng.permutation(len(keys)):
        key = keys[int(index)]
        pool = pools[key]
        phase = _phase(_endpoint(pool[0]), native)
        if phase in used_phases:
            continue
        order = rng.permutation(len(pool))
        # Multiple greedy starts avoid favoring a compass direction or position.
        for start in range(min(12, len(order))):
            chosen = []
            points = set()
            for offset in range(len(order)):
                candidate = pool[int(order[(start + offset) % len(order)])]
                point = _endpoint(candidate)
                if kind == "alias" and point in points:
                    continue
                if _compatible(candidate, selected + chosen):
                    chosen.append(candidate)
                    points.add(point)
                if len(chosen) == count:
                    return chosen, phase
    return None


def _layout(rng, native, shape, groups_spec, single_count):
    candidates = _candidates(shape)
    phases, exact = defaultdict(list), defaultdict(list)
    for a in candidates:
        phases[_phase(_endpoint(a), native)].append(a)
        exact[_endpoint(a)].append(a)
    selected, used, blocks = [], set(), []
    for kind, count in groups_spec:
        hit = _pick_group(rng, phases if kind == "alias" else exact,
                          count, selected, used, native, kind)
        if hit is None:
            return None
        group, phase = hit
        blocks.append(group)
        selected += group
        used.add(phase)
    for index in rng.permutation(len(candidates)):
        a = candidates[int(index)]
        phase = _phase(_endpoint(a), native)
        if phase not in used and _compatible(a, selected):
            selected.append(a)
            used.add(phase)
            blocks.append([a])
            single_count -= 1
            if single_count == 0:
                return blocks
    return None


def _carrier(rng, native, palette, teaching):
    n, m = native
    inks = np.asarray(rng.choice(palette, size=int(rng.integers(2, len(palette) + 1)),
                                 replace=False))
    rr, cc = np.indices(native)
    kind = int(rng.integers(3 if teaching else 7))
    if kind == 0:
        # Unequal contiguous regions, with independently placed cut points.
        labels = (rr >= int(rng.integers(1, n))).astype(int)
        labels += (cc >= int(rng.integers(1, m))).astype(int)
        labels %= len(inks)
    elif kind == 1:
        labels = (rr // int(rng.integers(1, n)) + cc // int(rng.integers(1, m))) % len(inks)
    elif kind == 2:
        labels = rng.integers(len(inks), size=native)
    elif kind == 3:
        labels = np.tile(rng.integers(len(inks), size=m), (n, 1))
    elif kind == 4:
        labels = np.tile(rng.integers(len(inks), size=(n, 1)), (1, m))
    elif kind == 5:
        labels = np.zeros(native, dtype=int)
    else:
        center = (int(rng.integers(n)), int(rng.integers(m)))
        labels = (abs(rr - center[0]) + abs(cc - center[1])) % len(inks)
    if rng.random() < .5:
        labels = labels[::-1]
    if rng.random() < .5:
        labels = labels[:, ::-1]
    return inks[labels].astype(np.int64)


def _assemble(field, carrier, background, marker):
    strip = np.full((carrier.shape[0], field.shape[1]), background, dtype=np.int64)
    strip[:, :carrier.shape[1]] = carrier
    separator = np.full((1, field.shape[1]), marker, dtype=np.int64)
    inp = np.concatenate((field, separator, strip), axis=0)
    return {"input": inp, "output": solve(inp)}


def _sample_pair(rng, params, lesson=None, novel_shape=None):
    teaching = lesson is not None
    for _ in range(36):
        if novel_shape is None:
            n = int(rng.integers(3 if teaching else 2, 8))
            m = int(rng.integers(3 if teaching else 2, 10))
        else:
            n, m = novel_shape
        lo_h = (15 if teaching else 8)
        lo_w = (16 if teaching else 8)
        kr_lo, kr_hi = max(2, (lo_h + n - 1) // n), (29 - n) // n
        kc_lo, kc_hi = max(2, (lo_w + m - 1) // m), 30 // m
        if kr_lo > kr_hi or kc_lo > kc_hi:
            continue
        kr = int(rng.integers(kr_lo, kr_hi + 1))
        kc = int(rng.integers(kc_lo, kc_hi + 1))
        native, shape = (n, m), (n * kr, m * kc)
        if lesson == "double":
            specs, singles = [("alias", 2)], int(rng.integers(2, 5))
        elif lesson == "triple_exact":
            specs, singles = [("alias", 3), ("exact", 2)], int(rng.integers(1, 4))
        else:
            mode = int(rng.integers(6))
            specs = ([], [("alias", 2)], [("exact", 2)], [("alias", 3)],
                     [("alias", 4)], [("alias", 2), ("alias", 2)])[mode]
            singles = int(rng.integers(1, min(7, n * m - len(specs)) + 1))
        blocks = _layout(rng, native, shape, specs, singles)
        if blocks is None:
            continue
        carrier = _carrier(rng, native, params["carrier"], teaching)
        if teaching:
            ink_pool = params["arrow"]
        else:
            ink_pool = params["arrow"] if rng.random() < .6 else params["payload"]
        field = np.full(shape, params["background"], dtype=np.int64)
        base_ink = int(rng.choice(ink_pool))
        # Crowded phases are monochrome in teaching, so cancellation cannot be
        # explained by conflicting ink colors. Some supplementary groups mix inks.
        for index, block in enumerate(blocks):
            color = int(ink_pool[(index + int(rng.integers(len(ink_pool)))) % len(ink_pool)])
            if not teaching and rng.random() < .35:
                color = base_ink
            for a in block:
                actual = color
                if not teaching and len(block) > 1 and rng.random() < .3:
                    actual = int(rng.choice(ink_pool))
                for p in a[2]:
                    field[p] = actual
        return _assemble(field, carrier, params["background"], params["marker"])
    return None


def _fallback_training(params, rng):
    """Constructive teaching witnesses, used only after random selection fails.

    Disjoint role pools make evidence invariant under every palette permutation.
    Reflections vary the witnesses without changing their phase-printing rule.
    """
    specifications = (
        ((18, 30), (6, 5), 3, 2,
         (((6, 19), (0, 1), 0), ((9, 5), (-1, -1), 1),
          ((12, 13), (1, 1), 2), ((12, 18), (1, -1), 2),
          ((13, 3), (-1, 0), 3), ((15, 28), (-1, -1), 0))),
        ((18, 21), (6, 7), 2, 5,
         (((1, 2), (0, 1), 0), ((2, 17), (1, 0), 0),
          ((3, 7), (1, 1), 1), ((7, 8), (0, -1), 0),
          ((7, 15), (0, -1), 0), ((8, 11), (0, -1), 0),
          ((9, 2), (1, 0), 2), ((13, 12), (-1, 0), 1))),
    )
    pairs = []
    for shape, native, cut_r, cut_c, specs in specifications:
        field = np.full(shape, params["background"], dtype=np.int64)
        for tip, direction, ink in specs:
            for p in _cells(tip, direction):
                field[p] = params["arrow"][ink]
        rr, cc = np.indices(native)
        labels = ((rr >= cut_r) ^ (cc >= cut_c)).astype(int)
        carrier = np.asarray(params["carrier"][:2])[labels]
        for axis in (0, 1):
            if rng.integers(2):
                field = np.flip(field, axis).copy()
                carrier = np.flip(carrier, axis).copy()
        pairs.append(_assemble(field, carrier, params["background"], params["marker"]))
    return pairs


def _distinct_simple_fallback(params, seen, native=(3, 4)):
    """Finite enumeration guarantees a fresh legal input for remaining slots."""
    n, m = native
    h, w = n * max(2, (8 + n - 1) // n), m * max(2, (8 + m - 1) // m)
    rr, cc = np.indices(native)
    carrier = np.asarray(params["carrier"][:2])[(rr + cc) % 2]
    for ink in params["arrow"]:
        for r in range(2, h - 1):
            for c in range(1, w - 5):
                field = np.full((h, w), params["background"], dtype=np.int64)
                for p in _cells((r, c), (0, 1)):
                    field[p] = ink
                pair = _assemble(field, carrier, params["background"], params["marker"])
                if _key(pair["input"]) not in seen:
                    return pair
    # Even the smallest native shape has >= 40 choices; at most 13 are used.
    raise AssertionError("simple fallback enumeration unexpectedly exhausted")


REQUIRED_EVIDENCE = frozenset({
    "no_arrows", "tips", "reverse", "physical_collision", "no_cancel", "parity",
    "tile_rows", "tile_columns", "enlarge_stencil", "fuse_then_enlarge",
    "finite_print", "opaque", "phase_rows", "phase_columns", "paths",
    "alias_double", "alias_triple", "exact_collision", "single", "multi_ink",
    "monochrome_double", "monochrome_triple",
})


def _evidence(pair):
    bg, _, field, carrier, arrows, inks = _decode(pair["input"])
    out = np.asarray(pair["output"])
    native, shape = carrier.shape, field.shape
    n, m = native
    groups = _writes(arrows, inks, native)
    rr, cc = np.indices(shape)
    sr, sc = rr // (shape[0] // n), cc // (shape[1] // m)
    base = carrier[sr, sc]
    cover = set()

    def differs(name, alternative):
        if not np.array_equal(out, alternative):
            cover.add(name)

    differs("no_arrows", base)
    differs("tips", _render(carrier, shape, _writes(arrows, inks, native, 0)))
    differs("reverse", _render(carrier, shape, _writes(arrows, inks, native, -5)))
    for mode in ("no_cancel", "parity", "opaque"):
        differs(mode, _render(carrier, shape, groups, mode=mode, background=bg))
    for axis in ("rows", "columns"):
        differs("tile_" + axis, _render(carrier, shape, groups, base_mode="tile_" + axis))
        shift = (1, 0) if axis == "rows" else (0, 1)
        differs("phase_" + axis, _render(carrier, shape, groups, phase_shift=shift))
    stencil = np.full(native, bg, dtype=np.int64)
    for p, colors in groups.items():
        if len(colors) == 1:
            stencil[p] = colors[0]
    differs("enlarge_stencil", np.where(stencil[sr, sc] == bg, base, stencil[sr, sc]))
    fused = np.where(stencil == bg, carrier, stencil)
    differs("fuse_then_enlarge", fused[sr, sc])
    finite = base.copy()
    physical = defaultdict(list)
    for a, color in zip(arrows, inks):
        physical[_endpoint(a)].append(color)
    physical_groups = defaultdict(list)
    for p, colors in physical.items():
        if len(colors) == 1:
            finite[p] = colors[0]
            physical_groups[_phase(p, native)].append(colors[0])
    differs("finite_print", finite)
    differs("physical_collision", _render(carrier, shape, physical_groups, mode="no_cancel"))
    differs("paths", _render(carrier, shape, _writes(arrows, inks, native, use_paths=True)))
    per_phase = defaultdict(list)
    for a in arrows:
        per_phase[_phase(_endpoint(a), native)].append(_endpoint(a))
    if any(len(p) == 2 and len(set(p)) == 2 for p in per_phase.values()):
        cover.add("alias_double")
    if any(len(p) >= 3 and len(set(p)) >= 3 for p in per_phase.values()):
        cover.add("alias_triple")
    if any(len(colors) == 2 and len(set(colors)) == 1 for colors in groups.values()):
        cover.add("monochrome_double")
    if any(len(colors) >= 3 and len(set(colors)) == 1 for colors in groups.values()):
        cover.add("monochrome_triple")
    if any(len(colors) >= 2 for colors in physical.values()):
        cover.add("exact_collision")
    if any(len(colors) == 1 for colors in groups.values()):
        cover.add("single")
    active_inks = {colors[0] for colors in groups.values() if len(colors) == 1}
    if len(active_inks) >= 2:
        cover.add("multi_ink")
    # Deliberately test folded distances even when their literal endpoints leave
    # the field. This is stronger than rejecting them solely on boundary grounds.
    for distance in range(1, 30):
        if distance != DISTANCE:
            differs("distance_" + str(distance),
                    _render(carrier, shape, _writes(arrows, inks, native, distance)))
    return cover


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError("training count must be 2..10 and test count 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    carrier_count = int(rng.integers(2, 5))
    params = {"background": palette[0], "marker": palette[1],
              "carrier": palette[2:2 + carrier_count],
              "arrow": palette[2 + carrier_count:], "payload": palette[2:]}
    needed = REQUIRED_EVIDENCE | {"distance_" + str(d) for d in range(1, 30) if d != 5}
    # Bounded teaching selection checks the pixels, rather than assuming a
    # named interaction alone makes the intended operations observable.
    pool = []
    train = None
    for _ in range(80):
        first = _sample_pair(rng, params, "double")
        second = _sample_pair(rng, params, "triple_exact")
        if first is None or second is None:
            continue
        proposed = [first, second]
        cover = _evidence(first) | _evidence(second)
        if (needed <= cover and _decode(first["input"])[3].shape != _decode(second["input"])[3].shape
                and first["output"].shape != second["output"].shape):
            train = proposed
            break
        pool.extend(proposed)
    if train is None:
        # Search accumulated candidates for complementary evidence, also bounded.
        covers = [_evidence(p) for p in pool]
        for i, j in combinations(range(len(pool)), 2):
            if (needed <= covers[i] | covers[j]
                    and _decode(pool[i]["input"])[3].shape != _decode(pool[j]["input"])[3].shape
                    and pool[i]["output"].shape != pool[j]["output"].shape):
                train = [pool[i], pool[j]]
                break
    if train is None:
        train = _fallback_training(params, rng)
        assert needed <= (_evidence(train[0]) | _evidence(train[1]))
    puzzle = {"train": train, "test": []}
    seen = {_key(p["input"]) for p in train}
    for _ in range(int(num_train) - 2):
        for attempt in range(80):
            pair = _sample_pair(rng, params)
            if pair is not None and _key(pair["input"]) not in seen:
                puzzle["train"].append(pair)
                seen.add(_key(pair["input"]))
                break
        else:
            pair = _distinct_simple_fallback(params, seen)
            puzzle["train"].append(pair)
            seen.add(_key(pair["input"]))
    observed = {_decode(p["input"])[3].shape for p in puzzle["train"]}
    unseen = [(n, m) for n in range(2, 8) for m in range(2, 10) if (n, m) not in observed]
    # All queries use a new native period, with no change to printing semantics.
    for _ in range(int(num_test)):
        for attempt in range(80):
            native = unseen[int(rng.integers(len(unseen)))]
            pair = _sample_pair(rng, params, novel_shape=native)
            if pair is not None and _key(pair["input"]) not in seen:
                puzzle["test"].append(pair)
                seen.add(_key(pair["input"]))
                break
        else:
            pair = _distinct_simple_fallback(params, seen, unseen[0])
            puzzle["test"].append(pair)
            seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, conventions, training_cover = set(), set(), set()
    periods = {s: set() for s in puzzle}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            bg, marker, field, carrier, arrows, colors = _decode(inp)
            conventions.add((bg, marker))
            periods[split].add(carrier.shape)
            assert out.shape == field.shape, "output shape must equal the arrow field"
            groups = _writes(arrows, colors, carrier.shape)
            kr, kc = field.shape[0] // carrier.shape[0], field.shape[1] // carrier.shape[1]
            # Explicit per-pixel invariant, separate from _render's array masks.
            for r in range(out.shape[0]):
                for c in range(out.shape[1]):
                    phase_inks = groups.get((r % carrier.shape[0], c % carrier.shape[1]), [])
                    expected = phase_inks[0] if len(phase_inks) == 1 else carrier[r // kr, c // kc]
                    assert out[r, c] == expected, "phase or carrier invariant violated"
            assert marker not in out, "separator is not output ink"
            assert np.array_equal(solve(inp.copy()), out), "input-only solve mismatch"
            if split == "train":
                training_cover |= _evidence(pair)
    assert len(conventions) == 1, "background and separator roles must agree within a puzzle"
    assert len(periods["train"]) >= 2, "need varied demonstrated native periods"
    assert not periods["train"] & periods["test"], "query must have an unseen native period"
    needed = REQUIRED_EVIDENCE | {"distance_" + str(d) for d in range(1, 30) if d != 5}
    assert needed <= training_cover, "missing teaching evidence: " + str(needed - training_cover)


def _features(pair):
    bg, marker, field, carrier, arrows, colors = _decode(pair["input"])
    groups = _writes(arrows, colors, carrier.shape)
    return {"native": carrier.shape, "shape": field.shape, "background": bg,
            "marker": marker, "count": len(arrows), "directions": {a[1] for a in arrows},
            "structure": _normalized(carrier), "output_structure": _normalized(pair["output"]),
            "replication": (field.shape[0] // carrier.shape[0], field.shape[1] // carrier.shape[1]),
            "max_hits": max(map(len, groups.values())),
            "collision_phases": sum(len(v) > 1 for v in groups.values()),
            "single_phases": sum(len(v) == 1 for v in groups.values()),
            "arrow_inks": len(set(colors)), "carrier_inks": len(np.unique(carrier)),
            "overlap": bool(set(colors) & set(map(int, np.unique(carrier))))}


def self_test():
    """Bounded checks of 42 seeds, structural variation, and semantic mutations."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({tuple(_key(p["input"]) for p in x["train"] + x["test"])
                for x in puzzles}) == 42
    features = [_features(p) for x in puzzles for s in x for p in x[s]]
    assert len({f["native"] for f in features}) >= 25
    assert len({f["shape"] for f in features}) >= 50
    assert len({f["replication"] for f in features}) >= 15
    assert len({f["structure"] for f in features}) >= 100
    assert len({f["output_structure"] for f in features}) >= 160
    assert len({f["background"] for f in features}) >= 8
    assert len({f["count"] for f in features}) >= 8
    assert {d for f in features for d in f["directions"]} == set(DIRECTIONS)
    assert any(f["count"] == 1 for f in features)
    assert any(f["max_hits"] >= 4 for f in features)
    assert any(f["carrier_inks"] == 1 for f in features)
    assert any(f["overlap"] for f in features)
    assert any(f["collision_phases"] >= 2 for f in features)
    assert any(f["arrow_inks"] >= 3 for f in features)
    needed = REQUIRED_EVIDENCE | {"distance_" + str(d) for d in range(1, 30) if d != 5}
    # Exercise fallback reflections and every role-pool size. These
    # are constructive witnesses, not a probabilistic hope of finding evidence.
    for carrier_count in (2, 3, 4):
        params = {"background": 0, "marker": 9,
                  "carrier": list(range(1, 1 + carrier_count)),
                  "arrow": list(range(1 + carrier_count, 9))}
        for fallback_seed in range(16):
            witnesses = _fallback_training(params, np.random.default_rng(fallback_seed))
            assert needed <= set().union(*(_evidence(p) for p in witnesses))
        seen = set()
        for native in ((2, 2), (2, 9), (7, 2), (7, 9)):
            for _ in range(14):
                pair = _distinct_simple_fallback(params, seen, native)
                assert _key(pair["input"]) not in seen
                seen.add(_key(pair["input"]))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            x = generate_puzzle(100 + a * 10 + b, a, b)
            assert len(x["train"]) == a and len(x["test"]) == b
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    assert all(_key(p[k]) == _key(q[k]) for s in a for p, q in zip(a[s], b[s])
               for k in p)
    sample = puzzles[0]["test"][0]
    inp = sample["input"].copy()
    original = inp.copy()
    assert np.array_equal(solve(inp), sample["output"]) and np.array_equal(inp, original)
    # Every singleton repeats; two or three writers cancel even if inks agree.
    carrier = np.array([[1, 2], [3, 4]])
    rendered = _render(carrier, (4, 6), {(0, 1): [6], (1, 0): [7, 7, 7]})
    assert np.all(rendered[::2, 1::2] == 6)
    assert rendered[1, 0] == 1 and rendered[3, 4] == 4
    # Exact specification example: two different physical endpoints alias;
    # adding a third shot at one literal endpoint keeps the phase transparent.
    native_carrier = np.array([[1, 2, 1, 3], [2, 3, 4, 1], [4, 1, 2, 3]])
    field = np.zeros((15, 20), dtype=np.int64)
    objects = [_arrow((2, 1), (0, 1)), _arrow((8, 1), (0, 1)),
               _arrow((11, 14), (0, -1))]
    fixture_inks = [6, 6, 8]
    for obj, color in zip(objects, fixture_inks):
        for p in obj[2]:
            field[p] = color
    strip = np.zeros((3, 20), dtype=np.int64)
    strip[:, :4] = native_carrier
    fixture = np.concatenate((field, np.full((1, 20), 9), strip))
    result = solve(fixture)
    assert result[2, 2] == 1 and result[8, 13] == 8
    assert np.array_equal(result[2::3, 1::4], np.full((5, 5), 8))
    cancelled_fixture = fixture.copy()
    for p in objects[-1][2]:
        cancelled_fixture[p] = 0
    enlarged = np.repeat(np.repeat(native_carrier, 5, axis=0), 5, axis=1)
    assert np.array_equal(solve(cancelled_fixture), enlarged)
    extra = _arrow((7, 6), (-1, 0))
    assert _compatible(extra, objects)
    for p in extra[2]:
        fixture[p] = 7
    assert np.array_equal(solve(fixture), result)
    objects.append(extra)
    fixture_inks.append(7)
    groups = _writes(objects, fixture_inks, (3, 4))
    reverse_groups = _writes(objects[::-1], fixture_inks[::-1], (3, 4))
    assert np.array_equal(_render(native_carrier, (15, 20), groups),
                          _render(native_carrier, (15, 20), reverse_groups))
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, every legal explicit count, semantic invariants)")
