"""Diagonal relay growth: a frame is both a boundary and a three-state key.

Only NumPy and the standard library are used.  solve() reads no generation state.
"""
from __future__ import annotations

import numpy as np


# Quadrant bits: south=2, east=1. Exterior markers occupy adjacent corners.
_SIDES = ((0, 1), (2, 3), (0, 2), (1, 3))  # top, bottom, left, right


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside its legal range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a


def _frame(h, w, patch, band_h, band_w, background, roles):
    """One-cell frame; corner runs, axial bands, and remaining edge segments."""
    corner, axial, edge = roles
    a = np.full((2 * h + 2, 2 * w + 2), background, dtype=np.int8)
    a[[0, -1], :] = edge
    a[:, [0, -1]] = edge
    for row in (0, a.shape[0] - 1):
        a[row, :patch] = corner
        a[row, -patch:] = corner
        a[row, w + 1 - band_w:w + 1 + band_w] = axial
    for col in (0, a.shape[1] - 1):
        a[:patch, col] = corner
        a[-patch:, col] = corner
        a[h + 1 - band_h:h + 1 + band_h, col] = axial
    return a


def _positions(top, bottom, left, right):
    return ((top - 1, left - 1), (top - 1, right + 1),
            (bottom + 1, left - 1), (bottom + 1, right + 1))


def _analyze(input_grid):
    """Reconstruct and check the entire visible input, including its role key."""
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 4:
        raise ValueError("a background and three distinct signal colors are required")
    # For h,w>=2 each individual border role has fewer cells than the cavity.
    # Thus the background is uniquely most frequent even with zero extra margin.
    bg = int(colors[np.argmax(counts)])
    foreground = a != bg
    rows = np.flatnonzero(foreground.sum(axis=1) >= 6)
    cols = np.flatnonzero(foreground.sum(axis=0) >= 6)
    if len(rows) != 2 or len(cols) != 2:
        raise ValueError("expected one complete rectangular frame")
    top, bottom = map(int, rows)
    left, right = map(int, cols)
    ih, iw = bottom - top - 1, right - left - 1
    if min(ih, iw) < 4 or ih % 2 or iw % 2:
        raise ValueError("cavity dimensions must be even and at least four")
    h, w = ih // 2, iw // 2
    corner = int(a[top, left])
    axial = int(a[top, left + w])
    remaining = set(map(int, colors)) - {bg, corner, axial}
    if corner == axial or bg in (corner, axial) or len(remaining) != 1:
        raise ValueError("frame key roles are not distinct")
    edge = remaining.pop()
    roles = (corner, axial, edge)
    top_edge = a[top, left:right + 1]
    patch = 0
    while patch < len(top_edge) and top_edge[patch] == corner:
        patch += 1
    band_w_count = int(np.count_nonzero(top_edge == axial))
    band_h_count = int(np.count_nonzero(a[top:bottom + 1, left] == axial))
    if band_w_count % 2 or band_h_count % 2:
        raise ValueError("axial bands must be centered and even")
    band_h, band_w = band_h_count // 2, band_w_count // 2
    if not (patch >= 1 and band_h >= 1 and band_w >= 1
            and patch + band_h <= h and patch + band_w <= w):
        raise ValueError("frame segments overlap or an edge role is missing")
    template = np.full(a.shape, bg, dtype=np.int8)
    template[top:bottom + 1, left:right + 1] = _frame(
        h, w, patch, band_h, band_w, bg, roles)
    exterior = foreground.copy()
    exterior[top:bottom + 1, left:right + 1] = False
    actual = {tuple(map(int, rc)) for rc in np.argwhere(exterior)}
    positions = _positions(top, bottom, left, right)
    matching = [s for s, corners in enumerate(_SIDES)
                if actual == {positions[q] for q in corners}]
    if len(actual) != 2 or len(matching) != 1:
        raise ValueError("exactly two adjacent exterior diagonal corner markers required")
    side = matching[0]
    markers = {q: int(a[positions[q]]) for q in _SIDES[side]}
    if len(set(markers.values())) != 2 or not set(markers.values()) <= set(roles):
        raise ValueError("markers must select distinct states of the frame key")
    for q, color in markers.items():
        template[positions[q]] = color
    if not np.array_equal(a, template):
        raise ValueError("input contains an incomplete frame, nonempty cavity, or noise")
    return dict(background=bg, roles=roles, h=h, w=w, top=top, bottom=bottom,
                left=left, right=right, patch=patch, band_h=band_h,
                band_w=band_w, side=side, markers=markers)


def _quadrant(info, q):
    h, w = info["h"], info["w"]
    r = info["top"] + 1 + (q // 2) * h
    c = info["left"] + 1 + (q % 2) * w
    return slice(r, r + h), slice(c, c + w)


def _orient(tile, source):
    # Growth travels away from the source's exterior corner in BOTH relay halves.
    if source & 2:
        tile = np.flipud(tile)
    if source & 1:
        tile = np.fliplr(tile)
    return tile


def _tile_formula(h, w, roles, start):
    r, c = np.indices((h, w))
    step = np.maximum(r, c)
    endpoint = (r == np.minimum(step, h - 1)) & (c == np.minimum(step, w - 1))
    phase = (start + step - (~endpoint).astype(int)) % 3
    return np.asarray(roles, dtype=np.int8)[phase]


def _paint(input_grid, info, mode="relay"):
    out = input_grid.copy()
    h, w, roles = info["h"], info["w"], info["roles"]
    duration = max(h, w)
    for source, color in info["markers"].items():
        start = roles.index(color)
        destination = source ^ 3
        if mode == "axis":
            # Ablation: transmit to the same column/row instead of opposite diagonal.
            destination = source ^ (2 if info["side"] < 2 else 1)
        if mode == "solid":
            near = far = np.full((h, w), color, dtype=np.int8)
        else:
            near = _orient(_tile_formula(h, w, roles, start), source)
            offset = 0 if mode == "reset" else duration
            far = _orient(_tile_formula(h, w, roles, start + offset), source)
        out[_quadrant(info, source)] = near
        out[_quadrant(info, destination)] = far
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the frame geometry, visible role key, and marker positions only."""
    a = _grid(input_grid)
    return _paint(a, _analyze(a))


def _margins(rng, length, before, after, intro):
    spare = 30 - length - before - after
    extra = (spare if not intro and rng.random() < 0.12 else
             int(rng.integers(0, min(spare, 2 if intro else 5) + 1)))
    first = int(rng.integers(0, extra + 1))
    return before + first, after + extra - first


def _draw_input(rng, h, w, profile, side, palette, starts, intro):
    bg, *roles = palette
    # At least one cell beyond each marked corner; the opposite side may touch an edge.
    mt = mb = ml = mr = 1
    if not intro:
        if side == 0:
            mb = 0
        elif side == 1:
            mt = 0
        elif side == 2:
            mr = 0
        else:
            ml = 0
    mt, mb = _margins(rng, 2 * h + 2, mt, mb, intro)
    ml, mr = _margins(rng, 2 * w + 2, ml, mr, intro)
    a = np.full((mt + 2 * h + 2 + mb, ml + 2 * w + 2 + mr), bg, dtype=np.int8)
    bottom, right = mt + 2 * h + 1, ml + 2 * w + 1
    a[mt:bottom + 1, ml:right + 1] = _frame(h, w, *profile, bg, roles)
    positions = _positions(mt, bottom, ml, right)
    for q, state in zip(_SIDES[side], starts):
        a[positions[q]] = roles[state]
    return a


def _profile(rng, h, w):
    if rng.random() < 0.35:
        return 1, 1, 1
    p = int(rng.integers(1, min(h, w)))
    return p, int(rng.integers(1, h - p + 1)), int(rng.integers(1, w - p + 1))


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ne = _integer(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base = tuple(map(int, rng.choice(10, 4, replace=False)))
    stable_palette = bool(rng.integers(2))
    first_side = int(rng.integers(4))
    second_side = int(rng.choice((2, 3) if first_side < 2 else (0, 1)))
    first_size = int(rng.choice((4, 5)))
    second_long = 9 - first_size  # opening durations are 4 and 5, in either order
    puzzle = {"train": [], "test": []}
    used = set()
    for i in range(nt + ne):
        split = "train" if i < nt else "test"
        for _attempt in range(100):
            if i == 0:
                h = w = first_size
                profile, side = (1, 1, 1), first_side
                starts = tuple(map(int, rng.permutation((0, 1))))
            elif i == 1:
                h, w = (3, second_long) if rng.integers(2) else (second_long, 3)
                profile = (2, 1 if h == 3 else 2, 1 if w == 3 else 2)
                side = second_side
                starts = tuple(map(int, rng.permutation((2, 0))))
            else:
                if split == "train":
                    h, w = map(int, rng.integers(2, 6, size=2))
                else:
                    long = int(rng.integers(6, 14))
                    short = int(rng.integers(2, long + 1))
                    h, w = (short, long) if rng.integers(2) else (long, short)
                profile, side = _profile(rng, h, w), int(rng.integers(4))
                starts = tuple(map(int, rng.choice(3, size=2, replace=False)))
            if stable_palette or i == 0:
                palette = base
            elif i == 1:
                palette = (base[0], base[2], base[3], base[1])
            elif rng.random() < 0.5:
                palette = tuple(map(int, rng.permutation(base)))
            else:
                palette = tuple(map(int, rng.choice(10, 4, replace=False)))
            a = _draw_input(rng, h, w, profile, side, palette, starts, i < 2)
            key = _signature(a)
            if key not in used:
                used.add(key)
                puzzle[split].append({"input": a, "output": solve(a)})
                break
        else:
            raise RuntimeError("failed to draw a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _iterated_tile(h, w, roles, start):
    """A separate recurrence used to check every arm, advancing corner, and clip."""
    tile = np.full((h, w), -1, dtype=np.int8)
    state = start
    tile[0, 0] = roles[state]
    old_h = old_w = 1
    for step in range(1, max(h, w)):
        new_h, new_w = min(step + 1, h), min(step + 1, w)
        if new_w > old_w:
            tile[:new_h, new_w - 1] = roles[state]
        if new_h > old_h:
            tile[new_h - 1, :new_w] = roles[state]
        state = (state + 1) % 3
        tile[new_h - 1, new_w - 1] = roles[state]
        old_h, old_w = new_h, new_w
    return tile, state


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "pairs must contain exactly input/output")
    a, out = _grid(pair["input"]), _grid(pair["output"])
    info = _analyze(a)
    _require(a.shape == out.shape, "output dimensions must stay fixed")
    mutable = np.zeros(a.shape, dtype=bool)
    mutable[info["top"] + 1:info["bottom"], info["left"] + 1:info["right"]] = True
    _require(np.array_equal(a[~mutable], out[~mutable]), "frame/exterior changed")
    _require(np.all(out[mutable] != info["background"]), "cavity was not completely filled")
    _require(set(map(int, np.unique(out))) == set(map(int, np.unique(a))), "palette changed")
    for source, color in info["markers"].items():
        start = info["roles"].index(color)
        near, terminal = _iterated_tile(info["h"], info["w"], info["roles"], start)
        far, _ = _iterated_tile(info["h"], info["w"], info["roles"], (terminal + 1) % 3)
        _require(np.array_equal(out[_quadrant(info, source)], _orient(near, source)),
                 "near quadrant violates inherited arms or successor corners")
        _require(np.array_equal(out[_quadrant(info, source ^ 3)], _orient(far, source)),
                 "opposite quadrant violates diagonal state continuation")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _signature(np.asarray(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    opening = infos["train"][:2]
    _require(opening[0]["side"] != opening[1]["side"], "opening examples need distinct marked sides")
    _require({max(i["h"], i["w"]) % 3 for i in opening} == {1, 2},
             "opening examples must distinguish relay continuation from reset")
    _require(any(i["h"] == i["w"] for i in opening)
             and any(i["h"] != i["w"] for i in opening), "show square and clipped rectangular growth")
    _require({i["roles"].index(c) for i in opening for c in i["markers"].values()} == {0, 1, 2},
             "opening markers must expose every starting state")
    for pair, info in zip(puzzle["train"][:2], opening):
        for mode in ("solid", "axis", "reset"):
            _require(not np.array_equal(pair["output"], _paint(np.asarray(pair["input"]), info, mode)),
                     f"opening demonstration fails to reject {mode} shortcut")
    train_max = max(max(i["h"], i["w"]) for i in infos["train"])
    _require(all(max(i["h"], i["w"]) > train_max for i in infos["test"]),
             "tests must require longer state propagation than all demonstrations")


def _puzzle_key(puzzle):
    return tuple(tuple((_signature(np.asarray(p["input"])), _signature(np.asarray(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed survey, recurrence checks, covariance, counts, corruptions."""
    shapes, cavities, profiles, sides, durations, phase_offsets = set(), set(), set(), set(), set(), set()
    structures, puzzle_keys, palettes = set(), set(), set()
    role_colors = [set() for _ in range(4)]
    edges = squares = rectangles = stable = mixed = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzle_keys.add(_puzzle_key(puzzle))
        local_palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                info = _check_pair(pair)
                _require(np.array_equal(solve(a.copy()), out), "input-only solve mismatch")
                shapes.add(a.shape)
                cavities.add((info["h"], info["w"]))
                profiles.add((info["patch"], info["band_h"], info["band_w"]))
                sides.add(info["side"])
                duration = max(info["h"], info["w"])
                durations.add(duration)
                phase_offsets.add(duration % 3)
                palette = (info["background"], *info["roles"])
                palettes.add(palette)
                local_palettes.add(palette)
                for observed, color in zip(role_colors, palette):
                    observed.add(color)
                mapping = {}
                structures.add((a.shape, tuple(mapping.setdefault(int(c), len(mapping)) for c in a.flat)))
                edges += (info["top"] == 0 or info["left"] == 0
                          or info["bottom"] == a.shape[0] - 1 or info["right"] == a.shape[1] - 1)
                squares += info["h"] == info["w"]
                rectangles += info["h"] != info["w"]
        stable += len(local_palettes) == 1
        mixed += len(local_palettes) > 1
        if seed < 5:
            _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "nondeterminism")
            a = puzzle["test"][0]["input"]
            out = puzzle["test"][0]["output"]
            for k in (1, 2, 3):
                _require(np.array_equal(solve(np.rot90(a, k)), np.rot90(out, k)), "rotation covariance")
            perm = np.random.default_rng(1000 + seed).permutation(10)
            _require(np.array_equal(solve(perm[a]), perm[out]), "literal color dependence")
    _require(len(puzzle_keys) == 42, "42 seeds must produce distinct puzzles")
    _require(len(shapes) >= 40 and len(cavities) >= 35 and len(structures) >= 100,
             "insufficient geometric variation")
    _require(len(profiles) >= 12 and sides == set(range(4)), "missing key/side variation")
    _require(phase_offsets == {0, 1, 2} and min(durations) == 2 and max(durations) >= 12,
             "missing durations or relay phases")
    _require(min(edges, squares, rectangles, stable, mixed) > 0, "missing family regimes")
    _require(all(c == set(range(10)) for c in role_colors), "some literal colors never change roles")
    # Probe extreme legal geometries directly, independently of the split sampler.
    # Each is solved solely from its input; the recurrence checks every output cell.
    boundary_cases = 0
    boundary_rng = np.random.default_rng(24680)
    for h, w in ((2, 2), (2, 13), (13, 2), (13, 13)):
        p = min(h, w) - 1
        for profile in sorted({(1, 1, 1), (p, h - p, w - p)}):
            for side in range(4):
                for first in range(3):
                    for second in range(3):
                        if first == second:
                            continue
                        a = _draw_input(boundary_rng, h, w, profile, side,
                                        (0, 2, 5, 7), (first, second), False)
                        _check_pair({"input": a, "output": solve(a)})
                        boundary_cases += 1
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(8128 + nt * 10 + ne, nt, ne)
            _require((len(p["train"]), len(p["test"])) == (nt, ne), "counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = generate_puzzle(99)
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    info = _analyze(damaged["test"][0]["input"])
    damaged["test"][0]["output"][info["top"] + 1, info["left"] + 1] = info["background"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted relay accepted")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(puzzle_keys), grid_shapes=len(shapes),
                quadrant_shapes=len(cavities), frame_profiles=len(profiles),
                normalized_inputs=len(structures), role_palettes=len(palettes),
                marked_sides=sorted(sides), durations=sorted(durations),
                relay_phase_offsets=sorted(phase_offsets), edge_pairs=int(edges),
                square_pairs=int(squares), rectangular_pairs=int(rectangles),
                stable_palette_puzzles=stable, mixed_palette_puzzles=mixed,
                extreme_geometry_cases=boundary_cases)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
