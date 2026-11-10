"""Recover a mirror-damaged unary parameter, then print its wrapped staircase.

Only standard library and NumPy are used. Black (0) always means missing;
other color roles and both fold positions are recovered from each input.
"""
from __future__ import annotations

from hashlib import sha256

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 rectangular integer color grid")
    return a.astype(np.int8, copy=True)


def _orbits(h, w, a, b):
    """Cartesian products of the in-bounds row and column reflection pairs."""
    seen = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            rows = {r, a - r} if 0 <= a - r < h else {r}
            cols = {c, b - c} if 0 <= b - c < w else {c}
            orbit = tuple(sorted((y, x) for y in rows for x in cols))
            seen.update(orbit)
            yield orbit


def _axis_candidates(grid, dimension):
    """Exhaust all nontrivial folds, including tiny overlaps and both parities.

    A fold is r -> a-r or c -> b-c; no coverage threshold hides alternatives.
    Erasures are wildcards here. Orbit restoration subsequently checks that
    every group actually has a witness, including diagonal-only groups.
    """
    g = grid if dimension == 0 else grid.T
    size = g.shape[0]
    candidates = []
    for fold in range(1, 2 * size - 2):
        first = np.arange(max(0, fold - size + 1), (fold + 1) // 2)
        second = fold - first
        left, right = g[first], g[second]
        visible = (left != 0) & (right != 0)
        if visible.any() and np.all(left[visible] == right[visible]):
            candidates.append(fold)
    return candidates


def _restore(grid, a, b):
    restored = grid.copy()
    for orbit in _orbits(*grid.shape, a, b):
        values = {int(grid[p]) for p in orbit if grid[p] != 0}
        if len(values) != 1:
            raise ValueError("reflection orbit is empty or contradictory")
        value = values.pop()
        for p in orbit:
            if grid[p] == 0:
                restored[p] = value
    return restored


def _read_input(value):
    grid = _grid(value)
    h, w = grid.shape
    if h < 8 or w < 6 or grid[0, 0] == 0:
        raise ValueError("missing visible upper-left seed anchor")
    counts = np.bincount(grid.ravel(), minlength=10)
    backgrounds = [c for c in range(1, 10) if counts[c] > grid.size // 2]
    if len(backgrounds) != 1:
        raise ValueError("background must occupy more than half the input")
    background, foreground = backgrounds[0], int(grid[0, 0])
    if background == foreground:
        raise ValueError("seed and background colors must differ")
    row_axes = _axis_candidates(grid, 0)
    col_axes = _axis_candidates(grid, 1)
    if len(row_axes) != 1 or len(col_axes) != 1:
        raise ValueError("surviving cells must identify exactly two folds")
    a, b = row_axes[0], col_axes[0]
    restored = _restore(grid, a, b)
    n = 0
    while n < w and restored[0, n] == foreground:
        n += 1
    if not (2 <= n <= 14 and 5 <= a < h and 2 * n + 1 <= b < w):
        raise ValueError("seed bars must have distinct in-bounds reflected copies")
    expected_foreground = np.zeros_like(grid, dtype=bool)
    for r in (0, a):
        expected_foreground[r, :n] = True
        expected_foreground[r, b - n + 1:b + 1] = True
    if not np.array_equal(restored == foreground, expected_foreground):
        raise ValueError("foreground must consist of the four reflected seed bars")
    if not np.all(np.isin(restored[[0, a]], (background, foreground))):
        raise ValueError("seed rows may contain only bar and background")
    if grid[0, n - 1] != 0 or grid[0, n] != 0:
        raise ValueError("seed damage must hide both sides of its end")
    if np.any(restored == 0) or not np.array_equal(
            restored[grid != 0], grid[grid != 0]):
        raise ValueError("restoration changed surviving evidence")
    return {"grid": grid, "clean": restored, "n": n, "background": background,
            "foreground": foreground, "axes": (a, b)}


def _length(n):
    return n * n + 2 * n - 2


def _emit(n, width, background, foreground):
    runs = list(range(1, n + 1)) + list(range(n - 1, 0, -1))
    cells = []
    for i, length in enumerate(runs):
        if i:
            cells.append(background)
        cells.extend([foreground] * length)
    rows = (len(cells) + width - 1) // width
    flat = np.full(rows * width, background, dtype=np.int8)
    flat[:len(cells)] = cells
    return flat.reshape(rows, width)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the folds, restore their orbits, and read n entirely from input."""
    info = _read_input(input_grid)
    return _emit(info["n"], info["grid"].shape[1],
                 info["background"], info["foreground"])


def _integer(name, value, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _width(rng, n, goal, excluded=()):
    candidates = list(range(2 * n + 2, 31))
    fresh = [w for w in candidates if w not in excluded]
    if fresh:
        candidates = fresh
    if goal == "single":
        preferred = [w for w in candidates if w >= _length(n)]
    elif goal == "wrapped":
        preferred = [w for w in candidates if w < _length(n)]
    elif goal == "exact":
        preferred = [w for w in candidates if _length(n) % w == 0]
    elif goal == "one":
        preferred = [w for w in candidates if (-_length(n)) % w == 1]
    elif goal == "long":
        preferred = [w for w in candidates if (-_length(n)) % w >= 5]
    else:
        preferred = candidates
    return int(rng.choice(preferred or candidates))


def _geometry(rng, n, width):
    h = int(rng.integers(8, 31))
    bottom = int(rng.integers(0, min(5, h - 6) + 1))
    right = int(rng.integers(0, min(5, width - 2 * n - 2) + 1))
    return h, width, h - 1 - bottom, width - 1 - right


def _draw_field(rng, geometry, n, colors, style):
    h, w, a, b = geometry
    bg, fg, *decor = colors
    clean = np.full((h, w), bg, dtype=np.int8)
    block = int(rng.integers(2, 5))
    cr, cc = int(rng.integers(1, a)), int(rng.integers(w))
    density = float(rng.uniform(0.16, 0.28))
    for orbit in _orbits(h, w, a, b):
        r, c = orbit[0]
        if r == 0:
            color = fg if c < n else bg
        else:
            # Coherent fragments are reflected, with sparse irregular landmarks
            # to distinguish the axes. Fringes get independent orbit values.
            if style == 0:
                probability = density
            elif style == 1:
                probability = 0.42 if (r // block + c // block) % 4 == 0 else 0.10
            elif style == 2:
                probability = 0.52 if abs((r - cr) - (c - cc)) <= 1 else 0.12
            else:
                distance = max(abs(r - cr), abs(c - cc))
                probability = 0.44 if distance % block == 0 else 0.10
            color = int(rng.choice(decor)) if rng.random() < probability else bg
        # The unpaired fringe row never gains foreground or a seed-like bar.
        for p in orbit:
            clean[p] = color
    return clean


def _seed_points(a, b, col):
    return ((0, col), (0, b - col), (a, col), (a, b - col))


def _damage(rng, clean, geometry, n, hard):
    h, w, a, b = geometry
    topology = int(rng.integers(5))
    mask = np.zeros((h, w), dtype=bool)
    if topology in (0, 4):
        mask = rng.random((h, w)) < rng.uniform(0.025, 0.075)
    if topology in (1, 2, 3, 4):
        for _ in range(int(rng.integers(1, 4))):
            rh = int(rng.integers(1, max(2, h // 4) + 1))
            cw = int(rng.integers(1, max(2, w // 5) + 1))
            if topology == 2:
                rh = 1
            if topology == 3:
                cw = 1
            r0 = int(rng.integers(h - rh + 1))
            c0 = int(rng.integers(w - cw + 1))
            mask[r0:r0 + rh, c0:c0 + cw] = True
    # The damaged interval straddles a foreground/background boundary, and
    # its endpoints are deliberately not a code for the hidden bar length.
    start = int(rng.integers(1, n))
    stop = int(rng.integers(n + 1, b // 2 + 2))
    mask[0, start:stop] = True
    protected = np.zeros_like(mask)
    endpoint = _seed_points(a, b, n - 1)
    if hard:
        keep = {3}  # Only the doubly reflected witness survives at the end.
    else:
        # A short anchor cannot be repaired by half-turn copying alone:
        # the diagonal is missing, with one or both direct mirrors surviving.
        keep = set(int(i) for i in rng.choice((1, 2), size=int(rng.integers(1, 3)),
                                             replace=False))
    for i, p in enumerate(endpoint):
        mask[p] = i not in keep
        protected[p] = i not in keep
    # Every reflected copy loses a foreground cell, so no intact bar bypasses
    # restoration. Its internal witness complements the endpoint witnesses.
    internal = int(rng.integers(0, n - 1))
    for i, p in enumerate(_seed_points(a, b, internal)):
        mask[p] = i in keep
        protected[p] = i in keep
    for i, p in enumerate(_seed_points(a, b, n)):
        mask[p] = (i != 3) if hard else (i == 0 or rng.random() < 0.25)
        protected[p] = bool(mask[p])
    # Always retain the diagonal background witness and the corner role marker.
    mask[a, b - n] = False
    protected[a, b - n] = False
    mask[0, 0] = protected[0, 0] = False
    for orbit in _orbits(h, w, a, b):
        if all(mask[p] for p in orbit):
            choices = [p for p in orbit if not protected[p]]
            if not choices:
                raise AssertionError("mandatory damage erased an entire orbit")
            mask[choices[int(rng.integers(len(choices)))]] = False
    grid = clean.copy()
    grid[mask] = 0
    return grid


def _make_pair(rng, n, goal, hard, used, colors=None, excluded_widths=()):
    style = int(rng.integers(4))
    for attempt in range(256):
        width = _width(rng, n, goal, excluded_widths)
        geometry = _geometry(rng, n, width)
        if colors is None:
            palette = [int(c) for c in rng.permutation(np.arange(1, 10))]
            pair_colors = palette[:int(rng.integers(3, 6))]
        else:
            pair_colors = colors
        clean = _draw_field(rng, geometry, n, pair_colors, style if attempt < 64 else 0)
        grid = _damage(rng, clean, geometry, n, hard)
        if _key(grid) in used:
            continue
        try:
            info = _read_input(grid)
        except ValueError:
            continue
        if info["n"] != n or not np.array_equal(info["clean"], clean):
            raise AssertionError("construction and restoration disagree")
        used.add(_key(grid))
        return {"input": grid, "output": _emit(n, width, *pair_colors[:2])}
    raise RuntimeError("bounded unique-fold sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    nt = _integer("num_train", num_train, 2, 10)
    ns = _integer("num_test", num_test, 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))
    small = 4 if regime == 1 else 3
    large = int(rng.integers(6, 9))
    if regime == 1:
        large = 6
    # Every legal request retains a single-row anchor and a hard wrapped anchor.
    plans = [(small, "single", False), (large, "wrapped", True)]
    if nt >= 3:
        plans.append((small, "wrapped", bool(rng.integers(2))))
    while len(plans) < nt:
        pool = [n for n in range(2, 9) if regime != 1 or n != 5]
        n = int(rng.choice(pool))
        goal = str(rng.choice(("free", "exact", "one", "long", "wrapped")))
        plans.append((n, goal, bool(rng.integers(2))))
    order = rng.permutation(nt)
    puzzle, used = {"train": [], "test": []}, set()
    stable_palette = None
    if rng.random() < 0.25:
        stable_palette = [int(c) for c in rng.permutation(np.arange(1, 10))[:4]]
    for i in order:
        n, goal, hard = plans[int(i)]
        anchor_widths = [p["input"].shape[1] for p in puzzle["train"]] if int(i) < 2 else ()
        puzzle["train"].append(_make_pair(rng, n, goal, hard, used, stable_palette,
                                              anchor_widths))
    train_infos = [_read_input(p["input"]) for p in puzzle["train"]]
    train_ns = [info["n"] for info in train_infos]
    for i in range(ns):
        if i == 0 and regime == 0:
            n = int(rng.integers(max(train_ns) + 1, 15))
        elif i == 0 and regime == 1:
            n = 5  # Held out between the demonstrated lengths 4 and 6.
        elif i == 0:
            # Reuse a length with an available *wrapped* width. In particular,
            # n=2 has no wrapped width under the separated-copy convention.
            eligible = [value for value in train_ns if any(
                w < _length(value) and w not in
                [info["grid"].shape[1] for info in train_infos if info["n"] == value]
                for w in range(2 * value + 2, 31))]
            n = int(rng.choice(eligible))
        else:
            n = int(rng.integers(2, 15))
        excluded = [info["grid"].shape[1] for info in train_infos if info["n"] == n]
        goal = "wrapped" if i == 0 else str(rng.choice(("free", "exact", "one", "long")))
        puzzle["test"].append(_make_pair(rng, n, goal, i == 0 or rng.random() < 0.7,
                                               used, stable_palette, excluded))
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    grid, clean = info["grid"], info["clean"]
    a, b = info["axes"]
    n, fg = info["n"], info["foreground"]
    diagonal = direct_only = 0
    for r in (0, a):
        for c in list(range(n)) + list(range(b - n + 1, b + 1)):
            if (grid[r, c] == 0 and grid[a - r, c] == 0
                    and grid[r, b - c] == 0 and grid[a - r, b - c] == fg):
                diagonal += 1
            if (grid[r, c] == 0 and grid[a - r, b - c] == 0
                    and (grid[a - r, c] == fg or grid[r, b - c] == fg)):
                direct_only += 1
    return {"diagonal": diagonal,
            "direct_only": direct_only,
            "diagonal_endpoint": bool(grid[0, n - 1] == grid[a, n - 1] ==
                                      grid[0, b - n + 1] == 0
                                      and grid[a, b - n + 1] == fg),
            "direct_endpoint": bool(grid[0, n - 1] == grid[a, b - n + 1] == 0
                                    and (grid[a, n - 1] == fg or grid[0, b - n + 1] == fg)),
            "erased_foreground": int(np.count_nonzero((grid == 0) & (clean == fg))),
            "erased_background": int(np.count_nonzero((grid == 0) & (clean == info["background"]))),
            "visible_seed": int(np.count_nonzero(grid[0, :n] == fg)),
            "padding": (-_length(n)) % grid.shape[1]}


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    info = _read_input(pair["input"])
    output = _grid(pair["output"])
    grid, clean = info["grid"], info["clean"]
    n, bg, fg = info["n"], info["background"], info["foreground"]
    h, w = grid.shape
    assert output.shape == ((_length(n) + w - 1) // w, w)
    assert set(int(c) for c in np.unique(output)) == {bg, fg}
    assert np.count_nonzero(output == fg) == n * n
    # Parse the actual returned string rather than comparing with _emit.
    flat = output.ravel()
    last = int(np.flatnonzero(flat == fg)[-1])
    assert last + 1 == _length(n)
    assert np.all(flat[last + 1:] == bg)
    lengths, separators, count = [], [], 0
    previous = fg
    for value in flat[:last + 1]:
        if value != previous:
            (lengths if previous == fg else separators).append(count)
            previous, count = int(value), 0
        count += 1
    lengths.append(count)
    assert lengths == list(range(1, n + 1)) + list(range(n - 1, 0, -1))
    assert separators == [1] * (2 * n - 2)
    assert not np.array_equal(output, grid)
    a, b = info["axes"]
    for r in (0, a):
        assert np.any(grid[r, :n] == 0) and np.any(grid[r, b - n + 1:b + 1] == 0)
    # Direct reflected lookup cross-check, including one-fold fringe orbits.
    for r, c in np.argwhere(grid == 0):
        witnesses = {int(grid[y, x]) for y, x in
                     ((a - r, c), (r, b - c), (a - r, b - c))
                     if 0 <= y < h and 0 <= x < w and grid[y, x] != 0}
        assert witnesses == {int(clean[r, c])}
    assert _axis_candidates(clean, 0) == [a] and _axis_candidates(clean, 1) == [b]
    return info


def validate_generated_puzzle(puzzle):
    """Check returned cells, causal necessity, unique folds, and demonstrations."""
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
        nt, ns = len(puzzle["train"]), len(puzzle["test"])
        assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
        seen, infos = set(), {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                key = _key(info["grid"])
                assert key not in seen
                seen.add(key)
                f = _features(info)
                assert f["erased_foreground"] > 0 and f["erased_background"] > 0
                assert f["visible_seed"] < info["n"]
                infos[split].append(info)
        train = infos["train"]
        assert len({x["n"] for x in train}) >= 2
        assert len({x["grid"].shape[1] for x in train}) >= 2
        assert any(_length(x["n"]) <= x["grid"].shape[1] for x in train)
        assert any(_length(x["n"]) > x["grid"].shape[1] for x in train)
        assert any(_features(x)["diagonal_endpoint"] for x in train)
        assert any(_features(x)["direct_endpoint"] for x in train), "training must disprove half-turn-only repair"
        first = infos["test"][0]
        assert _features(first)["diagonal_endpoint"]
        assert _length(first["n"]) > first["grid"].shape[1]
        train_ns = [x["n"] for x in train]
        if first["n"] in train_ns:
            assert first["grid"].shape[1] not in [
                x["grid"].shape[1] for x in train if x["n"] == first["n"]]
        else:
            assert first["n"] > min(train_ns)
    except (ValueError, TypeError, IndexError, KeyError) as error:
        raise AssertionError("malformed or ambiguous family instance") from error


def _puzzle_digest(puzzle):
    h = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(grid.shape))
                h.update(grid.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded streaming 42-seed checks; no files, caches, or stored traces."""
    signatures, shapes, lengths, palettes, offsets, parities = set(), set(), set(), set(), set(), set()
    backgrounds, foregrounds, structures = set(), set(), set()
    exact = one = long_pad = diagonal = direct_only = fringe_gaps = wrapped_peak = pairs = 0
    regimes = {"extrapolation": 0, "interpolation": 0, "reuse": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = _puzzle_digest(puzzle)
        assert digest == _puzzle_digest(generate_puzzle(seed))
        signatures.add(digest)
        train_ns = [_read_input(p["input"])["n"] for p in puzzle["train"]]
        first_n = _read_input(puzzle["test"][0]["input"])["n"]
        regimes["extrapolation"] += first_n > max(train_ns)
        regimes["interpolation"] += min(train_ns) < first_n < max(train_ns) and first_n not in train_ns
        regimes["reuse"] += first_n in train_ns
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                grid, clean = info["grid"], info["clean"]
                n, (a, b) = info["n"], info["axes"]
                f = _features(info)
                pairs += 1
                assert np.array_equal(solve(grid), pair["output"])
                assert np.array_equal(solve(grid.tolist()), pair["output"])
                shapes.add(grid.shape)
                lengths.add(n)
                palettes.add(len(np.unique(grid)))
                offsets.add((grid.shape[0] - 1 - a, grid.shape[1] - 1 - b))
                parities.add((a % 2, b % 2))
                backgrounds.add(info["background"])
                foregrounds.add(info["foreground"])
                # A role-normalized hash measures structure independently of palette.
                roles, normalized = {}, []
                for value in grid.flat:
                    normalized.append(roles.setdefault(int(value), len(roles)))
                structures.add((grid.shape, sha256(bytes(normalized)).hexdigest()))
                exact += f["padding"] == 0
                one += f["padding"] == 1
                long_pad += f["padding"] >= 5
                diagonal += f["diagonal"] > 0
                direct_only += f["direct_only"] > 0
                fringe_gaps += any((a - r < 0 or b - c < 0) for r, c in np.argwhere(grid == 0))
                peak_start = n * (n - 1) // 2 + n - 1
                wrapped_peak += any(peak_start < cut < peak_start + n
                                    for cut in range(grid.shape[1], _length(n), grid.shape[1]))
                # Omitting B: a visible-cell count has a strictly wrong parameter.
                assert f["visible_seed"] != n
                assert not np.array_equal(_emit(f["visible_seed"], grid.shape[1],
                                                info["background"], info["foreground"]), pair["output"])
                # Palette equivariance fixes black while permuting every payload role.
                mapping = np.array([0] + list(range(9, 0, -1)), dtype=np.int8)
                assert np.array_equal(solve(mapping[grid]), mapping[pair["output"]])
                # Revealing some holes cannot change the inferred answer.
                revealed = grid.copy()
                for r, c in np.argwhere(grid == 0)[::2]:
                    if (r, c) not in ((0, n - 1), (0, n)):
                        revealed[r, c] = clean[r, c]
                assert np.array_equal(solve(revealed), pair["output"])
    assert len(signatures) == 42 and len(structures) >= 150
    assert len(shapes) >= 50 and len(lengths) >= 11
    assert backgrounds == foregrounds == set(range(1, 10))
    assert len(offsets) >= 15 and len(parities) == 4 and len(palettes) >= 3
    assert exact >= 3 and one >= 3 and long_pad >= 30 and wrapped_peak >= 10
    assert diagonal >= 60 and fringe_gaps >= 10
    assert direct_only >= 42
    assert all(v >= 3 for v in regimes.values())
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(9000 + nt * 10 + ns, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    base = generate_puzzle(123, 2, 1)
    maximum = _make_pair(np.random.default_rng(1414), 14, "free", True, set())
    _inspect_pair(maximum)
    assert np.array_equal(solve(maximum["input"]), maximum["output"])
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
    bad["test"][0]["output"][0, 0] = 0
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output passed validation")
    inp = base["test"][0]["input"]
    info = _read_input(inp)
    a, b = info["axes"]
    bad_input = inp.copy()
    for p in _seed_points(a, b, info["n"] - 1):
        bad_input[p] = 0
    try:
        solve(bad_input)
    except ValueError:
        pass
    else:
        raise AssertionError("witness-free seed orbit accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "normalized_structures": len(structures),
            "lengths": sorted(lengths), "palette_sizes": sorted(palettes),
            "fold_offsets": len(offsets), "fold_parities": len(parities),
            "exact_fill": exact, "one_cell_padding": one, "long_padding": long_pad,
            "diagonal_seed_repair": diagonal, "fringe_erasure": fringe_gaps,
            "direct_only_seed_repair": direct_only,
            "peak_wrapping": wrapped_peak, "shifts": regimes}
