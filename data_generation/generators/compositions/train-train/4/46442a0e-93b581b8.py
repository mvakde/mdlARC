"""Opposite-corner color stencils: one mosaic, four filtered rotations.

Only the standard library and NumPy are used. No parent module is imported.
"""

from itertools import product

import numpy as np


# Destination order: upper-left, upper-right, lower-left, lower-right.
_TURNS = (0, -1, 1, 2)  # NumPy's positive turns are counterclockwise.
_OPPOSITE = (3, 2, 1, 0)
_FULL = (2, 2, 2, 2)
_PROFILES = tuple(p for p in product(range(3), repeat=4)
                  if max(p[:2]) > 0 and max(p[2:]) > 0)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= x <= 30 for x in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((array >= 0) & (array <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return array.astype(np.int8, copy=False)


def _parse(input_grid):
    """Recognize a solid square, without assuming a literal/modal background."""
    grid = _grid(input_grid)
    candidates = []
    for background in np.unique(grid):
        cells = np.argwhere(grid != background)
        if not len(cells):
            continue
        row, col = cells.min(axis=0).tolist()
        last_row, last_col = cells.max(axis=0).tolist()
        n = last_row - row + 1
        if n < 2 or last_col - col + 1 != n or len(cells) != n * n:
            continue
        motif = grid[row:row + n, col:col + n]
        corners = (int(motif[0, 0]), int(motif[0, -1]),
                   int(motif[-1, 0]), int(motif[-1, -1]))
        if len(set(corners)) != 4 or set(np.unique(motif)) != set(corners):
            continue
        candidates.append((grid, row, col, n, int(background), motif, corners))
    if len(candidates) != 1:
        raise ValueError("input needs one filled four-color square with distinct corners")
    return candidates[0]


def _origin(row, col, n, quadrant):
    return (row - n if quadrant < 2 else row + n,
            col - n if quadrant % 2 == 0 else col + n)


def _window(shape, row, col, mask):
    height, width = mask.shape
    r0, r1 = max(0, row), min(shape[0], row + height)
    c0, c1 = max(0, col), min(shape[1], col + width)
    if r0 >= r1 or c0 >= c1:
        return None
    return (slice(r0, r1), slice(c0, c1),
            mask[r0 - row:r1 - row, c0 - col:c1 - col])


def _render(input_grid, turns=_TURNS, mapping=_OPPOSITE, solid=False,
            tight=False, inward=False, whole=False):
    grid, row, col, n, background, motif, corners = _parse(input_grid)
    output = grid.copy()
    for quadrant in range(4):
        color = corners[mapping[quadrant]]
        mask = np.ones_like(motif, dtype=bool) if solid else motif == color
        if tight:
            cells = np.argwhere(mask)
            low, high = cells.min(axis=0), cells.max(axis=0) + 1
            mask = mask[low[0]:high[0], low[1]:high[1]]
        mask = np.rot90(mask, turns[quadrant])
        rr, cc = _origin(row, col, n, quadrant)
        if inward:
            rr = min(max(rr, 0), grid.shape[0] - mask.shape[0])
            cc = min(max(cc, 0), grid.shape[1] - mask.shape[1])
        window = _window(grid.shape, rr, cc, mask)
        if window is not None:
            rs, cs, visible_mask = window
            if whole:
                rotated = np.rot90(motif, turns[quadrant])
                full_window = _window(grid.shape, rr, cc, rotated)
                output[rs, cs] = full_window[2]
            else:
                output[rs, cs][visible_mask] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive selectors, square frame, positions, and clipping from input only."""
    return _render(input_grid)


def _reference(input_grid):
    """Cell-coordinate construction, separate from solve's rotated slices."""
    grid, row, col, n, _, motif, corners = _parse(input_grid)
    result = grid.copy()
    for quadrant in range(4):
        color = corners[3 - quadrant]
        rr, cc = _origin(row, col, n, quadrant)
        for u, v in np.argwhere(motif == color).tolist():
            x, y = ((u, v), (v, n - 1 - u),
                    (n - 1 - v, u), (n - 1 - u, n - 1 - v))[quadrant]
            if 0 <= rr + x < grid.shape[0] and 0 <= cc + y < grid.shape[1]:
                result[rr + x, cc + y] = color
    return result


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.tobytes()


def _views(mask):
    return [np.rot90(base, k) for base in (mask, np.fliplr(mask))
            for k in range(4)]


def _witness(motif):
    return all(len({_key(view) for view in _views(motif == color)}) == 8
               for color in np.unique(motif))


def _profile(parsed):
    grid, row, col, n, *_ = parsed
    margins = (row, grid.shape[0] - row - n, col, grid.shape[1] - col - n)
    return tuple(0 if x == 0 else 1 if x < n else 2 for x in margins)


def _satellite_stats(input_grid):
    grid, row, col, n, _, motif, corners = _parse(input_grid)
    stats = []
    for quadrant in range(4):
        mask = np.rot90(motif == corners[3 - quadrant], _TURNS[quadrant])
        rr, cc = _origin(row, col, n, quadrant)
        window = _window(grid.shape, rr, cc, mask)
        visible = 0 if window is None else int(window[2].sum())
        stats.append((int(mask.sum()), visible))
    return stats


def _labels(rng, n, style):
    """Geometric motifs, with the corners also serving as the visible key."""
    rows, cols = np.indices((n, n))
    if style == 0:
        labels = rng.integers(0, 4, (n, n), dtype=np.int8)
    elif style == 1:
        labels = np.full((n, n), int(rng.integers(4)), dtype=np.int8)
        for _ in range(int(rng.integers(3, n + 4))):
            r0, c0 = (int(v) for v in rng.integers(n, size=2))
            r1, c1 = int(rng.integers(r0 + 1, n + 1)), int(rng.integers(c0 + 1, n + 1))
            labels[r0:r1, c0:c1] = int(rng.integers(4))
    elif style == 2:
        strip = rng.integers(0, 4, n, dtype=np.int8)
        labels = np.repeat(strip[:, None], n, axis=1)
        if rng.random() < 0.5:
            labels = labels.T.copy()
    elif style == 3:
        a, b = (int(v) for v in rng.integers(1, max(2, n // 2 + 1), size=2))
        labels = ((rows // a + cols // b) % 4).astype(np.int8)
    elif style == 4:
        labels = np.full((n, n), int(rng.integers(4)), dtype=np.int8)
        for color in rng.permutation(4):
            r0, c0, r1, c1 = (int(v) for v in rng.integers(n, size=4))
            labels[min(r0, r1):max(r0, r1) + 1, c0] = color
            labels[r1, min(c0, c1):max(c0, c1) + 1] = color
    elif style == 5:
        anchors = rng.choice(n * n, size=4, replace=False)
        ar, ac = anchors // n, anchors % n
        distance = abs(rows[:, :, None] - ar) + abs(cols[:, :, None] - ac)
        labels = distance.argmin(axis=2).astype(np.int8)
    elif style == 6:
        depth = np.minimum.reduce((rows, cols, n - 1 - rows, n - 1 - cols))
        palette = rng.integers(0, 4, n, dtype=np.int8)
        labels = palette[depth]
    else:
        period = int(rng.integers(1, min(n, 4) + 1))
        strips = rng.integers(0, 4, (period, n), dtype=np.int8)
        labels = strips[np.arange(n) % period].copy()
    labels = np.rot90(labels, int(rng.integers(4))).copy()
    roles = rng.permutation(4)
    labels[0, 0], labels[0, -1] = roles[:2]
    labels[-1, 0], labels[-1, -1] = roles[2:]
    return labels


def _axis_margins(rng, n, a, b):
    values = [0 if p == 0 else int(rng.integers(1, n)) if p == 1 else n
              for p in (a, b)]
    capacity = 30 - n - sum(values)
    for index in rng.permutation(2):
        if (a, b)[index] == 2:
            extra = int(rng.integers(0, min(3, capacity) + 1))
            values[index] += extra
            capacity -= extra
    return values


def _make_input(rng, side_range, profile, background, need_witness,
                need_partial, used_geometry, avoid_n=None, avoid_corners=None):
    for _ in range(512):
        n = int(rng.integers(side_range[0], side_range[1] + 1))
        if n == avoid_n:
            continue
        top, bottom = _axis_margins(rng, n, *profile[:2])
        left, right = _axis_margins(rng, n, *profile[2:])
        geometry = (n + top + bottom, n + left + right, top, left, n)
        if geometry in used_geometry:
            continue
        labels = _labels(rng, n, int(rng.integers(8)))
        if need_witness and not _witness(labels):
            continue
        palette = rng.choice([c for c in range(10) if c != background], 4, replace=False)
        motif = palette[labels].astype(np.int8)
        corners = (int(motif[0, 0]), int(motif[0, -1]),
                   int(motif[-1, 0]), int(motif[-1, -1]))
        if avoid_corners is not None and any(a == b for a, b in zip(corners, avoid_corners)):
            continue
        grid = np.full(geometry[:2], background, dtype=np.int8)
        grid[top:top + n, left:left + n] = motif
        stats = _satellite_stats(grid)
        if not any(visible for _, visible in stats):
            continue
        if need_partial and not any(0 < visible < total for total, visible in stats):
            continue
        if need_partial and np.array_equal(solve(grid), _render(grid, inward=True)):
            continue
        if need_witness and np.array_equal(solve(grid), _render(grid, tight=True)):
            continue
        used_geometry.add(geometry)
        return grid
    raise RuntimeError("bounded mosaic construction exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    regime = seed % 3
    train_range, test_range = (((3, 5), (6, 10)),
                               ((3, 8), (3, 8)),
                               ((5, 9), (3, 4)))[regime]
    background = int(rng.integers(10))
    vary_background = bool(rng.integers(2))
    partial = list(_FULL)
    partial[int(rng.integers(4))] = 1
    train_profiles = [_FULL, tuple(partial)]
    pool = [p for p in _PROFILES if regime != 1 or sum(v < 2 for v in p) <= 1]
    train_profiles.extend(pool[int(rng.integers(len(pool)))] for _ in range(num_train - 2))
    # A novel combined edge exposure is a rule-preserving test shift.
    test_pool = [p for p in _PROFILES if sum(v < 2 for v in p) >= 2
                 and p not in train_profiles] if regime == 1 else list(_PROFILES)
    test_profiles = [test_pool[int(rng.integers(len(test_pool)))] for _ in range(num_test)]
    puzzle, used_geometry = {"train": [], "test": []}, set()
    witness_n = witness_corners = None
    for split, profiles, sides in (("train", train_profiles, train_range),
                                    ("test", test_profiles, test_range)):
        for index, profile in enumerate(profiles):
            local_background = int(rng.integers(10)) if vary_background else background
            witness = split == "train" and index == 0
            partial_witness = split == "train" and index == 1
            side_range = (max(4, sides[0]), sides[1]) if witness else sides
            inp = _make_input(rng, side_range, profile, local_background,
                              witness, partial_witness, used_geometry,
                              witness_n if partial_witness else None,
                              witness_corners if partial_witness else None)
            if witness:
                witness_n, witness_corners = _parse(inp)[3], _parse(inp)[6]
            puzzle[split].append({"input": inp, "output": _reference(inp)})
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal counts")
    seen, geometries = set(), set()
    training_sides = set()
    role_colors = [set() for _ in range(4)]
    full_witness = partial_witness = False
    shortcuts = {"no_rotation": False, "same_corner": False, "solid": False,
                 "whole_motif": False, "tight_frame": False, "inward_shift": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            parsed = _parse(inp)
            _, row, col, n, background, motif, corners = parsed
            _require(inp.shape == out.shape, "canvas dimensions must stay fixed")
            _require(np.array_equal(out, _reference(inp)), "wrong stencil mapping/rotation/clipping")
            _require(np.array_equal(out, solve(inp)), "solve disagrees with cellwise construction")
            _require(np.array_equal(out[row:row + n, col:col + n], motif), "source changed")
            _require(set(np.unique(out)) == set(np.unique(inp)), "palette changed")
            _require(np.all(out[row:row + n, :col] == background)
                     and np.all(out[row:row + n, col + n:] == background)
                     and np.all(out[:row, col:col + n] == background)
                     and np.all(out[row + n:, col:col + n] == background), "cardinal strips filled")
            _require(not np.array_equal(inp, out), "generated pair must show a visible satellite")
            key = _key(inp)
            geometry = (*inp.shape, row, col, n)
            _require(key not in seen and geometry not in geometries, "duplicate input/layout")
            seen.add(key)
            geometries.add(geometry)
            stats = _satellite_stats(inp)
            added = int(np.count_nonzero(out != background) - n * n)
            _require(added == sum(v for _, v in stats), "lost or duplicated stencil pixels")
            if _profile(parsed) == _FULL:
                _require(added == n * n, "complete stencils must partition the source")
                for color in corners:
                    _require(np.count_nonzero(out == color) == 2 * np.count_nonzero(inp == color),
                             "a complete color stencil was lost or copied twice")
            if split == "train":
                training_sides.add(n)
                for role, color in enumerate(corners):
                    role_colors[role].add(color)
                full_witness |= _profile(parsed) == _FULL and _witness(motif)
                partial_witness |= any(0 < v < total for total, v in stats)
                alternatives = {
                    "no_rotation": {"turns": (0, 0, 0, 0)},
                    "same_corner": {"mapping": (0, 1, 2, 3)},
                    "solid": {"solid": True}, "whole_motif": {"whole": True},
                    "tight_frame": {"tight": True}, "inward_shift": {"inward": True},
                }
                for name, options in alternatives.items():
                    shortcuts[name] |= not np.array_equal(out, _render(inp, **options))
    _require(full_witness, "training needs full, eight-view stencils for all four roles")
    _require(len(training_sides) >= 2, "training must demonstrate variable source scale")
    _require(all(len(colors) >= 2 for colors in role_colors), "corner roles need recoloring evidence")
    _require(partial_witness, "training must visibly truncate a stencil")
    _require(all(shortcuts.values()), f"training leaves shortcuts: {shortcuts}")


def _fingerprint(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _normalized(grid):
    roles = {}
    return grid.shape, tuple(roles.setdefault(int(v), len(roles)) for v in grid.flat)


def _rows(*rows):
    return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)


def self_test():
    # A hand-derived example checks absolute orientations, including empty cells.
    example = np.zeros((9, 9), dtype=np.int8)
    example[3:6, 3:6] = _rows("112", "324", "344")
    expected = _rows("000000330", "004000000", "044000000", "000112000",
                     "000324000", "000344000", "200000000", "020000000", "000000011")
    _require(np.array_equal(solve(example), expected), "hand-derived example failed")
    # Cropping a canvas commutes with solving when the entire source survives.
    # Exercise all four sides with absent, one-cell, partial, and complete views.
    for top, bottom, left, right in product((0, 1, 2, 3), repeat=4):
        if max(top, bottom) == 0 or max(left, right) == 0:
            continue
        rs, cs = slice(3 - top, 6 + bottom), slice(3 - left, 6 + right)
        _require(np.array_equal(solve(example[rs, cs]), expected[rs, cs]),
                 "clipping moved or rerotated a stencil")
    # Minimal 2x2 source and a source larger than sampled mosaics remain legal.
    for n in (2, 12, 28):
        grid = np.zeros((n + 2, n + 2), dtype=np.int8)
        grid[1:n + 1, 1:n + 1] = 1
        grid[1, n], grid[n, 1], grid[n, n] = 2, 3, 4
        _require(np.array_equal(solve(grid), _reference(grid)), "unsampled source size failed")
    identity = np.zeros((30, 30), dtype=np.int8)
    identity[:29, :29] = 1
    identity[0, 28], identity[28, 0], identity[28, 28] = 2, 3, 4
    _require(np.array_equal(solve(identity), identity), "legal fully cropped identity failed")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_fingerprint(p) for p in puzzles}) == 42, "seed variation collapsed")
    _require(_fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0)), "nondeterministic")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(10000 + nt * 10 + ns, nt, ns)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns), "counts ignored")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (4.0, 1), (True, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {nt}/{ns}")
    shapes, sides, profiles, backgrounds, structures, masks = set(), set(), set(), set(), set(), set()
    disappeared = truncated = full = symmetric = zero_as_foreground = 0
    for seed, puzzle in enumerate(puzzles):
        train_sides, test_sides, train_profiles, test_profiles = set(), set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                parsed = _parse(pair["input"])
                grid, _, _, n, bg, motif, _ = parsed
                shapes.add(grid.shape)
                sides.add(n)
                profiles.add(_profile(parsed))
                backgrounds.add(bg)
                structures.add(_normalized(grid))
                masks.add(_normalized(motif))
                zero_as_foreground += 0 in motif and bg != 0
                symmetric += not _witness(motif)
                for total, visible in _satellite_stats(grid):
                    disappeared += visible == 0
                    truncated += 0 < visible < total
                    full += visible == total
                (train_sides if split == "train" else test_sides).add(n)
                (train_profiles if split == "train" else test_profiles).add(_profile(parsed))
        if seed % 3 == 0:
            _require(min(test_sides) > max(train_sides), "larger-test shift lost")
        elif seed % 3 == 1:
            _require(not train_profiles & test_profiles, "new-edge-combination shift lost")
        else:
            _require(max(test_sides) < min(train_sides), "smaller-test shift lost")
    _require(sides == set(range(3, 11)), "sampled side-length coverage collapsed")
    _require(len(shapes) >= 70 and len(profiles) >= 25, "canvas/clipping diversity collapsed")
    _require(len(structures) >= 180 and len(masks) >= 150, "structural variation collapsed")
    _require(len(backgrounds) == 10 and zero_as_foreground >= 10, "palette roles collapsed")
    _require(min(disappeared, truncated, full, symmetric) >= 20, "visibility/symmetry coverage collapsed")
    return {"seeds": 42, "pairs": 210, "canvas_shapes": len(shapes),
            "source_sides": sorted(sides), "clipping_profiles": len(profiles),
            "normalized_inputs": len(structures), "normalized_mosaics": len(masks),
            "background_colors": len(backgrounds), "zero_as_foreground": zero_as_foreground,
            "satellites_absent": disappeared, "satellites_truncated": truncated,
            "satellites_complete": full, "non_witness_motifs": symmetric}
