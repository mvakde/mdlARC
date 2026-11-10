"""Complete a tile-scale copy of the distinctive ink stencil inside a glyph.

Geometry, palettes and designs are visible in each pair. The shared convention is
same-orientation registration followed by exact copying of the entire glyph.
"""
from functools import lru_cache

import numpy as np


def _validate_count(value, name, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(rng, num_train, num_test):
    _validate_count(num_train, "num_train", 2, 10)
    _validate_count(num_test, "num_test", 1, 4)
    if num_train is None:
        if num_test is None:
            num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _normalized(tile):
    roles = {}
    return tile.shape, tuple(roles.setdefault(int(c), len(roles)) for c in tile.flat)


def _design_symmetries(mask):
    """Dihedral rules indistinguishable from identity for this design."""
    def cropped_key(a):
        ys, xs = np.nonzero(a)
        return _key(a[ys.min():ys.max()+1, xs.min():xs.max()+1])
    target = cropped_key(mask)
    transforms = [np.rot90(mask, k) for k in range(4)]
    transforms += [np.fliplr(a) for a in transforms]
    return {i for i, a in enumerate(transforms) if cropped_key(a) == target}


def _candidate_origins(mask, rows, columns, positions):
    h, w = mask.shape
    return [(r, c) for r in range(rows - h + 1)
            for c in range(columns - w + 1)
            if all(0 <= y-r < h and 0 <= x-c < w and mask[y-r, x-c]
                   for y, x in positions)]


@lru_cache(maxsize=16)
def _subset_codes(n):
    codes = np.arange(1, 1 << n, dtype=np.uint32)
    counts = np.zeros(len(codes), dtype=np.uint8)
    for bit in range(n):
        counts += ((codes >> bit) & 1).astype(np.uint8)
    codes.flags.writeable = counts.flags.writeable = False
    return codes, counts


def _valid_fragments(mask, rows, columns, origin):
    """Enumerate all permitted subsets; reject every competing translation.

    A competing origin covers a bitset of the true stencil's active positions.
    A clue subset is unique iff it is not contained in any such bitset.
    """
    active = np.argwhere(mask)
    codes, counts = _subset_codes(len(active))
    limit = min(len(active) - 1, (rows * columns - 1) // 2)
    valid = codes[(counts >= 2) & (counts <= limit)].copy()
    h, w = mask.shape
    for r in range(rows - h + 1):
        for c in range(columns - w + 1):
            if (r, c) == origin:
                continue
            cover = 0
            for i, (y, x) in enumerate(active):
                dy, dx = int(y) + origin[0] - r, int(x) + origin[1] - c
                if 0 <= dy < h and 0 <= dx < w and mask[dy, dx]:
                    cover |= 1 << i
            if cover:
                valid = valid[(valid & np.uint32(cover)) != valid]
    return active, valid


def _choose_visible_fragment(mask, rows, columns, origin, rng, excluded=()):
    active, codes = _valid_fragments(mask, rows, columns, origin)
    if excluded:
        codes = np.array([code for code in codes if tuple(
            tuple(map(int, active[i])) for i in range(len(active))
            if int(code) & (1 << i)) not in excluded], dtype=np.uint32)
    if len(codes) == 0:
        return None
    sizes = np.array([int(code).bit_count() for code in codes])
    size = int(rng.choice(np.unique(sizes)))
    code = int(rng.choice(codes[sizes == size]))
    return active[[i for i in range(len(active)) if code & (1 << i)]]


_SHAPES = tuple((h, w) for h in range(2, 5) for w in range(2, 5))


def _geometry(rng, shape=None):
    if shape is None:
        shape = _SHAPES[int(rng.choice(len(_SHAPES), p=(.10, .10, .07, .10, .24, .10, .07, .10, .12)))]
    h, w = shape
    t = int(rng.choice((1, 2), p=(.7, .3)))
    # Independent axes retain all legal sizes, while giving roomier lattices
    # enough weight to exercise interior registrations and the pixel boundary.
    def axis_size(side):
        largest = (30-t)//(side+t)
        return largest if rng.random() < .35 else int(rng.integers(side, largest+1))
    rows, cols = axis_size(h), axis_size(w)
    if rows * cols == 4:  # Two clues must be fewer than the ordinary tiles.
        if rng.integers(2):
            rows += 1
        else:
            cols += 1
    return h, w, t, rows, cols


def _design(rng, shape):
    """A shared design distribution, not a split-specific alphabet."""
    h, w = shape
    mode = int(rng.integers(10))
    mask = np.zeros(shape, dtype=bool)
    if mode < 2:
        # Open cup, crossbar, triangle, hollow perimeter, diagonal, or checker.
        kind = int(rng.integers(6))
        if kind == 0:
            mask[:, (0, -1)] = True
            mask[-1] = True
        elif kind == 1:
            mask[:, (0, -1)] = True
            mask[h//2] = True
        elif kind == 2:
            mask[0, w//2] = True
            mask[1:, (0, -1)] = True
            mask[-1] = True
        elif kind == 3:
            mask[(0, -1), :] = True
            mask[:, (0, -1)] = True
        elif kind == 4:
            for r in range(h):
                mask[r, round(r * (w-1) / (h-1))] = True
        else:
            mask = np.indices(shape).sum(axis=0) % 2 == 0
    else:
        count = int(rng.integers(3, h*w))
        mask.flat[rng.choice(h*w, count, replace=False)] = True
    if int(mask.sum()) < 3 or mask.all():
        mask[:] = False
        mask.flat[rng.choice(h*w, int(rng.integers(3, h*w)), replace=False)] = True
    # These alter the design itself. Its later coarse realization uses it as-is.
    if rng.integers(2):
        mask = mask[::-1]
    if rng.integers(2):
        mask = mask[:, ::-1]
    if h == w:
        mask = np.rot90(mask, int(rng.integers(4)))
    return mask.copy()


_SOURCE_MOTIFS = (("000", "010", "000"), ("000", "101", "000"),
                  ("010", "101", "010"), ("010", "111", "000"))


def _ordinary(rng, shape, colors):
    k, n = len(colors), int(np.prod(shape))
    if shape == (3, 3) and k == 2 and rng.random() < .35:
        pattern = _SOURCE_MOTIFS[int(rng.integers(4))]
        labels = np.array([[int(c) for c in row] for row in pattern])
        if rng.integers(2):
            labels = 1-labels
    else:
        # All colors occur; all frequencies, including ties, are permitted.
        labels = np.concatenate((np.arange(k), rng.integers(k, size=n-k)))
        rng.shuffle(labels)
        labels = labels.reshape(shape)
    return np.array(colors, dtype=np.int8)[labels]


def _paint(geometry, divider, ordinary, glyph, positions):
    h, w, t, rows, columns = geometry
    grid = np.full((rows*(h+t)+t, columns*(w+t)+t), divider, dtype=np.int8)
    for r in range(rows):
        for c in range(columns):
            y, x = t+r*(h+t), t+c*(w+t)
            grid[y:y+h, x:x+w] = glyph if (r, c) in positions else ordinary
    return grid


def _assemble(geometry, palette, ordinary, mask, origin, visible, background):
    glyph = np.where(mask, palette[1], background).astype(np.int8)
    supplied = {(origin[0]+int(r), origin[1]+int(c)) for r, c in visible}
    complete = {(origin[0]+int(r), origin[1]+int(c)) for r, c in np.argwhere(mask)}
    return {"input": _paint(geometry, palette[0], ordinary, glyph, supplied),
            "output": _paint(geometry, palette[0], ordinary, glyph, complete)}


def _make_pair(rng, geometry, palette, forbidden_masks=(), familiar=None,
               excluded_fragments=(), forbidden_motif=None, prior_symmetries=None):
    h, w, t, rows, cols = geometry
    origin = (int(rng.integers(rows-h+1)), int(rng.integers(cols-w+1)))
    mask, visible = None, None
    if familiar is not None:
        mask = familiar.copy()
        visible = _choose_visible_fragment(mask, rows, cols, origin, rng, excluded_fragments)
        if visible is None:
            return None
    else:
        for _ in range(32):
            mask = _design(rng, (h, w))
            if (_key(mask) not in forbidden_masks and
                    (prior_symmetries is None or
                     _design_symmetries(mask) & prior_symmetries == {0})):
                visible = _choose_visible_fragment(mask, rows, cols, origin, rng)
                if visible is not None:
                    break
        if visible is None:
            # A three-corner design has a uniquely registering opposite-corner
            # pair in any legal lattice. Four variants suffice for train[1].
            for absent in range(4):
                mask = np.zeros((h, w), dtype=bool)
                corners = ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))
                for i, point in enumerate(corners):
                    mask[point] = i != absent
                if (_key(mask) not in forbidden_masks and
                        (prior_symmetries is None or
                         _design_symmetries(mask) & prior_symmetries == {0})):
                    visible = _choose_visible_fragment(mask, rows, cols, origin, rng)
                    if visible is not None:
                        break
            if visible is None and h >= 3 and w >= 3:
                # This three-point, non-square bounding box has no nontrivial
                # dihedral symmetry, even when the first demonstration did.
                mask = np.zeros((h, w), dtype=bool)
                mask[0, 0] = mask[h-1, 0] = mask[h-1, w-2] = True
                if _key(mask) not in forbidden_masks:
                    visible = _choose_visible_fragment(mask, rows, cols, origin, rng)
            if visible is None:
                return None
    ordinary = _ordinary(rng, (h, w), palette[2:])
    if _normalized(ordinary) == forbidden_motif:
        # Switching one cell changes the normalized pattern; keep all colors.
        for idx in range(h*w):
            trial = ordinary.copy()
            trial.flat[idx] = palette[2 + (palette[2:].index(int(trial.flat[idx]))+1) % (len(palette)-2)]
            if len(np.unique(trial)) == len(palette)-2 and _normalized(trial) != forbidden_motif:
                ordinary = trial
                break
    backgrounds = [c for c in palette[2:] if np.any(ordinary[~mask] != c)]
    background = int(rng.choice(backgrounds))
    return _assemble(geometry, palette, ordinary, mask, origin, visible, background)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _resolve_counts(rng, num_train, num_test)
    # These are presentation preferences, not hidden rules needed for solving.
    stable_geometry = bool(rng.random() < .25)
    stable_palette = bool(rng.random() < .25)
    base_geometry = _geometry(rng)
    base_palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    train_color_pool = rng.choice(10, 8 if rng.random() < .45 else 10, replace=False)
    puzzle, train_info, used = {"train": [], "test": []}, [], set()
    source_keys = {_key(p["input"]) for p in _source_pairs()}
    for split, count, regime in (("train", ntrain, "source_train"),
                                  ("test", ntest, "target_test")):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            pair = None
            for attempt in range(64):
                geometry = base_geometry if stable_geometry else _geometry(pair_rng)
                palette = base_palette
                if not stable_palette:
                    color_count = 5 if pair_rng.random() < .25 else 4
                    pool = train_color_pool if split == "train" else np.arange(10)
                    palette = tuple(map(int, pair_rng.choice(pool, color_count, replace=False)))
                    if train_info and not (split == "train" and index == 1) and pair_rng.random() < .15:
                        old = train_info[int(pair_rng.integers(len(train_info)))]
                        palette = (old["divider"], old["ink"], *map(int, np.unique(old["ordinary"])))
                    if split == "train" and index == 1 and palette[1] == train_info[0]["ink"]:
                        palette = (palette[1], palette[0], *palette[2:])
                familiar, excluded, forbidden_masks = None, (), set()
                if regime == "target_test" and pair_rng.random() < .4:
                    choices = [f for f in train_info if not stable_geometry or f["tile_shape"] == geometry[:2]]
                    if choices:
                        chosen = choices[int(pair_rng.integers(len(choices)))]
                        familiar = chosen["mask_array"]
                        if not stable_geometry:
                            geometry = _geometry(pair_rng, familiar.shape)
                        excluded = {f["fragment"] for f in train_info if f["mask"] == _key(familiar)}
                if regime == "target_test" and familiar is None:
                    forbidden_masks = {f["mask"] for f in train_info}
                elif split == "train" and index == 1:
                    forbidden_masks = {train_info[0]["mask"]}
                motif = train_info[0]["motif"] if split == "train" and index == 1 else None
                symmetries = (_design_symmetries(train_info[0]["mask_array"])
                              if split == "train" and index == 1 else None)
                pair = _make_pair(pair_rng, geometry, palette, forbidden_masks,
                                  familiar, excluded, motif, symmetries)
                if pair is not None and _key(pair["input"]) not in used | source_keys:
                    break
                pair = None
            if pair is None:
                # Deterministic finite fallback: 510 different ordinary patterns
                # on 3×3 tiles, versus at most 13 used inputs and 4 source inputs.
                geometry = (3, 3, 1, 7, 7)
                mask = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 1]], dtype=bool)
                if train_info and _key(mask) == train_info[0]["mask"]:
                    mask = mask[:, ::-1].copy()
                if split == "train" and index == 1:
                    prior = _design_symmetries(train_info[0]["mask_array"])
                    for k in range(4):
                        trial = np.rot90(mask, k)
                        if _design_symmetries(trial) & prior == {0}:
                            mask = trial.copy()
                            break
                visible = _choose_visible_fragment(mask, 7, 7, (1, 2), pair_rng)
                for code in range(1, 511):
                    ordinary = np.array([base_palette[2+(code >> i & 1)] for i in range(9)], dtype=np.int8).reshape(3, 3)
                    if split == "train" and index == 1 and _normalized(ordinary) == train_info[0]["motif"]:
                        continue
                    background = next(c for c in base_palette[2:] if np.any(ordinary[~mask] != c))
                    pair = _assemble(geometry, base_palette, ordinary, mask, (1, 2), visible, background)
                    if _key(pair["input"]) not in used | source_keys:
                        break
            puzzle[split].append(pair)
            used.add(_key(pair["input"]))
            if split == "train":
                train_info.append(_analyze_pair(pair))
    validate_generated_puzzle(puzzle)
    return puzzle


def _runs(indices):
    groups = []
    for i in map(int, indices):
        if groups and i == groups[-1][1] + 1:
            groups[-1] = (groups[-1][0], i)
        else:
            groups.append((i, i))
    return groups


def _detect_lattice(grid):
    divider = int(grid[0, 0])
    row_groups = _runs(np.flatnonzero(np.all(grid == divider, axis=1)))
    col_groups = _runs(np.flatnonzero(np.all(grid == divider, axis=0)))
    assert len(row_groups) >= 3 and len(col_groups) >= 3
    assert row_groups[0][0] == col_groups[0][0] == 0
    assert row_groups[-1][1] == grid.shape[0]-1 and col_groups[-1][1] == grid.shape[1]-1
    thicknesses = {b-a+1 for a, b in row_groups + col_groups}
    assert len(thicknesses) == 1
    t = thicknesses.pop()
    heights = {row_groups[i+1][0]-row_groups[i][1]-1 for i in range(len(row_groups)-1)}
    widths = {col_groups[i+1][0]-col_groups[i][1]-1 for i in range(len(col_groups)-1)}
    assert len(heights) == len(widths) == 1
    h, w = heights.pop(), widths.pop()
    assert h > 0 and w > 0
    return (h, w, t, len(row_groups)-1, len(col_groups)-1), divider


def _infer(input_grid):
    """Recover every parameter from the input, without consulting any output."""
    geometry, divider = _detect_lattice(input_grid)
    h, w, t, rows, cols = geometry
    records = {}
    for r in range(rows):
        for c in range(cols):
            y, x = t+r*(h+t), t+c*(w+t)
            tile = input_grid[y:y+h, x:x+w]
            assert divider not in tile
            records.setdefault(_key(tile), [tile.copy(), []])[1].append((r, c))
    assert len(records) == 2, "exactly two tile types are required"
    ordered = sorted(records.values(), key=lambda item: len(item[1]), reverse=True)
    ordinary, ordinary_positions = ordered[0]
    glyph, positions = ordered[1]
    assert len(ordinary_positions) > len(positions) >= 2, "ordinary tile must be uniquely modal"
    ordinary_colors, frequencies = np.unique(ordinary, return_counts=True)
    assert len(ordinary_colors) in (2, 3)
    ink_colors = set(map(int, np.unique(glyph))) - set(map(int, ordinary_colors))
    assert len(ink_colors) == 1
    ink = ink_colors.pop()
    mask = glyph == ink
    assert 3 <= int(mask.sum()) < h*w
    assert len(np.unique(glyph[~mask])) == 1
    assert int(glyph[~mask][0]) in ordinary_colors
    origins = _candidate_origins(mask, rows, cols, positions)
    assert len(origins) == 1, "supplied copies must determine exactly one translation"
    origin = origins[0]
    active = {(origin[0]+int(r), origin[1]+int(c)) for r, c in np.argwhere(mask)}
    assert set(positions) < active, "completion must add at least one whole tile"
    expected = input_grid.copy()
    for r, c in active - set(positions):
        y, x = t+r*(h+t), t+c*(w+t)
        expected[y:y+h, x:x+w] = glyph
    fragment = tuple(sorted((r-origin[0], c-origin[1]) for r, c in positions))
    missing = tuple(sorted((r-origin[0], c-origin[1]) for r, c in active-set(positions)))
    return expected, {
        "geometry": geometry, "tile_shape": (h, w), "thickness": t,
        "lattice": (rows, cols), "origin": origin, "mask": _key(mask),
        "mask_array": mask, "motif": _normalized(ordinary), "ordinary": ordinary,
        "glyph": glyph, "ink": ink, "divider": divider,
        "palette": tuple(map(int, np.unique(input_grid))),
        "roles": (divider, ink, int(glyph[~mask][0]), tuple(map(int, ordinary_colors))),
        "fragment": fragment, "missing": missing, "ink_count": int(mask.sum()),
        "equal_frequencies": len(set(map(int, frequencies))) < len(frequencies),
        "background_changes": bool(np.any(ordinary[~mask] != glyph[~mask])),
        "boundary_independent": _candidate_origins(mask, rows+2*h, cols+2*w,
                               [(r+h, c+w) for r, c in positions]) == [(origin[0]+h, origin[1]+w)],
    }


def _analyze_pair(pair):
    expected, info = _infer(pair["input"])
    assert np.array_equal(expected, pair["output"]), "output must copy entire glyph tiles exactly"
    return info


def solve(train_pairs, test_input):
    # Training establishes exact same-orientation tile copying, independent of
    # design or literal colors. Test inference sees only the copied test input.
    assert len(train_pairs) >= 2
    for pair in train_pairs:
        _analyze_pair(pair)
    return _infer(test_input.copy())[0]


@lru_cache(maxsize=1)
def _source_pairs():
    """Compact exact source fixtures, including independently recorded outputs."""
    fixtures = (
        (2, "444414444", "313313333", ((3, 2), (3, 3), (3, 4)),
         ((1, 2), (1, 4), (2, 2), (2, 4), (3, 2), (3, 3), (3, 4))),
        (2, "888383888", "181111181", ((2, 1), (2, 2), (2, 3)),
         ((1, 1), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 3))),
        (1, "424242424", "484888848", ((3, 1), (3, 2), (3, 3), (4, 1)),
         ((2, 2), (3, 1), (3, 2), (3, 3), (4, 1), (4, 3))),
        (8, "929222999", "949494444", ((3, 2), (4, 1), (4, 3)),
         ((3, 2), (4, 1), (4, 3), (5, 1), (5, 2), (5, 3))),
    )
    pairs = []
    for divider, ordinary, glyph, supplied, complete in fixtures:
        ordinary = np.array(list(map(int, ordinary)), dtype=np.int8).reshape(3, 3)
        glyph = np.array(list(map(int, glyph)), dtype=np.int8).reshape(3, 3)
        pair = {"input": _paint((3, 3, 1, 7, 7), divider, ordinary, glyph, set(supplied)),
                "output": _paint((3, 3, 1, 7, 7), divider, ordinary, glyph, set(complete))}
        for grid in pair.values():
            grid.flags.writeable = False
        pairs.append(pair)
    return tuple(pairs)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    source_keys = {(_key(p["input"]), _key(p["output"])) for p in _source_pairs()}
    seen, info = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            ikey, okey = _key(pair["input"]), _key(pair["output"])
            assert ikey not in seen and (ikey, okey) not in source_keys
            seen.add(ikey)
            info[split].append(_analyze_pair(pair))
    assert len({f["mask"] for f in info["train"]}) >= 2
    assert len({f["motif"] for f in info["train"]}) >= 2
    assert any(f["background_changes"] for f in info["train"])
    orientation_rules = set(range(8))
    for f in info["train"]:
        orientation_rules &= _design_symmetries(f["mask_array"])
    assert orientation_rules == {0}, "training must demonstrate preserved orientation"
    # There is no train-only location, mask alphabet, color, or size restriction.
    # Every test is inferred from its input after checking the demonstrations.
    train_copy = [{k: v.copy() for k, v in pair.items()} for pair in puzzle["train"]]
    for pair in puzzle["test"]:
        predicted = solve(train_copy, pair["input"].copy())
        assert np.array_equal(predicted, pair["output"])


def _puzzles_equal(left, right):
    return all(len(left[s]) == len(right[s]) and all(
        all(np.array_equal(a[k], b[k]) for k in ("input", "output"))
        for a, b in zip(left[s], right[s])) for s in ("train", "test"))


def _components(points):
    remaining, result = set(points), []
    while remaining:
        component, todo = set(), [remaining.pop()]
        while todo:
            r, c = todo.pop()
            component.add((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        result.append(component)
    return result


def review_measurements(puzzles):
    """Review axes measured from the returned grids, not sampling metadata."""
    pairs, within, conditional, design_classes = [], [], {}, {}
    for puzzle_index, puzzle in enumerate(puzzles):
        local = []
        train = [_analyze_pair(pair) for pair in puzzle["train"]]
        train_colors = set().union(*(set(f["palette"]) for f in train))
        for split in ("train", "test"):
            for pair_index, pair in enumerate(puzzle[split]):
                f = _analyze_pair(pair)
                h, w = f["tile_shape"]
                r, c = f["origin"]
                nr, nc = f["lattice"]
                clue = f["fragment"]
                mask = f["mask_array"]
                mask_id = str(mask.shape) + ':' + ''.join(map(str, mask.astype(int).flat))
                transforms = [np.rot90(mask, k) for k in range(4)]
                transforms += [np.fliplr(a) for a in transforms]
                designs = [str(a.shape) + ':' + ''.join(map(str, a.astype(int).flat)) for a in transforms]
                family = design_classes.setdefault(min(designs), {"orientations": set(), "rotation_orbits": set()})
                family["orientations"].add(mask_id)
                family["rotation_orbits"].add(min(designs[:4]))
                motif_id = str(f["motif"])
                edges = (r == 0, r+h == nr, c == 0, c+w == nc)
                inactive = _components(map(tuple, np.argwhere(~mask)))
                holes = sum(not any(y in (0, h-1) or x in (0, w-1) for y, x in comp) for comp in inactive)
                sides = sum((any(y == 0 for y, x in clue), any(y == h-1 for y, x in clue),
                             any(x == 0 for y, x in clue), any(x == w-1 for y, x in clue)))
                active_bottom = max(y for y, x in np.argwhere(mask))
                bottom = tuple((int(y), int(x)) for y, x in np.argwhere(mask) if y == active_bottom)
                familiar = [old for old in train if old["mask"] == f["mask"]]
                item = {
                    "puzzle": puzzle_index, "split": split, "pair": pair_index,
                    "shape": pair["input"].shape, "geometry": f["geometry"],
                    "tile_shape": (h, w), "thickness": f["thickness"],
                    "lattice": f["lattice"], "origin": f["origin"],
                    "mask": mask_id, "motif": motif_id, "palette": f["palette"],
                    "roles": f["roles"], "ink_count": f["ink_count"],
                    "density": round(f["ink_count"]/(h*w), 4),
                    "visible_count": len(clue), "missing_count": len(f["missing"]),
                    "fragment": clue, "missing": f["missing"],
                    "ordinary_colors": len(np.unique(f["ordinary"])),
                    "equal_frequencies": f["equal_frequencies"],
                    "background_changes": f["background_changes"],
                    "without_zero": 0 not in f["palette"],
                    "interior": not any(edges), "top": edges[0], "bottom": edges[1],
                    "left": edges[2], "right": edges[3],
                    "corner": any(edges[:2]) and any(edges[2:]),
                    "boundary_independent": f["boundary_independent"],
                    "single_row": len({y for y, x in clue}) == 1,
                    "single_column": len({x for y, x in clue}) == 1,
                    "diagonal_clues": len({y-x for y, x in clue}) == 1 or len({y+x for y, x in clue}) == 1,
                    "disconnected_clues": len(_components(clue)) > 1,
                    "several_sides": sides >= 3,
                    "one_missing": len(f["missing"]) == 1,
                    "sparse_clues": 2*len(clue) < f["ink_count"],
                    "bottom_row_only_missing": f["missing"] == bottom,
                    "asymmetric_mask": _design_symmetries(mask) == {0},
                    "diagonal_mask": (len({int(y-x) for y, x in np.argwhere(mask)}) == 1 or
                                      len({int(y+x) for y, x in np.argwhere(mask)}) == 1),
                    "disconnected_mask": len(_components(map(tuple, np.argwhere(mask)))) > 1,
                    "holes": holes,
                    "familiar_test": split == "test" and bool(familiar),
                    "familiar_new_fragment": split == "test" and bool(familiar) and all(clue != old["fragment"] for old in familiar),
                    "unseen_test": split == "test" and not familiar,
                    "new_test_colors": split == "test" and bool(set(f["palette"])-train_colors),
                }
                local.append(item)
                conditional.setdefault(mask_id, []).append(item)
        distinct = {name: len({str(f[name]) for f in local}) for name in
                    ("geometry", "tile_shape", "thickness", "palette", "roles", "motif", "mask", "fragment", "origin")}
        distinct["train_count"], distinct["test_count"] = len(puzzle["train"]), len(puzzle["test"])
        distinct["rich_pairs"] = sum(f["ordinary_colors"] == 3 and f["asymmetric_mask"] and
                                      f["disconnected_clues"] for f in local)
        distinct["stable_geometry_palette"] = distinct["geometry"] == distinct["palette"] == 1
        within.append(distinct)
        pairs.extend(local)
    bool_names = [name for name, value in pairs[0].items() if isinstance(value, (bool, np.bool_))]
    totals = {name: sum(bool(f[name]) for f in pairs) for name in bool_names}
    totals["holes"] = sum(f["holes"] > 0 for f in pairs)
    histograms = {}
    for name in ("shape", "tile_shape", "thickness", "lattice", "ink_count", "density",
                 "visible_count", "missing_count", "ordinary_colors", "palette"):
        histogram = {}
        for f in pairs:
            key = str(f[name])
            histogram[key] = histogram.get(key, 0) + 1
        histograms[name] = histogram
    return {"pairs": pairs, "within_puzzle": within, "totals": totals,
            "histograms": histograms,
            "distinct": {name: len({str(f[name]) for f in pairs}) for name in
                         ("geometry", "shape", "mask", "motif", "palette", "roles")},
            "design_classes": {key: {name: len(values) for name, values in group.items()}
                               for key, group in design_classes.items()},
            "conditional_masks": {mask: {"pairs": len(fs),
                "fragments": len({str(f["fragment"]) for f in fs}),
                "motifs": len({f["motif"] for f in fs}),
                "palettes": len({f["palette"] for f in fs}),
                "origins": len({f["origin"] for f in fs})} for mask, fs in conditional.items()}}


def self_test() -> None:
    for pair in _source_pairs():
        _analyze_pair(pair)
    source = _source_pairs()
    assert np.array_equal(solve(source[:3], source[3]["input"]), source[3]["output"])
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    records = [_analyze_pair(pair) for p in puzzles for s in ("train", "test") for pair in p[s]]
    assert len({f["mask"] for f in records}) >= 100
    assert len({f["motif"] for f in records}) >= 100
    assert len({f["geometry"] for f in records}) >= 60
    assert len({f["palette"] for f in records}) >= 60
    assert len({f["tile_shape"] for f in records}) == 9
    assert {f["thickness"] for f in records} == {1, 2}
    assert {len(f["palette"]) for f in records} == {4, 5}
    assert any(0 not in f["palette"] for f in records)
    assert any(f["equal_frequencies"] for f in records)
    assert sum(len(f["missing"]) == 1 for f in records) >= 15
    assert sum(2*len(f["fragment"]) < f["ink_count"] for f in records) >= 20
    assert sum(f["boundary_independent"] for f in records) >= 100
    for split in ("train", "test"):
        fs = [_analyze_pair(pair) for p in puzzles for pair in p[split]]
        assert any(f["origin"][0] == 0 for f in fs)
        assert any(f["origin"][1] == 0 for f in fs)
        assert any(f["origin"][0] == f["lattice"][0]-f["tile_shape"][0] for f in fs)
        assert any(f["origin"][1] == f["lattice"][1]-f["tile_shape"][1] for f in fs)
        assert any(all(0 < f["origin"][i] < f["lattice"][i]-f["tile_shape"][i] for i in (0, 1)) for f in fs)
    # Fixed glyph AND fixed geometry/origin: randomness must still vary clues.
    u = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    fragments = {tuple(map(tuple, _choose_visible_fragment(u, 7, 7, (1, 2), np.random.default_rng(s)))) for s in range(100)}
    assert len(fragments) >= 30
    assert any(len(f) == 2 for f in fragments) and any(len(f) == 6 for f in fragments)
    metrics = review_measurements(puzzles)
    totals, within = metrics["totals"], metrics["within_puzzle"]
    for name in ("single_row", "single_column", "diagonal_clues", "disconnected_clues",
                 "several_sides", "asymmetric_mask", "disconnected_mask", "holes",
                 "diagonal_mask", "familiar_new_fragment", "unseen_test", "new_test_colors"):
        assert totals[name] >= 3, (name, totals[name])
    assert totals["bottom_row_only_missing"] < len(records)//3
    assert sum(f["tile_shape"] > 1 for f in within) >= 15
    assert sum(f["thickness"] > 1 for f in within) >= 10
    assert sum(f["palette"] > 1 for f in within) >= 20
    assert sum(f["geometry"] == 1 for f in within) >= 3
    assert sum(f["palette"] == 1 for f in within) >= 3
    assert sum(f["stable_geometry_palette"] for f in within) >= 1
    assert sum(f["rich_pairs"] > 0 for f in within) >= 10
    assert sum(f["split"] == "train" and f["ordinary_colors"] == 3 and
               f["asymmetric_mask"] and f["disconnected_clues"]
               for f in metrics["pairs"]) >= 10
    assert sum(f["roles"] < f["train_count"]+f["test_count"] for f in within) >= 3
    assert all(f["motif"] > 1 for f in within)
    assert sum(f["fragments"] >= 2 for f in metrics["conditional_masks"].values()) >= 20
    assert sum(f["orientations"] > 1 for f in metrics["design_classes"].values()) >= 20
    assert sum(f["rotation_orbits"] > 1 for f in metrics["design_classes"].values()) >= 10
    observed_motifs = {f["motif"] for f in records}
    for rows in _SOURCE_MOTIFS:
        pattern = np.array([[int(c) for c in row] for row in rows])
        assert _normalized(pattern) in observed_motifs
    # Pixel-budget boundaries are exercised directly as well as sampled.
    for geometry in ((2, 2, 2, 7, 7), (4, 4, 1, 4, 5), (4, 4, 1, 5, 4)):
        pair = _make_pair(np.random.default_rng(100), geometry, (1, 2, 3, 4))
        assert pair is not None
        _analyze_pair(pair)
        assert max(pair["input"].shape) <= 30
    active, codes = _valid_fragments(u, 7, 7, (1, 2))
    assert sum(3 <= int(code).bit_count() <= 5 for code in codes) == 79
    for fragment in fragments:
        assert _candidate_origins(u, 7, 7, [(r+1, c+2) for r, c in fragment]) == [(1, 2)]
    assert _puzzles_equal(generate_puzzle(123456, 3, 2), generate_puzzle(123456, 3, 2))
    assert not _puzzles_equal(puzzles[0], puzzles[1])
    for args in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(3001, *args)
        for s, expected in zip(("train", "test"), args):
            assert expected is None or len(p[s]) == expected
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3.5, 1)):
        try:
            generate_puzzle(999, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("a changed divider was accepted")
    # A full-subset tie is ambiguity, regardless of identical overlap counts.
    ambiguous = _assemble((3, 3, 1, 7, 7), (1, 2, 3, 4),
                          np.array([[3, 3, 3], [3, 4, 3], [3, 3, 3]]),
                          u, (1, 2), np.array([[0, 0], [1, 0]]), 3)
    try:
        _analyze_pair(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError("an ambiguous translation was accepted")
    # Leaving a background pixel unchanged is an ink-only-copy error.
    pair = puzzles[0]["train"][0]
    f = _analyze_pair(pair)
    bad = {k: v.copy() for k, v in pair.items()}
    local_r, local_c = f["missing"][0]
    h, w, t, _, _ = f["geometry"]
    dy, dx = np.argwhere((f["ordinary"] != f["glyph"]) & ~f["mask_array"])[0]
    y = t + (f["origin"][0]+local_r)*(h+t) + int(dy)
    x = t + (f["origin"][1]+local_c)*(w+t) + int(dx)
    bad["output"][y, x] = f["ordinary"][dy, dx]
    try:
        _analyze_pair(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("an ink-only tile copy was accepted")
