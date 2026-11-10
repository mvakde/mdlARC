"""Two-color foreground geometry, with a learned puzzle-wide highlight."""

import numpy as np

_CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Exact source inputs, including shape; their dimensions remain available.
_SOURCE_INPUTS = frozenset((
    ((10, 12), '222244222222222224224242422222222242222422224422224222224422422222424442242422422422222222422222242222422422222442242424'),
    ((15, 17), '442224422244222444422244222442224422242222422224222224442244422444222224222242222422244242442424424244222422224222242222244422444224442222242222422224222442224422244222444422244222442224422242222422224222224442244422444222224222242222422244242442424424244'),
    ((12, 12), '222224442224222224442224222222222224222222222224222224444444222224442224422222442224442222244444444222444422444222422222222444422244222444444444'),
    ((22, 22), '4242442424424242242442224222242222222224222244444442442444424422422242222422224222242222424224222422424224244242422422244242442422422242222422224222242222444424424244444442244422422224222242222222224242422424424244242442424224242442222424244222422224222242222222224424444444444444444444224222242222222224222242424424244242422424424242422424422244242422222222242222422224222244444444424244444442422242222422224222242222424244242442422422244242424424242242442224422242222222224222242222'),
))


def _normalize_optional_count(
    value: int | None,
    name: str,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    low, high = (2, 10) if name == "num_train" else (1, 4)

    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")

    return value


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    n_train = _normalize_optional_count(num_train, "num_train")
    n_test = _normalize_optional_count(num_test, "num_test")

    if n_train is None and n_test is None:
        n_test = 1 + seed % 4
        low = max(2, n_test + 1)
        n_train = low + (seed // 4) % (11 - low)

    elif n_train is None:
        low = max(2, n_test + 1)
        n_train = low + seed % (11 - low)

    elif n_test is None:
        high = min(4, n_train - 1)
        if high < 1:
            raise ValueError(
                "no legal test count for the requested train count"
            )
        n_test = 1 + seed % high

    if n_train <= n_test:
        raise ValueError("num_train must be greater than num_test")

    return n_train, n_test


def _plus_centers_from_foreground(
    is_foreground: np.ndarray,
) -> np.ndarray:
    centers = np.zeros_like(is_foreground, dtype=bool)

    if is_foreground.shape[0] >= 3 and is_foreground.shape[1] >= 3:
        centers[1:-1, 1:-1] = (
            is_foreground[1:-1, 1:-1]
            & is_foreground[:-2, 1:-1]
            & is_foreground[2:, 1:-1]
            & is_foreground[1:-1, :-2]
            & is_foreground[1:-1, 2:]
        )

    return centers


def _plus_union_from_foreground(
    is_foreground: np.ndarray,
) -> np.ndarray:
    centers = _plus_centers_from_foreground(is_foreground)
    union = centers.copy()

    union[:-1, :] |= centers[1:, :]
    union[1:, :] |= centers[:-1, :]
    union[:, :-1] |= centers[:, 1:]
    union[:, 1:] |= centers[:, :-1]

    return union


def _apply_transformation(
    grid: np.ndarray,
    foreground: int,
    highlight: int,
) -> np.ndarray:
    output = grid.copy()
    union = _plus_union_from_foreground(grid == foreground)
    output[union] = highlight
    return output


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        grid.shape,
        np.asarray(grid, dtype=np.uint8).tobytes(),
    )


def _pair_key(pair: dict) -> tuple:
    return (
        _grid_key(pair["input"]),
        _grid_key(pair["output"]),
    )


def _visible_roles(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2, "two distinct input roles required"
    order = np.argsort(counts)
    assert counts[order[1]] > counts[order[0]]
    return int(colors[order[1]]), int(colors[order[0]])


def _scan(mask):
    """Validation detector: inspect crosses directly, without mask dilation."""
    height, width = mask.shape
    centers, cells = set(), set()
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if 0 < r < height - 1 and 0 < c < width - 1:
            if mask[r - 1:r + 2, c].all() and mask[r, c - 1:c + 2].all():
                centers.add((r, c))
                cells.update(((r, c), (r - 1, c), (r + 1, c),
                              (r, c - 1), (r, c + 1)))
    union = np.zeros_like(mask)
    for point in cells:
        union[point] = True
    return centers, union


def _sequential_union(mask):
    remaining = mask.copy()
    changed = np.zeros_like(mask)
    for r in range(1, mask.shape[0] - 1):
        for c in range(1, mask.shape[1] - 1):
            points = ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if all(remaining[p] for p in points):
                for point in points:
                    remaining[point] = False
                    changed[point] = True
    return changed


def _touches_unmarked(mask, union):
    for r, c in zip(*np.nonzero(union)):
        for dr, dc in _CARDINALS:
            a, b = r + dr, c + dc
            if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1]:
                if mask[a, b] and not union[a, b]:
                    return True
    return False


def _learn_highlight(train):
    changed_colors = set()
    for pair in train:
        inp, out = pair["input"], pair["output"]
        changed_colors.update(map(int, out[inp != out]))
    assert len(changed_colors) == 1, "positive training must reveal one highlight"
    return changed_colors.pop()


def _predict(train, inp):
    # Only demonstrations and the test input enter this prediction path.
    highlight = _learn_highlight(train)
    _, foreground = _visible_roles(inp)
    _, union = _scan(inp == foreground)
    answer = inp.copy()
    answer[union] = highlight
    return answer


def _source_copy(grid):
    if grid.shape not in {shape for shape, _ in _SOURCE_INPUTS}:
        return False
    return (grid.shape, "".join(str(int(v)) for v in grid.flat)) in _SOURCE_INPUTS


def _sample_shape(rng, band):
    low, high = {"compact": (5, 13), "large": (18, 30),
                 "middle": (10, 22), "any": (5, 30)}[band]
    kind = int(rng.integers(5))
    if kind == 0:
        side = int(rng.integers(low, high + 1))
        return side, side
    if kind == 1:
        narrow = int(rng.integers(3, 7))
        long = int(rng.integers(max(7, low), high + 1))
        return (narrow, long) if rng.random() < .5 else (long, narrow)
    return tuple(map(int, rng.integers(low, high + 1, size=2)))


def _axis(rng, size):
    count = int(rng.integers(1, 2 + (size - 3) // 3))
    if count == 1:
        return [int(rng.integers(1, size - 1))]
    if rng.random() < .5:
        gap = int(rng.integers(3, min(9, (size - 3) // (count - 1)) + 1))
        gaps = [gap] * (count - 1)
    else:
        gaps = [3] * (count - 1)
        spare = size - 3 - sum(gaps)
        for index in rng.permutation(count - 1):
            extra = int(rng.integers(0, min(6, spare) + 1))
            gaps[int(index)] += extra
            spare -= extra
    start = int(rng.integers(1, size - 1 - sum(gaps)))
    return [start] + [start + int(v) for v in np.cumsum(gaps)]


def _center(rng, shape):
    h, w = shape
    r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
    if rng.random() < .35:
        if rng.random() < .5:
            r = int(rng.choice((1, h - 2)))
        else:
            c = int(rng.choice((1, w - 2)))
    if rng.random() < .12:
        r, c = int(rng.choice((1, h - 2))), int(rng.choice((1, w - 2)))
    return r, c


def _add(mask, points, cap):
    points = {(int(r), int(c)) for r, c in points
              if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]}
    extra = sum(not mask[p] for p in points)
    if int(mask.sum()) + extra <= cap:
        for p in points:
            mask[p] = True
        return True
    return False


def _plus_points(r, c):
    return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _stamp(rng, shape, kind):
    h, w = shape
    r, c = int(rng.integers(h)), int(rng.integers(w))
    if kind == "pixels":
        return [(r, c)]
    if kind == "clipped":
        edge = int(rng.integers(4))
        if edge < 2:
            r, c = (0 if edge == 0 else h - 1), int(rng.integers(1, w - 1))
        else:
            r, c = int(rng.integers(1, h - 1)), (0 if edge == 2 else w - 1)
        return _plus_points(r, c)
    if kind in ("T", "X", "ring"):
        offsets = {"T": ((0, -1), (0, 0), (0, 1), (1, 0)),
                   "X": ((-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)),
                   "ring": tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                                 if (a, b) != (0, 0))}[kind]
        turns = int(rng.integers(4))
        points = []
        for a, b in offsets:
            for _ in range(turns):
                a, b = -b, a
            points.append((r + a, c + b))
        return points
    if kind in ("block", "loop"):
        a, b = int(rng.integers(2, 7)), int(rng.integers(2, 8))
        return [(r + i, c + j) for i in range(a) for j in range(b)
                if kind == "block" or i in (0, a - 1) or j in (0, b - 1)]
    # Variable-length straight, bent, and branching strokes.
    points = [(r, c)]
    starts = [(r, c)]
    for _ in range(int(rng.integers(1, 5)) if kind == "branch" else 1):
        r, c = starts[int(rng.integers(len(starts)))]
        direction = int(rng.integers(4))
        for step in range(int(rng.integers(2, 13))):
            if kind != "line" and step and rng.random() < .3:
                direction = (direction + int(rng.choice((-1, 1)))) % 4
            dr, dc = _CARDINALS[direction]
            r, c = r + dr, c + dc
            points.append((r, c))
            starts.append((r, c))
    return points


def _build_scene(rng, shape, richness, positive):
    mask = np.zeros(shape, dtype=bool)
    cap = int(mask.size * .46)
    # Layout and surroundings are independent pair-level choices.
    layout = str(rng.choice(("single", "scatter", "array", "overlap"),
                            p=(.30, .32, .26, .12)))
    candidates = []
    if positive:
        if layout == "single":
            candidates = [_center(rng, shape)]
        elif layout == "array":
            rows, columns = _axis(rng, shape[0]), _axis(rng, shape[1])
            candidates = [(r, c) for r in rows for c in columns]
            if rng.random() < .4 and len(candidates) > 2:
                keep = rng.uniform(.4, .9)
                candidates = [p for p in candidates if rng.random() < keep] or [candidates[0]]
        else:
            count = int(rng.integers(1, min(35, max(2, cap // 5)) + 1))
            candidates = [_center(rng, shape) for _ in range(count)]
        if layout == "overlap":
            r, c = _center(rng, shape)
            candidates = [(r, c)]
            # Adjacent centers or two centers sharing their intervening arm.
            distance = int(rng.integers(1, 3))
            for dr, dc in rng.permutation(np.array(_CARDINALS)):
                a, b = r + int(dr) * distance, c + int(dc) * distance
                if 0 < a < shape[0] - 1 and 0 < b < shape[1] - 1:
                    candidates.append((a, b))
                    break
            candidates += [_center(rng, shape) for _ in range(int(rng.integers(4)))]
    centers = []
    for r, c in candidates:
        if _add(mask, _plus_points(r, c), cap):
            centers.append((r, c))
    if positive and not centers:
        p = _center(rng, shape)
        _add(mask, _plus_points(*p), cap)
        centers = [p]

    surround = str(rng.choice(("bare", "diagonal", "arms", "both"),
                             p=(.4, .2, .2, .2)))
    for r, c in centers:
        if surround in ("diagonal", "both"):
            diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for i in rng.permutation(4)[:int(rng.integers(1, 5))]:
                dr, dc = diagonals[int(i)]
                _add(mask, [(r + dr, c + dc)], cap)
        if surround in ("arms", "both"):
            for i in rng.permutation(4)[:int(rng.integers(1, 5))]:
                dr, dc = _CARDINALS[int(i)]
                length = int(rng.integers(2, 10))
                _add(mask, [(r + dr * k, c + dc * k) for k in range(2, length + 1)], cap)

    probabilities = ((.48, .32, .15, .05), (.25, .35, .25, .15), (.12, .23, .30, .35))[richness]
    clutter = int(rng.choice(4, p=probabilities))
    intervals = ((0., 0.), (.03, .13), (.16, .31), (.36, .45))
    density = float(rng.uniform(*intervals[clutter]))
    budget = min(cap, max(int(mask.sum()), int(mask.size * density)))
    if clutter:
        kinds = ("pixels", "line", "bend", "branch", "block", "loop", "T", "X", "ring", "clipped")
        chosen = list(rng.choice(kinds, size=int(rng.integers(1, 5)), replace=False))
        if rng.random() < .25:
            # Repeated camouflage uses a freshly sampled tile and unequal crop.
            a, b = int(rng.integers(2, 7)), int(rng.integers(2, 7))
            tile = rng.random((a, b)) < rng.uniform(.25, .65)
            phase = (int(rng.integers(a)), int(rng.integers(b)))
            rr, cc = np.indices(shape)
            repeated = tile[(rr + phase[0]) % a, (cc + phase[1]) % b]
            points = np.argwhere(repeated & ~mask)
            for i in rng.permutation(len(points))[:max(0, budget - int(mask.sum()))]:
                mask[tuple(points[int(i)])] = True
        else:
            for _ in range(2 * mask.size):
                if int(mask.sum()) >= budget:
                    break
                _add(mask, _stamp(rng, shape, str(rng.choice(chosen))), budget)
                if rng.random() < .12:
                    _add(mask, _stamp(rng, shape, "pixels"), budget)
        # Finish any remaining budget without modifying the planted motifs.
        available = np.argwhere(~mask)
        for i in rng.permutation(len(available))[:max(0, budget - int(mask.sum()))]:
            mask[tuple(available[int(i)])] = True
    if not positive:
        # Removing all existing centers cannot create a new complete plus.
        mask[_plus_centers_from_foreground(mask)] = False
        if not mask.any():
            mask[int(rng.integers(shape[0])), int(rng.integers(shape[1]))] = True
    return mask


def _pair_from_mask(mask, roles, highlight):
    background, foreground = roles
    inp = np.full(mask.shape, background, dtype=np.int8)
    inp[mask] = foreground
    return {"input": inp, "output": _apply_transformation(inp, foreground, highlight)}


def _make_pair(rng, band, richness, roles, highlight, seen,
               positive=True, selective=False, overlap=False):
    for _ in range(60):
        mask = _build_scene(rng, _sample_shape(rng, band), richness, positive)
        _, union = _scan(mask)
        if selective and not _touches_unmarked(mask, union):
            continue
        if overlap and np.array_equal(union, _sequential_union(mask)):
            continue
        pair = _pair_from_mask(mask, roles, highlight)
        if _grid_key(pair["input"]) not in seen and not _source_copy(pair["input"]):
            return pair
    # Deterministic, finite fallback. More distinct dimensions than pair slots
    # guarantee an unused input without relying on random rejection succeeding.
    for width in range(10, 31):
        mask = np.zeros((10, width), dtype=bool)
        if overlap:
            mask[2:5, 2:6] = True  # 3x4 block: two adjacent centers, union size 8.
        else:
            for p in _plus_points(3, 3):
                mask[p] = True
        if selective:
            mask[3, 5:8] = True
        if not positive:
            mask[:] = False
            mask[2, 2:5] = True
        pair = _pair_from_mask(mask, roles, highlight)
        if _grid_key(pair["input"]) not in seen and not _source_copy(pair["input"]):
            return pair
    raise AssertionError("fallback uniqueness bound exceeded")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    """Sample source_train demonstrations and target_test applications.

    Only the highlight is a learned puzzle-wide color convention. Palette mode,
    size progression, and richness are distribution choices, never hidden rules.
    """
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    n_train, n_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    highlight = int(rng.integers(10))
    palette = [v for v in range(10) if v != highlight]
    initial = tuple(map(int, rng.choice(palette, size=2, replace=False)))
    palette_mode = str(rng.choice(("stable", "swaps", "free"), p=(.25, .25, .50)))
    richness = int(rng.integers(3))
    shift = str(rng.choice(("mixed", "comparable", "growing", "shrinking"), p=(.35, .25, .20, .20)))
    common_band = str(rng.choice(("compact", "middle", "large")))
    bands = {"mixed": ("any", "any"), "comparable": (common_band, common_band),
             "growing": ("compact", "large"), "shrinking": ("large", "compact")}[shift]
    puzzle = {"train": [], "test": []}
    seen = set()
    previous = initial
    for split, count, band in (("train", n_train, bands[0]), ("test", n_test, bands[1])):
        for index in range(count):
            if palette_mode == "stable" or (split == "train" and index == 0):
                roles = initial
            elif palette_mode == "swaps":
                roles = initial[::-1] if (split == "train" and index == 1) or rng.random() < .5 else initial
            else:
                roles = tuple(map(int, rng.choice(palette, size=2, replace=False)))
                if split == "train" and index == 1:
                    foreground = int(rng.choice([v for v in palette if v != initial[1]]))
                    background = int(rng.choice([v for v in palette if v != foreground]))
                    roles = (background, foreground)
                elif rng.random() < .25:
                    roles = previous[::-1]
            previous = roles
            positive = (split == "train" and index < 2) or rng.random() >= .08
            if split == "test" and shift == "comparable":
                positive = True
            pair = _make_pair(rng, band, richness, roles, highlight, seen, positive,
                              selective=(split == "train" and index == 0))
            if split == "test" and shift == "comparable":
                # Comparable complexity is an optional sampling regime, based
                # on visible demonstrations, not a constraint on the rule.
                train_inputs = [p["input"] for p in puzzle["train"]]
                target_area = np.mean([g.size for g in train_inputs]) * rng.uniform(.8, 1.2)
                target_count = np.mean([_plus_centers_from_foreground(g == _visible_roles(g)[1]).sum()
                                        for g in train_inputs]) * rng.uniform(.75, 1.25)
                def distance(candidate):
                    grid = candidate["input"]
                    count = _plus_centers_from_foreground(grid == roles[1]).sum()
                    return abs(np.log(grid.size / target_area)) + abs(np.log((count + 1) / (target_count + 1)))
                best = distance(pair)
                for _ in range(10):
                    if best < .4:
                        break
                    candidate = _make_pair(rng, band, richness, roles, highlight, seen)
                    score = distance(candidate)
                    if score < best:
                        pair, best = candidate, score
            seen.add(_grid_key(pair["input"]))
            puzzle[split].append(pair)
    def needs_simultaneous(pair):
        _, foreground = _visible_roles(pair["input"])
        mask = pair["input"] == foreground
        _, union = _scan(mask)
        return not np.array_equal(union, _sequential_union(mask))
    if any(needs_simultaneous(p) for p in puzzle["test"]) and not any(needs_simultaneous(p) for p in puzzle["train"]):
        # Preserve both positive color demonstrations and selective evidence.
        index = int(rng.integers(n_train))
        roles = _visible_roles(puzzle["train"][index]["input"])
        puzzle["train"][index] = _make_pair(rng, bands[0], richness, roles, highlight,
                                            seen, selective=True, overlap=True)
    # Scene kind and required witnesses never have a fixed demonstration index.
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(n_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, pairs = set(), set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= d <= 30 for d in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((grid >= 0) & (grid <= 9))
        assert pair["input"].shape == pair["output"].shape
        key, full = _grid_key(pair["input"]), _pair_key(pair)
        assert key not in seen and full not in pairs
        assert not _source_copy(pair["input"])
        seen.add(key)
        pairs.add(full)
    highlight = _learn_highlight(puzzle["train"])
    selective_training = False
    overlap_witness = {"train": False, "test": False}
    roles_by_split, positive_foregrounds = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            inp, out = pair["input"], pair["output"]
            background, foreground = _visible_roles(inp)
            assert highlight not in (background, foreground)
            roles_by_split.add((background, foreground))
            mask = inp == foreground
            _, union = _scan(mask)
            assert np.array_equal(inp != out, union)
            assert np.all(out[union] == highlight)
            assert np.array_equal(out[~union], inp[~union])
            overlap_witness[split] |= not np.array_equal(union, _sequential_union(mask))
            if split == "train":
                if union.any():
                    positive_foregrounds.add(foreground)
                selective_training |= _touches_unmarked(mask, union)
            else:
                assert np.array_equal(_predict(puzzle["train"], inp.copy()), out)
    assert selective_training, "training must distinguish component recoloring"
    assert positive_foregrounds
    if len(roles_by_split) > 1:
        assert len(positive_foregrounds) >= 2, "variable palettes need positive color witnesses"
    assert not overlap_witness["test"] or overlap_witness["train"]
    # No scene, size, density, or array restriction is inherent to either split.


def _clutter_features(mask, union):
    remaining = mask & ~union
    up = np.zeros_like(mask)
    down, left, right = up.copy(), up.copy(), up.copy()
    up[1:] = remaining[:-1]
    down[:-1] = remaining[1:]
    left[:, 1:] = remaining[:, :-1]
    right[:, :-1] = remaining[:, 1:]
    bend = bool(np.any(remaining & (up | down) & (left | right)))
    branch = bool(np.any(remaining & ((up.astype(int) + down + left + right) >= 3)))
    windows = np.lib.stride_tricks.sliding_window_view
    block = bool(windows(mask, (2, 2)).all(axis=(-1, -2)).any())
    loop, periodic = False, False
    for h in range(4, min(8, mask.shape[0]) + 1):
        for w in range(4, min(8, mask.shape[1]) + 1):
            patch = windows(mask, (h, w))
            boundary = (patch[:, :, 0, :].all(-1) & patch[:, :, -1, :].all(-1)
                        & patch[:, :, :, 0].all(-1) & patch[:, :, :, -1].all(-1))
            loop |= bool((boundary & ~patch[:, :, 1:-1, 1:-1].any((-1, -2))).any())
    # Observable repeated camouflage: four identical tiles, each containing
    # both roles and at least 30% foreground, in a local 4x4 to 10x10 patch.
    for a in range(2, min(5, mask.shape[0] // 2) + 1):
        for b in range(2, min(5, mask.shape[1] // 2) + 1):
            patch = windows(mask, (2 * a, 2 * b))
            tile = patch[:, :, :a, :b]
            same = ((tile == patch[:, :, a:, :b]).all((-1, -2))
                    & (tile == patch[:, :, :a, b:]).all((-1, -2))
                    & (tile == patch[:, :, a:, b:]).all((-1, -2)))
            density = tile.mean((-1, -2))
            periodic |= bool((same & (density >= .3) & (density <= .7)).any())
    return dict(bend=bend, branch=branch, block=block, loop=loop,
                periodic=periodic, dense_periodic=bool(periodic and mask.mean() >= .35))


def _features(pair):
    """Observable review features; no generation-time parameters are consulted."""
    inp = pair["input"]
    bg, fg = _visible_roles(inp)
    mask = inp == fg
    centers, union = _scan(mask)
    h, w = mask.shape
    rows, cols = sorted({r for r, c in centers}), sorted({c for r, c in centers})
    cartesian = bool(centers) and len(centers) == len(rows) * len(cols)
    gaps = [int(v) for axis in (rows, cols) for v in np.diff(axis)]
    patterns = {
        "near_T": np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        "near_X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
        "ring": np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    }
    near_misses = {name: False for name in patterns}
    for r in range(h - 2):
        for c in range(w - 2):
            patch = mask[r:r + 3, c:c + 3]
            for name, pattern in patterns.items():
                if not near_misses[name]:
                    near_misses[name] = any(np.array_equal(patch, np.rot90(pattern, k))
                                            for k in range(4 if name == "near_T" else 1))
    extensions, lengths, diagonals = [], [], []
    for r, c in centers:
        arm_lengths = []
        for dr, dc in _CARDINALS:
            k = 2
            while 0 <= r + k * dr < h and 0 <= c + k * dc < w and mask[r + k * dr, c + k * dc]:
                k += 1
            arm_lengths.append(k - 2)
        extensions.append(sum(v > 0 for v in arm_lengths))
        lengths.extend(arm_lengths)
        diagonals.append(sum(mask[r + dr, c + dc] for dr in (-1, 1) for dc in (-1, 1)))
    return dict(shape=inp.shape, roles=(bg, fg), density=float(mask.mean()), centers=len(centers),
                changed=int(union.sum()), clean=bool(centers) and np.array_equal(mask, union),
                attached=_touches_unmarked(mask, union),
                overlap=5 * len(centers) > int(union.sum()),
                sequential=not np.array_equal(union, _sequential_union(mask)),
                adjacent=any((r + 1, c) in centers or (r, c + 1) in centers for r, c in centers),
                shared_arm=any((r + 2, c) in centers or (r, c + 2) in centers for r, c in centers),
                edge=any(r in (1, h - 2) or c in (1, w - 2) for r, c in centers),
                two_edges=any(r in (1, h - 2) and c in (1, w - 2) for r, c in centers),
                clipped=any(mask[r, c] and all(mask[r + dr, c + dc] for dr, dc in _CARDINALS
                            if 0 <= r + dr < h and 0 <= c + dc < w)
                            for r, c in [(0, j) for j in range(1, w - 1)] + [(h - 1, j) for j in range(1, w - 1)]
                            + [(i, 0) for i in range(1, h - 1)] + [(i, w - 1) for i in range(1, h - 1)]),
                cartesian=cartesian, array_rows=len(rows), array_columns=len(cols), gaps=gaps,
                unequal=cartesian and any(len(set(map(int, np.diff(axis)))) > 1 for axis in (rows, cols)),
                partial=not cartesian and len(centers) >= 3 and any(sum(a == r for a, b in centers) >= 2 for r in rows)
                        and any(sum(b == c for a, b in centers) >= 2 for c in cols),
                extensions=extensions, lengths=lengths, diagonals=list(map(int, diagonals)),
                diagonal_no_extension=any(d >= 3 and e == 0 for d, e in zip(diagonals, extensions)),
                isolated_corner=any(mask[r, c] and not mask[r + dr, c] and not mask[r, c + dc]
                                    for r, c, dr, dc in ((0, 0, 1, 1), (0, w - 1, 1, -1),
                                                        (h - 1, 0, -1, 1), (h - 1, w - 1, -1, -1))),
                **near_misses, **_clutter_features(mask, union))


def _puzzle_fingerprint(puzzle):
    return tuple(_pair_key(p) for s in ("train", "test") for p in puzzle[s])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_fingerprint(p) for p in puzzles}) == 42
    assert _puzzle_fingerprint(puzzles[17]) == _puzzle_fingerprint(generate_puzzle(17))
    observed = [_features(pair) for p in puzzles for s in ("train", "test") for pair in p[s]]
    shapes = {f["shape"] for f in observed}
    assert len(shapes) >= 70 and any(30 in s for s in shapes)
    assert any(min(s) <= 4 and max(s) >= 12 for s in shapes)
    assert any(h == w for h, w in shapes)
    assert len({f["centers"] for f in observed}) >= 15
    assert min(f["density"] for f in observed) < .05
    assert max(f["density"] for f in observed) > .41
    for name in ("clean", "attached", "overlap", "adjacent", "shared_arm", "edge", "two_edges", "clipped", "unequal", "partial",
                 "near_T", "near_X", "ring", "diagonal_no_extension", "bend", "branch", "block", "loop", "dense_periodic"):
        assert sum(bool(f[name]) for f in observed) >= 3, name
    assert sum(f["isolated_corner"] for f in observed) < len(observed) // 2
    assert any(f["centers"] and not f["clean"] and not f["attached"] for f in observed)
    assert sum(f["density"] >= .35 and f["overlap"] and f["edge"] for f in observed) >= 10
    assert any(max(f["shape"]) <= 13 and f["centers"] >= 5 for f in observed)
    assert any(max(f["shape"]) >= 20 and f["centers"] == 1 for f in observed)
    assert sum(f["centers"] == 0 for f in observed) >= 3
    arrays = [f for f in observed if f["cartesian"] and f["centers"] > 1]
    assert any(min(f["array_rows"], f["array_columns"]) == 1 for f in arrays)
    assert any(max(f["array_rows"], f["array_columns"]) >= 4 for f in arrays)
    assert {0, 1, 2, 3, 4} <= {v for f in observed for v in f["extensions"]}
    assert {0, 1, 2, 3, 4} <= {v for f in observed for v in f["diagonals"]}
    assert max(v for f in observed for v in f["lengths"]) >= 5
    assert len({_learn_highlight(p["train"]) for p in puzzles}) >= 8
    stable, variable, swaps, dimensions, smaller, larger = 0, 0, 0, 0, 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        groups = {s: [_features(p) for p in puzzle[s]] for s in ("train", "test")}
        fs = groups["train"] + groups["test"]
        roles = {f["roles"] for f in fs}
        stable += len(roles) == 1
        variable += len(roles) > 1
        swaps += any((b, a) in roles for a, b in roles)
        dimensions += len({f["shape"] for f in fs}) > 1
        train_area = np.mean([np.prod(f["shape"]) for f in groups["train"]])
        test_area = np.mean([np.prod(f["shape"]) for f in groups["test"]])
        smaller += test_area < .7 * train_area
        larger += test_area > 1.4 * train_area
    assert min(stable, variable, swaps, smaller, larger) >= 3
    assert dimensions == 42
    for split in ("train", "test"):
        fs = [_features(pair) for p in puzzles for pair in p[split]]
        assert any(f["centers"] == 1 for f in fs) and any(f["centers"] >= 10 for f in fs)
        assert any(f["cartesian"] and f["centers"] > 1 for f in fs)
        assert any(not f["cartesian"] and f["centers"] > 1 for f in fs)
        assert min(min(f["shape"]) for f in fs) <= 4
        assert max(max(f["shape"]) for f in fs) == 30
    # Regression: production and validation scanners agree on all binary 3x3s.
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        _, expected = _scan(mask)
        assert np.array_equal(expected, _plus_union_from_foreground(mask))
    block = np.ones((3, 4), dtype=bool)
    centers, union = _scan(block)
    assert len(centers) == 2 and union.sum() == 8
    assert _sequential_union(block).sum() == 5
    clipped = np.zeros((4, 5), dtype=bool)
    clipped[0, 1:4] = True
    clipped[1, 2] = clipped[3, 2] = True
    assert not _plus_union_from_foreground(clipped).any(), "never wrap across canvas edges"
    # Original dimensions are legal: selective plus, same size, novel content.
    mask = np.zeros((10, 12), dtype=bool)
    for point in _plus_points(4, 4) + [(4, 6)]:
        mask[point] = True
    novel = {"train": [_pair_from_mask(mask, (2, 4), 8),
                         _pair_from_mask(np.pad(mask, ((1, 0), (0, 0))), (2, 4), 8)],
             "test": [_pair_from_mask(np.pad(mask, ((0, 2), (0, 0))), (2, 4), 8)]}
    validate_generated_puzzle(novel)
    # The validator must detect wrong labels, including destructive overlap
    # labels which used to escape the generator's self-test.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in novel[s]] for s in novel}
    bad["test"][0]["output"][0, 0] = 8
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted an extra highlight")
    overlapping = np.pad(block, ((2, 2), (2, 2)))
    overlap_pair = _pair_from_mask(overlapping, (2, 4), 8)
    bad["train"][0] = overlap_pair
    bad["test"][0] = _pair_from_mask(np.pad(overlapping, ((1, 0), (0, 0))), (2, 4), 8)
    wrong = bad["test"][0]["input"].copy()
    wrong[_sequential_union(wrong == 4)] = 8
    bad["test"][0]["output"] = wrong
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted destructive overlap labels")
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(90817, a, b)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
        assert _puzzle_fingerprint(puzzle) == _puzzle_fingerprint(generate_puzzle(90817, a, b))
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3.5, 1)):
        try:
            generate_puzzle(7, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
