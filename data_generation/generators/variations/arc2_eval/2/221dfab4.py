"""Edge-seeded period-six tracks over arbitrary foreground objects."""

import hashlib
import numpy as np


# Content fingerprints, not dimension exclusions. The source grids remain legal sizes.
_SOURCE_PAIRS = {
    ((29, 22), "8a857b74b891167733a432e8ce0a78e77c6aea5d049e721bcc3c5912f147b18e"),
    ((29, 22), "f9fadb8fb063a1ea22ec35a86adaee45a0e9c43f1fd66d81c9cb520d5390c687"),
    ((25, 25), "45d818a3ca066741235de43b74acd88a907cc79cdbedd9a214c74ea8ee45c8ef"),
    ((30, 30), "57fddbac43d6e92ba55c6f56f4fbca8e79b50c244f9a1441994e351abd0d2ad5"),
}
_SIDES = ("top", "right", "bottom", "left")
_TEACHING = ("clearing", "marker_overwrite", "highlight_overwrite",
             "remote_highlight", "outside_preserved", "outside_background")


def _as_seed(seed):
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    return int(seed)


def _given_count(name, value, low, high):
    if value is None:
        return None
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    nt = _given_count("num_train", num_train, 2, 10)
    ns = _given_count("num_test", num_test, 1, 4)
    if nt is None and ns is None:
        ns = 1 + seed % 4
        nt = ns + 1 + (seed // 4) % (10 - ns)
    elif nt is None:
        nt = ns + 1 + (seed // 4) % (10 - ns)
    elif ns is None:
        ns = 1 + seed % min(4, nt - 1)
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    return nt, ns


def _canonical(grid, side):
    """View with the bar at the top; rows are inward distances."""
    if side == "top":
        return grid
    if side == "bottom":
        return grid[::-1]
    if side == "left":
        return grid.T
    assert side == "right"
    return grid.T[::-1]


def _orient(grid, side):
    if side in ("top", "bottom"):
        return _canonical(grid, side).copy()
    if side == "left":
        return grid.T.copy()
    return grid[::-1].T.copy()


def _edge_bar_for_color(grid, color):
    points = np.argwhere(grid == color)
    if not len(points):
        return None
    rows, cols = points.T
    candidates = []
    for side, fixed, boundary, positions in (
        ("top", rows, 0, cols), ("bottom", rows, grid.shape[0] - 1, cols),
        ("left", cols, 0, rows), ("right", cols, grid.shape[1] - 1, rows),
    ):
        if np.all(fixed == boundary):
            start = int(positions.min())
            if np.array_equal(np.sort(positions), np.arange(start, start + len(points))):
                candidates.append((side, start, len(points)))
    # In particular, a singleton at a corner has two possible inward directions.
    return candidates[0] if len(candidates) == 1 else None


def _analyze_input(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert 3 <= len(values) <= 6, "background, unique bar, and 1-4 foreground colors"
    background = int(values[np.argmax(counts)])
    assert int(counts.max()) > grid.size / 2, "background must be evident"
    bars = [(int(c), _edge_bar_for_color(grid, c)) for c in values if c != background]
    bars = [(c, bar) for c, bar in bars if bar is not None]
    assert len(bars) == 1, "exactly one unambiguous edge bar"
    marker, (side, start, length) = bars[0]
    foreground = tuple(int(c) for c in values if c not in (background, marker))
    return dict(background=background, marker=marker, foreground=foreground,
                side=side, start=start, length=length)


def _transform(input_grid, background, foreground, marker, highlight, side, start, length):
    """The original overwrite/recolor rule, also treating all object colors alike."""
    original = _canonical(input_grid, side)
    result = original.copy()
    objects = np.isin(original, foreground)
    result[:, start:start + length] = background
    for distance in range(len(original)):
        phase = distance % 6
        if phase in (0, 2):
            result[distance, start:start + length] = marker
        elif phase == 4:
            result[distance, start:start + length] = highlight
            result[distance, objects[distance]] = highlight
    return _orient(result, side)


def _pixel_prediction(grid, info, highlight):
    """Validation oracle: direct cell coordinates, independent of track painting."""
    result = grid.copy()
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            side = info["side"]
            distance = {"top": r, "bottom": height - 1 - r,
                        "left": c, "right": width - 1 - c}[side]
            lateral = c if side in ("top", "bottom") else r
            inside = info["start"] <= lateral < info["start"] + info["length"]
            color = int(grid[r, c])
            if inside:
                result[r, c] = (info["background"] if distance % 2 else
                                highlight if distance % 6 == 4 else info["marker"])
            elif distance % 6 == 4 and color not in (info["background"], info["marker"]):
                result[r, c] = highlight
    return result


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    components = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack, cells = [root], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        components.append(cells)
    return components


def _grow(rng, allowed, size, path=False):
    points = np.argwhere(allowed)
    result = np.zeros_like(allowed)
    if not len(points):
        return result
    root = tuple(points[int(rng.integers(len(points)))])
    frontier = [root]
    queued = {root}
    for _ in range(min(size, len(points))):
        if not frontier:
            break
        index = len(frontier) - 1 if path and rng.random() < .92 else int(rng.integers(len(frontier)))
        r, c = frontier.pop(index)
        result[r, c] = True
        neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        for index in rng.permutation(4):
            rr, cc = neighbors[int(index)]
            if (0 <= rr < len(allowed) and 0 <= cc < allowed.shape[1]
                    and allowed[rr, cc] and (rr, cc) not in queued):
                frontier.append((rr, cc))
                queued.add((rr, cc))
    return result


def _object(rng, allowed, size):
    """Procedural connected growth, winding paths, rectangles, profiles and cutouts."""
    height, width = allowed.shape
    kind = int(rng.integers(5))
    if kind < 2:
        return _grow(rng, allowed, size, path=kind == 1)
    aspect = float(np.exp(rng.uniform(-1.35, 1.35)))
    h = min(height, max(2, int(np.sqrt(size * aspect))))
    w = min(width, max(2, int(np.ceil(size / h))))
    r = int(rng.integers(height - h + 1))
    c = int(rng.integers(width - w + 1))
    local = np.ones((h, w), dtype=bool)
    if kind == 3:
        left, right = 0, w
        for row in range(h):
            left = int(np.clip(left + rng.integers(-2, 3), 0, max(0, w // 2 - 1)))
            right = int(np.clip(right + rng.integers(-2, 3), max(left + 1, w // 2), w))
            local[row, :left] = False
            local[row, right:] = False
    elif kind == 4 and h >= 4 and w >= 4:
        a, b = int(rng.integers(1, h - 2)), int(rng.integers(1, w - 2))
        local[a:h - 1, b:w - 1] = False
        if rng.random() < .5:
            local[a:h, b:w - 1] = False  # an open concavity instead of a hole
    result = np.zeros_like(allowed)
    result[r:r + h, c:c + w] = local
    return result & allowed


def _scene(rng, depth, span, start, length, mode):
    allowed = np.ones((depth, span), dtype=bool)
    allowed[0, start:start + length] = False
    if rng.random() < .4:
        allowed[0] = allowed[-1] = False
        allowed[:, 0] = allowed[:, -1] = False
    end = start + length
    if mode in ("empty_corridor", "one_side"):
        allowed[:, start:end] = False
        if mode == "one_side":
            if start and (end == span or rng.random() < .5):
                allowed[:, end:] = False
            else:
                allowed[:, :start] = False
    elif mode == "inside":
        allowed[:] = False
        allowed[1:, start:end] = True
    elif mode == "no_out_highlight":
        # One natural horizontal band between consecutive highlighted sections.
        gaps = [(a, min(a + 5, depth)) for a in range(5, depth, 6)] + [(1, min(4, depth))]
        a, b = gaps[int(rng.integers(len(gaps)))]
        allowed[:a] = allowed[b:] = False

    # Single inside objects and restricted scenes do not require teaching witnesses.
    if mode == "inside":
        size = int(rng.integers(1, max(2, min(int(allowed.sum()), depth * span // 3) + 1)))
        return _grow(rng, allowed, size, path=bool(rng.integers(2)))
    density = float(rng.uniform(*((.28, .42) if mode == "dense" else
                                  ((.04, .12), (.13, .26), (.27, .35), (.35, .42))[int(rng.integers(4))])))
    target = max(3, min(int(depth * span * density), int(allowed.sum() * .7)))
    family = int(rng.integers(3))
    count = int(rng.integers(*((1, 4), (4, 9), (9, 23))[family]))
    count = max(1, min(count, target // 3))
    budgets = np.maximum(3, (rng.dirichlet(np.full(count, 1.8)) * target).astype(int))
    mask = np.zeros_like(allowed)
    # Overlaps are allowed: finished components, rather than drawn stamps, are measured.
    for size in budgets:
        candidate = mask | _object(rng, allowed, int(size))
        if candidate.sum() <= depth * span * .44:
            mask = candidate
    for _ in range(8):
        if mask.sum() >= .9 * target:
            break
        candidate = mask | _object(rng, allowed, max(3, target - int(mask.sum())))
        if candidate.sum() <= depth * span * .44:
            mask = candidate
    if not mask.any():
        mask = _grow(rng, allowed, 1)
    return mask


def _dimensions(rng, side, regime, first):
    minimum = 13 if first else 7
    if regime == "target_test" and rng.random() < .32:
        depth, span = int(rng.integers(3, 13)), int(rng.integers(6, 31))
    elif rng.random() < .12:
        h, w = ((29, 22), (25, 25), (30, 30))[int(rng.integers(3))]
        depth, span = (h, w) if side in ("top", "bottom") else (w, h)
    else:
        depth = int(rng.integers(minimum, 31))
        span = depth if rng.random() < .27 else int(rng.integers(6, 31))
    return depth, span


def _sample_bar(rng, span):
    u = rng.random()
    if u < .24:
        length = int(rng.integers(1, 3))
    elif u < .48:
        length = int(rng.integers(max(2, (span * 3) // 5), span))
    else:
        length = int(rng.integers(2, max(3, span // 2 + 1)))
    low, high = (1, span - 1) if length == 1 else (0, span - length + 1)
    start = int(rng.integers(low, high))
    return start, length


def _colored_pair(rng, mask, side, start, length, highlight, palette, multi):
    background, marker = palette[:2]
    parts = _components(mask)
    if not parts:
        return None
    count = min(len(parts), int(rng.integers(2, 4))) if multi else 1
    if multi and count < 2:
        return None
    colors = palette[2:2 + count]
    grid = np.full(mask.shape, background, dtype=np.int8)
    for i, cells in enumerate(parts):
        for r, c in cells:
            grid[r, c] = colors[i % count]
    grid[0, start:start + length] = marker
    grid = _orient(grid, side)
    try:
        info = _analyze_input(grid)
    except AssertionError:
        return None
    if info["marker"] != marker:
        return None
    output = _transform(grid, background, colors, marker, highlight, side, start, length)
    return dict(input=grid, output=output)


def _fallback_pair(rng, side, highlight, palette, multi, depth=None):
    """Bounded teaching construction, used only if natural scene proposals fail.

    Two independently sized rectangles: one crosses the track, the other is remote.
    Their positions range over many distances; there is no phase-aligned cell stamp.
    """
    depth = int(rng.integers(19, 31)) if depth is None else depth
    span = int(rng.integers(19, 31))
    start, length = int(rng.integers(3, 6)), int(rng.integers(2, 5))
    mask = np.zeros((depth, span), dtype=bool)
    a = int(rng.integers(1, 5))
    b = int(rng.integers(1, start))
    mask[a:a + int(rng.integers(7, 11)), b:start + length + 1] = True
    a = int(rng.integers(1, 5))
    b = int(rng.integers(start + length + 3, span - 2))
    mask[a:a + int(rng.integers(7, 12)), b:b + int(rng.integers(2, 5))] = True
    return _colored_pair(rng, mask, side, start, length, highlight, palette, multi)


def _evidence(grid):
    info = _analyze_input(grid)
    g = _canonical(grid, info["side"])
    fg = np.isin(g, info["foreground"])
    strip = np.zeros_like(fg)
    strip[:, info["start"]:info["start"] + info["length"]] = True
    distances = np.arange(len(g))[:, None]
    high = np.broadcast_to(distances % 6 == 4, g.shape)
    ordinary = np.broadcast_to((distances > 0) & ((distances % 6 == 0) | (distances % 6 == 2)), g.shape)
    remote = False
    for cells in _components(fg):
        rr, cc = np.array(cells).T
        if not strip[rr, cc].any():
            far = (cc < info["start"] - 1) | (cc > info["start"] + info["length"])
            remote |= bool(np.any(high[rr, cc] & far))
    multi_taught = len(info["foreground"]) > 1 and all(
        np.any((g == c) & high & ~strip) and np.any((g == c) & ~high & ~strip)
        for c in info["foreground"])
    return dict(clearing=bool(np.any(fg & strip & (distances % 2 == 1))),
                marker_overwrite=bool(np.any(fg & strip & ordinary)),
                highlight_overwrite=bool(np.any(fg & strip & high)),
                remote_highlight=remote, outside_preserved=bool(np.any(fg & ~strip & ~high)),
                outside_background=bool(np.any((g == info["background"]) & ~strip & high)),
                cycle=len(g) >= 13, multi_taught=multi_taught)


def _covered(evidence, multi):
    return all(any(e[k] for e in evidence) for k in _TEACHING + ("cycle",)) and (
        not multi or any(e["multi_taught"] for e in evidence))


def _input_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _source_copy(pair):
    signature = hashlib.sha256(pair["input"].astype(np.int8).tobytes() +
                               pair["output"].astype(np.int8).tobytes()).hexdigest()
    return (pair["input"].shape, signature) in _SOURCE_PAIRS


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _as_seed(seed)
    nt, ns = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed % (1 << 64))
    highlight = int(rng.integers(10))  # the only learned output convention
    multi_family = bool(rng.random() < .43)
    orientation_regime = int(rng.integers(6))
    base = int(rng.integers(4))
    if orientation_regime == 0:
        train_sides, test_sides = ["bottom"] * nt, ["top" if i % 2 == 0 else "left" for i in range(ns)]
    elif orientation_regime < 4:
        train_sides = [_SIDES[base]] * nt
        shift = (0, 0, 2, int(rng.choice([1, 3])))[orientation_regime]
        test_sides = [_SIDES[(base + shift) % 4]] * ns
    else:
        order = rng.permutation(4).tolist()
        train_sides = [_SIDES[order[i % 4]] for i in range(nt)]
        choices = train_sides if orientation_regime == 4 else list(_SIDES)
        test_sides = [str(rng.choice(choices)) for _ in range(ns)]

    puzzle = {"train": [], "test": []}
    seen = set()
    evidence = []
    previous_roles = None
    for split, sides in (("train", train_sides), ("test", test_sides)):
        for index, side in enumerate(sides):
            # Role colors vary per pair, independently of the orientation regime.
            palette = rng.permutation([c for c in range(10) if c != highlight]).tolist()
            if previous_roles is not None:
                for slot in range(3):
                    if palette[slot] == previous_roles[slot]:
                        j = next(j for j in range(3, 9) if palette[j] != previous_roles[slot])
                        palette[slot], palette[j] = palette[j], palette[slot]
            previous_roles = palette[:3]
            regime = "source_train" if split == "train" else "target_test"
            mode = "normal" if split == "train" else str(rng.choice(
                ["normal", "normal", "dense", "empty_corridor", "one_side", "inside", "no_out_highlight"]))
            multi = multi_family and mode != "inside" and (index == 0 or rng.random() < .55)
            last_train = split == "train" and index == nt - 1
            accepted = None
            for attempt in range(62):
                if attempt >= 48:
                    # Fourteen different depths exceed the number of preceding pairs,
                    # so even a collision cannot exhaust the teaching fallback.
                    pair = _fallback_pair(rng, side, highlight, palette,
                                          multi_family if last_train else multi, depth=17 + attempt - 48)
                else:
                    depth, span = _dimensions(rng, side, regime, split == "train" and index == 0)
                    start, length = _sample_bar(rng, span)
                    mask = _scene(rng, depth, span, start, length, mode)
                    pair = _colored_pair(rng, mask, side, start, length, highlight, palette,
                                         multi_family if last_train and not any(e["multi_taught"] for e in evidence) else multi)
                if pair is None or _input_key(pair["input"]) in seen or _source_copy(pair):
                    continue
                observed = _evidence(pair["input"]) if split == "train" else None
                if last_train and not _covered(evidence + [observed], multi_family):
                    continue
                accepted = pair
                if observed is not None:
                    evidence.append(observed)
                break
            assert accepted is not None, "bounded scene construction failed"
            seen.add(_input_key(accepted["input"]))
            puzzle[split].append(accepted)
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_highlight(train):
    extra = set()
    for pair in train:
        extra.update(set(map(int, np.unique(pair["output"]))) - set(map(int, np.unique(pair["input"]))))
    assert len(extra) == 1, "training must reveal one shared highlight"
    return next(iter(extra))


def predict_from_training(train_pairs, test_input):
    """Only demonstrations and the query input are used, including for short queries."""
    highlight = _training_highlight(train_pairs)
    return _pixel_prediction(test_input, _analyze_input(test_input), highlight)


def _validate_pair(pair, highlight, exclude_source=True):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    for grid in pair.values():
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert all(1 <= n <= 30 for n in grid.shape)
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((grid >= 0) & (grid <= 9))
    a, b = pair["input"], pair["output"]
    assert a.shape == b.shape and not np.array_equal(a, b)
    info = _analyze_input(a)
    assert highlight not in a
    assert np.array_equal(b, _pixel_prediction(a, info, highlight)), "incorrect output pixels"
    if exclude_source:
        assert not _source_copy(pair), "literal source pair"
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    # Check schema before attempting to learn the highlight from outputs.
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        assert all(isinstance(g, np.ndarray) for g in pair.values())
    highlight = _training_highlight(puzzle["train"])
    seen, evidence = set(), []
    multi_test = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _validate_pair(pair, highlight)
            key = _input_key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            if split == "train":
                evidence.append(_evidence(pair["input"]))
            else:
                multi_test |= len(info["foreground"]) > 1
                assert np.array_equal(predict_from_training(puzzle["train"], pair["input"]), pair["output"])
    assert _covered(evidence, multi_test), "training set does not distinguish all rule interactions"


def measure_pair(pair):
    """Inspect actual pixels; no sampling metadata is needed for these measurements."""
    info = _analyze_input(pair["input"])
    g = _canonical(pair["input"], info["side"])
    out = _canonical(pair["output"], info["side"])
    fg = np.isin(g, info["foreground"])
    components = _components(fg)
    strip = np.zeros_like(fg)
    strip[:, info["start"]:info["start"] + info["length"]] = True
    high = np.broadcast_to(np.arange(len(g))[:, None] % 6 == 4, g.shape)
    boundary = np.zeros_like(fg)
    boundary[0] = boundary[-1] = boundary[:, 0] = boundary[:, -1] = True
    rectangles = elongated = concave = 0
    for cells in components:
        rows, cols = np.array(cells).T
        h, w = int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)
        rectangles += len(cells) == h * w and len(cells) >= 4
        elongated += max(h, w) >= 3 * min(h, w) and max(h, w) >= 6
        concave += len(cells) >= 8 and len(cells) < .8 * h * w
    holes = sum(not any(r in (0, len(g) - 1) or c in (0, g.shape[1] - 1) for r, c in cells)
                for cells in _components(~fg))
    padded = np.pad(fg, 1)
    degrees = sum(padded[1 + dr:1 + dr + len(g), 1 + dc:1 + dc + g.shape[1]] for dr, dc in
                  ((-1, 0), (1, 0), (0, -1), (0, 1)))
    old_points = ([(d, c) for d in (1, 2) for c in range(info["start"], info["start"] + 3)] +
                  [(d, c) for d in (4, 6, 10) for c in (1, 2, g.shape[1] - 3, g.shape[1] - 2)])
    old_stamp = (info["length"] >= 3 and len(g) >= 11 and all(fg[d, c] for d, c in old_points))
    info.update(shape=tuple(pair["input"].shape), depth=len(g), span=g.shape[1],
                foreground_cells=int(fg.sum()), density=float(fg.mean()), components=len(components),
                largest_component=max(map(len, components)), boundary_cells=int((fg & boundary).sum()),
                rectangles=int(rectangles), elongated=int(elongated), concave=int(concave), holes=int(holes),
                thin_cells=int(np.sum(fg & (degrees <= 2))),
                margin=min(info["start"], g.shape[1] - info["start"] - info["length"]),
                empty_corridor=not bool(np.any(fg & strip)), inside_only=not bool(np.any(fg & ~strip)),
                one_side=bool(np.any(fg[:, :info["start"]])) != bool(np.any(fg[:, info["start"] + info["length"]:])),
                outside_highlight_cells=int(np.sum(fg & ~strip & high)),
                foreground_disappears=not bool(np.any(np.isin(out, info["foreground"]))),
                old_stamp_present=bool(old_stamp),
                evidence=_evidence(pair["input"]))
    return info


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(731, 7, 3)
    assert _puzzles_equal(a, generate_puzzle(731, 7, 3))
    assert _puzzles_equal(generate_puzzle(73), generate_puzzle(73))
    assert not _puzzles_equal(a, generate_puzzle(732, 7, 3))
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(1000 + i, nt, ns)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
        validate_generated_puzzle(p)
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(17, nt, ns)
        except ValueError:
            continue
        raise AssertionError("invalid pair counts accepted")

    records, puzzles = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        for split in ("train", "test"):
            for pair in p[split]:
                records.append((split, measure_pair(pair)))
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_input_key(pair["input"]) for p in puzzles for s in p for pair in p[s]}) >= .98 * len(records)
    assert len({m["shape"] for _, m in records}) >= 80
    assert all(any(m["shape"] == shape for _, m in records) for shape in ((29, 22), (25, 25), (30, 30)))
    assert sum(m["old_stamp_present"] for _, m in records) < .1 * len(records)
    for split in ("train", "test"):
        rows = [m for s, m in records if s == split]
        assert {m["side"] for m in rows} == set(_SIDES)
        assert {np.sign(m["shape"][0] - m["shape"][1]) for m in rows} == {-1, 0, 1}
        assert len({(m["side"], int(np.sign(m["shape"][0] - m["shape"][1]))) for m in rows}) == 12
        assert max(m["foreground_cells"] for m in rows) >= 220
        assert any(.27 <= m["density"] <= .35 for m in rows)
        assert any(m["density"] < .1 for m in rows)
        assert any(m["components"] >= 9 for m in rows)
        assert any(m["components"] == 1 for m in rows)
        assert any(m["largest_component"] >= 100 for m in rows)
        assert any(m["length"] == 1 for m in rows) and any(m["length"] == 2 for m in rows)
        assert any(m["length"] >= .6 * m["span"] for m in rows)
        assert any(m["margin"] == 0 for m in rows)
        assert any(m["boundary_cells"] for m in rows)
        assert all(any(m[k] for m in rows) for k in ("rectangles", "elongated", "holes", "concave"))
    tests = [m for s, m in records if s == "test"]
    assert any(m["depth"] < 5 for m in tests)
    assert any(5 <= m["depth"] <= 10 for m in tests)
    assert all(any(m[k] for m in tests) for k in ("empty_corridor", "inside_only", "one_side", "foreground_disappears"))
    assert any(m["outside_highlight_cells"] == 0 for m in tests)
    assert all({m[k] for _, m in records} == set(range(10)) for k in ("background", "marker"))
    assert {c for _, m in records for c in m["foreground"]} == set(range(10))
    assert {_training_highlight(p["train"]) for p in puzzles} == set(range(10))
    assert any(len({_analyze_input(pair["input"])["side"] for pair in p["train"]}) > 1 for p in puzzles)
    assert any(len({_analyze_input(pair["input"])["side"] for s in p for pair in p[s]}) == 1 for p in puzzles)
    for p in puzzles:
        assert len({_analyze_input(pair["input"])["marker"] for s in p for pair in p[s]}) > 1
    assert any(len(m["foreground"]) > 1 for _, m in records)
    assert any(len(m["foreground"]) == 1 and m["density"] > .3 for _, m in records)
    assert any(len(m["foreground"]) > 1 and m["density"] >= .27 and m["boundary_cells"] for _, m in records)

    # Exercise fallback and boundary interactions even when random proposals suffice.
    rng = np.random.default_rng(994)
    for side in _SIDES:
        for depth in (17, 23, 30):
            pair = _fallback_pair(rng, side, 3, [0, 1, 2, 4, 5, 6, 7, 8, 9], True, depth)
            assert pair is not None and _covered([_evidence(pair["input"])], True)
            _validate_pair(pair, 3)
        grid = np.zeros((15, 12), dtype=np.int8)
        grid[0, 3:5] = 1
        grid[4:6, 0] = 2
        grid[-1, 3:5] = 2
        inp = _orient(grid, side)
        pair = dict(input=inp, output=_transform(inp, 0, (2,), 1, 3, side, 3, 2))
        _validate_pair(pair, 3)
        out = _canonical(pair["output"], side)
        assert out[4, 0] == 3 and out[5, 0] == 2 and np.all(out[-1, 3:5] == 1)

    # A grid with no remaining foreground and no highlight is still a valid short test.
    for side in _SIDES:
        for length in (1, 2, 8):
            grid = np.zeros((4, 18), dtype=np.int8)
            start = 1 if length == 1 else 0
            grid[0, start:start + length] = 1
            grid[1:3, start:start + length] = 2
            inp = _orient(grid, side)
            pair = dict(input=inp, output=_transform(inp, 0, (2,), 1, 3, side, start, length))
            _validate_pair(pair, 3)
            assert not np.any(pair["output"] == 2)
    # The validator must detect one incorrect pixel, including outside the track.
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
