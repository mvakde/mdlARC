"""Bilateral-symmetry erasure puzzles."""

import hashlib

import numpy as np


_SHAPES = (
    (2, 4), (4, 2), (3, 3), (3, 5), (5, 3), (4, 4),
    (2, 7), (7, 2), (5, 5), (3, 7), (7, 3),
    (4, 8), (8, 4), (6, 6), (5, 7), (7, 5),
)

_SOURCE_PAIR_DIGESTS = {
    "927a80ac132f8dd4baeb378d780602419f8839de29ed9dea89952117dfe30e32",
    "257ed53c5721ed3e33f6b25eb658a8b36acae8a946b6bb0427f8fa7b8d9618e0",
    "934d43664d61db4840f8a38ba1f96f1b3df343e78f58f81d807201087e538abb",
    "bf5d3a213f1f70748dce742842cb9109dd5b544a7347c100206b49a69ec04888",
}


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(rng, num_train, num_test):
    if num_train is not None:
        num_train = _integer(num_train, "num_train")
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be 2..10")
    if num_test is not None:
        num_test = _integer(num_test, "num_test")
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be 1..4")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _latent(rng, height, width, background, colors):
    """Compose one quadrant, then reflect it, before selecting an erasure."""
    qh, qw = (height + 1) // 2, (width + 1) // 2
    rr, cc = np.indices((qh, qw))
    q = np.full((qh, qw), background, dtype=np.int8)
    style = int(rng.integers(0, 6))
    density = int(rng.integers(0, 3))
    if density == 2:
        q[:] = colors[0]

    if style == 0:
        # Broad nested regions, including bands of equal color.
        depth = np.minimum(rr, cc)
        cuts = sorted(rng.choice(np.arange(1, max(3, min(qh, qw))), size=2, replace=False))
        q[depth >= cuts[0]] = colors[0]
        q[depth >= cuts[1]] = colors[-1]
    elif style == 1:
        # Independent rectangular fields.
        for _ in range(int(rng.integers(2, 6))):
            r0 = int(rng.integers(0, qh))
            c0 = int(rng.integers(0, qw))
            r1 = int(rng.integers(r0 + 1, qh + 1))
            c1 = int(rng.integers(c0 + 1, qw + 1))
            q[r0:r1, c0:c1] = colors[int(rng.integers(0, len(colors)))]
    elif style == 2:
        # Rings and interior islands.
        q[:] = colors[0]
        for inset in range(1, min(qh, qw), int(rng.integers(2, 5))):
            value = background if (inset // 2) % 2 else colors[-1]
            q[inset:, inset:] = value
    elif style == 3:
        # Repeated blocks with independent horizontal and vertical scales.
        bh = int(rng.integers(1, 5))
        bw = int(rng.integers(1, 5))
        phase = int(rng.integers(0, 4))
        pattern = ((rr // bh + 2 * (cc // bw) + phase) % 4)
        q[pattern <= 1] = colors[0]
        if len(colors) > 1:
            q[pattern == 3] = colors[-1]
    elif style == 4:
        # Crossing horizontal and vertical fields.
        rcut = int(rng.integers(2, qh - 1))
        ccut = int(rng.integers(2, qw - 1))
        q[rr < rcut] = colors[0]
        q[cc >= ccut] = colors[-1]
        q[(rr >= rcut) & (cc < ccut)] = background
    else:
        # Diagonal and chevron-like large features.
        offset = int(rng.integers(-3, 4))
        q[rr + offset >= cc] = colors[0]
        q[np.abs(rr - cc) <= int(rng.integers(0, 3))] = colors[-1]

    # Periodic seams and gaps vary independently of the broad fields.
    period_r = int(rng.integers(2, 7))
    period_c = int(rng.integers(2, 7))
    phase_r = int(rng.integers(0, period_r))
    phase_c = int(rng.integers(0, period_c))
    if density == 0:
        q[((rr + phase_r) % period_r == 0) | ((cc + phase_c) % period_c == 0)] = background
    elif density == 1:
        q[((rr + phase_r) % period_r == 0) & ((cc + phase_c) % period_c == 0)] = background
    else:
        q[((rr + phase_r) % period_r == 0) & ((cc + phase_c) % period_c == 0)] = colors[-1]

    for _ in range(int(rng.integers(2, 7))):
        r0 = int(rng.integers(0, qh))
        c0 = int(rng.integers(0, qw))
        r1 = int(rng.integers(r0 + 1, qh + 1))
        c1 = int(rng.integers(c0 + 1, qw + 1))
        value = background if rng.random() < 0.35 else colors[int(rng.integers(len(colors)))]
        operation = int(rng.integers(0, 4))
        if operation == 0:
            q[r0:r1, c0:c1] = value
        elif operation == 1:
            q[r0, c0:c1] = value
            q[r1 - 1, c0:c1] = value
            q[r0:r1, c0] = value
            q[r0:r1, c1 - 1] = value
        elif operation == 2:
            q[r0:r1, c0] = value
        else:
            q[r0, c0:c1] = value

    rows = np.minimum(np.arange(height), height - 1 - np.arange(height))
    cols = np.minimum(np.arange(width), width - 1 - np.arange(width))
    return q[np.ix_(rows, cols)]


def _axes(rect, height, width):
    r0, r1, c0, c1 = rect
    crosses_h = r0 <= (height - 1) // 2 and r1 > height // 2
    crosses_v = c0 <= (width - 1) // 2 and c1 > width // 2
    return crosses_h, crosses_v


def _category(rect, height, width):
    r0, r1, c0, c1 = rect
    h, v = _axes(rect, height, width)
    if h and v:
        return None
    if h:
        return "h_left" if c0 + c1 < width else "h_right"
    if v:
        return "v_top" if r0 + r1 < height else "v_bottom"
    vertical = "top" if r0 + r1 < height else "bottom"
    horizontal = "left" if c0 + c1 < width else "right"
    near_row_edge = r0 <= 2 or height - r1 <= 2
    near_col_edge = c0 <= 2 or width - c1 <= 2
    if near_row_edge and near_col_edge:
        prefix = "corner"
    elif near_row_edge or near_col_edge:
        prefix = "edge"
    else:
        prefix = "inner"
    return f"{prefix}_{vertical}_{horizontal}"


def _options(height, width, shape):
    mh, mw = shape
    result = {}
    for r0 in range(height - mh + 1):
        for c0 in range(width - mw + 1):
            rect = (r0, r0 + mh, c0, c0 + mw)
            category = _category(rect, height, width)
            if category is not None:
                result.setdefault(category, []).append(rect)
    return result


def _recover(input_grid, color):
    positions = np.argwhere(input_grid == color)
    if not len(positions):
        return None
    r0, c0 = positions.min(axis=0)
    r1, c1 = positions.max(axis=0) + 1
    r0, r1, c0, c1 = map(int, (r0, r1, c0, c1))
    if len(positions) != (r1 - r0) * (c1 - c0):
        return None
    height, width = input_grid.shape
    output = np.empty((r1 - r0, c1 - c0), dtype=np.int8)
    for row in range(r0, r1):
        for col in range(c0, c1):
            orbit = {
                (row, col), (height - 1 - row, col),
                (row, width - 1 - col),
                (height - 1 - row, width - 1 - col),
            }
            visible = [
                int(input_grid[r, c]) for r, c in orbit
                if not (r0 <= r < r1 and c0 <= c < c1)
            ]
            if not visible or len(set(visible)) != 1:
                return None
            output[row - r0, col - c0] = visible[0]
    repaired = input_grid.copy()
    repaired[r0:r1, c0:c1] = output
    if not np.array_equal(repaired, repaired[::-1, :]) or not np.array_equal(repaired, repaired[:, ::-1]):
        return None
    if color in repaired:
        return None
    return output, (r0, r1, c0, c1), repaired


def _solve_input(input_grid):
    candidates = []
    for color in np.unique(input_grid):
        answer = _recover(input_grid, int(color))
        if answer is not None:
            candidates.append((int(color), *answer))
    return candidates[0] if len(candidates) == 1 else None


def _asymmetric(crop):
    return not np.array_equal(crop, crop[::-1, :]) and not np.array_equal(crop, crop[:, ::-1])


def _generate_pair(rng, parameters, desired, require_teaching):
    mask_color = parameters["mask_color"]
    for _ in range(70):
        source_geometry = rng.random() < 0.06
        if source_geometry or rng.random() < 0.10:
            height, width = 24, 24
        else:
            height = int(rng.integers(17, 31))
            width = int(rng.integers(17, 31))
        if source_geometry:
            shape = (5, 5)
        elif rng.random() < parameters["shape_reuse"]:
            shape = parameters["base_shape"]
        else:
            shape = _SHAPES[int(rng.integers(len(_SHAPES)))]
        options = _options(height, width, shape)
        allowed = [name for name in desired if name in options]
        if not allowed:
            continue
        available = [color for color in range(10) if color != mask_color]
        if rng.random() < parameters["palette_reuse"]:
            background, colors = parameters["base_palette"]
        else:
            if rng.random() < 0.7:
                background = parameters["neutral_anchor"]
            else:
                background = int(rng.choice(available))
            available.remove(background)
            count = int(rng.integers(1, 5))
            colors = [int(x) for x in rng.choice(available, size=count, replace=False)]
        latent = _latent(rng, height, width, background, colors)
        if len(np.unique(latent)) < 2:
            continue
        category = allowed[int(rng.integers(len(allowed)))]
        rects = options[category]
        crop_mode = 2 if require_teaching else int(rng.choice(3, p=[0.20, 0.30, 0.50]))
        rng.shuffle(rects)
        chosen = None
        for rect in rects:
            r0, r1, c0, c1 = rect
            crop = latent[r0:r1, c0:c1]
            palette_size = len(np.unique(crop))
            if require_teaching and (palette_size < 2 or not _asymmetric(crop)):
                continue
            if crop_mode == 0 and palette_size != 1:
                continue
            if crop_mode == 1 and palette_size < 3:
                continue
            if crop_mode == 2 and palette_size < 2:
                continue
            chosen = rect
            break
        if chosen is None:
            continue
        r0, r1, c0, c1 = chosen
        output = latent[r0:r1, c0:c1].copy()
        inp = latent.copy()
        inp[r0:r1, c0:c1] = mask_color
        solved = _solve_input(inp)
        if solved is None or not np.array_equal(solved[1], output):
            continue
        return {"input": inp, "output": output}, category
    raise RuntimeError("could not sample a recoverable pair")


def _key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    mask_color = int(rng.integers(1, 10))
    base_background = 0 if rng.random() < 0.8 else int(rng.choice([x for x in range(1, 10) if x != mask_color]))
    low_variation = rng.random() < 0.12
    base_colors = tuple(
        int(x) for x in rng.choice(
            [x for x in range(10) if x not in (mask_color, base_background)],
            size=1 if low_variation else int(rng.integers(1, 5)),
            replace=False,
        )
    )
    parameters = {
        "mask_color": mask_color,
        "neutral_anchor": 0,
        "base_shape": _SHAPES[int(rng.integers(len(_SHAPES)))],
        "base_palette": (base_background, base_colors),
        "shape_reuse": 1.0 if low_variation else (0.0, 0.5, 0.85)[int(rng.integers(3))],
        "palette_reuse": 1.0 if low_variation else (0.0, 0.5, 0.85)[int(rng.integers(3))],
    }
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    used_categories = set()
    all_categories = [
        f"{prefix}_{vertical}_{horizontal}"
        for prefix in ("corner", "edge", "inner")
        for vertical in ("top", "bottom")
        for horizontal in ("left", "right")
    ] + ["h_left", "h_right", "v_top", "v_bottom"]
    for index in range(train_count + test_count):
        train = index < train_count
        if index == 0:
            desired = ["h_left", "h_right"]
        elif index == 1:
            desired = ["v_top", "v_bottom"]
        elif train and index == 2:
            desired = [x for x in all_categories if x.startswith("corner")]
        elif train or index == train_count:
            desired = [x for x in all_categories if x not in used_categories]
        else:
            desired = all_categories
        for _ in range(20):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            pair, category = _generate_pair(
                pair_rng, parameters, desired,
                train and index < 2,
            )
            key = _key(pair["input"])
            if key not in used_inputs:
                used_inputs.add(key)
                puzzle["train" if train else "test"].append(pair)
                if train:
                    used_categories.add(category)
                break
        else:
            raise RuntimeError("could not sample a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    seen_inputs, seen_pairs = set(), set()
    mask_colors = set()
    categories = {"train": [], "test": []}
    shape_set, palette_set = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid type")
                _require(all(1 <= n <= 30 for n in grid.shape), "grid size")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "grid palette")
            inp, out = pair["input"], pair["output"]
            digest = hashlib.sha256(
                bytes(inp.shape) + inp.astype(np.int8).tobytes()
                + bytes(out.shape) + out.astype(np.int8).tobytes()
            ).hexdigest()
            _require(digest not in _SOURCE_PAIR_DIGESTS, "source pair copied")
            ikey, okey = _key(inp), _key(out)
            _require(ikey not in seen_inputs and (ikey, okey) not in seen_pairs, "duplicate")
            seen_inputs.add(ikey)
            seen_pairs.add((ikey, okey))
            solved = _solve_input(inp)
            _require(solved is not None, "mask not uniquely identifiable")
            color, predicted, rect, repaired = solved
            _require(np.array_equal(predicted, out), "output not recovered crop")
            _require(out.shape == (rect[1] - rect[0], rect[3] - rect[2]), "crop shape")
            _require(color not in out and color not in repaired, "mask annotation in latent")
            mask_colors.add(color)
            shape_set.add(out.shape)
            palette_set.add(tuple(int(x) for x in np.unique(repaired)))
            categories[split].append(_category(rect, *inp.shape))
    _require(len(mask_colors) == 1, "inconsistent mask color")
    _require(any(x.startswith("h_") for x in categories["train"]), "horizontal-axis teaching example")
    _require(any(x.startswith("v_") for x in categories["train"]), "vertical-axis teaching example")
    _require(all(_asymmetric(p["output"]) and len(np.unique(p["output"])) >= 2 for p in puzzle["train"][:2]), "orientation teaching")
    if nt >= 3:
        _require(any(x.startswith("corner") for x in categories["train"]), "corner teaching example")
    _require(categories["test"][0] not in categories["train"], "target placement must add a configuration")


def self_test() -> None:
    for args in ((None, None), (2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        first = generate_puzzle(12345, *args)
        second = generate_puzzle(12345, *args)
        validate_generated_puzzle(first)
        _require(all(np.array_equal(a[k], b[k]) for split in ("train", "test")
                     for a, b in zip(first[split], second[split]) for k in ("input", "output")), "determinism")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts = set(), set()
    input_shapes, output_shapes, palettes, categories, structures = set(), set(), set(), set(), set()
    odd_axes = even_masks = multicolor = uniform = dense = varied_shapes = varied_palettes = 0
    both_odd = zero_nonmodal = symmetric_nonuniform = test_unseen = combined = 0
    test_categories = set()
    for seed in range(20000, 20042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        per_shapes, per_palettes = set(), set()
        train_output_colors = set(int(x) for p in puzzle["train"] for x in np.unique(p["output"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                color, predicted, rect, repaired = _solve_input(inp)
                _require(np.array_equal(predicted, out), "independent reconstruction")
                _require(np.array_equal(repaired, repaired[::-1]) and np.array_equal(repaired, repaired[:, ::-1]), "symmetry")
                input_shapes.add(inp.shape)
                output_shapes.add(out.shape)
                palette = tuple(int(x) for x in np.unique(repaired))
                palettes.add(palette)
                per_shapes.add(out.shape)
                per_palettes.add(palette)
                categories.add(_category(rect, *inp.shape))
                if split == "test":
                    test_categories.add(_category(rect, *inp.shape))
                    test_unseen += int(bool(set(int(x) for x in np.unique(out)) - train_output_colors))
                structures.add((inp.shape, inp.tobytes()))
                odd_axes += int(inp.shape[0] % 2 or inp.shape[1] % 2)
                both_odd += int(inp.shape[0] % 2 and inp.shape[1] % 2)
                even_masks += int(out.shape[0] % 2 == 0 or out.shape[1] % 2 == 0)
                multicolor += int(len(np.unique(out)) >= 3)
                uniform += int(len(np.unique(out)) == 1)
                dense += int(np.unique(repaired, return_counts=True)[1].max() > repaired.size * 0.5)
                values, counts = np.unique(repaired, return_counts=True)
                zero_nonmodal += int(0 in values and int(values[np.argmax(counts)]) != 0)
                symmetric_nonuniform += int(len(np.unique(out)) >= 2 and
                                             (np.array_equal(out, out[::-1]) or np.array_equal(out, out[:, ::-1])))
                combined += int((inp.shape[0] % 2 or inp.shape[1] % 2)
                                and (out.shape[0] % 2 == 0 or out.shape[1] % 2 == 0)
                                and len(np.unique(out)) >= 3)
        varied_shapes += int(len(per_shapes) > 1)
        varied_palettes += int(len(per_palettes) > 1)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity")
    _require(len(input_shapes) >= 35 and (24, 24) in input_shapes, "canvas diversity")
    _require(len(output_shapes) >= 9, "mask diversity")
    _require(len(palettes) >= 30, "palette diversity")
    _require(len(categories) >= 12, "placement diversity")
    _require(len(structures) >= 200, "structure diversity")
    _require(odd_axes >= 35 and both_odd >= 20 and even_masks >= 35, "odd-axis/even-mask diversity")
    _require(multicolor >= 15 and uniform >= 5 and symmetric_nonuniform >= 5, "crop palette diversity")
    _require(zero_nonmodal >= 20 and dense >= 35, "dense design diversity")
    _require(test_unseen >= 5 and len(test_categories) >= 12, "test color and placement diversity")
    _require(combined >= 10, "combined axes, mask, and palette diversity")
    _require(varied_shapes >= 32 and varied_palettes >= 32, "within-puzzle diversity")
