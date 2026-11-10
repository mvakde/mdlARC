"""Nested-rectangle ARC puzzle generator."""

import hashlib
import numpy as np


# Numeric source inputs and independent digests of the literal task.json grids.
# Output digests come from task.json, not from the rule implementation.
_SOURCE = (
    ((16, 17, 3, 3, 3, 3, 2, 5, 2), "50e3c1a7a3c92cbe77f6a2e19a39ff8b237758a75a661a2c64c4ccff7cf73ebf", "2f16b2602ef5f659112b8da9324aea4dc0b840e367463f454b71c0d360cfa457"),
    ((13, 14, 3, 3, 1, 1, 2, 3, 1), "c4d9ef44c0ac92cb29a38d9e826061e4ab6e56fdb433891539419ae2b5340368", "30fb3b048dad35e5815075150ac13fc80e763767517b5d69cf0c23ffaf0edf30"),
    ((15, 14, 3, 3, 2, 2, 1, 6, 4), "9f6524714b23a9c2f53640daaaff887acc6185f7eacb5c251084f7efac6cf7ac", "b603ac1e36ef3e9f4aebecc28c1cb41260c139e9b712657f5e29436dc725770b"),
    ((12, 11, 3, 3, 1, 1, 1, 1, 2), "df0dc01bccd806d03753d186b9fce1485c210cfaef51b4807be0439321e0b30d", "d1389081e301b21a82b4b602213f1894d3930fa3f100dafb3790db42a0412f4a"),
    ((13, 19, 2, 2, 4, 3, 2, 4, 5), "8e71ec109b6963b60cce689fabfcbaa9c237f98a0bea21143cd58b7b9c35d86a", "6f1e5b2d3b6b18934dcc1d73b9ad0fba85e822aeff2b148703c8853dbc2d022c"),
    ((21, 21, 6, 5, 4, 4, 1, 3, 8), "1df0fe6ffc1168a5e06ca8162a9484fc5506743e6247fce66328926683ebea87", "35f03c50c9f5605b5d6420656f48d4c67bf2a3091c44093e06610b16f4497d69"),
)


def _digest(grid):
    return hashlib.sha256(np.asarray(grid, dtype=np.uint8).tobytes()).hexdigest()


def _check_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _check_count("num_train", num_train, 2, 10)
    num_test = _check_count("num_test", num_test, 1, 4)
    if num_train is None:
        if num_test is None:
            num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _source_copy(inp, out):
    return any(inp.shape == (spec[0], spec[1]) and
               _digest(inp) == input_hash and _digest(out) == output_hash
               for spec, input_hash, output_hash in _SOURCE)


def _paint_object(grid, r, c, h, w, thickness, frame, core):
    grid[r:r + h + 2 * thickness, c:c + w + 2 * thickness] = frame
    grid[r + thickness:r + thickness + h,
         c + thickness:c + thickness + w] = core


def _source_input(spec):
    height, width, r, c, h, w, thickness, frame, core = spec
    grid = np.zeros((height, width), dtype=np.int8)
    _paint_object(grid, r, c, h, w, thickness, frame, core)
    return grid


def _components(mask):
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    result = []
    for row, col in zip(*np.nonzero(mask)):
        if visited[row, col]:
            continue
        stack = [(int(row), int(col))]
        visited[row, col] = True
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not visited[rr, cc]:
                    visited[rr, cc] = True
                    stack.append((rr, cc))
        result.append(cells)
    return result


def _analyze_input(grid):
    height, width = grid.shape
    background = int(grid[0, 0])
    assert all(int(grid[r, c]) == background for r, c in
               ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)))
    components = _components(grid != background)
    assert 1 <= len(components) <= 3
    objects = []
    footprints = np.zeros(grid.shape, dtype=bool)
    for cells in components:
        rows, cols = zip(*cells)
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        assert 0 < r0 <= r1 < height - 1 and 0 < c0 <= c1 < width - 1
        region = grid[r0:r1 + 1, c0:c1 + 1]
        assert len(cells) == region.size and np.all(region != background)
        colors = [int(color) for color in np.unique(region)]
        assert len(colors) == 2
        candidates = []
        for color in colors:
            rr, cc = np.nonzero(region == color)
            a, b, d, e = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
            if (a > 0 and b < region.shape[0] - 1 and d > 0 and e < region.shape[1] - 1
                    and len(rr) == (b - a + 1) * (e - d + 1)
                    and np.all(region[a:b + 1, d:e + 1] == color)):
                candidates.append((color, a, b, d, e))
        assert len(candidates) == 1
        core, a, b, d, e = candidates[0]
        frame = colors[1] if colors[0] == core else colors[0]
        thickness = a
        assert thickness >= 1 and thickness == region.shape[0] - 1 - b
        assert thickness == d == region.shape[1] - 1 - e
        expected_region = np.full(region.shape, frame, dtype=grid.dtype)
        expected_region[a:b + 1, d:e + 1] = core
        assert np.array_equal(region, expected_region)
        core_height, core_width = b - a + 1, e - d + 1
        vertical = min(core_height, r0, height - r1 - 1)
        horizontal = min(core_width, c0, width - c1 - 1)
        assert vertical > 0 and horizontal > 0
        fr0, fr1 = r0 - vertical, r1 + vertical + 1
        fc0, fc1 = c0 - horizontal, c1 + horizontal + 1
        # A full background cell between enlarged objects leaves no collision rule.
        assert not np.any(footprints[max(0, fr0 - 1):min(height, fr1 + 1),
                                     max(0, fc0 - 1):min(width, fc1 + 1)])
        footprints[fr0:fr1, fc0:fc1] = True
        objects.append(dict(r0=r0, r1=r1, c0=c0, c1=c1,
                            core_height=core_height, core_width=core_width,
                            thickness=thickness, frame=frame, core=core,
                            vertical=vertical, horizontal=horizontal,
                            fr0=fr0, fr1=fr1, fc0=fc0, fc1=fc1))
    return dict(height=height, width=width, background=background, objects=objects)


def _solve_input(grid):
    info = _analyze_input(grid)
    output = grid.copy()
    for obj in info["objects"]:
        output[obj["fr0"]:obj["fr1"], obj["fc0"]:obj["fc1"]] = obj["core"]
        output[obj["r0"]:obj["r1"] + 1,
               obj["c0"]:obj["c1"] + 1] = grid[obj["r0"]:obj["r1"] + 1,
                                              obj["c0"]:obj["c1"] + 1]
    return output


def _analyze_pair(inp, out):
    info = _analyze_input(inp)
    assert inp.shape == out.shape and np.array_equal(out, _solve_input(inp))
    assert not np.array_equal(inp, out)
    return info


def _axis(rng, outer, ruler, limited, large=False, compact=False):
    growth = int(rng.integers(1, ruler)) if limited else ruler
    minimum = outer + 2 * growth
    assert minimum <= 30
    available = 30 - minimum
    choice = float(rng.random())
    if compact:
        extra = int(rng.integers(0, min(available, 3) + 1))
    elif choice < 0.18:
        extra = 0
    elif choice < (0.55 if not large else 0.32):
        extra = int(rng.integers(0, min(available, 3) + 1))
    else:
        extra = int(rng.integers(0, available + 1))
    if limited:
        margins = (growth, growth + extra)
        first = margins[int(rng.integers(0, 2))]
    else:
        first = growth + int(rng.integers(0, extra + 1))
    return minimum + extra, first


def _single_pair(rng, background, frame, core, style):
    for _ in range(500):
        if style == "large_square":
            h = w = int(rng.integers(4, 8))
            thickness = int(rng.integers(1, 3))
            mode = "full"
        elif style == "full_rect":
            h, w = int(rng.integers(2, 8)), int(rng.integers(2, 8))
            thickness = int(rng.integers(1, 5))
            mode = "full"
            if h == w:
                continue
        elif style == "one_axis":
            h, w = int(rng.integers(2, 8)), int(rng.integers(2, 8))
            thickness = int(rng.integers(1, 5))
            mode = ("vertical", "horizontal")[int(rng.integers(0, 2))]
        elif style == "tiny":
            h, w = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            thickness = int(rng.integers(1, 5))
            mode = ("full", "vertical", "horizontal", "both")[int(rng.integers(0, 4))]
        elif style == "compact":
            h, w = int(rng.integers(1, 5)), int(rng.integers(1, 5))
            thickness = int(rng.integers(1, 3))
            mode = ("full", "vertical", "horizontal", "both")[int(rng.integers(0, 4))]
        else:
            h, w = int(rng.integers(1, 9)), int(rng.integers(1, 9))
            thickness = int(rng.integers(1, 5))
            mode = ("full", "full", "vertical", "horizontal", "both")[int(rng.integers(0, 5))]
        lv, lh = mode in ("vertical", "both"), mode in ("horizontal", "both")
        if lv and h == 1 or lh and w == 1:
            continue
        outer_h, outer_w = h + 2 * thickness, w + 2 * thickness
        minimum_h = outer_h + 2 * (1 if lv else h)
        minimum_w = outer_w + 2 * (1 if lh else w)
        if minimum_h > 30 or minimum_w > 30:
            continue
        height, row = _axis(rng, outer_h, h, lv, style == "large_square", style == "compact")
        width, col = _axis(rng, outer_w, w, lh, style == "large_square", style == "compact")
        inp = np.full((height, width), background, dtype=np.int8)
        _paint_object(inp, row, col, h, w, thickness, frame, core)
        return inp, _solve_input(inp)
    raise AssertionError("single-object construction exhausted")


def _role(rng, background, previous=None):
    if (previous is not None and background not in previous
            and rng.random() < 0.38):
        return previous
    allowed = [c for c in range(10) if c != background]
    chosen = rng.choice(allowed, size=2, replace=False)
    return int(chosen[0]), int(chosen[1])


def _multi_pair(rng, background, first_role):
    for _ in range(500):
        count = 2 if rng.random() < 0.56 else 3
        thicknesses = [int(x) for x in rng.choice([1, 2, 3], size=count, replace=False)]
        shapes = []
        for index in range(count):
            for _ in range(20):
                shape = (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
                if shape not in shapes:
                    break
            shapes.append(shape)
        specs = [(h, w, thicknesses[i]) for i, (h, w) in enumerate(shapes)]
        if count == 2:
            slots = ([(0, 0), (0, 1)] if rng.random() < 0.5
                     else [(0, 0), (1, 0)])
        else:
            slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
            slots.pop(int(rng.integers(0, 4)))
            slots = [slots[int(i)] for i in rng.permutation(3)]
        row_heights = [0] * (1 + max(r for r, _ in slots))
        col_widths = [0] * (1 + max(c for _, c in slots))
        for (slot_r, slot_c), (h, w, thickness) in zip(slots, specs):
            row_heights[slot_r] = max(row_heights[slot_r], 3 * h + 2 * thickness)
            col_widths[slot_c] = max(col_widths[slot_c], 3 * w + 2 * thickness)
        gap_r, gap_c = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        height = sum(row_heights) + gap_r * (len(row_heights) - 1) + 2
        width = sum(col_widths) + gap_c * (len(col_widths) - 1) + 2
        if height > 30 or width > 30:
            continue
        extra_h = int(rng.integers(0, min(30 - height, 5) + 1))
        extra_w = int(rng.integers(0, min(30 - width, 5) + 1))
        height += extra_h
        width += extra_w
        base_r = 1 + int(rng.integers(0, extra_h + 1))
        base_c = 1 + int(rng.integers(0, extra_w + 1))
        inp = np.full((height, width), background, dtype=np.int8)
        roles = [first_role]
        while len(roles) < count:
            candidate = _role(rng, background)
            if candidate not in roles:
                roles.append(candidate)
        for (slot_r, slot_c), (h, w, thickness), (frame, core) in zip(slots, specs, roles):
            top = base_r + sum(row_heights[:slot_r]) + gap_r * slot_r
            left = base_c + sum(col_widths[:slot_c]) + gap_c * slot_c
            top += int(rng.integers(0, row_heights[slot_r] - (3 * h + 2 * thickness) + 1))
            left += int(rng.integers(0, col_widths[slot_c] - (3 * w + 2 * thickness) + 1))
            _paint_object(inp, top + h, left + w, h, w, thickness, frame, core)
        return inp, _solve_input(inp)
    raise AssertionError("multi-object construction exhausted")


def _signature(info):
    return tuple(sorted((o["core_height"], o["core_width"], o["thickness"],
                         o["vertical"], o["horizontal"]) for o in info["objects"]))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    mode_choice = float(rng.random())
    mode = ("fixed" if mode_choice < 0.25 else
            "multi" if mode_choice < 0.62 and num_train >= 3 else "varied")
    count = num_train + num_test
    if mode == "fixed":
        backgrounds = [int(rng.integers(0, 10))] * count
        fixed_role = _role(rng, backgrounds[0])
    else:
        first = int(rng.integers(0, 10))
        second = int(rng.choice([c for c in range(10) if c != first]))
        backgrounds = [first, second] + [int(rng.integers(0, 10)) for _ in range(count - 2)]
        fixed_role = None
    train_styles = ["full_rect", "one_axis"]
    if mode == "multi":
        train_styles.append("multi")
    if len(train_styles) < num_train:
        train_styles.append("tiny")
    train_styles += ["compact" if rng.random() < 0.25 else "mixed"
                     for _ in range(num_train - len(train_styles))]
    train_styles = [train_styles[int(i)] for i in rng.permutation(num_train)]
    test_styles = []
    for i in range(num_test):
        if mode == "multi" and (i == 0 or rng.random() < 0.45):
            test_styles.append("multi")
        else:
            choice = rng.random()
            test_styles.append("compact" if choice < 0.22 else
                               "large_square" if choice < 0.45 else "mixed")
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    train_signatures = set()
    previous_role = fixed_role
    for index, (split, style) in enumerate(
            [("train", x) for x in train_styles] + [("test", x) for x in test_styles]):
        background = backgrounds[index]
        role = fixed_role if fixed_role is not None else _role(rng, background, previous_role)
        previous_role = role
        for _ in range(200):
            child_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            inp, out = (_multi_pair(child_rng, background, role) if style == "multi"
                        else _single_pair(child_rng, background, *role, style))
            info = _analyze_pair(inp, out)
            signature = _signature(info)
            if (_grid_key(inp) not in seen_inputs and not _source_copy(inp, out)
                    and (split == "train" or signature not in train_signatures)):
                break
        else:
            raise AssertionError("unique pair construction exhausted")
        seen_inputs.add(_grid_key(inp))
        if split == "train":
            train_signatures.add(signature)
        puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _grid_key(pair["input"])
            pair_key = input_key, _grid_key(pair["output"])
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _source_copy(pair["input"], pair["output"])
            info[split].append(_analyze_pair(pair["input"], pair["output"]))
    train_objects = [obj for item in info["train"] for obj in item["objects"]]
    assert any(o["core_height"] != o["core_width"] and
               o["vertical"] == o["core_height"] and
               o["horizontal"] == o["core_width"] for o in train_objects)
    assert any((o["vertical"] < o["core_height"]) !=
               (o["horizontal"] < o["core_width"]) for o in train_objects)
    if any(len(item["objects"]) > 1 for item in info["test"]):
        assert any(len(item["objects"]) > 1 for item in info["train"])
    train_signatures = {_signature(item) for item in info["train"]}
    assert all(_signature(item) not in train_signatures for item in info["test"])


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    for spec, input_hash, output_hash in _SOURCE:
        inp = _source_input(spec)
        assert _digest(inp) == input_hash
        assert _digest(_solve_input(inp)) == output_hash
    assert _puzzles_equal(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3))
    assert not _puzzles_equal(generate_puzzle(31415, 6, 2), generate_puzzle(31416, 6, 2))
    for train, test in ((2, 1), (10, 1), (10, 4), (4, 3)):
        puzzle = generate_puzzle(1000 + 10 * train + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
        validate_generated_puzzle(puzzle)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2000 + (train or 0) + (test or 0), train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(9, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid counts accepted: {args}")
    train_counts, test_counts, backgrounds = set(), set(), set()
    shapes, core_shapes, thicknesses, roles, signatures = set(), set(), set(), set(), set()
    seen_puzzles = set()
    multi_train = multi_test = limited_test = exact_fit = 0
    varied_background = fixed_palette = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_key = tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                           for split in ("train", "test") for p in puzzle[split])
        seen_puzzles.add(puzzle_key)
        puzzle_colors = set()
        palette = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair["input"], pair["output"])
                shapes.add(pair["input"].shape)
                backgrounds.add(info["background"])
                puzzle_colors.add(info["background"])
                signatures.add(_signature(info))
                if len(info["objects"]) > 1:
                    if split == "train": multi_train += 1
                    else: multi_test += 1
                for o in info["objects"]:
                    core_shapes.add((o["core_height"], o["core_width"]))
                    thicknesses.add(o["thickness"])
                    roles.add((o["frame"], o["core"]))
                    palette.add((info["background"], o["frame"], o["core"]))
                    if split == "test" and (o["vertical"] < o["core_height"] or
                                             o["horizontal"] < o["core_width"]):
                        limited_test += 1
                    if o["fr0"] == 0 and o["fr1"] == info["height"] or \
                       o["fc0"] == 0 and o["fc1"] == info["width"]:
                        exact_fit += 1
        varied_background += len(puzzle_colors) > 1
        fixed_palette += len(palette) == 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(seen_puzzles) >= 40 and len(backgrounds) >= 8
    assert len(shapes) >= 50 and len(core_shapes) >= 25
    assert len(thicknesses) == 4 and len(roles) >= 40 and len(signatures) >= 100
    assert multi_train >= 3 and multi_test >= 2 and limited_test >= 5
    assert exact_fit >= 5 and varied_background >= 8 and fixed_palette >= 3
