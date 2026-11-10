"""Legend indexed, two by two tile substitution puzzles."""

import hashlib

import numpy as np


_SOURCE_DIGESTS = frozenset((
    "0da276f6b72bc6fc14d8e9ba9727a5b1aaa5862cac43de184ca8c5a48ca7bfdc",
    "fc005bb663e6fecffb6a9d42159992756ea54ba6d4f6553f4757347151592ef7",
    "237d1dc1a4c3c515f2918b347e60c79a4e9a785e56ed0f428d7c32f84d1745d5",
))


def _check_count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    if num_train is not None:
        num_train = _check_count("num_train", num_train, 2, 10)
    if num_test is not None:
        num_test = _check_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + int(seed) % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (int(seed) // 4) % (11 - minimum)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _mask(code):
    return np.array([[code & 1, (code >> 1) & 1],
                     [(code >> 2) & 1, (code >> 3) & 1]], dtype=bool)


def _digest_pair(inp, out):
    a = np.asarray(inp, dtype=np.uint8)
    b = np.asarray(out, dtype=np.uint8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes() +
                          bytes(b.shape) + b.tobytes()).hexdigest()


def _signature(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _sample_roles(rng, anchor=None, previous=None, exclude=()):
    available = [c for c in range(10) if c != anchor and c not in exclude]
    if previous is not None and rng.random() < 0.65:
        background = int(rng.choice([previous[1], previous[2]]))
    elif 0 in available and rng.random() < 0.30:
        background = 0
    else:
        background = int(rng.choice(available))
    remaining = [c for c in available if c != background]
    strokes = rng.choice(remaining, size=2, replace=False)
    roles = (background, int(strokes[0]), int(strokes[1]))
    if roles == previous:
        roles = (background, roles[2], roles[1])
    return roles


def _sample_selectors(rng, roles, anchor=None, new_color=None, mode="horizontal"):
    pool = [c for c in range(10) if c not in roles]
    count = int(rng.integers(3 if mode == "wrap" else 2, 7))
    required = []
    if anchor is not None:
        required.append(anchor)
    if new_color is not None and new_color in pool and new_color not in required:
        required.append(new_color)
    others = [c for c in pool if c not in required]
    picked = required + [int(x) for x in rng.choice(others, size=count-len(required), replace=False)]
    rng.shuffle(picked)
    return picked


def _sample_mapping(rng, selectors, changed=None):
    profile = int(rng.integers(0, 5))
    allowed_sizes = ((3,), (3, 4), (1, 2, 3, 4), (1, 2), (2, 3, 4))[profile]
    allowed = [m for m in range(1, 16) if m.bit_count() in allowed_sizes]
    mapping = {}
    for selector in selectors:
        if mapping and rng.random() < 0.10:
            choices = list(mapping.values())
        else:
            choices = allowed
            if rng.random() < 0.95:
                distinct = [m for m in choices if m not in mapping.values()]
                if distinct:
                    choices = distinct
        if changed is not None and selector == changed[0]:
            choices = [m for m in choices if m != changed[1]]
            if not choices:
                choices = [m for m in allowed if m != changed[1]]
        mapping[selector] = int(rng.choice(choices))
    return mapping


def _choose_layout(rng, mode, key_count):
    horizontal_pitch = int(rng.choice((4, 5)))
    vertical_pitch = int(rng.choice((4, 5)))
    if mode == "horizontal":
        legend_cols = key_count
    elif mode == "wrap":
        legend_cols = int(rng.integers(2, min(4, key_count - 1) + 1))
    else:
        legend_cols = 1
    legend_rows = (key_count + legend_cols - 1) // legend_cols
    legend_height = 3 + (legend_rows - 1) * vertical_pitch
    legend_width = 3 + (legend_cols - 1) * horizontal_pitch
    positions = [(i // legend_cols * vertical_pitch,
                  i % legend_cols * horizontal_pitch) for i in range(key_count)]
    top_gap = [(top, gap) for top in range(0, 4) for gap in range(2, 5)
               if top + legend_height + gap + 5 + 1 <= 30]
    if not top_gap:
        # A five-key vertical legend needs its compact four-cell row pitch.
        vertical_pitch = 4
        legend_height = 3 + (legend_rows - 1) * vertical_pitch
        positions = [(i // legend_cols * vertical_pitch,
                      i % legend_cols * horizontal_pitch) for i in range(key_count)]
        top_gap = [(top, gap) for top in range(0, 4) for gap in range(2, 5)
                   if top + legend_height + gap + 5 + 1 <= 30]
    assert top_gap
    legend_top, gap = top_gap[int(rng.integers(len(top_gap)))]
    legend_left = int(rng.integers(0, min(4, 29 - legend_width) + 1))
    array_left = int(rng.integers(0, 5))
    array_top = legend_top + legend_height + gap
    max_rows = min(7, (30 - array_top) // 3)
    max_cols = min(8, (30 - array_left) // 3)
    assert max_rows >= 2 and max_cols >= 2
    return (positions, legend_top, legend_left, array_top, array_left,
            legend_height, legend_width, max_rows, max_cols)


def _dimension_options(layout, first=False):
    max_rows, max_cols = layout[-2:]
    if first:
        return [(r, c) for r in range(3, min(5, max_rows) + 1)
                for c in range(3, min(6, max_cols) + 1)]
    return [(r, c) for r in range(2, max_rows + 1)
            for c in range(2, max_cols + 1)]


def _choose_dimensions(rng, options, relation=None, reference=None):
    if relation == "equal":
        eligible = [shape for shape in options if shape == reference]
    elif relation == "smaller":
        eligible = [shape for shape in options if shape[0] * shape[1] < reference[0] * reference[1]]
    elif relation == "larger":
        eligible = [shape for shape in options if shape[0] * shape[1] > reference[0] * reference[1]]
    else:
        eligible = options
    if not eligible:
        eligible = options
    return eligible[int(rng.integers(len(eligible)))]


def _sample_labels(rng, rows, cols, selectors, mandatory=None, first=False):
    total = rows * cols
    modes = ("full", "light", "mixed", "sparse", "blank_row", "blank_col", "cluster")
    mode = str(rng.choice(modes))
    if first and mode == "full":
        mode = "mixed"
    blank = np.zeros((rows, cols), dtype=bool)
    if mode in ("light", "mixed", "sparse"):
        lo, hi = {"light": (1, max(1, total // 5)),
                  "mixed": (max(1, total // 5), max(1, total // 2)),
                  "sparse": (max(1, total // 2), max(1, 4 * total // 5))}[mode]
        n = int(rng.integers(lo, hi + 1))
        blank.flat[rng.choice(total, size=n, replace=False)] = True
    elif mode == "blank_row":
        index = int(rng.choice((0, rows-1))) if rng.random() < 0.6 else int(rng.integers(rows))
        blank[index, :] = True
        blank |= rng.random((rows, cols)) < 0.20
    elif mode == "blank_col":
        index = int(rng.choice((0, cols-1))) if rng.random() < 0.6 else int(rng.integers(cols))
        blank[:, index] = True
        blank |= rng.random((rows, cols)) < 0.20
    elif mode == "cluster":
        r0 = int(rng.integers(rows))
        c0 = int(rng.integers(cols))
        radius = int(rng.integers(2, max(3, (rows + cols) // 2 + 1)))
        for r in range(rows):
            for c in range(cols):
                blank[r, c] = abs(r-r0) + abs(c-c0) <= radius
    if first and not blank.any():
        blank.flat[int(rng.integers(total))] = True
    must = [mandatory] if mandatory is not None else []
    if first:
        if mandatory is None:
            must = selectors[:2]
        else:
            must.append(next(s for s in selectors if s != mandatory))
    min_selected = max(1, len(must))
    if np.count_nonzero(~blank) < min_selected:
        for index in rng.permutation(total):
            blank.flat[int(index)] = False
            if np.count_nonzero(~blank) >= min_selected:
                break
    if first and not blank.any():
        blank.flat[int(rng.choice(np.flatnonzero(~blank.ravel())))] = True
    selected = np.flatnonzero(~blank.ravel())
    max_active = min(len(selectors), len(selected))
    unused = 0
    if max_active > max(1, len(must)) and rng.random() < 0.32:
        unused = int(rng.integers(1, min(2, max_active-max(1, len(must))) + 1))
    active_count = max(1, len(must), max_active-unused)
    choices = [s for s in selectors if s not in must]
    active = must + [int(x) for x in rng.choice(choices, size=active_count-len(must), replace=False)]
    labels = np.full((rows, cols), -1, dtype=np.int8)
    placement = str(rng.choice(("random", "row_band", "col_band", "patch")))
    row_keys = rng.choice(active, size=rows)
    col_keys = rng.choice(active, size=cols)
    patch_r, patch_c = int(rng.integers(rows)), int(rng.integers(cols))
    patch_key = int(rng.choice(active))
    for flat in selected:
        r, c = divmod(int(flat), cols)
        if placement == "row_band" and rng.random() < 0.78:
            value = row_keys[r]
        elif placement == "col_band" and rng.random() < 0.78:
            value = col_keys[c]
        elif placement == "patch" and abs(r-patch_r)+abs(c-patch_c) <= 2:
            value = patch_key
        else:
            value = rng.choice(active)
        labels[r, c] = int(value)
    rng.shuffle(selected)
    for selector, flat in zip(active, selected):
        labels.flat[int(flat)] = selector
    # Singleton use is deliberately possible; the legend still defines it.
    if len(active) > 1 and rng.random() < 0.28:
        singleton = int(rng.choice(active))
        positions = np.flatnonzero(labels.ravel() == singleton)
        for flat in positions[1:]:
            labels.flat[int(flat)] = int(rng.choice([s for s in active if s != singleton]))
    return labels


def _build_pair(rng, roles, selectors, mapping, mode, layout, dims, mandatory=None, first=False):
    background, stencil, carrier = roles
    rows, cols = dims
    (positions, legend_top, legend_left, array_top, array_left,
     legend_height, legend_width, _, _) = layout
    labels = _sample_labels(rng, rows, cols, selectors, mandatory, first)
    array_height, array_width = 3*rows-1, 3*cols-1
    right_extent = max(legend_left+legend_width, array_left+array_width)
    right_margin = int(rng.integers(1, min(3, 30-right_extent) + 1))
    bottom_margin = int(rng.integers(1, min(3, 30-array_top-array_height) + 1))
    inp = np.full((array_top+array_height+bottom_margin,
                   right_extent+right_margin), background, dtype=np.int8)
    order = [int(s) for s in rng.permutation(selectors)]
    for selector, (dr, dc) in zip(order, positions):
        r, c = legend_top+dr, legend_left+dc
        inp[r, c] = selector
        block = inp[r+1:r+3, c+1:c+3]
        block[_mask(mapping[selector])] = stencil
    out = np.full((array_height, array_width), background, dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            ir, ic = array_top+3*r, array_left+3*c
            inp[ir:ir+2, ic:ic+2] = carrier
            selector = int(labels[r, c])
            if selector >= 0:
                inp[ir+1, ic+1] = selector
                block = out[3*r:3*r+2, 3*c:3*c+2]
                block[_mask(mapping[selector])] = selector
    return {"input": inp, "output": out}


def _parse_input(inp):
    background = int(inp[-1, -1])
    assert np.all(inp[-1, :] == background) and np.all(inp[:, -1] == background)
    colors = [int(c) for c in np.unique(inp)]
    assert len(colors) >= 5
    solutions = []
    for carrier in colors:
        if carrier == background:
            continue
        cells = np.argwhere(inp == carrier)
        top, left = (int(x) for x in cells.min(axis=0))
        bottom, right = (int(x) for x in cells.max(axis=0))
        height, width = bottom-top+1, right-left+1
        if (height+1) % 3 or (width+1) % 3:
            continue
        rows, cols = (height+1)//3, (width+1)//3
        if not (2 <= rows <= 7 and 2 <= cols <= 8):
            continue
        labels = np.full((rows, cols), -1, dtype=np.int8)
        valid = True
        for r in range(height):
            if r % 3 == 2 and np.any(inp[top+r, left:right+1] != background):
                valid = False
        for c in range(width):
            if c % 3 == 2 and np.any(inp[top:bottom+1, left+c] != background):
                valid = False
        if not valid:
            continue
        for r in range(rows):
            for c in range(cols):
                rr, cc = top+3*r, left+3*c
                if not (inp[rr, cc] == carrier and inp[rr, cc+1] == carrier
                        and inp[rr+1, cc] == carrier):
                    valid = False
                    break
                corner = int(inp[rr+1, cc+1])
                if corner != carrier:
                    labels[r, c] = corner
            if not valid:
                break
        if not valid or not np.any(labels >= 0):
            continue
        used = set(int(x) for x in labels.flat if x >= 0)
        for stencil in colors:
            if stencil in (background, carrier) or stencil in used:
                continue
            selectors = [c for c in colors if c not in (background, carrier, stencil)]
            if not 2 <= len(selectors) <= 6 or not used.issubset(selectors):
                continue
            mapping = {}
            markers = {}
            for selector in selectors:
                occurrences = np.argwhere(inp == selector)
                outside = [(int(r), int(c)) for r, c in occurrences
                           if not (top <= r <= bottom and left <= c <= right)]
                if len(outside) != 1:
                    valid = False
                    break
                r, c = outside[0]
                if r+5 > top or c+3 >= inp.shape[1]:
                    valid = False
                    break
                block = inp[r+1:r+3, c+1:c+3]
                if block.shape != (2, 2) or not np.all((block == background) | (block == stencil)):
                    valid = False
                    break
                code = sum((int(block.flat[i] == stencil) << i) for i in range(4))
                if code == 0:
                    valid = False
                    break
                mapping[selector] = code
                markers[selector] = (r, c)
            if not valid:
                valid = True
                continue
            marker_rows = sorted(set(r for r, _ in markers.values()))
            row_columns = [sorted(c for r, c in markers.values() if r == row)
                           for row in marker_rows]
            ncols = max(len(x) for x in row_columns)
            if (len(marker_rows) > 1 and any(not 4 <= b-a <= 5
                                             for a, b in zip(marker_rows, marker_rows[1:]))) or (
                any(len(x) != ncols for x in row_columns[:-1]) or
                any(x != row_columns[0][:len(x)] for x in row_columns[1:]) or
                any(not 4 <= b-a <= 5 for x in row_columns for a, b in zip(x, x[1:]))
            ):
                continue
            mode = "horizontal" if len(marker_rows) == 1 else ("stack" if ncols == 1 else "wrap")
            expected = np.full_like(inp, background)
            for selector, (r, c) in markers.items():
                expected[r, c] = selector
                expected[r+1:r+3, c+1:c+3][_mask(mapping[selector])] = stencil
            for r in range(rows):
                for c in range(cols):
                    rr, cc = top+3*r, left+3*c
                    expected[rr:rr+2, cc:cc+2] = carrier
                    if labels[r, c] >= 0:
                        expected[rr+1, cc+1] = labels[r, c]
            if not np.array_equal(inp, expected):
                continue
            out = np.full((height, width), background, dtype=np.int8)
            for r in range(rows):
                for c in range(cols):
                    selector = int(labels[r, c])
                    if selector >= 0:
                        out[3*r:3*r+2, 3*c:3*c+2][_mask(mapping[selector])] = selector
            solutions.append({"output": out, "roles": (background, stencil, carrier),
                              "selectors": frozenset(selectors), "used": frozenset(used),
                              "mapping": mapping, "labels": labels, "mode": mode,
                              "rows": rows, "cols": cols, "blank": int(np.count_nonzero(labels < 0)),
                              "origin": (top, left)})
    assert len(solutions) == 1, f"expected one interpretation, found {len(solutions)}"
    return solutions[0]


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert 0 <= int(grid.min()) <= int(grid.max()) <= 9


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    parsed = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp)
            _validate_grid(out)
            signature = _signature(inp)
            assert signature not in seen
            seen.add(signature)
            assert _digest_pair(inp, out) not in _SOURCE_DIGESTS
            info = _parse_input(inp)
            assert np.array_equal(out, info["output"])
            assert inp.shape != out.shape or not np.array_equal(inp, out)
            parsed[split].append(info)
    training = parsed["train"]
    tests = parsed["test"]
    assert training[0]["blank"] > 0
    assert len(training[0]["used"]) >= 2
    assert training[0]["mode"] == "horizontal"
    assert training[0]["roles"] != training[1]["roles"]
    shared = training[0]["used"] & training[1]["used"]
    assert any(training[0]["mapping"][s] != training[1]["mapping"][s] for s in shared)
    trained_modes = {p["mode"] for p in training}
    assert all(p["mode"] in trained_modes for p in tests)
    if any(p["blank"] for p in tests):
        assert any(p["blank"] for p in training)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    # The source grammar is shared: a selector above-left of its stencil,
    # and a bottom-right index on two-by-two tiles with one-cell gutters.
    classic = bool(rng.random() < 0.16)
    stable_palette = bool(rng.random() < 0.28)
    size_relation = "larger" if classic else str(rng.choice(("smaller", "equal", "larger", "free")))
    additional_mode = str(rng.choice(("horizontal", "wrap", "stack")))
    train_modes = ["horizontal", additional_mode]
    for _ in range(2, num_train):
        train_modes.append(str(rng.choice(("horizontal", "wrap", "stack"))))
    puzzle = {"train": [], "test": []}
    seen = set()
    first_info = None
    first_roles = None
    reference_size = None
    train_selector_union = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for attempt in range(16):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))
                mode = train_modes[index] if split == "train" else (
                    "horizontal" if classic and index == 0 else str(pair_rng.choice(train_modes)))
                if stable_palette and split == "test" and index == 0:
                    mode = "horizontal"
                anchor = None
                if split == "train" and index == 1:
                    anchor = next(iter(first_info["used"]))
                same_palette = stable_palette and split == "test" and index == 0
                roles = _sample_roles(pair_rng, anchor,
                                      first_roles if split == "train" and index == 1 else None,
                                      first_info["selectors"] if same_palette else ())
                new_color = None
                if split == "test" and not same_palette and pair_rng.random() < 0.38:
                    unseen = [c for c in range(10) if c not in roles and c not in train_selector_union]
                    if unseen:
                        new_color = int(pair_rng.choice(unseen))
                selectors = (list(first_info["selectors"]) if same_palette else
                             _sample_selectors(pair_rng, roles, anchor, new_color, mode))
                if mode == "stack" and len(selectors) > 4:
                    selectors = selectors[:4]
                    if anchor is not None and anchor not in selectors:
                        selectors[-1] = anchor
                previous_mask = (anchor, first_info["mapping"][anchor]) if anchor is not None else None
                if same_palette:
                    changed_selector = next(iter(first_info["used"]))
                    previous_mask = (changed_selector, first_info["mapping"][changed_selector])
                mapping = _sample_mapping(pair_rng, selectors, previous_mask)
                layout = _choose_layout(pair_rng, mode, len(selectors))
                options = _dimension_options(layout, first=(split == "train" and index == 0))
                if classic and split == "train" and index == 0 and (3, 4) in options:
                    dims = (3, 4)
                elif classic and split == "test" and index == 0 and (4, 4) in options:
                    dims = (4, 4)
                else:
                    relation = size_relation if split == "test" and index == 0 else None
                    dims = _choose_dimensions(pair_rng, options, relation, reference_size)
                pair = _build_pair(pair_rng, roles, selectors, mapping, mode, layout,
                                   dims, anchor, first=(split == "train" and index == 0))
                signature = _signature(pair["input"])
                if signature not in seen and _digest_pair(pair["input"], pair["output"]) not in _SOURCE_DIGESTS:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    if split == "train":
                        train_selector_union.update(selectors)
                    if split == "train" and index == 0:
                        first_info = _parse_input(pair["input"])
                        first_roles = first_info["roles"]
                        reference_size = dims
                    break
            else:
                raise AssertionError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
        for x, y in zip(a[s], b[s])) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    input_signatures = set()
    shapes, boards, modes, roles, masks, blanks, key_counts = set(), set(), set(), set(), set(), set(), set()
    near_full_no_full = shared_geometry = singleton = unused = no_blank = blank_edge = 0
    near_full_with_full = mixed_masks = sparse_masks = sparse_board = 0
    tall = wide = combined = test_only = stable_palette = changed_palette = 0
    size_relations = set()
    role_exchange = 0
    within_size = within_background = within_mode = within_keys = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        infos = {split: [_parse_input(pair["input"]) for pair in puzzle[split]]
                 for split in ("train", "test")}
        if len({(x["rows"], x["cols"]) for x in infos["train"]+infos["test"]}) > 1:
            within_size += 1
        if len({x["roles"][0] for x in infos["train"]+infos["test"]}) > 1:
            within_background += 1
        if len({x["mode"] for x in infos["train"]+infos["test"]}) > 1:
            within_mode += 1
        if len({len(x["selectors"]) for x in infos["train"]+infos["test"]}) > 1:
            within_keys += 1
        stable_palette += infos["train"][0]["selectors"] == infos["test"][0]["selectors"]
        changed_palette += infos["train"][0]["selectors"] != infos["test"][0]["selectors"]
        train_colors = set().union(*(x["selectors"] for x in infos["train"]))
        test_only += any(x["selectors"] - train_colors for x in infos["test"])
        first_area = infos["train"][0]["rows"] * infos["train"][0]["cols"]
        test_area = infos["test"][0]["rows"] * infos["test"][0]["cols"]
        size_relations.add((test_area > first_area) - (test_area < first_area))
        train_roles = [x["roles"] for x in infos["train"]]
        if any(a in b for x in train_roles for y in train_roles if x != y for a, b in
               ((x[0], y[1:]), (x[1], (y[0], y[2])), (x[2], y[:2]))):
            role_exchange += 1
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                assert np.array_equal(pair["output"], info["output"])
                signature = _signature(pair["input"])
                input_signatures.add(signature)
                shapes.add(pair["input"].shape)
                boards.add((info["rows"], info["cols"]))
                modes.add(info["mode"])
                roles.add(info["roles"])
                masks.update(info["mapping"].values())
                blanks.add(info["blank"])
                key_counts.add(len(info["selectors"]))
                values = list(info["mapping"].values())
                near_full_no_full += all(m.bit_count() >= 3 for m in values) and 15 not in values
                near_full_with_full += all(m.bit_count() >= 3 for m in values) and 15 in values
                mixed_masks += any(m.bit_count() <= 2 for m in values) and any(m.bit_count() >= 3 for m in values)
                sparse_masks += all(m.bit_count() <= 2 for m in values)
                shared_geometry += len(values) != len(set(values))
                counts = [int(np.count_nonzero(info["labels"] == s)) for s in info["selectors"]]
                singleton += 1 in counts
                unused += 0 in counts
                no_blank += info["blank"] == 0
                sparse_board += info["blank"] > info["rows"] * info["cols"] // 2
                tall += info["rows"] > info["cols"]
                wide += info["cols"] > info["rows"]
                combined += (info["mode"] != "horizontal" and info["roles"][0] != 0
                             and 0 in counts and info["blank"] > info["rows"] * info["cols"] // 2)
                assert not (set(np.unique(pair["output"])) - (set(info["selectors"]) | {info["roles"][0]}))
                labels = info["labels"]
                blank_edge += (np.all(labels[0] < 0) or np.all(labels[-1] < 0) or
                               np.all(labels[:, 0] < 0) or np.all(labels[:, -1] < 0))
    assert len(input_signatures) >= 150
    assert len(shapes) >= 30 and len(boards) >= 20
    assert modes == {"horizontal", "wrap", "stack"}
    assert len(roles) >= 40 and len(masks) == 15
    assert len(key_counts) >= 4 and len(blanks) >= 10
    assert near_full_no_full > 0 and near_full_with_full > 0
    assert mixed_masks > 0 and sparse_masks > 0 and shared_geometry > 0
    assert singleton > 0 and unused > 0 and no_blank > 0 and blank_edge > 0
    assert sparse_board > 0 and tall > 0 and wide > 0 and combined > 0
    assert stable_palette > 0 and changed_palette > 0 and test_only > 0
    assert size_relations == {-1, 0, 1} and within_size >= 30 and role_exchange > 0
    assert within_background >= 30 and within_mode >= 20 and within_keys >= 20
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (4, 2), (7, 3)):
        p = generate_puzzle(2000+ntrain*10+ntest, ntrain, ntest)
        assert (len(p["train"]), len(p["test"])) == (ntrain, ntest)
    for ntrain, ntest in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(3000+(ntrain or 0)*10+(ntest or 0), ntrain, ntest)
        if ntrain is not None:
            assert len(p["train"]) == ntrain
        if ntest is not None:
            assert len(p["test"]) == ntest
    assert _puzzles_equal(generate_puzzle(777, 6, 3), generate_puzzle(777, 6, 3))
    assert not _puzzles_equal(generate_puzzle(777, 6, 3), generate_puzzle(778, 6, 3))
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(4000, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair counts accepted")
