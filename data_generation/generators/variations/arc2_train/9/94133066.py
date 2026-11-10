"""Solved ARC puzzles oriented by three duplicated, labeled corner keys."""

import hashlib
import numpy as np


TRANSFORMS = (
    "identity", "rot90", "rot180", "rot270",
    "main_diag", "anti_diag", "vertical", "horizontal",
)
CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
SOURCE_HASHES = frozenset({
    "d9205f0cd03b4f33ac6836f5a20cbf3cea69740c0b19b3aad82ac61f30554076",
    "da32e2282e285acef9b7fe4fce41a28dd68f5502e148637f562607b5fe7ee7d8",
    "17772c611c816a20853e4ce64da0c241c5da49d538160abc0a0d5cc5be442eaa",
    "004e8d79d535be4c9ff5db955b27ca9d530f7db2b59e3ea78d26d4d6fcfaaa8a",
})


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _resolve_counts(rng, num_train, num_test):
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        if rng.random() < 0.22:
            return 3, 1
        legal = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
        return legal[int(rng.integers(len(legal)))]
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _map_corner(name, corner):
    r, c = corner
    return {
        "identity": (r, c),
        "rot90": (1 - c, r),
        "rot180": (1 - r, 1 - c),
        "rot270": (c, 1 - r),
        "main_diag": (c, r),
        "anti_diag": (1 - c, 1 - r),
        "vertical": (1 - r, c),
        "horizontal": (r, 1 - c),
    }[name]


def _transform(panel, name):
    if name == "identity":
        result = panel
    elif name == "rot90":
        result = np.rot90(panel)
    elif name == "rot180":
        result = np.rot90(panel, 2)
    elif name == "rot270":
        result = np.rot90(panel, 3)
    elif name == "main_diag":
        result = panel.T
    elif name == "anti_diag":
        result = np.rot90(panel, 2).T
    elif name == "vertical":
        result = panel[::-1, :]
    elif name == "horizontal":
        result = panel[:, ::-1]
    else:
        raise ValueError("unknown transformation")
    return np.ascontiguousarray(result)


def _corner_cell(height, width, inset, corner):
    return (inset if corner[0] == 0 else height - 1 - inset,
            inset if corner[1] == 0 else width - 1 - inset)


def _aspect_shape(rng, aspect, low=7, high=13):
    if aspect == "square":
        side = int(rng.integers(low, high + 1))
        return side, side
    short = int(rng.integers(low, min(high - 1, 11) + 1))
    long = int(rng.integers(short + 1, high + 1))
    return (long, short) if aspect == "tall" else (short, long)


def _has_block(chosen, row, column):
    for dr in (-1, 0):
        for dc in (-1, 0):
            square = {(row + dr + a, column + dc + b)
                      for a in (0, 1) for b in (0, 1)}
            if square <= chosen | {(row, column)}:
                return True
    return False


def _motif(rng, height, width, blocked, size, style):
    allowed = {(r, c) for r in range(1, height - 1)
               for c in range(1, width - 1) if (r, c) not in blocked}
    chosen = set()
    last = None
    restart = {"stroke": 0.07, "separated": 0.32,
               "branch": 0.10, "compact": 0.03}[style]
    for _ in range(size):
        candidates = []
        if last is not None and rng.random() >= restart:
            bases = [last] if style in ("stroke", "separated") else list(chosen)
            for r, c in bases:
                for cell in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if cell not in allowed or cell in chosen:
                        continue
                    if style != "compact" and _has_block(chosen, *cell):
                        continue
                    candidates.append(cell)
        if not candidates:
            candidates = [cell for cell in allowed - chosen
                          if all(abs(cell[0]-r) + abs(cell[1]-c) > 1
                                 for r, c in chosen)]
        if not candidates:
            candidates = list(allowed - chosen)
        if not candidates:
            break
        last = candidates[int(rng.integers(len(candidates)))]
        chosen.add(last)
    return sorted(chosen)


def _layout(rng, panel, markers, background):
    height, width = panel.shape
    row_span = max(r for _, r, _ in markers)
    col_span = max(c for _, _, c in markers)
    candidates = []
    for side in range(4):
        for gap in range(2, 7):
            extent = row_span if side < 2 else col_span
            panel_extent = height if side < 2 else width
            for offset in range(-extent, panel_extent):
                if side < 2:
                    pt = max(0, -offset)
                    mt = pt + offset
                    pl = col_span + 1 + gap if side == 0 else 0
                    ml = 0 if side == 0 else width + gap
                else:
                    pl = max(0, -offset)
                    ml = pl + offset
                    pt = row_span + 1 + gap if side == 2 else 0
                    mt = 0 if side == 2 else height + gap
                used_h = max(pt + height, mt + row_span + 1)
                used_w = max(pl + width, ml + col_span + 1)
                if used_h + 2 <= 30 and used_w + 2 <= 30:
                    candidates.append((pt, pl, mt, ml, used_h, used_w))
    if not candidates:
        return None
    pt, pl, mt, ml, used_h, used_w = candidates[int(rng.integers(len(candidates)))]
    top = int(rng.integers(1, min(4, 30 - used_h)))
    left = int(rng.integers(1, min(4, 30 - used_w)))
    bottom = int(rng.integers(1, min(4, 31 - used_h - top)))
    right = int(rng.integers(1, min(4, 31 - used_w - left)))
    canvas = np.full((used_h + top + bottom, used_w + left + right),
                     background, dtype=np.int8)
    canvas[top+pt:top+pt+height, left+pl:left+pl+width] = panel
    for color, row, column in markers:
        canvas[top+mt+row, left+ml+column] = color
    return canvas


def _make_pair(rng, spec, convention):
    height, width = spec["shape"]
    inset = spec["inset"]
    keys = spec["keys"]
    excluded = set(keys)
    if convention["stable_palette"]:
        background, carrier = convention["background"], convention["carrier"]
    else:
        candidates = [c for c in convention["colors"] if c not in excluded]
        background = candidates[spec["index"] % len(candidates)]
        carrier = candidates[(spec["index"] + 1) % len(candidates)]
    if convention["simple"]:
        motif_palette = [convention["colors"][5]]
    else:
        motif_palette = [c for c in convention["colors"]
                         if c not in excluded | {background, carrier}]
        motif_palette = list(rng.permutation(motif_palette))
        motif_palette = motif_palette[:spec["motif_colors"]]
    missing = (convention["missing_corner"] + spec["index"]) % 4
    used_corners = [corner for i, corner in enumerate(CORNERS) if i != missing]
    used_corners = [used_corners[int(i)] for i in rng.permutation(3)]
    key_colors = [keys[int(i)] for i in rng.permutation(3)]
    key_to_corner = dict(zip(key_colors, used_corners))
    blocked = {_corner_cell(height, width, inset, corner) for corner in CORNERS}
    size = min(spec["motif_size"], (height - 2) * (width - 2) - 4)
    cells = _motif(rng, height, width, blocked, size, spec["style"])
    panel = np.full((height, width), carrier, dtype=np.int8)
    for rank, index in enumerate(rng.permutation(len(cells))):
        panel[cells[int(index)]] = motif_palette[rank % len(motif_palette)]
    for color, corner in key_to_corner.items():
        panel[_corner_cell(height, width, inset, corner)] = color
    markers = [(color, _map_corner(spec["transform"], corner))
               for color, corner in key_to_corner.items()]
    span_candidates = [(r, c) for r in range(2, 14) for c in range(2, 14)]
    for order in rng.permutation(len(span_candidates)):
        row_span, col_span = span_candidates[int(order)]
        external = [(color, bit[0]*row_span, bit[1]*col_span)
                    for color, bit in markers]
        canvas = _layout(rng, panel, external, background)
        if canvas is not None:
            return {"input": canvas,
                    "output": _transform(panel, spec["transform"])}
    raise AssertionError("no feasible marker layout")


def _components(mask):
    height, width = mask.shape
    seen = set()
    result = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            component = []
            while stack:
                x, y = stack.pop()
                component.append((x, y))
                for neighbor in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                    a, b = neighbor
                    if (0 <= a < height and 0 <= b < width and
                            mask[a, b] and neighbor not in seen):
                        seen.add(neighbor)
                        stack.append(neighbor)
            result.append(component)
    return result


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    background = int(inp[0, 0])
    components = sorted(_components(inp != background), key=len, reverse=True)
    assert len(components) == 4
    assert all(len(c) == 1 for c in components[1:])
    panel_cells = components[0]
    top = min(r for r, _ in panel_cells)
    bottom = max(r for r, _ in panel_cells)
    left = min(c for _, c in panel_cells)
    right = max(c for _, c in panel_cells)
    height, width = bottom-top+1, right-left+1
    assert len(panel_cells) == height*width and height >= 7 and width >= 7
    panel = inp[top:bottom+1, left:right+1]
    assert np.all(panel != background)
    border = np.concatenate((panel[0], panel[-1], panel[1:-1, 0], panel[1:-1, -1]))
    carrier = int(border[0])
    assert np.all(border == carrier) and carrier != background
    external = [component[0] for component in components[1:]]
    keys = [int(inp[cell]) for cell in external]
    assert len(set(keys)) == 3 and carrier not in keys and background not in keys
    internal = {}
    for color in keys:
        assert np.count_nonzero(inp == color) == 2
        positions = np.argwhere(panel == color)
        assert len(positions) == 1
        internal[color] = tuple(int(x) for x in positions[0])
    insets = [inset for inset in (1, 2)
              if all(r in (inset, height-1-inset) and
                     c in (inset, width-1-inset) for r, c in internal.values())]
    assert len(insets) == 1
    inset = insets[0]
    inner_corners = {color: (int(r != inset), int(c != inset))
                     for color, (r, c) in internal.items()}
    assert len(set(inner_corners.values())) == 3
    rows = sorted({r for r, _ in external})
    cols = sorted({c for _, c in external})
    assert len(rows) == 2 and len(cols) == 2
    row_span, col_span = rows[1]-rows[0], cols[1]-cols[0]
    assert row_span >= 2 and col_span >= 2
    outer_corners = {int(inp[r, c]): (int(r == rows[1]), int(c == cols[1]))
                     for r, c in external}
    assert len(set(outer_corners.values())) == 3
    matches = [name for name in TRANSFORMS
               if all(_map_corner(name, inner_corners[color]) == outer_corners[color]
                      for color in keys)]
    assert len(matches) == 1
    name = matches[0]
    expected = _transform(panel, name)
    assert np.array_equal(out, expected)
    assert np.array_equal(np.bincount(out.ravel(), minlength=10),
                          np.bincount(panel.ravel(), minlength=10))
    motif_mask = panel != carrier
    for color in keys:
        motif_mask &= panel != color
    motif_colors = tuple(int(x) for x in np.unique(panel[motif_mask]))
    assert 1 <= len(motif_colors) <= 3
    assert all(x not in keys and x not in (background, carrier) for x in motif_colors)
    motif_cells = np.argwhere(motif_mask)
    assert len(motif_cells) >= 4
    assert np.all((motif_cells[:, 0] > 0) & (motif_cells[:, 0] < height-1))
    assert np.all((motif_cells[:, 1] > 0) & (motif_cells[:, 1] < width-1))
    blocks = sum(bool(np.all(motif_mask[r:r+2, c:c+2]))
                 for r in range(height-1) for c in range(width-1))
    side = ("left" if max(c for _, c in external) < left else
            "right" if min(c for _, c in external) > right else
            "above" if max(r for r, _ in external) < top else "below")
    if side == "below":
        assert min(r for r, _ in external) > bottom
    if side == "left":
        gap = left - max(c for _, c in external) - 1
        offset = min(r for r, _ in external) - top
    elif side == "right":
        gap = min(c for _, c in external) - right - 1
        offset = min(r for r, _ in external) - top
    elif side == "above":
        gap = top - max(r for r, _ in external) - 1
        offset = min(c for _, c in external) - left
    else:
        gap = min(r for r, _ in external) - bottom - 1
        offset = min(c for _, c in external) - left
    assert gap >= 2
    return {"background": background, "carrier": carrier, "inset": inset,
            "transform": name, "shape": panel.shape, "keys": frozenset(keys),
            "inner_corners": inner_corners, "motif_colors": motif_colors,
            "motif_count": len(motif_cells),
            "motif_components": len(_components(motif_mask)),
            "blocks": blocks, "row_span": row_span, "col_span": col_span,
            "side": side, "gap": gap, "offset": offset,
            "motif_mask": motif_mask}


def _signature(array):
    a = np.ascontiguousarray(array.astype(np.int8))
    return (a.shape, a.tobytes())


def _pair_hash(pair):
    digest = hashlib.sha256()
    for key in ("input", "output"):
        array = np.ascontiguousarray(pair[key].astype(np.int8))
        digest.update(bytes(array.shape))
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs, info = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_sig = _signature(pair["input"])
            pair_sig = (input_sig, _signature(pair["output"]))
            assert input_sig not in inputs and pair_sig not in pairs
            inputs.add(input_sig)
            pairs.add(pair_sig)
            assert _pair_hash(pair) not in SOURCE_HASHES
            info[split].append(_analyze(pair))
    assert len({x["transform"] for x in info["train"]}) >= 2
    assert info["train"][0]["keys"] == info["train"][1]["keys"]
    assert info["train"][0]["inner_corners"] != info["train"][1]["inner_corners"]
    # The only puzzle-wide convention is the visible, labeled corner mapping.
    # Each test operation is independently and uniquely selected by its own keys.
    for item in info["test"]:
        assert item["transform"] in TRANSFORMS


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    colors = tuple(int(x) for x in rng.permutation(10))
    simple = bool(rng.random() < 0.28)
    stable_palette = simple or bool(rng.integers(2))
    convention = {"colors": colors, "background": colors[0],
                  "carrier": colors[1], "stable_palette": stable_palette,
                  "missing_corner": int(rng.integers(4)), "simple": simple}
    first_keys = tuple(colors[2:5])
    operation_pool = (("main_diag", "anti_diag", "vertical")
                      if simple else TRANSFORMS)
    transforms = [operation_pool[int(i)] for i in rng.permutation(len(operation_pool))]
    train_transforms = (transforms * 4)[:num_train]
    train_set = set(train_transforms)
    mode = int(rng.integers(2 if simple else 3))
    # Modes: larger target panels, overlapping sizes, and new test operations.
    test_transforms = []
    for _ in range(num_test):
        pool = ([name for name in TRANSFORMS if name not in train_set]
                if mode == 2 else [name for name in TRANSFORMS
                                   if name in train_set])
        if not pool:
            pool = list(TRANSFORMS)
        test_transforms.append(pool[int(rng.integers(len(pool)))])
    first_aspect = ("square", "tall", "wide")[int(rng.integers(3))]
    pairs = {"train": [], "test": []}
    used_keys = [first_keys, first_keys]
    base_key_pool = ([c for c in colors if c not in colors[:2]]
                     if stable_palette else list(colors))
    for index in range(2, num_train + num_test):
        selected = (first_keys if simple else
                    tuple(sorted(int(x) for x in rng.choice(base_key_pool, 3,
                                                             replace=False))))
        used_keys.append(selected)
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=num_train+num_test, dtype=np.int64)
    for index in range(num_train + num_test):
        local = np.random.default_rng(int(pair_seeds[index]))
        train = index < num_train
        split = "train" if train else "test"
        if index < 2:
            aspect = first_aspect
        else:
            aspect = ("square", "tall", "wide")[int(local.integers(3))]
        if not train and mode == 0:
            shape = _aspect_shape(local, aspect, 9, 15)
            if max(shape) < 12:
                shape = (12, shape[1]) if aspect == "tall" else (
                    shape[0], 12) if aspect == "wide" else (12, 12)
        else:
            shape = _aspect_shape(local, aspect, 7, 13 if mode else 11)
        style_pool = ("stroke", "separated", "branch", "compact")
        style = style_pool[int(local.integers(2 if simple else 4))]
        motif_size = int(local.integers(5, 16 if simple else 26))
        motif_colors = 1 if simple else int(local.integers(1, 4))
        spec = {"index": index, "shape": shape,
                "transform": train_transforms[index] if train else
                             test_transforms[index-num_train],
                "keys": used_keys[index],
                "inset": 1 if simple else int(local.integers(1, 3)),
                "style": style, "motif_size": motif_size,
                "motif_colors": motif_colors}
        pairs[split].append(_make_pair(local, spec, convention))
    validate_generated_puzzle(pairs)
    return pairs


def self_test() -> None:
    examples = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in examples:
        validate_generated_puzzle(puzzle)
    explicit = generate_puzzle(1234, 10, 4)
    validate_generated_puzzle(explicit)
    assert len(explicit["train"]) == 10 and len(explicit["test"]) == 4
    for index, (tr, te) in enumerate(((2, 1), (10, 1), (10, 4),
                                      (None, 1), (None, 4),
                                      (2, None), (10, None))):
        puzzle = generate_puzzle(200 + index, tr, te)
        validate_generated_puzzle(puzzle)
        if tr is not None:
            assert len(puzzle["train"]) == tr
        if te is not None:
            assert len(puzzle["test"]) == te
    for tr, te in ((1, None), (11, None), (None, 0),
                   (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(300, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair counts accepted")
    first = generate_puzzle(8080, 6, 2)
    again = generate_puzzle(8080, 6, 2)
    different = generate_puzzle(8081, 6, 2)
    assert all(_signature(a[key]) == _signature(b[key])
               for split in ("train", "test")
               for a, b in zip(first[split], again[split])
               for key in ("input", "output"))
    assert any(_signature(a["input"]) != _signature(b["input"])
               for a, b in zip(first["train"], different["train"]))
    infos = [_analyze(pair) for puzzle in examples
             for split in ("train", "test") for pair in puzzle[split]]
    train_counts = {len(p["train"]) for p in examples}
    test_counts = {len(p["test"]) for p in examples}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert {x["transform"] for x in infos} == set(TRANSFORMS)
    assert {x["side"] for x in infos} == {"left", "right", "above", "below"}
    assert max(x["row_span"] for x in infos) >= 9
    assert max(x["col_span"] for x in infos) >= 10
    assert any(x["motif_components"] > 1 for x in infos)
    assert any(x["blocks"] == 0 for x in infos)
    assert any(x["blocks"] > 0 for x in infos)
    assert len({x["shape"] for x in infos}) >= 25
    assert len({(x["background"], x["carrier"], x["inset"],
                 len(x["motif_colors"])) for x in infos}) >= 25
    assert {x["gap"] for x in infos} == {2, 3, 4, 5, 6}
    assert len({x["offset"] for x in infos}) >= 15
    assert sum(x["row_span"] != x["col_span"] for x in infos) >= 100
    assert any(len(x["motif_colors"]) == 3 for x in infos)
    assert sum(len(p["train"]) == 3 and len(p["test"]) == 1
               for p in examples) >= 3
    aspect_operations = {
        (x["transform"], "square" if x["shape"][0] == x["shape"][1]
         else "tall" if x["shape"][0] > x["shape"][1] else "wide")
        for x in infos
    }
    assert len(aspect_operations) == 24
    assert any(pair["input"].shape in
               {(22, 21), (23, 19), (22, 25), (19, 30)}
               for puzzle in examples for split in ("train", "test")
               for pair in puzzle[split])
    has_novel_test_operation = False
    has_larger_test_motif = False
    has_smaller_test_motif = False
    has_similar_test_motif = False
    has_overlapping_panel_shape = False
    stable_role_palette = 0
    changed_background = 0
    combined_variation = 0
    for puzzle in examples:
        train = [_analyze(pair) for pair in puzzle["train"]]
        test = [_analyze(pair) for pair in puzzle["test"]]
        all_info = train + test
        train_ops = {x["transform"] for x in train}
        train_sizes = [x["motif_count"] for x in train]
        train_shapes = {x["shape"] for x in train}
        has_novel_test_operation |= any(
            x["transform"] not in train_ops for x in test)
        has_larger_test_motif |= any(
            x["motif_count"] > max(train_sizes) for x in test)
        has_smaller_test_motif |= any(
            x["motif_count"] < min(train_sizes) for x in test)
        has_similar_test_motif |= any(
            abs(x["motif_count"] - n) <= 2
            for x in test for n in train_sizes)
        has_overlapping_panel_shape |= any(
            x["shape"] in train_shapes for x in test)
        backgrounds = {x["background"] for x in all_info}
        changed_background += len(backgrounds) > 1
        stable_role_palette += len({
            (x["background"], x["carrier"], x["keys"],
             x["motif_colors"], x["inset"]) for x in all_info
        }) == 1
        combined_variation += (
            len(backgrounds) > 1 and
            any(x["motif_components"] > 1 for x in all_info) and
            any(max(x["row_span"], x["col_span"]) >= 7
                for x in all_info) and
            len({x["transform"] for x in all_info}) >= 2
        )
    assert has_novel_test_operation and has_larger_test_motif
    assert has_smaller_test_motif and has_similar_test_motif
    assert has_overlapping_panel_shape
    assert stable_role_palette >= 3 and changed_background >= 3
    assert combined_variation >= 3
