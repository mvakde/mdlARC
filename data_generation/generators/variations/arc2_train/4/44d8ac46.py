import numpy as np


_SOURCE_INPUT_ROWS = (
    (
        "000000000000",
        "055555000000",
        "050005000000",
        "050005055550",
        "050005050050",
        "050005050050",
        "055555055550",
        "000000000000",
        "000555555000",
        "000500005000",
        "000500005000",
        "000555555000",
    ),
    (
        "000000000000",
        "055550000000",
        "055050000000",
        "055550555555",
        "055550500005",
        "000000500005",
        "000000500005",
        "555550500005",
        "555550555555",
        "500550000000",
        "500550000000",
        "555550000000",
    ),
    (
        "000000000000",
        "055550000000",
        "050050000000",
        "055550000000",
        "055550000000",
        "000000000000",
        "000555550000",
        "000500550000",
        "000500550000",
        "000500050000",
        "000555550000",
        "000000000000",
    ),
    (
        "000000000000",
        "000000000000",
        "055550000000",
        "050050000000",
        "050050055550",
        "055550050550",
        "000000050050",
        "000000055550",
        "005555000000",
        "005555000000",
        "005055000000",
        "005555000000",
    ),
    (
        "000000000000",
        "555550055550",
        "500050055050",
        "500050050050",
        "500050050050",
        "555550055550",
        "000000000000",
        "000000000000",
        "005555550000",
        "005000050000",
        "005000050000",
        "005555550000",
    ),
)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _validate_optional_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_optional_count(
        "num_train", num_train, 2, 10
    )
    num_test = _validate_optional_count(
        "num_test", num_test, 1, 4
    )

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = low + seed % (11 - low)
    elif num_test is None:
        high = min(4, num_train - 1)
        num_test = 1 + seed % high

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    return {"highlight": int(rng.integers(0, 10))}


def _connected_mask(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    start = tuple(int(x) for x in cells[0])
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                    and (nr, nc) not in seen):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _make_hole(rng: np.random.Generator, kind: str,
               side: int | None = None) -> dict:
    if kind == "square":
        h = w = int(side if side is not None else rng.integers(1, 8))
        mask = np.ones((h, w), dtype=bool)
    elif kind == "rectangle":
        h = int(rng.integers(1, 7))
        w = int(rng.integers(1, 7))
        if h == w:
            w = 1 + w % 6
        mask = np.ones((h, w), dtype=bool)
    else:
        h = int(rng.integers(2, 7))
        w = h if kind == "irregular_square_box" else int(rng.integers(2, 7))
        if kind == "irregular_non_square_box" and h == w:
            w = 2 + w % 5
        mask = np.ones((h, w), dtype=bool)
        target = int(rng.integers(1, min(7, h * w // 3 + 1)))
        removed = 0
        for _ in range(80):
            if removed >= target:
                break
            boundary = [(r, c) for r in range(h) for c in range(w)
                        if mask[r, c] and (r in (0, h - 1) or c in (0, w - 1)
                            or any(0 <= r + dr < h and 0 <= c + dc < w
                                   and not mask[r + dr, c + dc]
                                   for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))))]
            if not boundary:
                break
            r, c = boundary[int(rng.integers(0, len(boundary)))]
            mask[r, c] = False
            if (not np.any(mask[0]) or not np.any(mask[-1])
                    or not np.any(mask[:, 0]) or not np.any(mask[:, -1])
                    or not _connected_mask(mask)):
                mask[r, c] = True
            else:
                removed += 1
        if removed == 0:
            mask[0, 0] = False
        assert _connected_mask(mask)
    return {"kind": kind, "mask": mask, "padded": False}


def _make_object(rng: np.random.Generator, holes: list[dict],
                 compact: bool = False) -> np.ndarray:
    horizontal = bool(rng.integers(0, 2))
    gap = int(rng.integers(1, 4))
    origins = []
    cursor = 0
    for hole in holes:
        hh, ww = hole["mask"].shape
        offset = int(rng.integers(0, 3))
        origins.append((offset, cursor) if horizontal else (cursor, offset))
        cursor += (ww if horizontal else hh) + gap
    content_h = max(r + hole["mask"].shape[0]
                    for (r, _), hole in zip(origins, holes))
    content_w = max(c + hole["mask"].shape[1]
                    for (_, c), hole in zip(origins, holes))
    walls = [int(rng.integers(1, 3 if compact else 4)) for _ in range(4)]
    if any(hole["padded"] for hole in holes):
        walls[int(rng.integers(0, 4))] = int(rng.integers(2, 5))
    if len(holes) == 1 and holes[0]["kind"] == "rectangle" and rng.random() < .35:
        outer_h = content_h + walls[0] + walls[1]
        outer_w = content_w + walls[2] + walls[3]
        if outer_h < outer_w:
            walls[0] += outer_w - outer_h
        elif outer_w < outer_h:
            walls[2] += outer_h - outer_w
    top, bottom, left, right = walls
    block = np.ones((content_h + top + bottom, content_w + left + right), dtype=bool)
    for (r, c), hole in zip(origins, holes):
        hh, ww = hole["mask"].shape
        block[top + r:top + r + hh, left + c:left + c + ww] &= ~hole["mask"]
    if rng.random() < (.25 if compact else .55):
        padded = np.pad(block, 3)
        bh, bw = block.shape
        for _ in range(int(rng.integers(1, 4))):
            side = int(rng.integers(0, 4))
            if side < 2:
                length = int(rng.integers(1, min(5, bw + 1)))
                start = int(rng.integers(0, bw - length + 1))
                depth = int(rng.integers(1, 4))
                rows = slice(3 - depth, 3) if side == 0 else slice(3 + bh, 3 + bh + depth)
                padded[rows, 3 + start:3 + start + length] = True
            else:
                length = int(rng.integers(1, min(5, bh + 1)))
                start = int(rng.integers(0, bh - length + 1))
                depth = int(rng.integers(1, 4))
                cols = slice(3 - depth, 3) if side == 2 else slice(3 + bw, 3 + bw + depth)
                padded[3 + start:3 + start + length, cols] = True
        rr, cc = np.where(padded)
        block = padded[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    return block


def _partition_holes(rng: np.random.Generator, holes: list[dict]) -> list[list[dict]]:
    n = len(holes)
    count = int(rng.integers(1, min(5, n) + 1))
    order = [int(i) for i in rng.permutation(n)]
    groups = [[holes[order[i]]] for i in range(count)]
    for i in order[count:]:
        groups[int(rng.integers(0, count))].append(holes[i])
    return groups


def _separated(a: tuple[int, int, int, int],
               b: tuple[int, int, int, int]) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return (ar + ah + 1 <= br or br + bh + 1 <= ar
            or ac + aw + 1 <= bc or bc + bw + 1 <= ac)


def _place_objects(rng: np.random.Generator,
                   blocks: list[np.ndarray],
                   compact: bool = False) -> tuple[int, int, list[tuple[int, int]]] | None:
    rects = [(0, 0, *blocks[0].shape)]
    for block in blocks[1:]:
        h, w = block.shape
        choices = []
        for _ in range(100):
            er, ec, eh, ew = rects[int(rng.integers(0, len(rects)))]
            gap = int(rng.integers(1, 6))
            direction = int(rng.integers(0, 4))
            if direction == 0:
                r, c = er + int(rng.integers(-h + 1, eh)), ec + ew + gap
            elif direction == 1:
                r, c = er + int(rng.integers(-h + 1, eh)), ec - w - gap
            elif direction == 2:
                r, c = er + eh + gap, ec + int(rng.integers(-w + 1, ew))
            else:
                r, c = er - h - gap, ec + int(rng.integers(-w + 1, ew))
            trial = (r, c, h, w)
            if not all(_separated(trial, other) for other in rects):
                continue
            merged = rects + [trial]
            height = max(x[0] + x[2] for x in merged) - min(x[0] for x in merged)
            width = max(x[1] + x[3] for x in merged) - min(x[1] for x in merged)
            if height <= 29 and width <= 29:
                choices.append(trial)
        if not choices:
            return None
        rects.append(choices[int(rng.integers(0, len(choices)))])
    min_r = min(x[0] for x in rects)
    min_c = min(x[1] for x in rects)
    height = max(x[0] + x[2] for x in rects) - min_r
    width = max(x[1] + x[3] for x in rects) - min_c
    if height > 30 or width > 30:
        return None
    extra_h = 30 - height
    extra_w = 30 - width
    margin_cap = 2 if compact else (4 if rng.random() < .5 else 10)
    top = int(rng.integers(0, min(margin_cap, extra_h) + 1))
    left = int(rng.integers(0, min(margin_cap, extra_w) + 1))
    bottom = int(rng.integers(0, min(margin_cap, extra_h - top) + 1))
    right = int(rng.integers(0, min(margin_cap, extra_w - left) + 1))
    return (height + top + bottom, width + left + right,
            [(r - min_r + top, c - min_c + left) for r, c, _, _ in rects])


def _background_components(grid: np.ndarray, background: int) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    components = []
    for start_r in range(height):
        for start_c in range(width):
            if seen[start_r, start_c] or grid[start_r, start_c] != background:
                continue
            seen[start_r, start_c] = True
            stack = [(start_r, start_c)]
            cells = []
            min_r = max_r = start_r
            min_c = max_c = start_c
            boundary = False
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                min_r, max_r = min(min_r, r), max(max_r, r)
                min_c, max_c = min(min_c, c), max(max_c, c)
                boundary |= r in (0, height - 1) or c in (0, width - 1)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < height and 0 <= nc < width and not seen[nr, nc]
                            and grid[nr, nc] == background):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            box_h, box_w = max_r - min_r + 1, max_c - min_c + 1
            fills = len(cells) == box_h * box_w
            components.append({"cells": cells, "touches_boundary": boundary,
                               "box_h": box_h, "box_w": box_w,
                               "fills_box": fills, "is_square": fills and box_h == box_w,
                               "bbox": (min_r, min_c, max_r, max_c)})
    return components


def _apply_rule(grid: np.ndarray, background: int, highlight: int) -> np.ndarray:
    output = grid.copy()
    for region in _background_components(grid, background):
        if not region["touches_boundary"] and region["is_square"]:
            for r, c in region["cells"]:
                output[r, c] = highlight
    return output


def _grid_signature(grid: np.ndarray) -> tuple:
    return (grid.shape, grid.tobytes())


def _matches_source_input(grid: np.ndarray) -> bool:
    if grid.shape != (12, 12):
        return False
    return tuple("".join(str(int(v)) for v in row) for row in grid) in _SOURCE_INPUT_ROWS


def _holes_for_pair(rng: np.random.Generator, regime: str,
                    slot: int, mode: int, train_side: int) -> list[dict]:
    if regime == "source_train" and slot == 0:
        holes = [_make_hole(rng, "square", 1),
                 _make_hole(rng, "square", train_side),
                 _make_hole(rng, "rectangle"),
                 _make_hole(rng, "irregular_square_box")]
        holes[0]["padded"] = True
    elif regime == "source_train" and slot == 1:
        holes = [_make_hole(rng, "rectangle"),
                 _make_hole(rng, "irregular_square_box"),
                 _make_hole(rng, "irregular_non_square_box")]
    elif regime == "target_test" and slot == 0 and mode == 0:
        holes = [_make_hole(rng, "square", 3),
                 _make_hole(rng, "rectangle"),
                 _make_hole(rng, "irregular_non_square_box")]
    else:
        count = int(rng.integers(1, 6))
        selected = int(rng.integers(0, min(3, count) + 1))
        if regime == "target_test" and slot == 0:
            if mode == 1:
                count = max(2, count)
                selected = min(count - 1, max(1, selected))
            elif mode == 2:
                selected = max(1, selected)
            else:
                selected = 0 if rng.random() < .5 else max(2, selected)
        holes = []
        for _ in range(selected):
            if regime == "target_test" and mode == 2:
                side = int(rng.integers(5, 8))
            elif regime == "target_test" and mode == 1:
                side = int(rng.choice((1, train_side)))
            elif regime == "source_train" and mode == 0:
                side = int(rng.choice((1, 2, 4, 5, 6, 7)))
            else:
                side = int(rng.integers(1, 8))
            holes.append(_make_hole(rng, "square", side))
        kinds = ("rectangle", "irregular_square_box", "irregular_non_square_box")
        for _ in range(count - selected):
            holes.append(_make_hole(rng, kinds[int(rng.integers(0, 3))]))
    return [holes[int(i)] for i in rng.permutation(len(holes))]


def _make_scene(rng: np.random.Generator, holes: list[dict],
                highlight: int, equal_outer: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
    if equal_outer:
        selected = next(h for h in holes if h["kind"] == "square")
        decoy = next(h for h in holes if h["kind"] != "square")
        remaining = [h for h in holes if h is not selected and h is not decoy]
        groups = [[selected], [decoy]] + ([remaining] if remaining else [])
    else:
        groups = _partition_holes(rng, holes)
    compact = len(holes) <= 2 and rng.random() < .6
    blocks = [_make_object(rng, group, compact) for group in groups]
    if equal_outer:
        target_h = max(blocks[0].shape[0], blocks[1].shape[0])
        target_w = max(blocks[0].shape[1], blocks[1].shape[1])
        blocks[:2] = [np.pad(block, ((0, target_h - block.shape[0]),
                                      (0, target_w - block.shape[1])),
                             constant_values=True) for block in blocks[:2]]
    if any(max(block.shape) > 29 for block in blocks):
        return None
    placement = _place_objects(rng, blocks, compact)
    if placement is None:
        return None
    height, width, positions = placement
    palette = [c for c in range(10) if c != highlight]
    background = palette[int(rng.integers(0, len(palette)))]
    wall_palette = [c for c in palette if c != background]
    if rng.random() < .45:
        color = wall_palette[int(rng.integers(0, len(wall_palette)))]
        wall_colors = [color] * len(blocks)
    else:
        wall_colors = [wall_palette[int(rng.integers(0, len(wall_palette)))]
                       for _ in blocks]
    grid = np.full((height, width), background, dtype=np.int8)
    for block, (r, c), color in zip(blocks, positions, wall_colors):
        patch = grid[r:r + block.shape[0], c:c + block.shape[1]]
        patch[block] = color
    if any(int(grid[r, c]) != background for r, c in
           ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))):
        return None
    output = _apply_rule(grid, background, highlight)
    expected_holes = len(holes)
    regions = [x for x in _background_components(grid, background)
               if not x["touches_boundary"]]
    if len(regions) != expected_holes:
        return None
    return grid, output


def _pair_features(pair: dict) -> dict:
    grid = pair["input"]
    bg = int(grid[0, 0])
    regions = [x for x in _background_components(grid, bg)
               if not x["touches_boundary"]]
    return {"regions": regions, "selected": [x for x in regions if x["is_square"]],
            "rectangle": any(x["fills_box"] and not x["is_square"] for x in regions),
            "irregular_square": any(not x["fills_box"] and x["box_h"] == x["box_w"] for x in regions),
            "irregular_nonsquare": any(not x["fills_box"] and x["box_h"] != x["box_w"] for x in regions)}


def _object_features(grid: np.ndarray, regions: list[dict]) -> dict:
    bg = int(grid[0, 0])
    height, width = grid.shape
    seen = set()
    objects = []
    owner = {}
    for r in range(height):
        for c in range(width):
            if grid[r, c] == bg or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                owner[rr, cc] = len(objects)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (0 <= nr < height and 0 <= nc < width
                            and grid[nr, nc] != bg and (nr, nc) not in seen):
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            rows, cols = [p[0] for p in cells], [p[1] for p in cells]
            objects.append((min(rows), min(cols), max(rows), max(cols)))
    cavity_owners = []
    padded = 0
    padded_one = 0
    square_in_nonsquare = 0
    nonsquare_in_square = 0
    selected_outer = []
    decoy_outer = []
    irregular_subsquare = 0
    for region in regions:
        neighbors = {owner[nr, nc] for r, c in region["cells"]
                     for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                     for nr, nc in ((r + dr, c + dc),) if (nr, nc) in owner}
        assert len(neighbors) == 1
        object_id = next(iter(neighbors))
        cavity_owners.append(object_id)
        top, left, bottom, right = objects[object_id]
        outer_h, outer_w = bottom - top + 1, right - left + 1
        if region["is_square"]:
            side = region["box_h"]
            is_padded = outer_h > side + 2 or outer_w > side + 2
            padded += is_padded
            padded_one += is_padded and side == 1
            square_in_nonsquare += outer_h != outer_w
            selected_outer.append((object_id, outer_h, outer_w))
        else:
            nonsquare_in_square += outer_h == outer_w
            decoy_outer.append((object_id, outer_h, outer_w))
            if not region["fills_box"]:
                cells = set(region["cells"])
                irregular_subsquare += any(
                    {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} <= cells
                    for r, c in cells)
    larger = smaller = equal = False
    for sid, sh, sw in selected_outer:
        for did, dh, dw in decoy_outer:
            if sid == did:
                continue
            larger |= dh * dw > sh * sw
            smaller |= dh * dw < sh * sw
            equal |= (dh, dw) == (sh, sw)
    return {"object_count": len(objects),
            "multi_cavity_object": len(set(cavity_owners)) < len(cavity_owners),
            "padded_positive": padded, "padded_one": padded_one,
            "square_in_nonsquare": square_in_nonsquare,
            "nonsquare_in_square": nonsquare_in_square,
            "irregular_subsquare": irregular_subsquare,
            "decoy_outer_larger": larger,
            "decoy_outer_smaller": smaller,
            "decoy_outer_equal": equal}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    signatures = set()
    added_train_colors = set()
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        grid, output = pair["input"], pair["output"]
        for array in (grid, output):
            assert isinstance(array, np.ndarray) and array.ndim == 2
            assert np.issubdtype(array.dtype, np.integer)
            assert 1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30
            assert np.all((array >= 0) & (array <= 9))
        assert grid.shape == output.shape
        signature = _grid_signature(grid)
        assert signature not in signatures
        signatures.add(signature)
        assert not _matches_source_input(grid)
        bg = int(grid[0, 0])
        assert all(int(grid[r, c]) == bg for r, c in
                   ((0, 0), (0, -1), (-1, 0), (-1, -1)))
        assert len(set(int(x) for x in np.unique(grid))) >= 2
        changed = grid != output
        assert np.all(grid[changed] == bg)
        assert np.all(output[~changed] == grid[~changed])
        if any(pair is item for item in train):
            added_train_colors.update(int(x) for x in np.unique(output[changed]))
    assert len(added_train_colors) == 1
    highlight = next(iter(added_train_colors))
    saw_unchanged = False
    saw_positive = False
    saw_one = False
    saw_rect = False
    saw_irregular_sq = False
    saw_irregular_ns = False
    saw_multiple = False
    saw_padded_one = False
    for pair in train + test:
        grid, output = pair["input"], pair["output"]
        bg = int(grid[0, 0])
        assert highlight not in grid
        assert np.array_equal(_apply_rule(grid, bg, highlight), output)
        features = _pair_features(pair)
        assert len(features["regions"]) >= 1
        if any(pair is item for item in train):
            object_features = _object_features(grid, features["regions"])
            saw_padded_one |= object_features["padded_one"] > 0
            saw_unchanged |= len(features["selected"]) == 0
            saw_positive |= len(features["selected"]) > 0
            saw_one |= any(x["box_h"] == 1 for x in features["selected"])
            saw_rect |= features["rectangle"]
            saw_irregular_sq |= features["irregular_square"]
            saw_irregular_ns |= features["irregular_nonsquare"]
            saw_multiple |= len(features["selected"]) >= 2
    assert all((saw_unchanged, saw_positive, saw_one, saw_rect,
                saw_irregular_sq, saw_irregular_ns, saw_multiple,
                saw_padded_one))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    params = _sample_puzzle_parameters(rng)
    mode = int(rng.integers(0, 4))
    train_side = 4 if mode == 0 else int(rng.integers(2, 7))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        regime = "source_train" if split == "train" else "target_test"
        for slot in range(count):
            for _ in range(160):
                child_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
                pair_rng = np.random.default_rng(child_seed)
                holes = _holes_for_pair(pair_rng, regime, slot, mode, train_side)
                scene = _make_scene(pair_rng, holes, params["highlight"],
                                    regime == "target_test" and slot == 0 and mode == 1)
                if scene is None:
                    continue
                grid, output = scene
                signature = _grid_signature(grid)
                if signature in seen or _matches_source_input(grid):
                    continue
                seen.add(signature)
                puzzle[split].append({"input": grid, "output": output})
                break
            else:
                raise ValueError("could not place distinct enclosed regions")
    order = [int(x) for x in rng.permutation(len(puzzle["train"]))]
    puzzle["train"] = [puzzle["train"][i] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a["input"], b["input"]) and
        np.array_equal(a["output"], b["output"])
        for a, b in zip(first[s], second[s])) for s in ("train", "test"))


def self_test() -> None:
    source_changes = (
        (1, {(2, 3)} | {(r, c) for r in range(4, 8) for c in range(7, 11)}
         | {(r, c) for r in range(9, 11) for c in range(1, 3)}),
        (3, {(r, c) for r in range(3, 5) for c in range(2, 4)} | {(10, 3)}),
        (4, {(r, c) for r in range(2, 5) for c in range(1, 4)}),
    )
    for index, expected in source_changes:
        grid = np.array([[int(value) for value in row]
                         for row in _SOURCE_INPUT_ROWS[index]], dtype=np.int8)
        changed = _apply_rule(grid, 0, 2) != grid
        assert {tuple(int(x) for x in cell) for cell in np.argwhere(changed)} == expected
    train_counts, test_counts, shapes, layouts, palettes = set(), set(), set(), set(), set()
    test_selected, square_sizes, wall_color_counts = set(), set(), set()
    coverage = {key: 0 for key in (
        "compact", "spacious", "edge", "one_object", "many_objects",
        "multi_cavity", "multi_wall", "padded", "padded_one",
        "square_in_nonsquare", "nonsquare_in_square", "subsquare",
        "equal_outer", "larger_decoy", "smaller_decoy",
        "heldout_three", "familiar_test", "larger_test",
        "first_unchanged")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        coverage["first_unchanged"] += not np.any(
            puzzle["train"][0]["input"] != puzzle["train"][0]["output"])
        highlight = next(int(x) for p in puzzle["train"] for x in np.unique(p["output"])
                         if x not in p["input"])
        backgrounds = set()
        train_sizes = {x["box_h"] for pair in puzzle["train"]
                       for x in _pair_features(pair)["selected"]}
        test_sizes = {x["box_h"] for pair in puzzle["test"]
                      for x in _pair_features(pair)["selected"]}
        coverage["heldout_three"] += 3 in test_sizes and 3 not in train_sizes and 4 in train_sizes
        coverage["familiar_test"] += bool(train_sizes & test_sizes)
        coverage["larger_test"] += bool(test_sizes) and max(test_sizes) > max(train_sizes)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                bg = int(grid[0, 0])
                backgrounds.add(bg)
                assert np.array_equal(_apply_rule(grid, bg, highlight), pair["output"])
                shapes.add(grid.shape)
                role = np.where(grid == bg, 0, 1).astype(np.int8)
                layouts.add(_grid_signature(role))
                palettes.add(tuple(sorted(int(x) for x in np.unique(grid))))
                wall_color_counts.add(len(np.unique(grid)) - 1)
                features = _pair_features(pair)
                square_sizes.update(x["box_h"] for x in features["selected"])
                objects = _object_features(grid, features["regions"])
                coverage["compact"] += max(grid.shape) <= 12
                coverage["spacious"] += max(grid.shape) >= 25
                coverage["edge"] += bool(
                    np.any(grid[0] != bg) or np.any(grid[-1] != bg)
                    or np.any(grid[:, 0] != bg) or np.any(grid[:, -1] != bg))
                coverage["one_object"] += objects["object_count"] == 1
                coverage["many_objects"] += objects["object_count"] >= 4
                coverage["multi_cavity"] += objects["multi_cavity_object"]
                coverage["multi_wall"] += len(np.unique(grid)) >= 3
                coverage["padded"] += objects["padded_positive"] > 0
                coverage["padded_one"] += objects["padded_one"] > 0
                coverage["square_in_nonsquare"] += objects["square_in_nonsquare"] > 0
                coverage["nonsquare_in_square"] += objects["nonsquare_in_square"] > 0
                coverage["subsquare"] += objects["irregular_subsquare"] > 0
                if split == "test":
                    test_selected.add(len(features["selected"]))
                    coverage["equal_outer"] += objects["decoy_outer_equal"]
                    coverage["larger_decoy"] += objects["decoy_outer_larger"]
                    coverage["smaller_decoy"] += objects["decoy_outer_smaller"]
        assert len(backgrounds) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 40 and len(layouts) >= 100 and len(palettes) >= 12
    assert len(square_sizes) >= 6 and {0, 1, 2} <= test_selected
    assert {1, 2} <= wall_color_counts
    assert coverage["compact"] >= 10 and coverage["spacious"] >= 100
    assert coverage["edge"] >= 100 and coverage["one_object"] >= 40
    assert coverage["many_objects"] >= 10 and coverage["multi_cavity"] >= 40
    assert coverage["multi_wall"] >= 20 and coverage["padded"] >= 50
    assert coverage["padded_one"] >= 20
    assert coverage["square_in_nonsquare"] >= 50
    assert coverage["nonsquare_in_square"] >= 10
    assert coverage["subsquare"] >= 20
    assert coverage["equal_outer"] >= 5
    assert coverage["larger_decoy"] >= 5 and coverage["smaller_decoy"] >= 5
    assert coverage["heldout_three"] >= 2 and coverage["familiar_test"] >= 5
    assert coverage["larger_test"] >= 2 and coverage["first_unchanged"] >= 5
    assert _puzzles_equal(generate_puzzle(777, 6, 2), generate_puzzle(777, 6, 2))
    assert not _puzzles_equal(generate_puzzle(777), generate_puzzle(778))
    for seed, ntrain, ntest in ((100, 2, 1), (101, 10, 1), (102, 10, 4),
                                (103, 5, 2), (200, None, 1), (201, None, 4),
                                (202, 2, None), (203, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(300, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
