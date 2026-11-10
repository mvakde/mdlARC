"""Generate color-keyed rectangle extraction puzzles."""

import numpy as np


_SOURCE_INPUT_CODES = {
    "13x13:0000000000000/0088888880000/0088248880000/0082244480000/0088888880000/0000000000000/0000000000000/0140000000000/0000000000000/0000000320000/0000000000000/0000000000000/0000000000000",
    "13x13:0000000000000/0000000000000/0000068000000/0000000000000/0000000002100/0000000000000/0333333304500/0355553300000/0331153300000/0381133300000/0388883300000/0333333300000/0333333300000",
    "13x13:0000000000000/0000002400000/0000000000000/0000000000000/0144100000000/0884100000000/0881100000000/0111100000000/0000000000000/0000000380000/0000000000000/0000000000000/0000000000000",
    "13x13:0000000000000/0230000000000/0000000000000/0140000000000/0000000000000/0000000000000/0000888888000/0000884888000/0000844488000/0000834388000/0000833388000/0000888888000/0000000000000",
    "13x13:0000000000000/0000000023000/0000000000000/0011311000000/0013331000000/0018881000000/0017771000000/0011711004800/0011111000000/0000000000000/0000000670000/0000000000000/0000000000000",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid_code(grid: np.ndarray) -> str:
    return f"{grid.shape[0]}x{grid.shape[1]}:" + "/".join(
        "".join(str(int(value)) for value in row) for row in grid
    )


def _count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _palette(rng: np.random.Generator, background: int, count: int) -> tuple[int, list[int], list[int]]:
    available = [color for color in range(10) if color != background]
    if background and rng.random() < 0.6:
        chosen = [0] + [int(x) for x in rng.choice([c for c in available if c], 2 * count, replace=False)]
        rng.shuffle(chosen)
    else:
        chosen = [int(x) for x in rng.choice(available, 1 + 2 * count, replace=False)]
    return chosen[0], chosen[1:1 + count], chosen[1 + count:]


def _shape(rng: np.random.Generator, regime: str, count: int) -> tuple[int, int]:
    weights = (0.28, 0.42, 0.30) if regime == "source_train" else (0.28, 0.25, 0.47)
    aspect = int(rng.choice(3, p=weights))
    if aspect == 0:
        side = int(rng.integers(5 if count == 3 else 4, 9))
        return side, side
    short = int(rng.integers(4, 8))
    long = int(rng.integers(max(short + 1, 5 if count == 3 else 4), min(10, short + 4)))
    return (short, long) if aspect == 1 else (long, short)


def _grow_patch(rng: np.random.Generator, free: np.ndarray, size: int, edge: bool) -> list[tuple[int, int]] | None:
    starts = list(zip(*np.where(free)))
    if edge:
        boundary = [(r, c) for r, c in starts if r in (0, free.shape[0] - 1) or c in (0, free.shape[1] - 1)]
        if boundary:
            starts = boundary
    if not starts:
        return None
    start = starts[int(rng.integers(len(starts)))]
    patch = [start]
    used = {start}
    while len(patch) < size:
        frontier = set()
        for row, col in patch:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < free.shape[0] and 0 <= nc < free.shape[1] and free[nr, nc] and (nr, nc) not in used:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        options = sorted(frontier)
        distances = np.array([abs(r - start[0]) + abs(c - start[1]) for r, c in options])
        weights = 1.0 / (1.0 + distances)
        weights /= weights.sum()
        next_cell = options[int(rng.choice(len(options), p=weights))]
        used.add(next_cell)
        patch.append(next_cell)
    return patch


def _patch_panel(rng: np.random.Generator, height: int, width: int, carrier: int, sources: list[int], framed: bool) -> np.ndarray | None:
    allowed = np.ones((height, width), dtype=bool)
    if framed:
        allowed[0, :] = allowed[-1, :] = False
        allowed[:, 0] = allowed[:, -1] = False
    capacity = int(allowed.sum())
    if capacity < 3 * len(sources):
        return None
    extra_limit = max(0, min(capacity - 3 * len(sources), 4 * len(sources),
                             int(capacity * 0.58) - 3 * len(sources)))
    for _ in range(90):
        panel = np.full((height, width), carrier, dtype=np.int8)
        free = allowed.copy()
        sizes = [3] * len(sources)
        extra = int(rng.integers(0, extra_limit + 1))
        for _ in range(extra):
            available = [index for index, size in enumerate(sizes) if size < 7]
            sizes[available[int(rng.integers(len(available)))]] += 1
        order = rng.permutation(len(sources))
        success = True
        for index in order:
            index = int(index)
            patch = _grow_patch(rng, free, sizes[index], bool(not framed and rng.random() < 0.40))
            if patch is None:
                success = False
                break
            for row, col in patch:
                panel[row, col] = sources[index]
                free[row, col] = False
        if success and np.any(panel == carrier):
            return panel
    return None


def _make_panel(rng: np.random.Generator, height: int, width: int, carrier: int, sources: list[int]) -> np.ndarray:
    mode = int(rng.choice(5, p=(0.58, 0.18, 0.09, 0.08, 0.07)))
    if mode in (0, 1):
        panel = _patch_panel(rng, height, width, carrier, sources, mode == 1)
        if panel is not None:
            return panel
    if mode in (2, 3):
        length = height if mode == 2 else width
        labels = np.array([carrier] + sources + [carrier] * (length - 1 - len(sources)), dtype=np.int8)
        rng.shuffle(labels)
        return np.repeat(labels[:, None], width, axis=1) if mode == 2 else np.repeat(labels[None, :], height, axis=0)
    if mode == 4:
        colors = np.array([carrier] + sources, dtype=np.int8)
        panel = rng.choice(colors, size=(height, width), p=[0.48] + [0.52 / len(sources)] * len(sources)).astype(np.int8)
        spots = rng.choice(panel.size, len(colors), replace=False)
        for spot, color in zip(spots, colors):
            panel.flat[int(spot)] = color
        return panel
    panel = _patch_panel(rng, height, width, carrier, sources, False)
    if panel is None:
        raise RuntimeError("could not make a connected panel pattern")
    return panel


def _can_place_key(occupied: np.ndarray, row: int, col: int) -> bool:
    height, width = occupied.shape
    if not (0 <= row < height and 0 <= col < width - 1):
        return False
    if (row, col) in ((0, 0), (height - 1, 0)) or (row, col + 1) in ((0, width - 1), (height - 1, width - 1)):
        return False
    return not bool(np.any(occupied[max(0, row - 1):min(height, row + 2), max(0, col - 1):min(width, col + 3)]))


def _candidates(occupied: np.ndarray) -> list[tuple[int, int]]:
    return [(r, c) for r in range(occupied.shape[0]) for c in range(occupied.shape[1] - 1) if _can_place_key(occupied, r, c)]


def _side(key: tuple[int, int], box: tuple[int, int, int, int]) -> str:
    row, col = key
    r0, r1, c0, c1 = box
    if row < r0 - 1:
        return "above"
    if row > r1 + 1:
        return "below"
    if col + 1 < c0 - 1:
        return "left"
    if col > c1 + 1:
        return "right"
    return "near"


def _choose_keys(rng: np.random.Generator, occupied: np.ndarray, box: tuple[int, int, int, int], count: int, style: str) -> list[tuple[int, int]] | None:
    keys: list[tuple[int, int]] = []
    target_side = None
    if style == "same_side":
        first = _candidates(occupied)
        sides = [side for side in ("above", "below", "left", "right") if any(_side(key, box) == side for key in first)]
        if not sides:
            return None
        target_side = sides[int(rng.integers(len(sides)))]
    for index in range(count):
        choices = _candidates(occupied)
        if target_side is not None:
            choices = [key for key in choices if _side(key, box) == target_side]
        if index and style == "aligned_row":
            choices = [key for key in choices if key[0] == keys[0][0]]
        if index and style == "aligned_column":
            choices = [key for key in choices if key[1] == keys[0][1]]
        if not choices:
            return None
        if index and style in ("compact", "same_side", "aligned_row", "aligned_column"):
            distances = [min(abs(r - kr) + abs(c - kc) for kr, kc in keys) for r, c in choices]
            threshold = min(distances) + (1 if style == "compact" else 2)
            choices = [key for key, distance in zip(choices, distances) if distance <= threshold]
        elif index and style == "scattered":
            distances = [min(abs(r - kr) + abs(c - kc) for kr, kc in keys) for r, c in choices]
            threshold = max(distances) - 2
            choices = [key for key, distance in zip(choices, distances) if distance >= threshold]
        key = choices[int(rng.integers(len(choices)))]
        keys.append(key)
        occupied[key[0], key[1]:key[1] + 2] = True
    return keys


def _place_layout(rng: np.random.Generator, canvas: tuple[int, int], panel_shape: tuple[int, int], count: int) -> tuple[int, int, list[tuple[int, int]]]:
    height, width = canvas
    ph, pw = panel_shape
    positions = [(r, c) for r in range(height - ph + 1) for c in range(width - pw + 1)
                 if not ((r == 0 or r + ph == height) and (c == 0 or c + pw == width))]
    styles = ("compact", "same_side", "aligned_row", "aligned_column", "scattered", "random")
    style = str(rng.choice(styles, p=(0.26, 0.21, 0.13, 0.12, 0.13, 0.15)))
    for current_style in (style, "random"):
        for index in rng.permutation(len(positions)):
            row, col = positions[int(index)]
            occupied = np.zeros(canvas, dtype=bool)
            occupied[row:row + ph, col:col + pw] = True
            box = (row, row + ph - 1, col, col + pw - 1)
            keys = _choose_keys(rng, occupied, box, count, current_style)
            if keys is not None:
                return row, col, keys
    raise RuntimeError("could not place detached legends")


def _make_pair(rng: np.random.Generator, regime: str, count: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = _shape(rng, regime, count)
    background = 0 if rng.random() < 0.70 else int(rng.integers(1, 10))
    carrier, sources, replacements = _palette(rng, background, count)
    panel = _make_panel(rng, height, width, carrier, sources)
    for attempt in range(12):
        if attempt == 0 and rng.random() < 0.34:
            canvas = (13, 13)
        else:
            canvas = (int(rng.integers(max(9, height + 2), 19)), int(rng.integers(max(9, width + 2), 19)))
        if canvas[0] < height or canvas[1] < width:
            continue
        try:
            row, col, keys = _place_layout(rng, canvas, panel.shape, count)
            break
        except RuntimeError:
            continue
    else:
        canvas = (19, 19)
        row, col, keys = _place_layout(rng, canvas, panel.shape, count)
    grid = np.full(canvas, background, dtype=np.int8)
    grid[row:row + height, col:col + width] = panel
    for (kr, kc), source, replacement in zip(keys, sources, replacements):
        grid[kr, kc] = replacement
        grid[kr, kc + 1] = source
    answer = panel.copy()
    for source, replacement in zip(sources, replacements):
        answer[panel == source] = replacement
    return grid, answer


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    puzzle = {"train": [], "test": []}
    used = set()
    # Replacement-left, source-right is the only puzzle-wide learned convention.
    # The first two examples expose both key counts; all geometry and color roles vary per pair.
    for split, counts in (("train", [2, 3] + [int(rng.integers(2, 4)) for _ in range(num_train - 2)]),
                          ("test", [int(rng.integers(2, 4)) for _ in range(num_test)])):
        regime = "source_train" if split == "train" else "target_test"
        for count in counts:
            for _ in range(60):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                grid, answer = _make_pair(pair_rng, regime, count)
                code = _grid_code(grid)
                if code not in used and code not in _SOURCE_INPUT_CODES:
                    used.add(code)
                    puzzle[split].append({"input": grid, "output": answer})
                    break
            else:
                raise RuntimeError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for row, col in zip(*np.where(mask)):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        seen[row, col] = True
        stack = [(row, col)]
        component = []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
        result.append(component)
    return result


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    unseen = set(zip(*np.where(mask)))
    result = []
    while unseen:
        start = unseen.pop()
        piece = [start]
        stack = [start]
        while stack:
            row, col = stack.pop()
            for next_cell in ((row - 1, col), (row + 1, col),
                              (row, col - 1), (row, col + 1)):
                if next_cell in unseen:
                    unseen.remove(next_cell)
                    piece.append(next_cell)
                    stack.append(next_cell)
        result.append(piece)
    return result


def _analyze_pair(grid: np.ndarray, answer: np.ndarray) -> dict:
    corners = (int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1]))
    _require(len(set(corners)) == 1, "canvas corners must expose one background")
    background = corners[0]
    components = _components(grid != background)
    large = [component for component in components if len(component) >= 16]
    _require(len(large) == 1, "there must be one uniquely large panel")
    component = large[0]
    rows, cols = zip(*component)
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    panel = grid[r0:r1 + 1, c0:c1 + 1]
    _require(len(component) == panel.size and np.all(panel != background), "panel must be a full rectangle")
    _require(4 <= panel.shape[0] <= 9 and 4 <= panel.shape[1] <= 9, "panel dimensions outside supported range")
    keys = [piece for piece in components if piece is not component]
    _require(len(keys) in (2, 3), "must have two or three detached keys")
    panel_colors = set(int(x) for x in np.unique(panel))
    mapping = {}
    positions = []
    replacements = set()
    for key in keys:
        _require(len(key) == 2, "each legend must contain two cells")
        (kr, kc), (rr, rc) = sorted(key)
        _require(kr == rr and rc == kc + 1, "keys must be horizontal")
        replacement, source = int(grid[kr, kc]), int(grid[rr, rc])
        _require(source in panel_colors and replacement not in panel_colors, "legend roles must be visible")
        _require(source not in mapping and replacement not in replacements, "key colors must be unique")
        mapping[source] = replacement
        replacements.add(replacement)
        positions.append((kr, kc))
    carrier = panel_colors - set(mapping)
    _require(len(carrier) == 1, "panel must have one unkeyed carrier")
    expected = panel.copy()
    for source, replacement in mapping.items():
        expected[panel == source] = replacement
    _require(answer.shape == panel.shape and np.array_equal(answer, expected), "answer must be exactly the recolored crop")
    _require(set(int(x) for x in np.unique(grid)) == panel_colors | replacements | {background}, "unrelated input colors")
    _require(set(int(x) for x in np.unique(answer)) == carrier | replacements, "wrong output palette")
    framed = bool(np.all(panel[0] == next(iter(carrier))) and np.all(panel[-1] == next(iter(carrier))) and
                  np.all(panel[:, 0] == next(iter(carrier))) and np.all(panel[:, -1] == next(iter(carrier))))
    connected = sum(len(_components4(panel == source)) == 1 for source in mapping)
    distances = [abs(a[0] - b[0]) + abs(a[1] - b[1]) for i, a in enumerate(positions) for b in positions[i + 1:]]
    return {"background": background, "panel": panel, "shape": panel.shape, "box": (r0, r1, c0, c1),
            "count": len(keys), "framed": framed, "connected": connected, "positions": positions,
            "min_key_distance": min(distances), "mapping": mapping}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal pair counts")
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grids must be 2D arrays")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "grid size outside ARC bounds")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "grid color outside ARC palette")
            code = _grid_code(pair["input"])
            _require(code not in seen and code not in _SOURCE_INPUT_CODES, "duplicate or source input")
            seen.add(code)
            analyses[split].append(_analyze_pair(pair["input"], pair["output"]))
    _require({item["count"] for item in analyses["train"]} == {2, 3}, "training must show both key counts")
    # All keys are decoded from the observed panel colors, so no hidden color map
    # or geometric convention is needed to solve a test example.
    for item in analyses["test"]:
        _require(item["count"] in {2, 3}, "test uses an unseen key count")


def _same(a: dict, b: dict) -> bool:
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
        for x, y in zip(a[split], b[split])) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts = set(), set()
    canvas_shapes, panel_shapes, layouts = set(), set(), set()
    aspect_by_split = {"train": set(), "test": set()}
    count_by_split = {"train": set(), "test": set()}
    backgrounds = set()
    nonzero_background = zero_foreground = compact = framed_count = connected = total_sources = 0
    aligned = same_side = wide_legend = edge_panel = small_canvas = large_canvas = 0
    source_size_ge_three = source_touches_edge = stripes = fragmented_mosaics = 0
    mixed_frames = mixed_canvas = mixed_aspect = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_frames, local_canvases, local_aspects = set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                item = _analyze_pair(pair["input"], pair["output"])
                h, w = item["shape"]
                aspect = (h > w) - (h < w)
                aspect_by_split[split].add(aspect)
                local_aspects.add(aspect)
                count_by_split[split].add(item["count"])
                canvas_shapes.add(pair["input"].shape)
                panel_shapes.add(item["shape"])
                layouts.add((pair["input"].shape, item["box"], tuple(sorted(item["positions"]))))
                backgrounds.add(item["background"])
                local_frames.add(item["framed"])
                local_canvases.add(pair["input"].shape)
                nonzero_background += item["background"] != 0
                zero_foreground += item["background"] != 0 and bool(np.any(pair["input"] == 0))
                compact += item["min_key_distance"] <= 4
                wide_legend += item["min_key_distance"] >= 7
                aligned += (len({r for r, c in item["positions"]}) == 1 or
                            len({c for r, c in item["positions"]}) == 1)
                same_side += len({_side(key, item["box"]) for key in item["positions"]}) == 1
                r0, r1, c0, c1 = item["box"]
                ch, cw = pair["input"].shape
                edge_panel += r0 == 0 or c0 == 0 or r1 == ch - 1 or c1 == cw - 1
                small_canvas += ch < 13 or cw < 13
                large_canvas += ch > 13 or cw > 13
                framed_count += item["framed"]
                connected += item["connected"]
                total_sources += item["count"]
                source_size_ge_three += sum(int(np.sum(item["panel"] == source)) >= 3
                                            for source in item["mapping"])
                source_touches_edge += sum(bool(np.any(item["panel"][0] == source) or
                                                np.any(item["panel"][-1] == source) or
                                                np.any(item["panel"][:, 0] == source) or
                                                np.any(item["panel"][:, -1] == source))
                                           for source in item["mapping"])
                row_stripe = all(np.all(row == row[0]) for row in item["panel"])
                col_stripe = all(np.all(column == column[0]) for column in item["panel"].T)
                stripes += row_stripe or col_stripe
                fragmented_mosaics += not (row_stripe or col_stripe) and item["connected"] < item["count"]
        mixed_frames += len(local_frames) > 1
        mixed_canvas += len(local_canvases) > 1
        mixed_aspect += len(local_aspects) > 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack diversity")
    _require(len(canvas_shapes) >= 15 and len(panel_shapes) >= 15 and len(layouts) >= 150, "geometry lacks diversity")
    _require(aspect_by_split["train"] == {-1, 0, 1} and aspect_by_split["test"] == {-1, 0, 1}, "aspects lack range")
    _require(count_by_split["train"] == count_by_split["test"] == {2, 3}, "key counts lack range")
    _require(len(backgrounds) >= 5 and nonzero_background >= 20 and zero_foreground >= 10, "palette roles lack range")
    _require(compact >= 20 and wide_legend >= 10 and aligned >= 10 and same_side >= 10,
             "legend arrangements lack coverage")
    _require(edge_panel >= 10 and small_canvas >= 10 and large_canvas >= 10,
             "panel positions or canvas sizes lack range")
    _require(framed_count >= 10 and connected / total_sources >= 0.75 and
             source_size_ge_three / total_sources >= 0.90, "panel regions lack coverage")
    _require(source_touches_edge >= 10 and stripes >= 10 and fragmented_mosaics >= 3,
             "panel pattern modes lack coverage")
    _require(mixed_frames >= 8 and mixed_canvas >= 15 and mixed_aspect >= 15,
             "within-puzzle variation is too narrow")
    _require(_same(generate_puzzle(8675309, 7, 3), generate_puzzle(8675309, 7, 3)), "explicit determinism")
    _require(_same(generate_puzzle(314159), generate_puzzle(314159)), "automatic determinism")
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(10000 + 100 * train + test, train, test)
        _require(len(puzzle["train"]) == train and len(puzzle["test"]) == test, "explicit counts ignored")
    for index, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(20000 + index, train, test)
        _require(train is None or len(puzzle["train"]) == train, "partial train count ignored")
        _require(test is None or len(puzzle["test"]) == test, "partial test count ignored")
    for index, args in enumerate(({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                                  {"num_test": 5}, {"num_train": 2, "num_test": 2},
                                  {"num_train": 3, "num_test": 4})):
        try:
            generate_puzzle(30000 + index, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
