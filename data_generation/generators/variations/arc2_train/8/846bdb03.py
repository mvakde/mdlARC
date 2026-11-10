import numpy as np


# The five original inputs are retained as exact-content guards. Matching an
# output shape or a palette is fine; copying an original pair is not.
_SOURCE_INPUT_SIGNATURES = frozenset(
    "".join(block.split())
    for block in (
        """
        0000000000000
        0040000400000
        0020000100000
        0020000100000
        0040000400000
        0000000000000
        0000000000000
        0000000000000
        0000000000000
        0001120000000
        0000122000000
        0000000000000
        0000000000000
        """,
        """
        0000000000000
        0000030808000
        0000333888000
        0000030808000
        0000033888000
        0000000808000
        0400000040000
        0800000030000
        0800000030000
        0800000030000
        0800000030000
        0800000030000
        0400000040000
        """,
        """
        0000000000000
        0000040000400
        0000070000300
        0000070000300
        0000070000300
        0000040000400
        0000000000000
        0000000000000
        0000077030000
        0000077330000
        0000007030000
        0000000000000
        """,
        """
        0000000000000
        0000000000000
        0000000000000
        0022010000000
        0002011100000
        0002210000000
        0000200000000
        0000040000004
        0000020000001
        0000020000001
        0000020000001
        0000020000001
        0000040000004
        """,
        """
        0000000000000
        0400000040000
        0200000080000
        0200000080000
        0200000080000
        0200000080000
        0400000040000
        0000000000000
        0000000000000
        0808200000000
        0888222000000
        0800020000000
        0800022000000
        """,
    )
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _array_key(array: np.ndarray) -> tuple:
    return tuple(array.shape), tuple(int(v) for v in array.ravel())


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        (split, tuple((_array_key(p["input"]), _array_key(p["output"]))
                      for p in puzzle[split]))
        for split in ("train", "test")
    )


def _empty_socket(height: int, width: int, background: int, marker: int,
                  left: int, right: int) -> np.ndarray:
    socket = np.full((height, width), background, dtype=np.int8)
    socket[0, 0] = socket[0, -1] = marker
    socket[-1, 0] = socket[-1, -1] = marker
    socket[1:-1, 0] = left
    socket[1:-1, -1] = right
    return socket


def _neighbors(y: int, x: int, height: int, width: int):
    if y:
        yield y - 1, x
    if y + 1 < height:
        yield y + 1, x
    if x:
        yield y, x - 1
    if x + 1 < width:
        yield y, x + 1


def _make_oriented_payload(rng: np.random.Generator, height: int, width: int,
                           background: int, left: int, right: int) -> np.ndarray:
    # Four connected skeletons yield stepped paths, bands, solid blocks, and
    # irregular ribbons. Frontier growth varies thickness and leaves holes.
    occupied = np.zeros((height, width), dtype=bool)
    skeleton = int(rng.integers(4))
    if skeleton == 0:
        start = 0 if int(rng.integers(2)) else height - 1
        end = height - 1 - start
        row = start
        for column in range(width):
            new_row = int(round(start + (end - start) *
                                column / max(1, width - 1)))
            occupied[min(row, new_row):max(row, new_row) + 1, column] = True
            row = new_row
    elif skeleton == 1:
        band_start = int(rng.integers(height))
        band_end = min(height, band_start + int(rng.integers(1, 4)))
        occupied[band_start:band_end, :] = True
    elif skeleton == 2:
        y0 = int(rng.integers(height))
        y1 = int(rng.integers(y0 + 1, height + 1))
        x0 = int(rng.integers(max(1, width // 2)))
        x1 = int(rng.integers(max(x0 + 1, width // 2), width + 1))
        occupied[y0:y1, x0:x1] = True
        bridge_row = int(rng.integers(y0, y1))
        occupied[bridge_row, :x0 + 1] = True
        occupied[bridge_row, x1 - 1:] = True
    else:
        row = int(rng.integers(height))
        for column in range(width):
            new_row = int(np.clip(row + int(rng.integers(-2, 3)),
                                  0, height - 1))
            occupied[min(row, new_row):max(row, new_row) + 1, column] = True
            row = new_row

    for target_row in (0, height - 1):
        column = int(rng.integers(width))
        existing = np.flatnonzero(occupied[:, column])
        anchor = int(existing[0] if target_row == 0 else existing[-1])
        occupied[min(anchor, target_row):max(anchor, target_row) + 1, column] = True

    mode = int(rng.choice(3, p=[0.18, 0.55, 0.27]))
    ranges = ((0.23, 0.39), (0.40, 0.64), (0.65, 0.84))
    low, high = ranges[mode]
    target = max(int(np.count_nonzero(occupied)),
                 int(round(float(rng.uniform(low, high)) * height * width)))
    frontier = set()
    for y, x in np.argwhere(occupied):
        for cell in _neighbors(int(y), int(x), height, width):
            if not occupied[cell]:
                frontier.add(cell)
    while np.count_nonzero(occupied) < target and frontier:
        choices = sorted(frontier)
        cell = choices[int(rng.integers(len(choices)))]
        frontier.remove(cell)
        occupied[cell] = True
        for neighbor in _neighbors(cell[0], cell[1], height, width):
            if not occupied[neighbor]:
                frontier.add(neighbor)

    payload = np.full((height, width), background, dtype=np.int8)
    split = int(rng.integers(1, width))
    drift = 0
    for y in range(height):
        drift = int(np.clip(drift + int(rng.integers(-1, 2)), -2, 2))
        boundary = int(np.clip(split + drift, 1, width - 1))
        for x in range(width):
            if occupied[y, x]:
                payload[y, x] = left if x < boundary else right
    return payload


def _sample_output_size(rng: np.random.Generator) -> tuple[int, int]:
    # Familiar source-sized examples coexist with tall, narrow and larger
    # sockets. Height and width are independent within the broader branch.
    if float(rng.random()) < 0.42:
        return int(rng.integers(4, 9)), int(rng.integers(5, 11))
    return int(rng.integers(4, 14)), int(rng.integers(5, 16))


def _generate_pair(rng: np.random.Generator, background: int, marker: int,
                   left: int, right: int, flip: bool, layout: str,
                   size: tuple[int, int], flush_bottom: bool = False) -> dict:
    height, width = size
    inner_height, inner_width = height - 2, width - 2
    oriented = _make_oriented_payload(
        rng, inner_height, inner_width, background, left, right
    )
    visible = np.fliplr(oriented).copy() if flip else oriented.copy()
    frame = _empty_socket(height, width, background, marker, left, right)

    if layout in ("frame_above", "payload_above"):
        gap = int(rng.integers(1, min(4, 30 - (height + inner_height)) + 1))
        base_height = height + inner_height + gap
        spare = min(9, 30 - base_height)
        top_margin = int(rng.integers(0, min(5, spare) + 1))
        bottom_margin = (0 if flush_bottom else
                         int(rng.integers(0, min(5, spare - top_margin) + 1)))
        canvas_height = base_height + top_margin + bottom_margin
        canvas_width = width + int(rng.integers(0, min(9, 30 - width) + 1))
        if layout == "frame_above":
            frame_top = top_margin
            payload_top = frame_top + height + gap
        else:
            payload_top = top_margin
            frame_top = payload_top + inner_height + gap
        frame_left = int(rng.integers(0, canvas_width - width + 1))
        payload_left = int(rng.integers(0, canvas_width - inner_width + 1))
    else:
        _require(layout in ("frame_left", "payload_left"), "unknown layout")
        maximum_gap = min(4, 30 - (width + inner_width))
        _require(maximum_gap >= 1, "side layout cannot fit on the canvas")
        gap = int(rng.integers(1, maximum_gap + 1))
        base_width = width + inner_width + gap
        spare = min(8, 30 - base_width)
        left_margin = int(rng.integers(0, min(4, spare) + 1))
        right_margin = int(rng.integers(0, min(4, spare - left_margin) + 1))
        canvas_width = base_width + left_margin + right_margin
        canvas_height = height + int(rng.integers(0, min(8, 30 - height) + 1))
        frame_top = int(rng.integers(0, canvas_height - height + 1))
        payload_top = int(rng.integers(0, canvas_height - inner_height + 1))
        if layout == "frame_left":
            frame_left = left_margin
            payload_left = frame_left + width + gap
        else:
            payload_left = left_margin
            frame_left = payload_left + inner_width + gap

    input_grid = np.full((canvas_height, canvas_width), background, dtype=np.int8)
    input_grid[frame_top:frame_top + height, frame_left:frame_left + width] = frame
    input_grid[payload_top:payload_top + inner_height,
               payload_left:payload_left + inner_width] = visible
    output_grid = frame.copy()
    output_grid[1:-1, 1:-1] = oriented
    return {"input": input_grid, "output": output_grid}


def _normalize_count(value: int | None, name: str,
                     minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(rng: np.random.Generator, seed: int,
                    num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _normalize_count(num_train, "num_train", 2, 10)
    test = _normalize_count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _training_plans(rng: np.random.Generator, count: int) -> list[tuple[str, bool]]:
    first_order = ("frame_above" if int(rng.integers(2)) else "payload_above")
    other_order = ("payload_above" if first_order == "frame_above" else
                   "frame_above")
    if count == 2:
        plans = [(first_order, False), (first_order, True)]
    elif count == 3:
        plans = [(first_order, False), (first_order, True),
                 (other_order, bool(rng.integers(2)))]
    else:
        plans = [(order, flip) for order in ("frame_above", "payload_above")
                 for flip in (False, True)]
        for _ in range(count - 4):
            if float(rng.random()) < 0.43:
                layout = ("frame_left" if int(rng.integers(2)) else "payload_left")
            else:
                layout = ("frame_above" if int(rng.integers(2)) else
                          "payload_above")
            plans.append((layout, bool(rng.integers(2))))
    rng.shuffle(plans)
    return plans


def _training_color_pairs(rng: np.random.Generator,
                          palette: list[int], count: int) -> list[tuple[int, int]]:
    pairs = [(palette[0], palette[1])]
    for index in range(2, len(palette)):
        pairs.append((palette[index], palette[int(rng.integers(index))]))
    while len(pairs) < count:
        choice = rng.choice(len(palette), size=2, replace=False)
        pairs.append((palette[int(choice[0])], palette[int(choice[1])]))
    rng.shuffle(pairs)
    return [(right, left) if int(rng.integers(2)) else (left, right)
            for left, right in pairs]


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)

    # No literal color has a fixed role across the source pairs. The only
    # puzzle-wide convention is the visible rail-to-edge matching rule.
    colors = [int(v) for v in rng.permutation(10)]
    palette_size = int(rng.integers(2, min(6, train_count + 1) + 1))
    rail_palette = colors[:palette_size]
    other_colors = colors[palette_size:]
    base_background, base_marker = other_colors[:2]
    train_color_pairs = _training_color_pairs(rng, rail_palette, train_count)
    train_plans = _training_plans(rng, train_count)

    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    train_sizes = []
    maximum_seed = np.iinfo(np.int64).max

    def choose_background_marker() -> tuple[int, int]:
        if float(rng.random()) < 0.52:
            return base_background, base_marker
        chosen = rng.choice(other_colors, size=2, replace=False)
        return int(chosen[0]), int(chosen[1])

    def create_pair(colors_for_pair: tuple[int, int], layout: str,
                    flip: bool, size: tuple[int, int],
                    flush_bottom: bool = False) -> dict:
        for _ in range(32):
            background, marker = choose_background_marker()
            pair_rng = np.random.default_rng(
                int(rng.integers(0, maximum_seed, dtype=np.int64))
            )
            pair = _generate_pair(pair_rng, background, marker,
                                  colors_for_pair[0], colors_for_pair[1],
                                  flip, layout, size, flush_bottom)
            key = _array_key(pair["input"])
            if (key not in seen_inputs and len(_input_candidates(pair["input"])) == 1
                and (
                pair["input"].shape != (13, 13)
                or "".join(str(int(v)) for v in pair["input"].ravel())
                   not in _SOURCE_INPUT_SIGNATURES
            )):
                seen_inputs.add(key)
                return pair
        raise RuntimeError("could not construct a unique pair")

    for color_pair, (layout, flip) in zip(train_color_pairs, train_plans):
        size = _sample_output_size(rng)
        if layout in ("frame_left", "payload_left") and size[1] > 14:
            size = (size[0], int(rng.integers(5, 15)))
        pair = create_pair(color_pair, layout, flip, size)
        puzzle["train"].append(pair)
        train_sizes.append(size)

    for index in range(test_count):
        if float(rng.random()) < 0.46:
            size = train_sizes[int(rng.integers(train_count))]
        else:
            size = _sample_output_size(rng)
            for _ in range(12):
                if size not in train_sizes:
                    break
                size = _sample_output_size(rng)
            if size in train_sizes:
                size = next((candidate for candidate in
                             ((h, w) for h in range(4, 14)
                              for w in range(5, 16))
                             if candidate not in train_sizes))

        if float(rng.random()) < 0.24 and size[1] <= 14:
            layout = ("frame_left" if int(rng.integers(2)) else "payload_left")
        else:
            layout = ("frame_above" if int(rng.integers(2)) else
                      "payload_above")
        flip = bool(rng.integers(2))
        if test_count >= 2 and index == 1:
            # Both decisions appear within a multi-test puzzle, while layout
            # remains independently sampled.
            first_info = _analyze_pair(
                puzzle["test"][0]["input"], puzzle["test"][0]["output"]
            )
            flip = not first_info["needs_flip"]
        if float(rng.random()) < 0.45:
            left, right = train_color_pairs[int(rng.integers(train_count))]
            color_pair = (right, left)
        else:
            selected = rng.choice(rail_palette, size=2, replace=False)
            color_pair = (int(selected[0]), int(selected[1]))
        flush = layout == "frame_above" and flip and float(rng.random()) < 0.32
        puzzle["test"].append(create_pair(color_pair, layout, flip, size, flush))

    validate_generated_puzzle(puzzle)
    return puzzle


def _component_count(mask: np.ndarray) -> int:
    remaining = {tuple(map(int, cell)) for cell in np.argwhere(mask)}
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            y, x = stack.pop()
            for neighbor in _neighbors(y, x, mask.shape[0], mask.shape[1]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def _input_candidates(grid: np.ndarray) -> list[np.ndarray]:
    """Infer all sockets and completions using only one input grid."""
    candidates = []
    for marker in np.unique(grid):
        points = np.argwhere(grid == marker)
        if points.shape != (4, 2):
            continue
        rows = sorted({int(v) for v in points[:, 0]})
        columns = sorted({int(v) for v in points[:, 1]})
        if len(rows) != 2 or len(columns) != 2:
            continue
        top, bottom = rows
        left, right = columns
        if {tuple(map(int, cell)) for cell in points} != {
            (top, left), (top, right), (bottom, left), (bottom, right)
        }:
            continue
        height, width = bottom - top + 1, right - left + 1
        if not (4 <= height <= 13 and 5 <= width <= 15):
            continue
        background = int(grid[top, left + 1])
        rail_left = int(grid[top + 1, left])
        rail_right = int(grid[top + 1, right])
        if len({int(marker), background, rail_left, rail_right}) != 4:
            continue
        frame = _empty_socket(height, width, background, int(marker),
                              rail_left, rail_right)
        if not np.array_equal(grid[top:bottom + 1, left:right + 1], frame):
            continue
        outside = grid != background
        outside[top:bottom + 1, left:right + 1] = False
        cells = np.argwhere(outside)
        if len(cells) == 0:
            continue
        payload_top, payload_left = (int(v) for v in np.min(cells, axis=0))
        payload_bottom, payload_right = (int(v) for v in np.max(cells, axis=0))
        payload = grid[payload_top:payload_bottom + 1,
                       payload_left:payload_right + 1]
        if payload.shape != (height - 2, width - 2):
            continue
        if not (payload_bottom + 1 < top or bottom + 1 < payload_top or
                payload_right + 1 < left or right + 1 < payload_left):
            continue
        if not set(map(int, np.unique(payload))) <= {
            background, rail_left, rail_right
        }:
            continue
        left_key = {int(v) for v in payload[:, 0] if int(v) != background}
        right_key = {int(v) for v in payload[:, -1] if int(v) != background}
        if left_key == {rail_left} and right_key == {rail_right}:
            oriented = payload
        elif left_key == {rail_right} and right_key == {rail_left}:
            oriented = np.fliplr(payload)
        else:
            continue
        answer = frame.copy()
        answer[1:-1, 1:-1] = oriented
        candidates.append(answer)
    return candidates


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    height, width = output_grid.shape
    _require(4 <= height <= 13 and 5 <= width <= 15, "socket size is invalid")
    marker = int(output_grid[0, 0])
    background = int(output_grid[0, 1])
    left = int(output_grid[1, 0])
    right = int(output_grid[1, -1])
    _require(len({background, marker, left, right}) == 4,
             "semantic colors must be distinct")
    expected_frame = _empty_socket(height, width, background, marker, left, right)
    _require(np.all(output_grid[0] == expected_frame[0]) and
             np.all(output_grid[-1] == expected_frame[-1]) and
             np.all(output_grid[1:-1, 0] == left) and
             np.all(output_grid[1:-1, -1] == right),
             "output border does not preserve the socket")
    _require(np.count_nonzero(output_grid == marker) == 4,
             "marker occurs beyond output corners")

    markers = np.argwhere(input_grid == marker)
    _require(markers.shape == (4, 2), "input needs exactly four markers")
    rows = sorted({int(v) for v in markers[:, 0]})
    columns = sorted({int(v) for v in markers[:, 1]})
    _require(len(rows) == len(columns) == 2, "markers are not rectangle corners")
    top, bottom = rows
    frame_left, frame_right = columns
    _require({tuple(map(int, cell)) for cell in markers} ==
             {(top, frame_left), (top, frame_right),
              (bottom, frame_left), (bottom, frame_right)},
             "markers do not form all four corners")
    _require((bottom - top + 1, frame_right - frame_left + 1) == (height, width),
             "socket and output sizes disagree")
    _require(np.array_equal(input_grid[top:bottom + 1,
                                       frame_left:frame_right + 1], expected_frame),
             "input socket is not empty")

    outside = input_grid != background
    outside[top:bottom + 1, frame_left:frame_right + 1] = False
    cells = np.argwhere(outside)
    _require(len(cells) > 0, "separate payload is missing")
    payload_top, payload_left = (int(v) for v in np.min(cells, axis=0))
    payload_bottom, payload_right = (int(v) for v in np.max(cells, axis=0))
    payload = input_grid[payload_top:payload_bottom + 1,
                         payload_left:payload_right + 1]
    _require(payload.shape == (height - 2, width - 2),
             "payload does not match socket interior")
    _require(np.any(payload[0] != background) and
             np.any(payload[-1] != background) and
             np.any(payload[:, 0] != background) and
             np.any(payload[:, -1] != background),
             "payload bounding box is not tight")
    _require(set(map(int, np.unique(input_grid))) ==
             {background, marker, left, right}, "input color roles are invalid")
    _require(set(map(int, np.unique(output_grid))) ==
             {background, marker, left, right}, "output color roles are invalid")
    _require(set(map(int, np.unique(payload))) <= {background, left, right},
             "payload has an unrelated color")

    if payload_bottom + 1 < top:
        layout = "payload_above"
    elif bottom + 1 < payload_top:
        layout = "frame_above"
    elif payload_right + 1 < frame_left:
        layout = "payload_left"
    elif frame_right + 1 < payload_left:
        layout = "frame_left"
    else:
        raise AssertionError("payload and frame have no clear background gap")

    left_key = {int(v) for v in payload[:, 0] if int(v) != background}
    right_key = {int(v) for v in payload[:, -1] if int(v) != background}
    if left_key == {left} and right_key == {right}:
        flip = False
        oriented = payload
    elif left_key == {right} and right_key == {left}:
        flip = True
        oriented = np.fliplr(payload)
    else:
        raise AssertionError("payload edge keys are ambiguous")
    expected_output = expected_frame.copy()
    expected_output[1:-1, 1:-1] = oriented
    _require(np.array_equal(output_grid, expected_output),
             "output is not the correctly oriented socket completion")
    input_only_answers = _input_candidates(input_grid)
    _require(len(input_only_answers) == 1 and
             np.array_equal(input_only_answers[0], output_grid),
             "input does not determine one socket completion")

    canvas_height, canvas_width = input_grid.shape
    return {
        "background": background, "marker": marker,
        "left_color": left, "right_color": right,
        "needs_flip": flip, "layout": layout,
        "socket_shape": (height, width),
        "canvas_shape": (canvas_height, canvas_width),
        "frame_bbox": (top, frame_left, bottom, frame_right),
        "payload_bbox": (payload_top, payload_left,
                         payload_bottom, payload_right),
        "components": _component_count(payload != background),
        "density": float(np.mean(payload != background)),
        "frame_touches_edge": (top == 0 or frame_left == 0 or
                               bottom == canvas_height - 1 or
                               frame_right == canvas_width - 1),
        "payload_touches_edge": (payload_top == 0 or payload_left == 0 or
                                 payload_bottom == canvas_height - 1 or
                                 payload_right == canvas_width - 1),
        "payload_bottom_flush": payload_bottom == canvas_height - 1,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
                 "puzzle must contain exactly train and test")
        _require(isinstance(puzzle["train"], list) and
                 isinstance(puzzle["test"], list), "splits must be lists")
        train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
        _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and
                 train_count > test_count, "invalid pair counts")
        seen_inputs = set()
        infos = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _require(isinstance(pair, dict) and
                         set(pair) == {"input", "output"}, "invalid pair schema")
                for name in ("input", "output"):
                    grid = pair[name]
                    _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                             "grid must be a 2D NumPy array")
                    _require(1 <= grid.shape[0] <= 30 and
                             1 <= grid.shape[1] <= 30, "grid size out of range")
                    _require(np.issubdtype(grid.dtype, np.integer),
                             "grid must use integer colors")
                    _require(int(np.min(grid)) >= 0 and int(np.max(grid)) <= 9,
                             "grid colors out of range")
                input_grid, output_grid = pair["input"], pair["output"]
                key = _array_key(input_grid)
                _require(key not in seen_inputs, "duplicate input/pair")
                seen_inputs.add(key)
                _require(not (input_grid.shape == output_grid.shape and
                              np.array_equal(input_grid, output_grid)),
                         "input equals output")
                if input_grid.shape == (13, 13):
                    signature = "".join(str(int(v)) for v in input_grid.ravel())
                    _require(signature not in _SOURCE_INPUT_SIGNATURES,
                             "pair copies a source input")
                infos[split].append(_analyze_pair(input_grid, output_grid))

        training = infos["train"]
        tests = infos["test"]
        _require({info["needs_flip"] for info in training} == {False, True},
                 "training must show preservation and reflection")
        # A shared object order exhibits both decisions. Object order cannot
        # therefore stand in for the edge-key rule, even with two examples.
        _require(any({info["needs_flip"] for info in training
                      if info["layout"] == layout} == {False, True}
                     for layout in ("frame_above", "payload_above")),
                 "training makes layout a proxy for the decision")
        if train_count >= 4:
            _require({(info["layout"], info["needs_flip"]) for info in training}
                     >= {(layout, flip)
                         for layout in ("frame_above", "payload_above")
                         for flip in (False, True)},
                     "long training set must cross order and reflection")
        train_colors = {color for info in training
                        for color in (info["left_color"], info["right_color"])}
        _require(2 <= len(train_colors) <= 6, "invalid rail palette size")
        _require(all({info["left_color"], info["right_color"]} <= train_colors
                     for info in tests), "test uses unseen rail colors")
        if test_count >= 2:
            _require({info["needs_flip"] for info in tests} == {False, True},
                     "multiple tests must exercise both decisions")
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("invalid generated puzzle structure") from exc


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    _require(len({_puzzle_key(p) for p in puzzles}) >= 40,
             "cross-seed puzzles lack diversity")
    all_infos = []
    training_infos = []
    testing_infos = []
    palette_sizes = set()
    background_changes = marker_changes = 0
    new_test_sizes = 0
    new_test_aspects = side_swaps = repeated_pairings = 0
    source_like_tests = combined_tests = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        training_infos.extend(train)
        testing_infos.extend(test)
        all_infos.extend(train + test)
        palette_sizes.add(len({c for info in train
                               for c in (info["left_color"], info["right_color"])}))
        background_changes += len({i["background"] for i in train + test}) > 1
        marker_changes += len({i["marker"] for i in train + test}) > 1
        train_sizes = {i["socket_shape"] for i in train}
        new_test_sizes += sum(i["socket_shape"] not in train_sizes for i in test)
        train_aspects = {h / w for h, w in train_sizes}
        new_test_aspects += sum(
            i["socket_shape"][0] / i["socket_shape"][1] not in train_aspects
            for i in test
        )
        side_swaps += bool({i["left_color"] for i in train} &
                           {i["right_color"] for i in train})
        repeated_pairings += (len({frozenset((i["left_color"], i["right_color"]))
                                   for i in train}) < len(train))
        source_like_tests += sum(i["layout"] == "frame_above" and
                                 i["needs_flip"] and i["payload_bottom_flush"]
                                 for i in test)
        combined_tests += sum(i["layout"] in ("frame_left", "payload_left") and
                              i["needs_flip"] and i["frame_touches_edge"]
                              for i in test)
    _require({2, 3} <= palette_sizes and any(v >= 4 for v in palette_sizes),
             "rail palette diversity is missing")
    _require(background_changes >= 10 and marker_changes >= 10,
             "pair-local backgrounds or markers do not vary")
    _require(new_test_sizes >= 10, "test sizes are always copied from training")
    _require(new_test_aspects >= 10, "test aspect ratios lack novelty")
    _require(side_swaps >= 15 and repeated_pairings >= 10,
             "rail pairings do not repeat and swap sides")
    _require(source_like_tests >= 1 and combined_tests >= 1,
             "test shifts do not combine supported axes")
    _require({i["layout"] for i in testing_infos} ==
             {"frame_above", "payload_above", "frame_left", "payload_left"},
             "test layouts lack diversity")
    _require({i["needs_flip"] for i in testing_infos} == {False, True},
             "test decisions lack diversity")
    _require(any(i["payload_bottom_flush"] for i in testing_infos),
             "source-like bottom-flush test is absent")
    _require(any(i["frame_touches_edge"] for i in training_infos) and
             any(i["payload_touches_edge"] for i in training_infos),
             "training does not use the canvas boundary")
    _require(sum(i["frame_bbox"][3] == i["canvas_shape"][1] - 1
                 for i in training_infos) >= 10,
             "training omits the source's right-edge socket")
    _require(sum(i["components"] == 1 for i in all_infos) >=
             int(0.85 * len(all_infos)), "payload motifs are too fragmented")
    sizes = {i["socket_shape"] for i in all_infos}
    _require(len(sizes) >= 35 and
             any(h >= 10 and w <= 8 for h, w in sizes) and
             any(h >= 10 and w >= 11 for h, w in sizes),
             "socket dimensions lack variety")
    _require(len({(i["layout"], i["needs_flip"], i["socket_shape"])
                  for i in all_infos}) >= 70,
             "structural combinations lack diversity")
    normalized_masks = set()
    enclosed_background = 0
    for puzzle in puzzles:
        for pair in puzzle["train"] + puzzle["test"]:
            info = _analyze_pair(pair["input"], pair["output"])
            top, left, bottom, right = info["payload_bbox"]
            motif = pair["input"][top:bottom + 1, left:right + 1]
            mask = motif != info["background"]
            normalized_masks.add(
                (mask.shape, tuple(bool(v) for v in mask.ravel()))
            )
            enclosed_background += any(
                not mask[y, x] and np.any(mask[y, :x]) and
                np.any(mask[y, x + 1:]) and np.any(mask[:y, x]) and
                np.any(mask[y + 1:, x])
                for y in range(1, mask.shape[0] - 1)
                for x in range(1, mask.shape[1] - 1)
            )
    _require(len(normalized_masks) >= 200 and enclosed_background >= 20,
             "motifs lack distinct silhouettes or holes")
    _require(sum(i["density"] < 0.4 for i in all_infos) >= 8 and
             sum(i["density"] >= 0.65 for i in all_infos) >= 40,
             "sparse and dense motif regimes are not both represented")

    for seed, train, test in ((2000, 2, 1), (2001, 10, 1),
                              (2002, 10, 4), (2003, 6, 2)):
        puzzle = generate_puzzle(seed, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit count was ignored")
        validate_generated_puzzle(puzzle)
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed, train, test)),
                 "explicit generation is not deterministic")
    for seed, train, test in ((3000, None, 1), (3001, None, 4),
                              (3002, 2, None), (3003, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        _require(train is None or len(puzzle["train"]) == train,
                 "partial training count was ignored")
        _require(test is None or len(puzzle["test"]) == test,
                 "partial test count was ignored")
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed, train, test)),
                 "partial generation is not deterministic")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(4000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts were accepted: {kwargs}")
