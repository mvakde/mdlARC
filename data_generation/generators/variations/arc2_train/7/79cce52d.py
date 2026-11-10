import numpy as np


_ORIENTATIONS = ("tl", "tr", "bl", "br")

_DIRECTION_SIGNS = {
    "tl": (1, 1),
    "tr": (1, -1),
    "bl": (-1, 1),
    "br": (-1, -1),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.reshape(-1)),
    )


def _pair_signature(pair: dict) -> tuple:
    return (
        _grid_signature(pair["input"]),
        _grid_signature(pair["output"]),
    )


def _puzzle_fingerprint(puzzle: dict) -> tuple:
    return tuple(
        (
            split,
            _grid_signature(pair["input"]),
            _grid_signature(pair["output"]),
        )
        for split in ("train", "test")
        for pair in puzzle[split]
    )


def _puzzles_equal(first: dict, second: dict) -> bool:
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    if set(first) != {"train", "test"} or set(second) != {"train", "test"}:
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for pair_a, pair_b in zip(first[split], second[split]):
            if set(pair_a) != {"input", "output"}:
                return False
            if set(pair_b) != {"input", "output"}:
                return False
            if not np.array_equal(pair_a["input"], pair_b["input"]):
                return False
            if not np.array_equal(pair_a["output"], pair_b["output"]):
                return False

    return True


def _grid_from_strings(rows: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = (
    (
        _grid_from_strings(
            (
                "7266666",
                "2888444",
                "6988844",
                "6933354",
                "6939355",
                "6999355",
                "6111115",
            )
        ),
        _grid_from_strings(
            (
                "888444",
                "988844",
                "933354",
                "939355",
                "999355",
                "111115",
            )
        ),
    ),
    (
        _grid_from_strings(
            (
                "7662666",
                "6559988",
                "6555981",
                "2844981",
                "6844991",
                "6883331",
                "6833311",
            )
        ),
        _grid_from_strings(
            (
                "318833",
                "118333",
                "885599",
                "815559",
                "818449",
                "918449",
            )
        ),
    ),
    (
        _grid_from_strings(
            (
                "7666266",
                "6355555",
                "2335999",
                "6334949",
                "6114449",
                "6114849",
                "6118888",
            )
        ),
        _grid_from_strings(
            (
                "888118",
                "555355",
                "999335",
                "949334",
                "449114",
                "849114",
            )
        ),
    ),
    (
        _grid_from_strings(
            (
                "7626666",
                "6553399",
                "6555398",
                "6133398",
                "6114488",
                "6114448",
                "2114448",
            )
        ),
        _grid_from_strings(
            (
                "855539",
                "813339",
                "811448",
                "811444",
                "811444",
                "955339",
            )
        ),
    ),
)


def _is_exact_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    for source_input, source_output in _SOURCE_PAIRS:
        if (
            input_grid.shape == source_input.shape
            and output_grid.shape == source_output.shape
            and np.array_equal(input_grid, source_input)
            and np.array_equal(output_grid, source_output)
        ):
            return True
    return False


def _validate_optional_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return value


def _resolve_pair_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_optional_count("num_train", num_train, 2, 10)
    test_count = _validate_optional_count("num_test", num_test, 1, 4)

    if train_count is None and test_count is None:
        test_count = int(_choice(rng, (1, 1, 1, 1, 2, 2, 3, 4)))
        minimum_train = max(2, test_count + 1)
        common = tuple(x for x in (2, 2, 3, 3, 3, 3, 3, 4, 5, 6, 8, 10) if x >= minimum_train)
        train_count = int(_choice(rng, common))
    elif train_count is None:
        minimum_train = max(2, int(test_count) + 1)
        if minimum_train > 10:
            raise ValueError("no legal num_train exists for the requested num_test")
        train_count = int(rng.integers(minimum_train, 11))
    elif test_count is None:
        maximum_test = min(4, int(train_count) - 1)
        if maximum_test < 1:
            raise ValueError("no legal num_test exists for the requested num_train")
        test_count = int(rng.integers(1, maximum_test + 1))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return int(train_count), int(test_count)


def _choice(rng: np.random.Generator, values: tuple):
    return values[int(rng.integers(len(values)))]


def _sample_dimensions(rng: np.random.Generator) -> tuple[int, int]:
    mode = _choice(rng, ("compact", "compact", "compact", "medium", "medium", "thin_h", "thin_w", "large"))
    if mode == "compact":
        return int(rng.integers(3, 9)), int(rng.integers(3, 9))
    if mode == "medium":
        return int(rng.integers(7, 17)), int(rng.integers(7, 17))
    if mode == "thin_h":
        return int(rng.integers(3, 7)), int(rng.integers(11, 27))
    if mode == "thin_w":
        return int(rng.integers(11, 27)), int(rng.integers(3, 7))
    return int(rng.integers(16, 30)), int(rng.integers(16, 30))


def _sample_colors(rng: np.random.Generator, shared_roles: tuple[int, int, int] | None = None,
                   shared_palette: tuple[int, ...] | None = None) -> tuple[int, int, int, tuple[int, ...]]:
    if shared_roles is None:
        order = [int(x) for x in rng.permutation(10)]
        roles = tuple(order[:3])
    else:
        roles = shared_roles
    if shared_palette is None:
        available = [x for x in range(10) if x not in roles]
        colors = [int(x) for x in rng.permutation(available)]
        palette_size = int(rng.integers(2, 8))
        palette = tuple(colors[:palette_size])
    else:
        palette = shared_palette
    return roles[0], roles[1], roles[2], palette


def _geometry(height: int, width: int, orientation: str) -> tuple:
    top = orientation[0] == "t"
    left = orientation[1] == "l"
    corner = (0 if top else height, 0 if left else width)
    payload_rows = slice(1, height + 1) if top else slice(0, height)
    payload_columns = slice(1, width + 1) if left else slice(0, width)
    vertical_arm = tuple((i if top else height - i, corner[1]) for i in range(1, height + 1))
    horizontal_arm = tuple((corner[0], j if left else width - j) for j in range(1, width + 1))
    return corner, payload_rows, payload_columns, vertical_arm, horizontal_arm


def _make_payload(rng: np.random.Generator, height: int, width: int,
                  palette: tuple[int, ...]) -> np.ndarray:
    colors = np.asarray(palette, dtype=np.int8)
    size = len(colors)
    mode = int(rng.integers(0, 6))
    rows = np.arange(height)[:, None]
    columns = np.arange(width)[None, :]
    if mode == 0:  # irregular Voronoi regions
        sites = rng.choice(height * width, size=size, replace=False)
        rr, cc = sites // width, sites % width
        weights = rng.integers(1, 5, size=(size, 2))
        distance = np.stack([
            weights[k, 0] * np.abs(rows - rr[k]) + weights[k, 1] * np.abs(columns - cc[k])
            for k in range(size)
        ])
        payload = colors[np.argmin(distance, axis=0)].copy()
    elif mode == 1:  # overlapping rectangular patches
        payload = np.full((height, width), colors[0], dtype=np.int8)
        order = [int(x) for x in rng.permutation(size)]
        for k in order + [int(x) for x in rng.integers(0, size, size=size)]:
            r0 = int(rng.integers(height))
            c0 = int(rng.integers(width))
            r1 = int(rng.integers(r0 + 1, height + 1))
            c1 = int(rng.integers(c0 + 1, width + 1))
            payload[r0:r1, c0:c1] = colors[k]
    elif mode == 2:  # offset nested rectangles and accents
        payload = np.full((height, width), colors[0], dtype=np.int8)
        r0, c0, r1, c1 = 0, 0, height, width
        for k in range(1, size + 3):
            if r0 >= r1 or c0 >= c1:
                break
            payload[r0:r1, c0:c1] = colors[k % size]
            r0 += int(rng.integers(0, 2))
            c0 += int(rng.integers(0, 2))
            r1 -= int(rng.integers(0, 2))
            c1 -= int(rng.integers(0, 2))
        for _ in range(size):
            payload[int(rng.integers(height)), int(rng.integers(width))] = colors[int(rng.integers(size))]
    elif mode == 3:  # interrupted bands
        rb = int(rng.integers(1, min(5, height) + 1))
        cb = int(rng.integers(1, min(5, width) + 1))
        a = int(rng.integers(1, size))
        b = int(rng.integers(1, size))
        payload = colors[((rows // rb) * a + (columns // cb) * b + int(rng.integers(size))) % size].copy()
        for _ in range(int(rng.integers(1, 4))):
            r = int(rng.integers(height))
            c = int(rng.integers(width))
            payload[r:min(height, r + int(rng.integers(1, 4))), c:min(width, c + int(rng.integers(1, 4)))] = colors[int(rng.integers(size))]
    elif mode == 4:  # sparse details on a broad field
        payload = np.full((height, width), colors[0], dtype=np.int8)
        for k in range(1, size):
            for _ in range(int(rng.integers(1, 4))):
                r = int(rng.integers(height))
                c = int(rng.integers(width))
                rh = int(rng.integers(1, min(4, height - r) + 1))
                cw = int(rng.integers(1, min(4, width - c) + 1))
                payload[r:r + rh, c:c + cw] = colors[k]
    else:  # block mosaic with occasional thin overlays
        rh = int(rng.integers(1, min(5, height) + 1))
        cw = int(rng.integers(1, min(5, width) + 1))
        blocks = rng.integers(0, size, size=((height + rh - 1) // rh, (width + cw - 1) // cw))
        payload = colors[np.repeat(np.repeat(blocks, rh, axis=0), cw, axis=1)[:height, :width]].copy()
        if rng.integers(2):
            payload[int(rng.integers(height)), :] = colors[int(rng.integers(size))]
        else:
            payload[:, int(rng.integers(width))] = colors[int(rng.integers(size))]
    # Every chosen color occurs even when patches overlap or a small grid crops a motif.
    slots = rng.choice(height * width, size=size, replace=False)
    for slot, color in zip(slots, colors):
        payload.flat[int(slot)] = color
    return payload


def _unique_phase(payload: np.ndarray) -> bool:
    height, width = payload.shape
    for dr in range(height):
        for dc in range(width):
            if (dr or dc) and np.array_equal(payload, np.roll(payload, (dr, dc), (0, 1))):
                return False
    return True


def _witness_shift(rng: np.random.Generator, length: int) -> int:
    legal = [value for value in range(1, length) if 2 * value != length]
    return int(_choice(rng, tuple(legal)))


def _sample_shift(rng: np.random.Generator, height: int, width: int,
                  intent: str) -> tuple[int, int]:
    if intent == "vertical":
        return _witness_shift(rng, height), 0
    if intent == "horizontal":
        return 0, _witness_shift(rng, width)
    if intent == "zero":
        return 0, 0
    if intent == "equal":
        value = int(rng.integers(1, min(height, width)))
        return value, value
    if intent == "mixed":
        return int(rng.integers(1, height)), int(rng.integers(1, width))
    if intent == "interior":
        return int(rng.integers(1, height - 1)), int(rng.integers(1, width - 1))
    if intent == "edge_small":
        if rng.integers(2):
            return height - 1, int(rng.integers(1, min(4, width)))
        return int(rng.integers(1, min(4, height))), width - 1
    return int(rng.integers(height)), int(rng.integers(width))


def _make_pair(rng: np.random.Generator, height: int, width: int, orientation: str,
               colors: tuple, intent: str) -> dict:
    guide, corner_color, marker, palette = colors
    payload = _make_payload(rng, height, width, palette)
    vertical, horizontal = _sample_shift(rng, height, width, intent)
    corner, payload_rows, payload_cols, vertical_arm, horizontal_arm = _geometry(height, width, orientation)
    inp = np.full((height + 1, width + 1), guide, dtype=np.int8)
    inp[payload_rows, payload_cols] = payload
    inp[corner] = corner_color
    inp[vertical_arm[vertical]] = marker
    inp[horizontal_arm[horizontal]] = marker
    row_sign, col_sign = _DIRECTION_SIGNS[orientation]
    out = np.roll(payload, (row_sign * vertical, col_sign * horizontal), (0, 1)).astype(np.int8, copy=False)
    return {"input": inp, "output": out}


def _decode_input(input_grid: np.ndarray) -> dict:
    height, width = input_grid.shape[0] - 1, input_grid.shape[1] - 1
    matches = []
    for orientation in _ORIENTATIONS:
        corner, payload_rows, payload_cols, vertical_arm, horizontal_arm = _geometry(height, width, orientation)
        vertical_values = [int(input_grid[pos]) for pos in vertical_arm]
        horizontal_values = [int(input_grid[pos]) for pos in horizontal_arm]
        if set(vertical_values) != set(horizontal_values) or len(set(vertical_values)) != 2:
            continue
        for marker in set(vertical_values):
            guide = next(iter(set(vertical_values) - {marker}))
            if vertical_values.count(marker) != 1 or horizontal_values.count(marker) != 1:
                continue
            key = int(input_grid[corner])
            if len({guide, marker, key}) != 3:
                continue
            if np.count_nonzero(input_grid == key) != 1 or np.count_nonzero(input_grid == marker) != 2:
                continue
            if np.count_nonzero(input_grid == guide) != height + width - 2:
                continue
            payload = input_grid[payload_rows, payload_cols]
            payload_colors = set(int(x) for x in np.unique(payload))
            if not 2 <= len(payload_colors) <= 7 or payload_colors & {guide, marker, key}:
                continue
            matches.append({"orientation": orientation, "guide_color": guide,
                            "corner_color": key, "marker_color": marker,
                            "palette": tuple(sorted(payload_colors)),
                            "height": height, "width": width,
                            "vertical_shift": vertical_values.index(marker),
                            "horizontal_shift": horizontal_values.index(marker),
                            "payload": payload})
    _require(len(matches) == 1, f"expected one input-only L interpretation; found {len(matches)}")
    return matches[0]


def _roll_from_info(info: dict, vertical_sign: int = 1, horizontal_sign: int = 1) -> np.ndarray:
    row_sign, col_sign = _DIRECTION_SIGNS[info["orientation"]]
    return np.roll(info["payload"],
                   (row_sign * vertical_sign * info["vertical_shift"],
                    col_sign * horizontal_sign * info["horizontal_shift"]),
                   (0, 1))


def _infer_direction(train_pairs: list[dict]) -> tuple[int, int]:
    possibilities = []
    for vertical_sign in (1, -1):
        for horizontal_sign in (1, -1):
            if all(np.array_equal(_roll_from_info(_decode_input(pair["input"]), vertical_sign, horizontal_sign),
                                  pair["output"]) for pair in train_pairs):
                possibilities.append((vertical_sign, horizontal_sign))
    _require(possibilities == [(1, 1)], "training must uniquely establish both corner-relative directions")
    return possibilities[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "split lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    used_inputs, used_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid array")
                _require(np.issubdtype(grid.dtype, np.integer), "integer grid")
                _require(all(1 <= n <= 30 for n in grid.shape), "grid size")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "ARC colors")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape == (out.shape[0] + 1, out.shape[1] + 1), "guide adds row and column")
            _require(3 <= out.shape[0] <= 29 and 3 <= out.shape[1] <= 29, "decodable payload size")
            input_sig, pair_sig = _grid_signature(inp), _pair_signature(pair)
            _require(input_sig not in used_inputs and pair_sig not in used_pairs, "duplicate pair")
            used_inputs.add(input_sig)
            used_pairs.add(pair_sig)
            _require(not _is_exact_source_pair(inp, out), "source pair copied")
            info = _decode_input(inp)
            _require(_unique_phase(info["payload"]), "payload has ambiguous cyclic phases")
            _require(set(int(x) for x in np.unique(out)) == set(info["palette"]), "output palette")
            _require(np.array_equal(_roll_from_info(info), out), "incorrect corner-relative cyclic shift")
            if info["vertical_shift"]:
                _require(not np.array_equal(out, _roll_from_info(info, 0, 1)), "invisible vertical shift")
            if info["horizontal_shift"]:
                _require(not np.array_equal(out, _roll_from_info(info, 1, 0)), "invisible horizontal shift")
            infos[split].append(info)
    _infer_direction(puzzle["train"])
    for pair, info in zip(puzzle["test"], infos["test"]):
        _require(np.array_equal(_roll_from_info(info), pair["output"]), "test not predictable from training")
    train_orientations = {info["orientation"] for info in infos["train"]}
    _require(all(info["orientation"] in train_orientations for info in infos["test"]), "test orientation unseen in training")
    # The local key/arm geometry defines color roles; no literal color or size is a learned convention.


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_pair_counts(rng, num_train, num_test)
    size_mode = _choice(rng, ("fixed", "varied", "varied"))
    fixed_size = ((6, 6) if rng.integers(4) == 0 else _sample_dimensions(rng)) if size_mode == "fixed" else None
    color_mode = _choice(rng, ("shared_all", "shared_all", "shared_roles", "local", "local", "local"))
    shared_colors = _sample_colors(rng)
    anchor_orientation = _choice(rng, _ORIENTATIONS)
    orientation_mode = "mixed" if train_count >= 3 and rng.integers(3) else "fixed"
    intents = ["vertical", "horizontal"]
    intents.extend(_choice(rng, ("zero", "equal", "mixed", "mixed", "vertical", "horizontal", "random"))
                   for _ in range(train_count - 2))
    rng.shuffle(intents)
    # Both axis witnesses share a placement; further training pairs can demonstrate rotations.
    train_orientations = [anchor_orientation] * train_count
    if orientation_mode == "mixed":
        other = _choice(rng, tuple(o for o in _ORIENTATIONS if o != anchor_orientation))
        witness_indices = {next(i for i, intent in enumerate(intents) if intent == axis)
                           for axis in ("vertical", "horizontal")}
        extras = [i for i in range(train_count) if i not in witness_indices]
        train_orientations[extras[0]] = other
        for i in extras[1:]:
            train_orientations[i] = _choice(rng, (anchor_orientation, other))
    test_intents = [_choice(rng, ("zero", "vertical", "horizontal", "equal", "interior",
                                   "edge_small", "edge_small", "random", "mixed")) for _ in range(test_count)]
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for i in range(count):
            orientation = train_orientations[i] if split == "train" else _choice(rng, tuple(sorted(set(train_orientations))))
            intent = intents[i] if split == "train" else test_intents[i]
            for _ in range(150):
                height, width = fixed_size if fixed_size else _sample_dimensions(rng)
                if color_mode == "shared_all":
                    colors = shared_colors
                elif color_mode == "shared_roles":
                    colors = _sample_colors(rng, shared_colors[:3])
                else:
                    colors = _sample_colors(rng)
                candidate = _make_pair(rng, height, width, orientation, colors, intent)
                signature = _grid_signature(candidate["input"])
                if signature in used_inputs or _is_exact_source_pair(candidate["input"], candidate["output"]):
                    continue
                try:
                    info = _decode_input(candidate["input"])
                except AssertionError:
                    continue
                if not _unique_phase(info["payload"]):
                    continue
                if intent in ("vertical", "horizontal"):
                    if np.array_equal(_roll_from_info(info), _roll_from_info(info, -1, -1)):
                        continue
                used_inputs.add(signature)
                puzzle[split].append(candidate)
                break
            else:
                raise RuntimeError("failed to construct a decodable, phase-diagnostic pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _require(_puzzles_equal(generate_puzzle(123), generate_puzzle(123)), "automatic determinism")
    _require(_puzzles_equal(generate_puzzle(123, 8, 3), generate_puzzle(123, 8, 3)), "explicit determinism")
    _require(not _puzzles_equal(generate_puzzle(123), generate_puzzle(124)), "cross-seed variation")
    for train, test in ((2, 1), (10, 1), (10, 4), (6, 2), (9, 3)):
        puzzle = generate_puzzle(1000 + train * 10 + test, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test), "explicit counts")
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2000 + (train or 0) * 10 + (test or 0), train, test)
        _require((train is None or len(puzzle["train"]) == train) and
                 (test is None or len(puzzle["test"]) == test), "partial counts")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(3000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    _require(sum((len(p["train"]), len(p["test"])) == (3, 1) for p in puzzles) >= 4,
             "source-like compact puzzle coverage")
    _require(sum((len(p["train"]), len(p["test"])) == (2, 1) for p in puzzles) >= 2,
             "two-witness compact puzzle coverage")
    _require(len({_puzzle_fingerprint(p) for p in puzzles}) == 42, "puzzle diversity")
    all_infos = [[_decode_input(pair["input"]) for split in ("train", "test") for pair in puzzle[split]]
                 for puzzle in puzzles]
    _require({i["orientation"] for infos in all_infos for i in infos} == set(_ORIENTATIONS), "orientation coverage")
    _require(sum(len({i["orientation"] for i in infos}) > 1 for infos in all_infos) >= 5, "within-puzzle orientations")
    _require(sum(len({(i["height"], i["width"]) for i in infos}) == 1 for infos in all_infos) >= 5,
             "fixed-size puzzles")
    _require(any(all((i["height"], i["width"]) == (6, 6) for i in infos)
                 for infos in all_infos), "source-like fixed 6-by-6 payload")
    _require(sum(len({(i["height"], i["width"]) for i in infos}) > 1 for infos in all_infos) >= 15,
             "varied-size puzzles")
    _require(sum(len({(i["guide_color"], i["corner_color"], i["marker_color"]) for i in infos}) > 1
                 for infos in all_infos) >= 5, "local role colors")
    _require(sum(len({len(i["palette"]) for i in infos}) > 1 for infos in all_infos) >= 8,
             "local palette sizes")
    _require(any(i["height"] <= 5 and i["width"] >= 11 for infos in all_infos for i in infos), "thin landscape")
    _require(any(i["width"] <= 5 and i["height"] >= 11 for infos in all_infos for i in infos), "thin portrait")
    _require(any(i["height"] >= 16 and i["width"] >= 16 for infos in all_infos for i in infos), "large payload")
    _require(any(i["vertical_shift"] == 0 and i["horizontal_shift"] == 0
                 for infos in all_infos for i in infos), "zero phase")
    _require(any(i["vertical_shift"] == i["horizontal_shift"] > 0
                 for infos in all_infos for i in infos), "equal phase")
    _require(any(i["vertical_shift"] > 0 and i["horizontal_shift"] > 0 and
                 i["vertical_shift"] != i["horizontal_shift"] for infos in all_infos for i in infos), "unequal phase")
    _require(sum(len({(i["height"], i["width"]) for i in infos}) > 1 and
                 len({i["orientation"] for i in infos}) > 1 and
                 len({(i["guide_color"], i["corner_color"], i["marker_color"]) for i in infos}) > 1
                 for infos in all_infos) >= 4, "combined local geometry, color, and orientation variation")
    test_categories = set()
    familiar_tests = 0
    for puzzle in puzzles:
        training_phases = {(info["vertical_shift"], info["horizontal_shift"])
                           for pair in puzzle["train"] for info in (_decode_input(pair["input"]),)}
        for pair in puzzle["test"]:
            info = _decode_input(pair["input"])
            v, h = info["vertical_shift"], info["horizontal_shift"]
            familiar_tests += (v, h) in training_phases
            if v == h == 0:
                test_categories.add("zero")
            elif v == 0 or h == 0:
                test_categories.add("one_axis")
            elif v == h:
                test_categories.add("equal")
            elif v == info["height"] - 1 or h == info["width"] - 1:
                test_categories.add("near_edge")
            else:
                test_categories.add("interior_unequal")
    _require(len(test_categories) == 5, "test phase regime diversity")
    _require(familiar_tests >= 3, "familiar test phase combinations")
    normalized_structures = set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                role_map = {}
                normalized = []
                for cell in grid.flat:
                    color = int(cell)
                    if color not in role_map:
                        role_map[color] = len(role_map)
                    normalized.append(role_map[color])
                normalized_structures.add((grid.shape, tuple(normalized)))
    _require(len(normalized_structures) >= 250, "color-normalized structural diversity")
