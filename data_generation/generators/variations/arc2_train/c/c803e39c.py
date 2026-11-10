import numpy as np


_SOURCE_INPUT_ROWS = {
    (
        "00000500000500000500000",
        "01010502200503330509990",
        "01010502020503330509990",
        "01110500220503330509990",
        "00000500000500000500000",
    ),
    (
        "00000500000500000500000",
        "01110502220504440508880",
        "00100500200504440508880",
        "01110500200504440508880",
        "00000500000500000500000",
    ),
    (
        "000000500000050000005000000",
        "011110502200050777705088880",
        "000100500220050777705088880",
        "001000500220050777705088880",
        "011110500022050777705088880",
        "000000500000050000005000000",
    ),
    (
        "000000050000000500000005000000",
        "011100050222220503333305044444",
        "010100050000020503333305044444",
        "011111050222220503333305044444",
        "000101050200000503333305044444",
        "000111050222220503333305044444",
        "000000050000000500000005000000",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _integer_argument(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        num_train = _integer_argument(num_train, "num_train")
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")

    if num_test is not None:
        num_test = _integer_argument(num_test, "num_test")
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = num_test + 1
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        low = num_test + 1
        num_train = low + seed % (11 - low)
    elif num_test is None:
        high = min(4, num_train - 1)
        num_test = 1 + seed % high

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _grid_rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _value_signature(
    grid: np.ndarray,
) -> tuple[tuple[int, int], bytes]:
    return (
        tuple(grid.shape),
        grid.astype(np.int8, copy=False).tobytes(),
    )

def _d4_nonidentity(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, k) for k in (1, 2, 3)] + [
        np.fliplr(np.rot90(mask, k)) for k in range(4)
    ]


def _is_orientation_diagnostic(tile: np.ndarray, placement: np.ndarray) -> bool:
    if any(np.array_equal(tile, x) for x in _d4_nonidentity(tile)):
        return False
    if any(np.array_equal(placement, x) for x in _d4_nonidentity(placement)):
        return False
    return not np.array_equal(np.kron(placement, tile), np.kron(tile, placement))


def _random_mask(rng: np.random.Generator, n: int, count: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    if count:
        mask.flat[rng.choice(n * n, size=count, replace=False)] = True
    return mask


def _sample_mask(rng: np.random.Generator, n: int) -> np.ndarray:
    """Parametric motifs mixed with unconstrained masks."""
    style = int(rng.integers(0, 9))
    mask = np.zeros((n, n), dtype=bool)
    if style == 0:
        return _random_mask(rng, n, int(rng.integers(1, n * n)))
    if style == 1:
        target = int(rng.integers(2, n * n))
        mask.flat[int(rng.integers(0, n * n))] = True
        while int(mask.sum()) < target:
            frontier = np.zeros_like(mask)
            frontier[1:] |= mask[:-1]
            frontier[:-1] |= mask[1:]
            frontier[:, 1:] |= mask[:, :-1]
            frontier[:, :-1] |= mask[:, 1:]
            choices = np.argwhere(frontier & ~mask)
            r, c = choices[int(rng.integers(0, len(choices)))]
            mask[int(r), int(c)] = True
    elif style == 2:
        rows = rng.random(n)[:, None]
        cols = rng.random(n)[None, :]
        scores = rows + cols + .6 * rng.random((n, n))
        count = int(rng.integers(1, n * n))
        mask.flat[np.argsort(scores.ravel())[-count:]] = True
    elif style == 3:
        horizontal = bool(rng.integers(0, 2))
        start = int(rng.integers(0, n))
        width = int(rng.integers(1, max(2, n // 2 + 1)))
        if horizontal:
            mask[start:min(n, start + width), :] = True
        else:
            mask[:, start:min(n, start + width)] = True
        if bool(rng.integers(0, 3)):
            a = int(rng.integers(0, n))
            b = int(rng.integers(a + 1, n + 1))
            if horizontal:
                mask[start:min(n, start + width), :a] = False
                mask[start:min(n, start + width), b:] = False
            else:
                mask[:a, start:min(n, start + width)] = False
                mask[b:, start:min(n, start + width)] = False
    elif style == 4:
        r = int(rng.integers(0, n))
        c = int(rng.integers(0, n))
        mask[r, :] = True
        mask[:, c] = True
        if bool(rng.integers(0, 3)):
            mask[r, :c] = False
    elif style == 5:
        r0 = int(rng.integers(0, n - 1))
        c0 = int(rng.integers(0, n - 1))
        r1 = int(rng.integers(r0 + 1, n))
        c1 = int(rng.integers(c0 + 1, n))
        mask[r0, c0:c1 + 1] = True
        mask[r1, c0:c1 + 1] = True
        mask[r0:r1 + 1, c0] = True
        mask[r0:r1 + 1, c1] = True
        if bool(rng.integers(0, 3)):
            mask[r0, c0 + 1:c1] = False
    elif style == 6:
        offset = int(rng.integers(-n + 2, n - 1))
        for r in range(n):
            c = r + offset
            if 0 <= c < n:
                mask[r, c] = True
            if bool(rng.integers(0, 2)) and 0 <= n - 1 - r + offset < n:
                mask[r, n - 1 - r + offset] = True
    elif style == 7:
        pattern = rng.random(n) < rng.uniform(.25, .8)
        if not pattern.any():
            pattern[int(rng.integers(0, n))] = True
        for r in range(int(rng.integers(0, 2)), n, 2):
            mask[r] = pattern
    else:
        for r in range(n):
            for c in range(n):
                rr, cc = n - 1 - r, n - 1 - c
                if (r, c) <= (rr, cc):
                    value = bool(rng.integers(0, 2))
                    mask[r, c] = value
                    mask[rr, cc] = value
    if not mask.any():
        if style == 8:
            mask[0, 0] = mask[-1, -1] = True
        else:
            mask[int(rng.integers(0, n)), int(rng.integers(0, n))] = True
    if mask.all():
        if style == 8:
            mask[0, 0] = mask[-1, -1] = False
        else:
            mask[int(rng.integers(0, n)), int(rng.integers(0, n))] = False
    return mask


def _canonical_diagnostic(n: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((n, n), dtype=bool)
    b = np.zeros((n, n), dtype=bool)
    for r, c in ((0, 0), (0, 1), (1, 1), (2, 2)):
        a[r, c] = True
    for r, c in ((0, 2), (1, 0), (2, 1), (2, 2)):
        b[r, c] = True
    return a, b


def _diagnostic_masks(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(1000):
        a = _random_mask(rng, n, int(rng.integers(2, n * n - 1)))
        b = _random_mask(rng, n, int(rng.integers(2, n * n - 1)))
        if _is_orientation_diagnostic(a, b):
            return a, b
    return _canonical_diagnostic(n)


def _symmetric_mask(rng: np.random.Generator, n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    for r in range(n):
        for c in range(n):
            rr, cc = n - 1 - r, n - 1 - c
            if (r, c) <= (rr, cc):
                value = bool(rng.integers(0, 2))
                mask[r, c] = value
                mask[rr, cc] = value
    if not mask.any():
        mask[0, 0] = mask[-1, -1] = True
    if mask.all():
        mask[0, 0] = mask[-1, -1] = False
    return mask


def _sample_mask_pair(
    rng: np.random.Generator, n: int, mode: str,
    prior: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "diagnostic":
        return _diagnostic_masks(rng, n)
    if mode == "symmetric_tile":
        return _symmetric_mask(rng, n), _sample_mask(rng, n)
    if mode == "symmetric_placement":
        return _sample_mask(rng, n), _symmetric_mask(rng, n)
    if mode == "both_symmetric":
        return _symmetric_mask(rng, n), _symmetric_mask(rng, n)
    eligible = [item for item in prior if item["n"] == n]
    if mode in ("reuse_tile", "reuse_placement", "mirror_placement") and not eligible:
        mode = "ordinary"
    if mode == "equal_count":
        count = int(rng.integers(1, n * n))
        return _random_mask(rng, n, count), _random_mask(rng, n, count)
    if mode == "identical":
        tile = _sample_mask(rng, n)
        return tile, tile.copy()
    if mode == "singleton_tile":
        return _random_mask(rng, n, 1), _sample_mask(rng, n)
    if mode == "singleton_placement":
        return _sample_mask(rng, n), _random_mask(rng, n, 1)
    if mode == "full_tile":
        return np.ones((n, n), dtype=bool), _sample_mask(rng, n)
    if mode == "full_placement":
        return _sample_mask(rng, n), np.ones((n, n), dtype=bool)
    if mode == "empty_placement":
        return _sample_mask(rng, n), np.zeros((n, n), dtype=bool)
    if mode in ("reuse_tile", "reuse_placement", "mirror_placement"):
        item = eligible[int(rng.integers(0, len(eligible)))]
        if mode == "reuse_tile":
            return item["tile"].copy(), _sample_mask(rng, n)
        if mode == "reuse_placement":
            return _sample_mask(rng, n), item["placement"].copy()
        return item["tile"].copy(), np.fliplr(item["placement"]).copy()
    return _sample_mask(rng, n), _sample_mask(rng, n)


def _compose_input(
    tile: np.ndarray, placement: np.ndarray, neutral: int,
    divider: int, tile_marker: int, placement_marker: int,
    foreground: int, background: int, clip_side: str,
) -> np.ndarray:
    n = tile.shape[0]
    grid = np.full((n + 2, 4 * (n + 2) + 3), neutral, dtype=np.int8)
    panels = (
        np.where(tile, tile_marker, neutral),
        np.where(placement, placement_marker, neutral),
        np.full((n, n), foreground),
        np.full((n, n), background),
    )
    for i, panel in enumerate(panels):
        start = i * (n + 3)
        grid[1:n + 1, start + 1:start + n + 1] = panel
        if i < 3:
            grid[:, start + n + 2] = divider
    if clip_side == "left":
        grid = grid[:, 1:]
    elif clip_side == "right":
        grid = grid[:, :-1]
    elif clip_side != "none":
        raise ValueError("invalid clip side")
    return grid.copy()


def _construct_output(
    tile: np.ndarray, placement: np.ndarray, foreground: int, background: int,
) -> np.ndarray:
    return np.where(np.kron(placement, tile), foreground, background).astype(np.int8)


def _parse_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    h, w = input_grid.shape
    n = h - 2
    _require(n in (3, 4, 5), "panel side must be 3, 4, or 5")
    neutral = int(input_grid[0, 0])
    dividers = [c for c in range(w) if int(input_grid[0, c]) != neutral]
    _require(len(dividers) == 3, "exactly three dividers are needed")
    divider = int(input_grid[0, dividers[0]])
    _require(all(int(input_grid[0, c]) == divider for c in dividers), "divider colors disagree")
    starts = [0] + [c + 1 for c in dividers]
    ends = dividers + [w]
    lengths = [b - a for a, b in zip(starts, ends)]
    full = n + 2
    if lengths == [full] * 4:
        clip = "none"
    elif lengths == [full - 1, full, full, full]:
        clip = "left"
    elif lengths == [full, full, full, full - 1]:
        clip = "right"
    else:
        raise AssertionError("invalid panel widths or border clipping")
    _require(n != 5 or clip != "none", "n=5 needs one clipped border")
    panels = []
    for i, (a, b) in enumerate(zip(starts, ends)):
        left = 0 if i == 0 and clip == "left" else 1
        right = 0 if i == 3 and clip == "right" else 1
        panel = input_grid[1:n + 1, a + left:b - right if right else b]
        _require(panel.shape == (n, n), "panel interiors must be square")
        panels.append(panel)
    markers = []
    masks = []
    for panel in panels[:2]:
        colors = {int(x) for x in np.unique(panel)} - {neutral}
        _require(len(colors) <= 1, "mask has more than one marker color")
        marker = next(iter(colors)) if colors else None
        markers.append(marker)
        masks.append(panel != neutral)
    tile, placement = masks
    _require(tile.any(), "tile must contain an active cell")
    _require(len(np.unique(panels[2])) == 1 and len(np.unique(panels[3])) == 1,
             "both swatches must be solid")
    foreground = int(panels[2][0, 0])
    background = int(panels[3][0, 0])
    observed = [neutral, divider, foreground, background] + [m for m in markers if m is not None]
    _require(len(set(observed)) == len(observed), "visible color roles must be distinct")
    used = set(observed)
    marker_values = []
    for marker in markers:
        if marker is None:
            marker = next(color for color in range(10) if color not in used)
            used.add(marker)
        marker_values.append(marker)
    reconstructed = _compose_input(tile, placement, neutral, divider,
                                   marker_values[0], marker_values[1],
                                   foreground, background, clip)
    _require(np.array_equal(input_grid, reconstructed), "input margins or panels are malformed")
    _require(output_grid.shape == (n * n, n * n), "output size must be n squared")
    _require(np.array_equal(output_grid, _construct_output(tile, placement,
                                                           foreground, background)),
             "output violates block substitution")
    return {"n": n, "neutral": neutral, "divider": divider,
            "tile_marker": markers[0], "placement_marker": markers[1],
            "foreground": foreground, "background": background,
            "clip_side": clip, "tile": tile, "placement": placement}


def _validate_semantics(train: list[dict], test: list[dict]) -> None:
    _require(any(_is_orientation_diagnostic(x["tile"], x["placement"])
                 for x in train), "training lacks an orientation and panel-order example")
    _require(len({(x["foreground"], x["background"]) for x in train}) >= 2,
             "training swatches need at least two assignments")
    _require(len({(x["n"], x["tile"].tobytes(), x["placement"].tobytes())
                  for x in train}) >= 2, "training needs different mask layouts")
    for item in train + test:
        _require(item["n"] in (3, 4, 5), "invalid panel scale")
        _require(item["n"] != 5 or item["clip_side"] in ("left", "right"),
                 "n=5 must clip an outer border")


def _choose_sizes(seed: int, num_train: int, num_test: int,
                  rng: np.random.Generator) -> tuple[list[int], list[int]]:
    phase = seed % 5
    if phase == 0:
        train = [3, 4] + [int(rng.integers(3, 5)) for _ in range(num_train - 2)]
        test = [5] * num_test
    elif phase == 1:
        train = [3, 5] + [int(rng.choice([3, 5])) for _ in range(num_train - 2)]
        test = [5] + [int(rng.choice([3, 5])) for _ in range(num_test - 1)]
    elif phase == 2:
        train = [5, 4] + [int(rng.choice([4, 5])) for _ in range(num_train - 2)]
        test = [4] + [int(rng.choice([3, 4])) for _ in range(num_test - 1)]
    elif phase == 3:
        shared = int(rng.integers(3, 6))
        train = [shared, shared] + [int(rng.integers(3, 6)) for _ in range(num_train - 2)]
        test = [shared] + [int(rng.choice(train)) for _ in range(num_test - 1)]
    else:
        train = [4, 3] + [int(rng.integers(3, 6)) for _ in range(num_train - 2)]
        test = [int(rng.integers(3, 6)) for _ in range(num_test)]
    return train, test


def _choose_swatches(
    rng: np.random.Generator, history: list[dict],
    split: str, seed: int, index: int,
) -> tuple[int, int]:
    if not history:
        colors = rng.choice(10, size=2, replace=False)
        return int(colors[0]), int(colors[1])
    prior = [(x["foreground"], x["background"]) for x in history]
    if split == "train" and index == 1:
        options = [c for c in range(10) if c != prior[0][0]]
        fg = int(rng.choice(options))
        bg = int(rng.choice([c for c in range(10) if c != fg]))
        return fg, bg
    mode = (seed + index) % 5 if split == "test" else int(rng.integers(0, 5))
    if mode == 0:
        foregrounds = list(dict.fromkeys(a for a, _ in prior))
        if len(foregrounds) >= 2:
            colors = rng.choice(foregrounds, size=2, replace=False)
            return int(colors[0]), int(colors[1])
    if mode == 1:
        fg, bg = prior[int(rng.integers(0, len(prior)))]
        return bg, fg
    if mode == 2:
        return prior[int(rng.integers(0, len(prior)))]
    if mode == 3:
        seen = {color for pair in prior for color in pair}
        unused = [color for color in range(10) if color not in seen]
        if len(unused) >= 2:
            colors = rng.choice(unused, size=2, replace=False)
            return int(colors[0]), int(colors[1])
    colors = rng.choice(10, size=2, replace=False)
    return int(colors[0]), int(colors[1])


def _choose_palette(rng: np.random.Generator, foreground: int,
                    background: int) -> tuple[int, int, int, int]:
    choices = [color for color in range(10) if color not in (foreground, background)]
    values = rng.choice(choices, size=4, replace=False)
    return tuple(int(x) for x in values)


def generate_puzzle(
    seed: int, num_train: int | None = None, num_test: int | None = None,
) -> dict:
    seed = _integer_argument(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    train_sizes, test_sizes = _choose_sizes(seed, num_train, num_test, rng)
    child_seeds = rng.integers(0, 2**63, size=num_train + num_test, dtype=np.int64)
    result = {"train": [], "test": []}
    history: list[dict] = []
    signatures = set()
    for split, sizes in (("train", train_sizes), ("test", test_sizes)):
        for index, n in enumerate(sizes):
            pair_rng = np.random.default_rng(int(child_seeds[len(history)]))
            fg, bg = _choose_swatches(pair_rng, history, split, seed, index)
            neutral, divider, tile_marker, placement_marker = _choose_palette(pair_rng, fg, bg)
            if n == 5:
                clip = "left" if bool(pair_rng.integers(0, 2)) else "right"
            else:
                clip = ("none", "none", "left", "right")[int(pair_rng.integers(0, 4))]
            prior = [x for x in history if x["n"] == n]
            if split == "train" and index == 0:
                mode = "diagnostic"
            elif split == "train" and index == 1:
                mode = ("ordinary", "symmetric_tile", "symmetric_placement",
                        "both_symmetric", "diagnostic")[int(pair_rng.integers(0, 5))]
            elif split == "test" and seed % 5 == 3 and index == 0 and prior:
                mode = ("reuse_tile", "reuse_placement", "mirror_placement")[(seed // 5) % 3]
            else:
                modes = ("ordinary", "ordinary", "ordinary", "equal_count", "identical",
                         "singleton_tile", "singleton_placement", "full_tile",
                         "full_placement", "empty_placement", "reuse_tile",
                         "reuse_placement", "mirror_placement")
                mode = modes[int(pair_rng.integers(0, len(modes)))]
            for attempt in range(100):
                active_mode = ("diagnostic" if split == "train" and index == 0
                               else mode if attempt == 0 else "ordinary")
                tile, placement = _sample_mask_pair(pair_rng, n, active_mode, prior)
                if split == "train" and index == 1 and prior and any(
                    np.array_equal(tile, old["tile"]) and
                    np.array_equal(placement, old["placement"]) for old in prior
                ):
                    continue
                inp = _compose_input(tile, placement, neutral, divider,
                                     tile_marker, placement_marker, fg, bg, clip)
                signature = _value_signature(inp)
                if signature not in signatures and _grid_rows(inp) not in _SOURCE_INPUT_ROWS:
                    break
            else:
                if split == "train" and index == 0:
                    tile, placement = _canonical_diagnostic(n)
                    inp = _compose_input(tile, placement, neutral, divider,
                                         tile_marker, placement_marker, fg, bg, clip)
                    signature = _value_signature(inp)
                else:
                    for tile_cell in range(n * n):
                        for placement_cell in range(n * n + 1):
                            tile = np.zeros((n, n), dtype=bool)
                            placement = np.zeros((n, n), dtype=bool)
                            tile.flat[tile_cell] = True
                            if placement_cell < n * n:
                                placement.flat[placement_cell] = True
                            if split == "train" and index == 1 and prior and any(
                                np.array_equal(tile, old["tile"]) and
                                np.array_equal(placement, old["placement"]) for old in prior
                            ):
                                continue
                            inp = _compose_input(tile, placement, neutral, divider,
                                                 tile_marker, placement_marker, fg, bg, clip)
                            signature = _value_signature(inp)
                            if signature not in signatures and _grid_rows(inp) not in _SOURCE_INPUT_ROWS:
                                break
                        else:
                            continue
                        break
                    else:
                        raise RuntimeError("deterministic pair fallback exhausted")
            out = _construct_output(tile, placement, fg, bg)
            result[split].append({"input": inp, "output": out})
            signatures.add(signature)
            history.append({"n": n, "tile": tile, "placement": placement,
                            "foreground": fg, "background": bg})
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(all(isinstance(puzzle[k], list) for k in ("train", "test")),
             "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "invalid pair counts")
    seen_inputs = set()
    seen_pairs = set()
    metadata = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                         "grid dimensions out of range")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "invalid ARC color")
            inp_sig = _value_signature(pair["input"])
            pair_sig = (inp_sig, _value_signature(pair["output"]))
            _require(inp_sig not in seen_inputs and pair_sig not in seen_pairs,
                     "duplicate input or pair")
            _require(_grid_rows(pair["input"]) not in _SOURCE_INPUT_ROWS,
                     "source input copied")
            seen_inputs.add(inp_sig)
            seen_pairs.add(pair_sig)
            metadata[split].append(_parse_pair(pair["input"], pair["output"]))
    _validate_semantics(metadata["train"], metadata["test"])


def _predict_from_input(input_grid: np.ndarray) -> np.ndarray:
    """Independent read of the visible panels for self-test prediction."""
    n = input_grid.shape[0] - 2
    neutral = int(input_grid[0, 0])
    cuts = [c for c in range(input_grid.shape[1]) if int(input_grid[0, c]) != neutral]
    blocks = []
    for i, (left, right) in enumerate(zip([0] + [c + 1 for c in cuts],
                                          cuts + [input_grid.shape[1]])):
        segment = input_grid[1:n + 1, left:right]
        if i == 0 and segment.shape[1] == n + 1:
            blocks.append(segment[:, :n])
        elif i == 3 and segment.shape[1] == n + 1:
            blocks.append(segment[:, 1:])
        else:
            blocks.append(segment[:, 1:n + 1])
    tile = blocks[0] != neutral
    placement = blocks[1] != neutral
    fg, bg = int(blocks[2][0, 0]), int(blocks[3][0, 0])
    return np.where(np.kron(placement, tile), fg, bg).astype(np.int8)


def _puzzles_equal(a: dict, b: dict) -> bool:
    return all(np.array_equal(x[key], y[key]) for split in ("train", "test")
               for x, y in zip(a[split], b[split]) for key in ("input", "output"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    signatures = set()
    layouts = set()
    dimensions = set()
    counts = set()
    symmetric_tile = symmetric_placement = both_symmetric = 0
    equal_counts = singleton_tile = singleton_placement = 0
    full_tile = full_placement = empty_placement = 0
    shared_sizes = larger_tests = n5_train = 0
    clipped_small = complete_small = 0
    repeated_tile = repeated_placement = mirrored_placement = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        infos = []
        train_info = []
        test_info = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _require(np.array_equal(_predict_from_input(pair["input"]), pair["output"]),
                         "input-only prediction failed")
                info = _parse_pair(pair["input"], pair["output"])
                infos.append(info)
                (train_info if split == "train" else test_info).append(info)
                dimensions.add((pair["input"].shape, pair["output"].shape))
                layouts.add((info["n"], info["tile"].tobytes(), info["placement"].tobytes()))
                counts.add((info["n"], int(info["tile"].sum()), int(info["placement"].sum())))
                signatures.add((_value_signature(pair["input"]), _value_signature(pair["output"])))
                t, p = info["tile"], info["placement"]
                ts = np.array_equal(t, np.rot90(t, 2))
                ps = np.array_equal(p, np.rot90(p, 2))
                symmetric_tile += ts
                symmetric_placement += ps
                both_symmetric += ts and ps
                equal_counts += int(t.sum()) == int(p.sum())
                singleton_tile += int(t.sum()) == 1
                singleton_placement += int(p.sum()) == 1
                full_tile += t.all()
                full_placement += p.all()
                empty_placement += not p.any()
                if info["n"] < 5:
                    clipped_small += info["clip_side"] != "none"
                    complete_small += info["clip_side"] == "none"
        shared_sizes += bool({x["n"] for x in train_info} & {x["n"] for x in test_info})
        larger_tests += max(x["n"] for x in test_info) > max(x["n"] for x in train_info)
        n5_train += any(x["n"] == 5 for x in train_info)
        for i, later in enumerate(infos):
            for earlier in infos[:i]:
                if later["n"] != earlier["n"]:
                    continue
                same_t = np.array_equal(later["tile"], earlier["tile"])
                same_p = np.array_equal(later["placement"], earlier["placement"])
                repeated_tile += same_t and not same_p
                repeated_placement += same_p and not same_t
                mirrored_placement += same_t and not same_p and np.array_equal(
                    later["placement"], np.fliplr(earlier["placement"]))
    _require(len(signatures) >= 150 and len(layouts) >= 140 and len(counts) >= 40,
             "insufficient cross-seed structural variety")
    _require(len(dimensions) == 5, "missing size or clip regimes")
    _require(min(symmetric_tile, symmetric_placement, both_symmetric, equal_counts,
                 singleton_tile, singleton_placement, full_tile, full_placement,
                 empty_placement, shared_sizes, larger_tests, n5_train,
                 clipped_small, complete_small, repeated_tile,
                 repeated_placement, mirrored_placement) > 0,
             "a requested variation regime was not observed")
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(700 + ntrain + ntest, ntrain, ntest)
        _require((len(p["train"]), len(p["test"])) == (ntrain, ntest),
                 "explicit counts were not honored")
        _require(_puzzles_equal(p, generate_puzzle(700 + ntrain + ntest, ntrain, ntest)),
                 "generation is not deterministic")
    for i, (ntrain, ntest) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(900 + i, ntrain, ntest)
        _require((ntrain is None or len(p["train"]) == ntrain) and
                 (ntest is None or len(p["test"]) == ntest), "partial counts failed")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": 2.0, "num_test": 1}):
        try:
            generate_puzzle(1000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count request accepted: {kwargs}")
