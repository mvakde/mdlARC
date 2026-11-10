import numpy as np


_SOURCE_INPUT_TEXTS = (
    """58888888888888888888
88888888888888888888
51818181818181818181
88888888888888888888
81818181818181818181
88888888888888888888
81818181818181818181
81818181818181818181
88888888888888888888
81818181818181818181
81818181818181818181
88888888888888888888
81818181818181818181
88888888888888888888
81818181818181818181
88888888888888888888
88888888888888888888
88888888888888888888
88888888888888888888
88888888888888888888""",
    """8888888885
8787878788
8888888885
8787878788
8888888888
8888888888
8787878788
8888888888
8787878788
8888888888
8787878788
8888888888
8888888888
8787878788
8888888888
8787878788
8888888888
8888888888
8888888888
8888888888""",
    """888888888888888888888888888888
888888888888888888888888888888
883838388888888888888888888888
888888888838383888383888383838
883838388888888888888888888888
888888888838383888383888383838
883838388888888888888888888888
888888888838383888383888383838
883838388888888888888888888888
888888888838383888383888383838
883838388888888888888888888888
888888888838383888383888383838
883838388888888888888888888888
588888888888888888888888888888
888888888888888888888888888888
588888888888888888888888888888""",
    """5858888888888888888888
8888888888888888888888
8848484888888888888888
8888888888888484848888
8848484888888888888888
8888888888888484848888
8848484888888888888888
8888888888888484848888
8888888888888888888888
8888888888888888888888""",
    """89898585
88888888
89898989
88888888""",
    """8888888888888888888888888888
8888888888888888888888888888
8882828282828282828282828288
8888888888888888888888888888
8882828282828282828282828288
8888888888888888888888888888
8882828282828282828282828288
8888888888888888888888888888
8882828282828282828282828288
8888888888888888888888888585""",
)


def _decode_grid(text: str) -> np.ndarray:
    return np.asarray(
        [[int(character) for character in row] for row in text.split()],
        dtype=np.int8,
    )


_SOURCE_INPUTS = tuple(_decode_grid(text) for text in _SOURCE_INPUT_TEXTS)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _is_source_input(grid: np.ndarray) -> bool:
    return any(
        grid.shape == source.shape and np.array_equal(grid, source)
        for source in _SOURCE_INPUTS
    )


def _check_requested_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _check_requested_count("num_train", num_train, 2, 10)
    _check_requested_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(
            rng.integers(max(2, test_count + 1), 11)
        )
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(
            rng.integers(max(2, test_count + 1), 11)
        )
    elif num_test is None:
        train_count = int(num_train)
        maximum_test = min(4, train_count - 1)
        test_count = int(rng.integers(1, maximum_test + 1))
    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = tuple(int(x) for x in rng.permutation(10))
    modes = ("source_like", "mixed", "side_only", "stacked_only")
    mode = modes[int(rng.choice(4, p=(0.16, 0.38, 0.24, 0.22)))]
    return {
        "mode": mode,
        "fixed_roles": bool(rng.random() < 0.42),
        "background": colors[0],
        "marker": colors[1],
        "palette": colors,
        "multicolor": bool(rng.random() < 0.55),
    }


def _specs(count: int, rng: np.random.Generator, mode: str, split: str,
           train_specs: list[dict] | None = None) -> list[dict]:
    if split == "train":
        if mode == "source_like":
            specs = [dict(orientation="stacked", selected="bottom"),
                     dict(orientation="side", selected="left")]
            if count >= 3:
                specs.append(dict(orientation="stacked", selected="top"))
            while len(specs) < count:
                orientation = "stacked" if rng.random() < .65 else "side"
                selected = ("top" if rng.random() < .5 else "bottom") if orientation == "stacked" else "left"
                specs.append(dict(orientation=orientation, selected=selected))
        elif mode in ("side_only", "stacked_only"):
            orientation = "side" if mode == "side_only" else "stacked"
            sides = ("left", "right") if orientation == "side" else ("top", "bottom")
            specs = [dict(orientation=orientation, selected=sides[0]),
                     dict(orientation=orientation, selected=sides[1])]
            while len(specs) < count:
                specs.append(dict(orientation=orientation,
                                  selected=sides[int(rng.integers(0, 2))]))
        else:
            if rng.random() < .5:
                specs = [dict(orientation="stacked", selected="top"),
                         dict(orientation="side", selected="right")]
            else:
                specs = [dict(orientation="stacked", selected="bottom"),
                         dict(orientation="side", selected="left")]
            while len(specs) < count:
                orientation = "stacked" if rng.random() < .5 else "side"
                sides = ("top", "bottom") if orientation == "stacked" else ("left", "right")
                existing = {s["selected"] for s in specs if s["orientation"] == orientation}
                missing = [side for side in sides if side not in existing]
                selected = missing[0] if missing and len(specs) == 2 else sides[int(rng.integers(0, 2))]
                specs.append(dict(orientation=orientation, selected=selected))
        return [specs[int(i)] for i in rng.permutation(count)]

    assert train_specs is not None
    if mode == "source_like":
        return [dict(orientation="side", selected="right") for _ in range(count)]
    orientations = sorted({s["orientation"] for s in train_specs})
    specs = []
    for _ in range(count):
        orientation = orientations[int(rng.integers(0, len(orientations)))]
        sides = ("top", "bottom") if orientation == "stacked" else ("left", "right")
        train_sides = [s["selected"] for s in train_specs if s["orientation"] == orientation]
        if rng.random() < .55:
            selected = min(sides, key=lambda side: train_sides.count(side))
        else:
            selected = sides[int(rng.integers(0, 2))]
        specs.append(dict(orientation=orientation, selected=selected))
    return specs


def _size(rng: np.random.Generator, orientation: str, split: str,
          source_like: bool) -> tuple[int, int]:
    if source_like and orientation == "side":
        if split == "train":
            panel = int(rng.integers(6, 12))
        else:
            panel = int(rng.integers(3, 5)) if rng.random() < .5 else int(rng.integers(13, 16))
    else:
        probabilities = (.22, .53, .25) if split == "train" else (.26, .40, .34)
        band = int(rng.choice(3, p=probabilities))
        intervals = ((3, 5), (5, 13), (13, 16))
        low, high = intervals[band]
        panel = int(rng.integers(low, high))
    other = int(rng.integers(4, 31))
    if orientation == "stacked":
        return panel * 2, other
    return other, panel * 2


def _motif_mask(rng: np.random.Generator, height: int, width: int,
                style: str | None = None) -> np.ndarray:
    if style is None:
        style = ("row_bands", "column_bands", "checker", "runs", "field")[int(rng.integers(0, 5))]
    yy, xx = np.indices((height, width))
    if style == "row_bands":
        row_period = int(rng.integers(2, 5))
        row_phase = int(rng.integers(0, row_period))
        col_period = int(rng.integers(2, 4))
        col_phase = int(rng.integers(0, col_period))
        mask = ((yy + row_phase) % row_period == 0) & ((xx + col_phase) % col_period == 0)
        if rng.random() < .45:
            start = int(rng.integers(0, height))
            stop = int(rng.integers(start + 1, height + 1))
            mask[:start] = False
            mask[stop:] = False
    elif style == "column_bands":
        col_period = int(rng.integers(2, 5))
        col_phase = int(rng.integers(0, col_period))
        row_period = int(rng.integers(2, 4))
        row_phase = int(rng.integers(0, row_period))
        mask = ((xx + col_phase) % col_period == 0) & ((yy + row_phase) % row_period == 0)
        if rng.random() < .45:
            start = int(rng.integers(0, width))
            stop = int(rng.integers(start + 1, width + 1))
            mask[:, :start] = False
            mask[:, stop:] = False
    elif style == "checker":
        mask = (yy + xx + int(rng.integers(0, 2))) % 2 == 0
        top = int(rng.integers(0, max(1, height - 1)))
        bottom = int(rng.integers(top + 1, height + 1))
        left = int(rng.integers(0, max(1, width - 1)))
        right = int(rng.integers(left + 1, width + 1))
        mask[:top] = False
        mask[bottom:] = False
        mask[:, :left] = False
        mask[:, right:] = False
    elif style == "runs":
        mask = np.zeros((height, width), dtype=bool)
        for _ in range(int(rng.integers(2, 8))):
            if rng.random() < .5:
                row = int(rng.integers(0, height))
                left = int(rng.integers(0, width))
                right = int(rng.integers(left + 1, width + 1))
                mask[row, left:right:2] = True
            else:
                col = int(rng.integers(0, width))
                top = int(rng.integers(0, height))
                bottom = int(rng.integers(top + 1, height + 1))
                mask[top:bottom:2, col] = True
    else:
        mask = rng.random((height, width)) < float(rng.uniform(.07, .28))
    if not np.any(mask):
        mask[int(rng.integers(0, height)), int(rng.integers(0, width))] = True
    return mask


def _sparse_mask(rng: np.random.Generator, shape: tuple[int, int],
                 orientation: str, count: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if count == 0:
        return mask
    h, w = shape
    # Keep rare payload colors away from the compatible marker borders.
    positions = [(r, c) for r in range(h) for c in range(w)
                 if (0 < c < w - 1 if orientation == "stacked" else 0 < r < h - 1)]
    indices = rng.choice(len(positions), size=min(count, len(positions)), replace=False)
    for index in np.atleast_1d(indices):
        mask[positions[int(index)]] = True
    return mask


def _make_masks(rng: np.random.Generator, height: int, width: int,
                orientation: str, relationship: str,
                allow_blank_selected: bool, selected_index: int) -> np.ndarray:
    shape = (height // 2, width) if orientation == "stacked" else (height, width // 2)
    if relationship == "continuous":
        mask = _motif_mask(rng, height, width, ("row_bands", "column_bands", "checker")[int(rng.integers(0, 3))])
        return mask
    if relationship == "related":
        first = _motif_mask(rng, *shape)
        second = np.roll(first, int(rng.integers(-1, 2)), axis=int(rng.integers(0, 2))).copy()
        edits = int(rng.integers(1, max(3, first.size // 12 + 2)))
        positions = rng.choice(first.size, size=min(edits, first.size), replace=False)
        second.flat[positions] = ~second.flat[positions]
    elif relationship == "sparse":
        retained_count = int(rng.choice((0, 1, 2), p=(.20, .42, .38))) if allow_blank_selected else int(rng.integers(1, 3))
        discarded_count = int(rng.choice((0, 1, 2, 3, 5), p=(.08, .18, .25, .29, .20)))
        if retained_count + discarded_count == 0:
            discarded_count = 1
        retained = _sparse_mask(rng, shape, orientation, retained_count)
        discarded = _sparse_mask(rng, shape, orientation, discarded_count)
        first, second = (retained, discarded) if selected_index == 0 else (discarded, retained)
    elif relationship == "salient_distractor":
        retained = _sparse_mask(rng, shape, orientation, int(rng.integers(0 if allow_blank_selected else 1, 4)))
        discarded = _motif_mask(rng, *shape, style="checker")
        if np.count_nonzero(discarded) <= np.count_nonzero(retained):
            discarded = _motif_mask(rng, *shape, style="row_bands")
        first, second = (retained, discarded) if selected_index == 0 else (discarded, retained)
    else:
        first = _motif_mask(rng, *shape)
        second = _motif_mask(rng, *shape)
    return np.vstack((first, second)) if orientation == "stacked" else np.hstack((first, second))


def _marker_candidates(grid: np.ndarray) -> list[tuple[int, int, str, str, tuple[tuple[int, int], tuple[int, int]]]]:
    h, w = grid.shape
    candidates = []
    values, counts = np.unique(grid, return_counts=True)
    for color, count in zip(values, counts):
        if count != 2:
            continue
        positions = np.argwhere(grid == color)
        (r0, c0), (r1, c1) = (tuple(map(int, p)) for p in positions)
        if c0 == c1 and abs(r1 - r0) == 2 and c0 in (0, w - 1) and h % 2 == 0:
            middle = ((r0 + r1) // 2, c0)
            if (r0 < h // 2) == (r1 < h // 2):
                background = int(grid[middle])
                if background != color:
                    selected = "top" if r0 < h // 2 else "bottom"
                    candidates.append((int(color), background, "stacked", selected, ((r0, c0), (r1, c1))))
        if r0 == r1 and abs(c1 - c0) == 2 and r0 in (0, h - 1) and w % 2 == 0:
            middle = (r0, (c0 + c1) // 2)
            if (c0 < w // 2) == (c1 < w // 2):
                background = int(grid[middle])
                if background != color:
                    selected = "left" if c0 < w // 2 else "right"
                    candidates.append((int(color), background, "side", selected, ((r0, c0), (r1, c1))))
    return candidates


def _crop(grid: np.ndarray, candidate: tuple) -> np.ndarray:
    marker, background, orientation, selected, _ = candidate
    h, w = grid.shape
    if orientation == "stacked":
        output = grid[:h // 2, :].copy() if selected == "top" else grid[h // 2:, :].copy()
    else:
        output = grid[:, :w // 2].copy() if selected == "left" else grid[:, w // 2:].copy()
    output[output == marker] = background
    return output


def _generate_pair(rng: np.random.Generator, spec: dict, split: str,
                   parameters: dict, multicolor: bool, relationship: str,
                   forced_payload: int | None = None) -> dict:
    orientation, selected = spec["orientation"], spec["selected"]
    height, width = _size(rng, orientation, split, parameters["mode"] == "source_like")
    selected_index = 0 if selected in ("top", "left") else 1
    if parameters["fixed_roles"]:
        background, marker = parameters["background"], parameters["marker"]
    else:
        background, marker = (int(x) for x in rng.choice(10, size=2, replace=False))
    available = [color for color in parameters["palette"] if color not in (background, marker)]
    if forced_payload in available:
        payloads = [forced_payload]
        if multicolor:
            payloads.append(int(rng.choice([x for x in available if x != forced_payload])))
    else:
        payloads = [int(x) for x in rng.choice(available, size=2 if multicolor else 1, replace=False)]
    mask = _make_masks(rng, height, width, orientation, relationship,
                       split == "test" or relationship == "sparse", selected_index)
    grid = np.full((height, width), background, dtype=np.int8)
    grid[mask] = payloads[0]
    if orientation == "stacked":
        panel = height // 2
        local = int(rng.integers(0, panel - 2))
        col = 0 if rng.random() < .5 else width - 1
        start = selected_index * panel + local
        line = [(start + offset, col) for offset in range(3)]
    else:
        panel = width // 2
        local = int(rng.integers(0, panel - 2))
        row = 0 if rng.random() < .5 else height - 1
        start = selected_index * panel + local
        line = [(row, start + offset) for offset in range(3)]
    for r, c in line:
        grid[r, c] = background
    grid[line[0]] = marker
    grid[line[2]] = marker
    # A sparse draw may otherwise lose its only payload under the marker.
    if not np.any(np.isin(grid, payloads)):
        candidates = [(r, c) for r in range(height) for c in range(width)
                      if (r, c) not in line and grid[r, c] == background
                      and (0 < c < width - 1 if orientation == "stacked" else 0 < r < height - 1)]
        grid[candidates[int(rng.integers(0, len(candidates)))]] = payloads[0]
    if multicolor:
        if orientation == "stacked":
            retained = grid[:height // 2] if selected_index == 0 else grid[height // 2:]
        else:
            retained = grid[:, :width // 2] if selected_index == 0 else grid[:, width // 2:]
        retained_cells = np.argwhere(retained == payloads[0])
        if len(retained_cells) < 2:
            available_cells = np.argwhere(retained == background)
            for r, c in available_cells:
                if len(retained_cells) >= 2:
                    break
                if orientation == "stacked" and c in (0, width - 1):
                    continue
                if orientation == "side" and r in (0, height - 1):
                    continue
                retained[r, c] = payloads[0]
                retained_cells = np.argwhere(retained == payloads[0])
        cells = np.argwhere(grid == payloads[0])
        if len(cells) >= 2:
            chosen = rng.choice(len(cells), size=max(1, len(cells) // 3), replace=False)
            for index in np.atleast_1d(chosen):
                r, c = cells[int(index)]
                grid[r, c] = payloads[1]
        if not np.any(retained == payloads[1]):
            retained_cells = np.argwhere(retained == payloads[0])
            retained[tuple(retained_cells[0])] = payloads[1]
    candidates = _marker_candidates(grid)
    if len(candidates) != 1 or candidates[0][:4] != (marker, background, orientation, selected):
        raise AssertionError("ambiguous marker in sampled field")
    output = _crop(grid, candidates[0])
    return {"input": grid, "output": output}


def _array_key(array: np.ndarray) -> tuple:
    return array.shape, array.tobytes()


def _analyze_pair(pair: dict) -> dict:
    grid, output = pair["input"], pair["output"]
    candidates = _marker_candidates(grid)
    _require(len(candidates) == 1, "marker crop must be unique from the input")
    marker, background, orientation, selected, positions = candidates[0]
    expected = _crop(grid, candidates[0])
    _require(np.array_equal(expected, output), "output is not the exact marker-selected crop")
    colors = {int(x) for x in np.unique(grid)}
    payloads = colors - {marker, background}
    _require(1 <= len(payloads) <= 2, "expected one or two payload colors")
    counts = {int(color): int(np.count_nonzero(grid == color)) for color in colors}
    _require(counts[background] > max(counts[color] for color in colors if color != background),
             "background must be visually dominant")
    _require(not np.any(output == marker), "marker remains in output")
    h, w = grid.shape
    if orientation == "stacked":
        half = h // 2
        retained = grid[:half] if selected == "top" else grid[half:]
        discarded = grid[half:] if selected == "top" else grid[:half]
        edge = "left" if positions[0][1] == 0 else "right"
    else:
        half = w // 2
        retained = grid[:, :half] if selected == "left" else grid[:, half:]
        discarded = grid[:, half:] if selected == "left" else grid[:, :half]
        edge = "top" if positions[0][0] == 0 else "bottom"
    output_payloads = {int(x) for x in np.unique(output)} - {background}
    return {"orientation": orientation, "selected": selected, "edge": edge,
            "background": background, "marker": marker, "payloads": payloads,
            "output_payloads": output_payloads, "panel_shape": output.shape,
            "retained_count": int(np.count_nonzero(output != background)),
            "discarded_count": int(np.count_nonzero((discarded != background) & (discarded != marker))),
            "halves_equal": bool(np.array_equal(output, discarded))}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[k], list) for k in ("train", "test")), "split lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    seen_inputs = set()
    seen_pairs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid array")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "grid size")
                _require(np.issubdtype(grid.dtype, np.integer), "integer grid")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "ARC colors")
            _require(not _is_source_input(pair["input"]), "source copy")
            ik = _array_key(pair["input"])
            pk = (ik, _array_key(pair["output"]))
            _require(ik not in seen_inputs and pk not in seen_pairs, "duplicate pair")
            seen_inputs.add(ik)
            seen_pairs.add(pk)
            analyses[split].append(_analyze_pair(pair))
    train, tests = analyses["train"], analyses["test"]
    _require({i["selected"] in ("bottom", "right") for i in train} == {False, True},
             "training must show marker location can change retained half")
    _require({i["orientation"] for i in tests} <= {i["orientation"] for i in train},
             "test axis was not demonstrated in training")
    _require(any(i["retained_count"] >= 3 for i in train), "training lacks payload copying")
    if any(len(i["payloads"]) == 2 for i in tests):
        _require(any(len(i["output_payloads"]) == 2 for i in train),
                 "two-color pass-through absent from training")
    # Color roles can remain fixed or be relabeled per pair. Each input exposes
    # its roles locally through a dominant background and a unique marker pair.


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    mode = parameters["mode"]
    train_specs = _specs(train_count, rng, mode, "train")
    test_specs = _specs(test_count, rng, mode, "test", train_specs)
    puzzle = {"train": [], "test": []}
    used = set()
    train_payloads = []
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for index, spec in enumerate(specs):
            if split == "train" and index == 0:
                relationship = "continuous"
            elif split == "train" and index == 1:
                relationship = "related"
            else:
                relationship = ("continuous", "related", "independent", "sparse", "salient_distractor")[int(rng.choice(5, p=(.25, .19, .20, .20, .16)))]
            # First training example demonstrates two-color copying when the
            # puzzle uses this extension. Sparse cases remain possible later.
            multicolor = parameters["multicolor"] and (split == "train" and index == 0 or rng.random() < .35)
            forced_payload = (int(rng.choice(train_payloads)) if split == "test" and train_payloads and rng.random() < .55 else None)
            for _ in range(48):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                try:
                    pair = _generate_pair(pair_rng, spec, split, parameters, multicolor,
                                          relationship, forced_payload)
                    info = _analyze_pair(pair)
                    if split == "train" and index == 0 and info["retained_count"] < 3:
                        continue
                except AssertionError:
                    continue
                key = _array_key(pair["input"])
                if key not in used and not _is_source_input(pair["input"]):
                    break
            else:
                raise RuntimeError("could not make a unique, unambiguous pair")
            puzzle[split].append(pair)
            used.add(key)
            if split == "train":
                train_payloads.extend(_analyze_pair(pair)["payloads"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple((split, tuple((_array_key(p["input"]), _array_key(p["output"]))
                                for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    signatures = {_puzzle_signature(p) for p in automatic}
    _require(len(signatures) == 42, "seed diversity")
    _require(len({len(p["train"]) for p in automatic}) >= 4, "train-count diversity")
    _require(len({len(p["test"]) for p in automatic}) >= 3, "test-count diversity")
    features = {"train": [], "test": []}
    shapes = set()
    role_pairs = set()
    coverage = {"same_axis": 0, "role_relabel": 0, "fixed_roles": 0,
                "recurring_payload": 0, "new_payload": 0,
                "size_inside": 0, "size_outside": 0,
                "within_train_size_variation": 0,
                "related": 0, "different": 0, "repeated_rows": 0,
                "left_to_right": 0, "right_to_left": 0,
                "top_to_bottom": 0, "bottom_to_top": 0,
                "multi_salient": 0, "source_like": 0}
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        train_info = [_analyze_pair(pair) for pair in puzzle["train"]]
        test_info = [_analyze_pair(pair) for pair in puzzle["test"]]
        coverage["same_axis"] += (len({i["orientation"] for i in train_info}) ==
                                  len({i["orientation"] for i in test_info}) == 1)
        roles = {(i["background"], i["marker"]) for i in train_info + test_info}
        coverage["role_relabel"] += len(roles) > 1
        coverage["fixed_roles"] += len(roles) == 1
        coverage["within_train_size_variation"] += len({i["panel_shape"] for i in train_info}) > 1
        train_payloads = set().union(*(i["payloads"] for i in train_info))
        coverage["recurring_payload"] += any(i["payloads"] & train_payloads for i in test_info)
        coverage["new_payload"] += any(i["payloads"] - train_payloads for i in test_info)
        train_sides = {i["selected"] for i in train_info}
        test_sides = {i["selected"] for i in test_info}
        coverage["left_to_right"] += "left" in train_sides and "right" in test_sides
        coverage["right_to_left"] += "right" in train_sides and "left" in test_sides
        coverage["top_to_bottom"] += "top" in train_sides and "bottom" in test_sides
        coverage["bottom_to_top"] += "bottom" in train_sides and "top" in test_sides
        coverage["source_like"] += ("right" not in train_sides and
                                    any(i["orientation"] == "stacked" for i in train_info) and
                                    all(i["selected"] == "right" for i in test_info))
        for item in test_info:
            axis = item["orientation"]
            dimension = 1 if axis == "side" else 0
            sizes = [i["panel_shape"][dimension] for i in train_info
                     if i["orientation"] == axis]
            panel = item["panel_shape"][dimension]
            coverage["size_inside"] += min(sizes) <= panel <= max(sizes)
            coverage["size_outside"] += panel < min(sizes) or panel > max(sizes)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                # Independent reconstruction uses only the input's unique marker.
                _require(np.array_equal(_crop(pair["input"], _marker_candidates(pair["input"])[0]),
                                        pair["output"]), "crop reconstruction")
                features[split].append(info)
                shapes.add((pair["input"].shape, pair["output"].shape))
                role_pairs.add((info["background"], info["marker"]))
                coverage["multi_salient"] += (len(info["payloads"]) == 2 and
                                               info["discarded_count"] > info["retained_count"])
                clean = pair["input"].copy()
                clean[clean == info["marker"]] = info["background"]
                h, w = clean.shape
                first, second = ((clean[:h // 2], clean[h // 2:])
                                 if info["orientation"] == "stacked" else
                                 (clean[:, :w // 2], clean[:, w // 2:]))
                mismatch = np.count_nonzero(first != second) / first.size
                coverage["related"] += 0 < mismatch <= .15
                coverage["different"] += mismatch > .15
                rows = [tuple(row != info["background"]) for row in clean]
                coverage["repeated_rows"] += any(rows.count(row) >= 3 and sum(row) >= 3
                                                 for row in set(rows))
    _require(len(shapes) >= 70 and len(role_pairs) >= 18, "structural or role diversity")
    _require(coverage["same_axis"] >= 5, "same-orientation train/test sets absent")
    _require(coverage["role_relabel"] >= 8 and coverage["fixed_roles"] >= 8,
             "color-role regimes are narrow")
    _require(coverage["recurring_payload"] >= 8 and coverage["new_payload"] >= 8,
             "payload recurrence and novelty are narrow")
    _require(coverage["size_inside"] >= 8 and coverage["size_outside"] >= 8,
             "size overlap and extrapolation are narrow")
    _require(coverage["within_train_size_variation"] >= 36,
             "within-puzzle panel sizes are too uniform")
    _require(coverage["related"] >= 30 and coverage["different"] >= 30 and
             coverage["repeated_rows"] >= 20, "motif relationships are narrow")
    _require(all(coverage[direction] >= 4 for direction in
                 ("left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top")),
             "retained-side shifts are narrow")
    _require(coverage["source_like"] >= 2 and coverage["multi_salient"] >= 5,
             "source-like or combined cases are absent")
    for split in ("train", "test"):
        infos = features[split]
        _require({i["selected"] for i in infos} == {"top", "bottom", "left", "right"},
                 f"{split} missing a retained side")
        _require({i["orientation"] for i in infos} == {"stacked", "side"}, f"{split} axis")
        _require(any(i["retained_count"] <= 2 for i in infos), f"{split} lacks sparse output")
        _require(all(any(i["retained_count"] == count for i in infos) for count in (0, 1, 2)),
                 f"{split} lacks blank, one-cell, or two-cell outputs")
        _require(any(i["discarded_count"] == 0 for i in infos), f"{split} lacks blank distractor")
        _require(any(i["discarded_count"] > i["retained_count"] for i in infos), f"{split} lacks salient distractor")
        _require(any(len(i["payloads"]) == 2 for i in infos), f"{split} lacks two-color fields")
        for orientation in ("stacked", "side"):
            _require(len({i["panel_shape"] for i in infos if i["orientation"] == orientation}) >= 12,
                     f"{split} size diversity")
            dimension = 1 if orientation == "side" else 0
            bands = {"compact" if i["panel_shape"][dimension] <= 4 else
                     "middle" if i["panel_shape"][dimension] <= 12 else "wide"
                     for i in infos if i["orientation"] == orientation}
            _require(bands == {"compact", "middle", "wide"}, f"{split} lacks size band")
    _require(any(i["retained_count"] == 0 for i in features["test"]), "blank retained test absent")
    explicit = generate_puzzle(91001, 7, 3)
    _require(_puzzle_signature(explicit) == _puzzle_signature(generate_puzzle(91001, 7, 3)),
             "nondeterminism")
    _require(_puzzle_signature(explicit) != _puzzle_signature(generate_puzzle(91002, 7, 3)),
             "different seeds matched")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(92000 + train_count * 10 + test_count, train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count),
                 "explicit counts")
    for index, (train_count, test_count) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(93000 + index, train_count, test_count)
        _require(train_count is None or len(puzzle["train"]) == train_count, "partial train count")
        _require(test_count is None or len(puzzle["test"]) == test_count, "partial test count")
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(94000, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
