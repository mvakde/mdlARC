import numpy as np


_SOURCE_PAIR_KEYS = {
    (
        (15, 13),
        (
            "0000000000000"
            "0011000000000"
            "0228000000000"
            "0328000000000"
            "0300000110000"
            "0000001200000"
            "0008000200000"
            "0088800000000"
            "0055500000000"
            "0000000000000"
            "0000000000800"
            "0008000006800"
            "0088800066800"
            "0055500004400"
            "0000000004000"
        ),
        (3, 3),
        "080888555",
    ),
    (
        (10, 13),
        (
            "0058800000000"
            "0353800000000"
            "0033000000700"
            "0000000007720"
            "0000000000220"
            "0035200000000"
            "0022000005880"
            "0032000035380"
            "0000000003300"
            "0000000000000"
        ),
        (3, 4),
        "058835380330",
    ),
    (
        (15, 15),
        (
            "002800000005900"
            "008200000775900"
            "088000000057000"
            "000000000000000"
            "000000006000000"
            "000000066000000"
            "000000333000000"
            "000000000000000"
            "000000000000000"
            "004000000000000"
            "034300000000000"
            "004000000000590"
            "000000000077590"
            "000000000005700"
            "000000000000000"
        ),
        (3, 4),
        "005977590570",
    ),
    (
        (15, 15),
        (
            "000000000000000"
            "000000000000880"
            "000555000085580"
            "000595000000000"
            "000090000000000"
            "000000038300000"
            "000000034300000"
            "000000004000000"
            "022000004000000"
            "072000000003830"
            "007700000003430"
            "000700002000400"
            "000000012000400"
            "000000011000000"
            "000000000000000"
        ),
        (4, 3),
        "383343040040",
    ),
}


def _grid_key(grid: np.ndarray) -> tuple:
    arr = np.asarray(grid, dtype=np.uint8)
    return int(arr.shape[0]), int(arr.shape[1]), arr.tobytes()


def _source_pair_key(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple:
    return (
        tuple(int(x) for x in input_grid.shape),
        "".join(str(int(x)) for x in input_grid.ravel()),
        tuple(int(x) for x in output_grid.shape),
        "".join(str(int(x)) for x in output_grid.ravel()),
    )


def _is_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    return _source_pair_key(input_grid, output_grid) in _SOURCE_PAIR_KEYS


def _clean_count(
    value: int,
    name: str,
    low: int,
    high: int,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")

    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")

    return result


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train = (
        None
        if num_train is None
        else _clean_count(num_train, "num_train", 2, 10)
    )
    test = (
        None
        if num_test is None
        else _clean_count(num_test, "num_test", 1, 4)
    )

    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))

    if train <= test:
        raise ValueError("num_train must be greater than num_test")

    return train, test



def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    """Only a palette preference and a home background persist across pairs."""
    order = tuple(int(x) for x in rng.permutation(10))
    home = int(rng.integers(10))
    pool = tuple(x for x in order if x != home)
    return {"home": home, "pool": pool, "colors": order}


def _neighbors(row: int, col: int, height: int, width: int):
    for nr, nc in ((row - 1, col), (row + 1, col),
                   (row, col - 1), (row, col + 1)):
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    height, width = mask.shape
    start = tuple(int(x) for x in cells[0])
    seen = {start}
    stack = [start]
    while stack:
        row, col = stack.pop()
        for next_cell in _neighbors(row, col, height, width):
            if mask[next_cell] and next_cell not in seen:
                seen.add(next_cell)
                stack.append(next_cell)
    return len(seen) == len(cells)


def _mask(rng: np.random.Generator, height: int, width: int,
          density: float, hole: bool) -> np.ndarray:
    """Grow a connected, tight mask from a random root and four edges."""
    for _ in range(70):
        root = (int(rng.integers(height)), int(rng.integers(width)))
        mask = np.zeros((height, width), dtype=bool)
        mask[root] = True
        anchors = [(0, int(rng.integers(width))),
                   (height - 1, int(rng.integers(width))),
                   (int(rng.integers(height)), 0),
                   (int(rng.integers(height)), width - 1)]
        for anchor in anchors:
            row, col = anchor
            mask[row, col] = True
            while (row, col) != root:
                moves = []
                if row != root[0]:
                    moves.append((row + (1 if row < root[0] else -1), col))
                if col != root[1]:
                    moves.append((row, col + (1 if col < root[1] else -1)))
                row, col = moves[int(rng.integers(len(moves)))]
                mask[row, col] = True
        maximum = height * width - int(hole)
        if int(mask.sum()) > maximum:
            continue
        desired = min(maximum, max(int(mask.sum()),
                                   int(round(height * width * density))))
        while int(mask.sum()) < desired:
            frontier = [(r, c) for r in range(height) for c in range(width)
                        if not mask[r, c] and any(mask[nr, nc] for nr, nc
                                                  in _neighbors(r, c, height, width))]
            if not frontier:
                break
            mask[frontier[int(rng.integers(len(frontier)))]] = True
        if int(mask.sum()) >= 3 and (not hole or not mask.all()):
            return mask
    raise RuntimeError("could not form a tight connected mask")


def _pattern(rng: np.random.Generator, shape: tuple[int, int],
             palette: tuple[int, ...], background: int,
             density: float, hole: bool) -> np.ndarray:
    mask = _mask(rng, *shape, density, hole)
    if int(mask.sum()) < len(palette):
        raise RuntimeError("palette exceeds motif")
    result = np.full(shape, background, dtype=np.int8)
    cells = np.argwhere(mask)
    assignments = rng.integers(len(palette), size=len(cells))
    for index, (row, col) in enumerate(cells):
        result[row, col] = palette[int(assignments[index])]
    for color, index in zip(palette, rng.permutation(len(cells))):
        row, col = cells[int(index)]
        result[row, col] = color
    return result


def _target_shape(rng: np.random.Generator, aspect: str) -> tuple[int, int]:
    if aspect == "square":
        side = int(rng.integers(3, 6))
        return side, side
    short = int(rng.integers(2, 5))
    long = int(rng.integers(short + 1, min(7, short + 3)))
    return (short, long) if aspect == "landscape" else (long, short)


def _palette(rng: np.random.Generator, params: dict,
             background: int, count: int) -> tuple[int, ...]:
    common = [x for x in params["pool"][:5] if x != background]
    available = [x for x in params["colors"] if x != background]
    if rng.random() < 0.68 and len(common) >= count:
        return tuple(int(x) for x in rng.choice(common, count, replace=False))
    return tuple(int(x) for x in rng.choice(available, count, replace=False))


def _dimensions_for_relation(rng: np.random.Generator,
                             target_area: int, relation: str) -> tuple[int, int]:
    choices = [(h, w) for h in range(2, 7) for w in range(2, 7)
               if (h * w < target_area if relation == "largest" else
                   h * w == target_area if relation == "tied" else
                   h * w > target_area)]
    if not choices:
        raise RuntimeError("no distractor dimensions")
    return choices[int(rng.integers(len(choices)))]


def _variant(rng: np.random.Generator, target: np.ndarray,
             background: int, kind: str, params: dict) -> np.ndarray:
    result = target.copy()
    cells = np.argwhere(target != background)
    if kind == "recolor":
        row, col = cells[int(rng.integers(len(cells)))]
        options = [x for x in params["colors"] if x != background
                   and x != int(result[row, col])]
        result[row, col] = int(options[int(rng.integers(len(options)))])
        return result
    if kind == "rearrange":
        if len(cells) >= 2:
            for _ in range(20):
                indices = rng.choice(len(cells), 2, replace=False)
                a = tuple(cells[int(indices[0])])
                b = tuple(cells[int(indices[1])])
                if result[a] != result[b]:
                    result[a], result[b] = result[b], result[a]
                    return result
        return _variant(rng, target, background, "recolor", params)
    # Change one foreground cell without disconnecting or loosening the crop.
    mask = target != background
    candidates = []
    for row, col in cells:
        altered = mask.copy()
        altered[row, col] = False
        if (_connected(altered) and altered[0].any() and altered[-1].any()
                and altered[:, 0].any() and altered[:, -1].any()):
            candidates.append((int(row), int(col)))
    if candidates:
        row, col = candidates[int(rng.integers(len(candidates)))]
        result[row, col] = background
        return result
    return _variant(rng, target, background, "recolor", params)


def _components(grid: np.ndarray, background: int) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    objects = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) == background:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in _neighbors(r, c, height, width):
                    if not seen[nr, nc] and int(grid[nr, nc]) != background:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            top = min(r for r, _ in cells)
            bottom = max(r for r, _ in cells)
            left = min(c for _, c in cells)
            right = max(c for _, c in cells)
            crop = grid[top:bottom + 1, left:right + 1].copy()
            objects.append({"crop": crop, "bbox": (top, left, bottom, right),
                            "cells": cells})
    return objects


def _place(rng: np.random.Generator, objects: list[np.ndarray],
           background: int, compact: bool, close: bool) -> np.ndarray:
    total = sum(obj.size for obj in objects)
    minimum = max(10, int(np.ceil(np.sqrt(total * (1.4 if compact else 2.8)))))
    for trial in range(100):
        growth = trial // 20
        if compact:
            height = min(30, int(rng.integers(minimum, min(23, minimum + 5))) + growth)
            width = min(30, int(rng.integers(minimum, min(23, minimum + 5))) + growth)
        else:
            height = min(30, int(rng.integers(max(16, minimum), 28)) + growth)
            width = min(30, int(rng.integers(max(16, minimum), 28)) + growth)
        if height < max(obj.shape[0] for obj in objects) or width < max(obj.shape[1] for obj in objects):
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        boxes = []
        order = [0, 1] + [int(x) for x in rng.permutation(np.arange(2, len(objects)))]
        success = True
        for index in order:
            obj = objects[index]
            oh, ow = obj.shape
            locations = [(r, c) for r in range(height - oh + 1)
                         for c in range(width - ow + 1)]
            rng.shuffle(locations)
            if index == 1 and boxes:
                first_r, first_c, first_h, first_w = boxes[0]
                center_r = first_r + first_h / 2
                center_c = first_c + first_w / 2
                locations.sort(key=lambda rc: abs(rc[0] + oh / 2 - center_r)
                               + abs(rc[1] + ow / 2 - center_c), reverse=not close)
            placed = False
            for top, left in locations:
                # Crops may lie close or touch diagonally but never overlap.
                if any(not (top + oh <= r or r + h <= top or
                            left + ow <= c or c + w <= left)
                       for r, c, h, w in boxes):
                    continue
                foreground = np.argwhere(obj != background)
                collision = False
                for dr, dc in foreground:
                    row, col = top + int(dr), left + int(dc)
                    if any(grid[nr, nc] != background for nr, nc in
                           _neighbors(row, col, height, width)):
                        collision = True
                        break
                if collision:
                    continue
                region = grid[top:top + oh, left:left + ow]
                region[obj != background] = obj[obj != background]
                boxes.append((top, left, oh, ow))
                placed = True
                break
            if not placed:
                success = False
                break
        if success:
            return grid
    raise RuntimeError("could not place objects")


def _make_pair(rng: np.random.Generator, params: dict, aspect: str,
               count: int, relation: str, decoy: str,
               compact: bool, close: bool, background: int,
               preferred_palette: tuple[int, ...] | None = None) -> dict:
    shape = _target_shape(rng, aspect)
    color_count = int(rng.choice([1, 2, 2, 3, 3, 3, 4]))
    palette = (preferred_palette if preferred_palette is not None else
               _palette(rng, params, background, color_count))
    target = _pattern(rng, shape, palette, background,
                      1.0 if rng.random() < 0.28 else float(rng.uniform(0.48, 0.88)),
                      bool(rng.random() < 0.50))
    objects = [target.copy(), target.copy()]
    used = {_grid_key(target)}
    if relation != "largest" and decoy != "none":
        variant = _variant(rng, target, background, decoy, params)
        if not np.array_equal(variant, target):
            objects.append(variant)
            used.add(_grid_key(variant))
    forced = False
    while len(objects) < count:
        for _ in range(100):
            if not forced:
                wanted = relation
            elif relation == "largest":
                wanted = "largest"
            elif relation == "tied":
                wanted = str(rng.choice(["largest", "tied"]))
            else:
                wanted = str(rng.choice(["largest", "tied", "smaller"]))
            dimensions = _dimensions_for_relation(rng, target.size, wanted)
            color_n = int(rng.choice([1, 2, 2, 3, 3, 4]))
            colors = palette if rng.random() < 0.42 and len(palette) <= dimensions[0] * dimensions[1] else _palette(rng, params, background, color_n)
            try:
                crop = _pattern(rng, dimensions, colors, background,
                                float(rng.uniform(0.43, 0.88)), bool(rng.random() < 0.55))
            except RuntimeError:
                continue
            key = _grid_key(crop)
            if key not in used:
                objects.append(crop)
                used.add(key)
                forced = True
                break
        else:
            raise RuntimeError("could not make unique distractor")
    grid = _place(rng, objects, background, compact, close)
    return {"input": grid, "output": target}


def _infer(pair: dict, regime: str) -> dict:
    grid = pair["input"]
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > grid.size / 2
    objects = _components(grid, background)
    assert len(objects) in ((4, 5, 6) if regime == "source_train" else (4, 5, 6, 7))
    groups = {}
    for obj in objects:
        crop = obj["crop"]
        assert np.count_nonzero(crop != background) == len(obj["cells"])
        assert _connected(crop != background)
        groups.setdefault(_grid_key(crop), []).append(obj)
    duplicates = [group for group in groups.values() if len(group) > 1]
    assert len(duplicates) == 1 and len(duplicates[0]) == 2
    answer = duplicates[0][0]["crop"]
    assert np.array_equal(answer, pair["output"])
    assert 2 <= answer.shape[0] <= 6 and 2 <= answer.shape[1] <= 6
    assert 1 <= len(np.unique(answer[answer != background])) <= 4
    singletons = [group[0]["crop"] for group in groups.values() if len(group) == 1]
    relation = ("smaller" if any(crop.size > answer.size for crop in singletons)
                else "tied" if any(crop.size == answer.size for crop in singletons)
                else "largest")
    mask_decoy = any(crop.shape == answer.shape and
                     np.array_equal(crop != background, answer != background)
                     for crop in singletons)
    boxes = [obj["bbox"] for obj in duplicates[0]]
    distance = abs(boxes[0][0] - boxes[1][0]) + abs(boxes[0][1] - boxes[1][1])
    return {"background": background, "count": len(objects),
            "aspect": "square" if answer.shape[0] == answer.shape[1] else
            "landscape" if answer.shape[0] < answer.shape[1] else "portrait",
            "relation": relation, "mask_decoy": mask_decoy,
            "distance": distance, "background_fraction": counts[background] / grid.size,
            "palette": frozenset(int(x) for x in answer[answer != background]),
            "shape": answer.shape}


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4
        assert len(puzzle["train"]) > len(puzzle["test"])
        inputs = set()
        train_info = []
        for section, regime in (("train", "source_train"), ("test", "target_test")):
            for pair in puzzle[section]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                for grid in pair.values():
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert all(1 <= side <= 30 for side in grid.shape)
                    assert bool(np.all((grid >= 0) & (grid <= 9)))
                assert pair["input"].shape[0] > pair["output"].shape[0]
                assert pair["input"].shape[1] > pair["output"].shape[1]
                assert not _is_source_pair(pair["input"], pair["output"])
                key = _grid_key(pair["input"])
                assert key not in inputs
                inputs.add(key)
                info = _infer(pair, regime)
                if section == "train":
                    train_info.append(info)
        # Both a same-mask false match and an answer whose size is not a
        # reliable selector are visible in the demonstrations.
        assert any(info["mask_decoy"] for info in train_info)
        assert any(not info["mask_decoy"] for info in train_info)
        assert {info["relation"] for info in train_info} >= {"largest", "smaller"}
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("invalid generated puzzle") from exc


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    params = sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    train_aspects = [str(x) for x in rng.permutation(["square", "landscape", "portrait"])[:2]]
    train_aspects.extend(str(rng.choice(["square", "landscape", "portrait"]))
                         for _ in range(train_count - 2))
    relations = ["smaller", "largest"] + [str(rng.choice(["largest", "tied", "smaller"]))
                                          for _ in range(train_count - 2)]
    rng.shuffle(relations)
    # Assign recolor to a pair with a larger distractor; a largest-answer
    # example necessarily has no same-sized false match.
    decoys = ["recolor" if relation == "smaller" and i == relations.index("smaller")
              else str(rng.choice(["none", "recolor", "rearrange", "outline"]))
              for i, relation in enumerate(relations)]
    for i, relation in enumerate(relations):
        if relation == "largest":
            decoys[i] = "none"
    seen = set()
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if section == "train":
                aspect = train_aspects[index]
                relation = relations[index]
                decoy = decoys[index]
                object_count = int(rng.integers(4, 7))
            else:
                unseen = [a for a in ("square", "landscape", "portrait")
                          if a not in train_aspects]
                aspect = str(rng.choice(unseen)) if unseen and rng.random() < 0.38 else str(rng.choice(["square", "landscape", "portrait"]))
                relation = str(rng.choice(["largest", "tied", "smaller"]))
                decoy = str(rng.choice(["none", "recolor", "rearrange", "outline"]))
                object_count = int(rng.integers(4, 8))
            background = params["home"] if rng.random() < 0.72 else int(rng.integers(10))
            preferred_palette = None
            if section == "test" and index == 0:
                first = puzzle["train"][0]
                background = int(np.argmax(np.bincount(first["input"].ravel(), minlength=10)))
                preferred_palette = tuple(int(x) for x in np.unique(first["output"])
                                          if int(x) != background)
            compact = bool(rng.random() < 0.52)
            close = bool(rng.random() < 0.5)
            for _ in range(35):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                try:
                    pair = _make_pair(pair_rng, params, aspect, object_count,
                                      relation, decoy, compact, close, background,
                                      preferred_palette)
                    key = _grid_key(pair["input"])
                    if key in seen or _is_source_pair(pair["input"], pair["output"]):
                        continue
                    _infer(pair, "source_train" if section == "train" else "target_test")
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
                except (RuntimeError, AssertionError):
                    continue
            else:
                raise RuntimeError("could not construct a valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle: dict) -> tuple:
    return tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                 for section in ("train", "test") for pair in puzzle[section])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_fingerprint(p) for p in puzzles}) == 42
    infos = []
    within = {"background": 0, "aspect": 0, "count": 0, "palette": 0,
              "relation": 0}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        measured = [_infer(pair, "source_train" if section == "train" else "target_test")
                    for section in ("train", "test") for pair in puzzle[section]]
        infos.extend(measured)
        for name in within:
            within[name] += len({str(info[name]) for info in measured}) > 1
    assert len({x["shape"] for x in infos}) >= 12
    assert {x["aspect"] for x in infos} == {"square", "landscape", "portrait"}
    assert {x["relation"] for x in infos} == {"largest", "tied", "smaller"}
    assert len({x["background"] for x in infos}) >= 5
    assert any(x["background_fraction"] < 0.85 for x in infos)
    assert any(x["background_fraction"] > 0.85 for x in infos)
    assert any(x["mask_decoy"] for x in infos)
    assert any(not x["mask_decoy"] for x in infos)
    assert all(within[name] >= 8 for name in within)
    for seed, ntrain, ntest in ((901, 2, 1), (902, 10, 1), (903, 10, 4),
                               (904, None, 1), (905, None, 4),
                               (906, 2, None), (907, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    assert _fingerprint(generate_puzzle(8123, 7, 3)) == _fingerprint(generate_puzzle(8123, 7, 3))
    assert _fingerprint(generate_puzzle(8123, 7, 3)) != _fingerprint(generate_puzzle(8124, 7, 3))
    for bad_train, bad_test in ((1, None), (11, None), (None, 0),
                                (None, 5), (2, 2), (3, 3), (2, 4),
                                (True, 1), (3, False)):
        try:
            generate_puzzle(999, bad_train, bad_test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
