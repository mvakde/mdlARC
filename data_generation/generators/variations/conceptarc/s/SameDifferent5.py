import numpy as np
from collections import Counter


_SOURCE_INPUTS = {
    """00000000000
02220333000
02420313000
02220333000
00000000000
00066600777
00064600787
00066600777
00000000000
00000000000
00000000000""",
    """00000000044440
03333000045540
03223000045540
03223000044440
03333000000000
00000008888000
00000008778000
00000008778000
00000008888000
00999900000000
00944900111100
00944900122100
00999900122100
00000000111100""",
    """0000000000
0000000000
0030000000
0373005550
0373005750
0333005550
0000000000
0033330000
0034430000
0033330000""",
    """0000000444
0111000484
1333100444
0111000000
0000000000
0000000000
0666666000
0688886000
0688886000
0666666000""",
    """00000000000000
07777700333330
07000700300030
07020700308030
07000700300030
07777700333330
00000000000000
00000000000000
04444400111110
04000400100010
04020400104010
04000400100010
04444400111110
00000000000000""",
    """00000000000000
00000000222000
00000000252000
00003000222000
00035300000000
00355530000000
00035300700000
00003007470000
00000074447000
00000007470555
00000000700525
00000000000555""",
}


def _key(grid: np.ndarray) -> str:
    return "\n".join("".join(str(int(v)) for v in row) for row in grid)


def _components(grid: np.ndarray, background: int) -> list[tuple[int, set[tuple[int, int]]]]:
    height, width = grid.shape
    unseen = {(r, c) for r in range(height) for c in range(width)
              if int(grid[r, c]) != background}
    components = []
    while unseen:
        first = min(unseen)
        color = int(grid[first])
        cells = {first}
        pending = [first]
        unseen.remove(first)
        while pending:
            row, col = pending.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    near = (row + dr, col + dc)
                    if near in unseen and int(grid[near]) == color:
                        unseen.remove(near)
                        cells.add(near)
                        pending.append(near)
        components.append((color, cells))
    return components


def _connected_four(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    seen = {min(cells)}
    pending = list(seen)
    while pending:
        r, c = pending.pop()
        for near in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if near in cells and near not in seen:
                seen.add(near)
                pending.append(near)
    return seen == cells


def _masks(spec: tuple[str, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    kind, height, width, turns = spec
    outside = np.zeros((height, width), dtype=bool)
    inside = np.zeros_like(outside)
    if kind == "rectangle":
        outside[0, :] = outside[-1, :] = True
        outside[:, 0] = outside[:, -1] = True
        inside[1:-1, 1:-1] = True
    elif kind == "diamond":
        rows, cols = np.indices((height, width))
        radius = height // 2
        distance = abs(rows - radius) + abs(cols - radius)
        outside = distance == radius
        inside = distance < radius
    elif kind == "pointed":
        outside[0, width // 2] = True
        outside[1:-1, 0] = outside[1:-1, -1] = True
        outside[-1, :] = True
        inside[1:-1, width // 2] = True
    elif kind == "capped":
        outside[0, 1:-1] = outside[-1, 1:-1] = True
        outside[1:-1, 0] = outside[1:-1, -1] = True
        inside[1:-1, 1:-1] = True
    else:
        raise AssertionError(kind)
    return np.rot90(outside, turns), np.rot90(inside, turns)


def _shape_candidates(height: int, width: int):
    yield ("rectangle", height, width, 0)
    if height == width and height in (5, 7, 9):
        yield ("diamond", height, width, 0)
    for length in range(3, 7):
        for turns in range(4):
            spec = ("pointed", length, 3, turns)
            if _masks(spec)[0].shape == (height, width):
                yield spec
    for cap_height in range(3, 6):
        for cap_width in range(3, 9):
            for turns in (0, 1):
                spec = ("capped", cap_height, cap_width, turns)
                if _masks(spec)[0].shape == (height, width):
                    yield spec


def _analyze_input(grid: np.ndarray) -> tuple[int, list[dict]]:
    background = int(grid[0, 0])
    components = _components(grid, background)
    objects = []
    coverage: dict[tuple[int, int], int] = {}
    for color, cells in components:
        top = min(r for r, _ in cells)
        bottom = max(r for r, _ in cells) + 1
        left = min(c for _, c in cells)
        right = max(c for _, c in cells) + 1
        height, width = bottom - top, right - left
        if height < 3 or width < 3:
            continue
        local = {(r - top, c - left) for r, c in cells}
        block = grid[top:bottom, left:right]
        for spec in _shape_candidates(height, width):
            outline_mask, interior_mask = _masks(spec)
            outline_cells = set(zip(*np.where(outline_mask)))
            if local != outline_cells:
                continue
            interior = block[interior_mask]
            accents = {int(v) for v in interior if int(v) not in (background, color)}
            if len(accents) != 1:
                continue
            accent = accents.pop()
            if any(int(v) not in (background, accent) for v in interior):
                continue
            accent_cells = {(top + int(r), left + int(c))
                            for r, c in zip(*np.where(interior_mask & (block == accent)))}
            if not _connected_four(accent_cells):
                continue
            all_cells = cells | accent_cells
            for point in all_cells:
                coverage[point] = coverage.get(point, 0) + 1
            objects.append({
                "kind": spec[0], "height": height, "width": width,
                "outline": color, "accent": accent,
                "accent_area": len(accent_cells),
                "capacity": int(np.count_nonzero(interior_mask)),
                "interior_zeros": int(np.count_nonzero(block[interior_mask] == background)),
                "origin": (top, left), "cells": all_cells,
            })
            break
    occupied = {(int(r), int(c)) for r, c in zip(*np.where(grid != background))}
    assert set(coverage) == occupied and all(v == 1 for v in coverage.values())
    assert 3 <= len(objects) <= 5
    counts = Counter(obj["accent"] for obj in objects)
    repeated = [color for color, count in counts.items() if count == 2]
    assert len(repeated) == 1 and all(count in (1, 2) for count in counts.values())
    return repeated[0], objects


def _connected_patch(allowed: np.ndarray, size: int,
                     rng: np.random.Generator) -> set[tuple[int, int]]:
    points = set(zip(*np.where(allowed)))
    assert 1 <= size <= len(points)
    start = sorted(points)[int(rng.integers(len(points)))]
    chosen = {start}
    while len(chosen) < size:
        frontier = {near for r, c in chosen
                    for near in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))}
        options = sorted((frontier & points) - chosen)
        assert options
        chosen.add(options[int(rng.integers(len(options)))])
    return chosen


def _sample_spec(rng: np.random.Generator, regime: str,
                 composition: str, minimum_area: int = 1):
    for _ in range(80):
        if composition == "hollow_only":
            kind = "rectangle"
        else:
            kinds = ("rectangle", "rectangle", "rectangle", "diamond",
                     "pointed", "capped")
            kind = kinds[int(rng.integers(len(kinds)))]
        if kind == "rectangle":
            if regime == "source_train" or composition == "compact":
                h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
            elif composition == "hollow_only":
                h, w = int(rng.integers(5, 10)), int(rng.integers(5, 10))
            else:
                h, w = int(rng.integers(3, 11)), int(rng.integers(3, 11))
            spec = (kind, h, w, 0)
        elif kind == "diamond":
            side = 5 if regime == "source_train" or composition == "compact" else (5, 7, 9)[int(rng.integers(3))]
            spec = (kind, side, side, 0)
        elif kind == "pointed":
            length = int(rng.integers(3, 6 if regime == "source_train" or composition == "compact" else 7))
            spec = (kind, length, 3, int(rng.integers(4)))
        else:
            cap_height = int(rng.integers(3, 6))
            cap_width = int(rng.integers(3, 6 if regime == "source_train" or composition == "compact" else 9))
            spec = (kind, cap_height, cap_width, int(rng.integers(2)))
        outline, inside = _masks(spec)
        if regime == "source_train" and max(outline.shape) > 5:
            continue
        hollow = bool(kind in ("rectangle", "diamond") and rng.random() <
                      (0.75 if composition == "hollow_only" else 0.24))
        allowed = inside.copy()
        if hollow:
            if kind == "rectangle":
                allowed[:2, :] = allowed[-2:, :] = False
                allowed[:, :2] = allowed[:, -2:] = False
            else:
                rows, cols = np.indices(allowed.shape)
                radius = allowed.shape[0] // 2
                allowed &= abs(rows - radius) + abs(cols - radius) <= radius - 2
        if np.count_nonzero(allowed) >= minimum_area:
            return spec, allowed
    # Deterministic compact fallback, also used by witness construction.
    spec = ("rectangle", 5, 5, 0)
    return spec, _masks(spec)[1]


def _paint_object(spec, allowed: np.ndarray, outline_color: int,
                  accent_color: int, area: int,
                  rng: np.random.Generator) -> np.ndarray:
    outline_mask, _ = _masks(spec)
    obj = np.full(outline_mask.shape, -1, dtype=np.int8)
    obj[outline_mask] = outline_color
    for r, c in _connected_patch(allowed, area, rng):
        obj[r, c] = accent_color
    return obj


def _placement_property(placements: list[tuple[int, int, np.ndarray]]) -> bool:
    boxes = [(r, r + obj.shape[0], c, c + obj.shape[1])
             for r, c, obj in placements]
    # Interleaved spans in both axes defeat the old row/column lane parser.
    return any(max(a[0], b[0]) < min(a[1], b[1]) and
               max(a[2], b[2]) < min(a[3], b[3])
               for i, a in enumerate(boxes) for b in boxes[i + 1:])


def _arrange_objects(objects: list[np.ndarray], rng: np.random.Generator,
                     seek_interleaved: bool) -> np.ndarray:
    area = sum((obj.shape[0] + 1) * (obj.shape[1] + 1) for obj in objects)
    scale = float(rng.uniform(1.25, 2.05))
    aspect = float(rng.uniform(0.75, 1.35))
    height = min(28, max(max(obj.shape[0] for obj in objects) + 2,
                         int(np.ceil(np.sqrt(area * scale * aspect)))))
    width = min(28, max(max(obj.shape[1] for obj in objects) + 2,
                        int(np.ceil(area * scale / height))))
    for restart in range(40):
        board_h = min(28, height + restart // 8)
        board_w = min(28, width + restart // 8)
        occupied = np.zeros((board_h, board_w), dtype=bool)
        placements = []
        order = sorted(range(len(objects)), key=lambda i: -np.count_nonzero(objects[i] >= 0))
        success = True
        for index in order:
            obj = objects[index]
            h, w = obj.shape
            blocked = np.zeros_like(occupied)
            for y, x in zip(*np.where(occupied)):
                blocked[max(0, int(y) - 1):min(board_h, int(y) + 2),
                        max(0, int(x) - 1):min(board_w, int(x) + 2)] = True
            options = [(r, c) for r in range(board_h - h + 1)
                       for c in range(board_w - w + 1)]
            rng.shuffle(options)
            found = False
            for r, c in options:
                mask = obj >= 0
                if np.any(blocked[r:r + h, c:c + w] & mask):
                    continue
                occupied[r:r + h, c:c + w] |= mask
                placements.append((r, c, obj))
                found = True
                break
            if not found:
                success = False
                break
        if not success or (seek_interleaved and not _placement_property(placements)):
            continue
        ys, xs = np.where(occupied)
        top, bottom = int(min(ys)), int(max(ys)) + 1
        left, right = int(min(xs)), int(max(xs)) + 1
        pad_top = int(rng.integers(3))
        pad_bottom = int(rng.integers(3))
        pad_left = int(rng.integers(3))
        pad_right = int(rng.integers(3))
        if pad_top == pad_left == 0 and occupied[top, left]:
            pad_top = 1
        pad_top = min(pad_top, 30 - (bottom - top))
        pad_bottom = min(pad_bottom, 30 - (bottom - top) - pad_top)
        pad_left = min(pad_left, 30 - (right - left) - 1)
        pad_right = min(pad_right, 30 - (right - left) - pad_left)
        canvas = np.full((bottom - top + pad_top + pad_bottom,
                          right - left + pad_left + pad_right), -1, dtype=np.int8)
        for r, c, obj in placements:
            y, x = r - top + pad_top, c - left + pad_left
            view = canvas[y:y + obj.shape[0], x:x + obj.shape[1]]
            mask = obj >= 0
            assert not np.any(view[mask] >= 0)
            view[mask] = obj[mask]
        return canvas
    raise RuntimeError("Object placement failed")


def _generate_pair(rng: np.random.Generator, regime: str,
                   background: int, answer: int, witness: str | None,
                   force_role_exchange: bool, force_outline_repeat: bool,
                   seek_interleaved: bool,
                   forbid_role_exchange: bool) -> tuple[np.ndarray, np.ndarray]:
    is_train = regime == "source_train"
    count = int(rng.integers(3, 6 if is_train else 5))
    if is_train:
        composition = "compact"
    else:
        composition = ("compact", "hollow_only", "mixed", "mixed", "mixed")[int(rng.integers(5))]
    available = [color for color in range(10) if color != background]
    other_accents = [color for color in available if color != answer]
    rng.shuffle(other_accents)
    accents = [answer, answer] + other_accents[:count - 2]
    rng.shuffle(accents)
    repeated = [i for i, accent in enumerate(accents) if accent == answer]
    singleton = [i for i, accent in enumerate(accents) if accent != answer]
    outlines = [int(rng.choice([c for c in available
                               if c != accent and
                               (not forbid_role_exchange or c != answer)]))
                for accent in accents]
    if force_outline_repeat:
        chosen = list(rng.choice(count, size=2, replace=False))
        shared = int(rng.choice([c for c in available
                                 if c not in (accents[chosen[0]], accents[chosen[1]])
                                 and (not forbid_role_exchange or c != answer)]))
        outlines[chosen[0]] = outlines[chosen[1]] = shared
    if force_role_exchange:
        chosen = singleton[int(rng.integers(len(singleton)))]
        outlines[chosen] = answer
    specs = []
    for i in range(count):
        minimum = 1
        if witness == "unequal" and i == repeated[1]:
            minimum = 3
        if witness == "dominance" and i == singleton[0]:
            minimum = 5
        if witness == "dominance" and i == singleton[0]:
            spec, allowed = ("rectangle", 5, 5, 0), _masks(("rectangle", 5, 5, 0))[1]
        else:
            spec, allowed = _sample_spec(rng, regime, composition, minimum)
        specs.append((spec, allowed))
    areas = []
    for i, (_, allowed) in enumerate(specs):
        capacity = int(np.count_nonzero(allowed))
        if witness == "unequal" and i == repeated[0]:
            area = 1
        elif witness == "unequal" and i == repeated[1]:
            area = min(capacity, int(rng.integers(2, 5)))
        elif witness == "dominance" and i in repeated:
            area = 1
        elif witness == "dominance" and i == singleton[0]:
            area = capacity
        else:
            mode = int(rng.integers(5))
            if mode == 0:
                area = 1
            elif mode == 1:
                area = capacity
            else:
                area = int(rng.integers(1, capacity + 1))
        areas.append(area)
    objects = [_paint_object(spec, allowed, outlines[i], accents[i], areas[i], rng)
               for i, (spec, allowed) in enumerate(specs)]
    raw = _arrange_objects(objects, rng, seek_interleaved)
    input_grid = np.where(raw < 0, background, raw).astype(np.int8)
    inferred, _ = _analyze_input(input_grid)
    assert inferred == answer
    return input_grid, np.array([[inferred]], dtype=np.int8)


def _check_count(value: int | None, low: int, high: int, name: str) -> None:
    if value is not None and (isinstance(value, bool) or
                              not isinstance(value, (int, np.integer)) or
                              not low <= value <= high):
        raise ValueError(name)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    _check_count(num_train, 2, 10, "num_train")
    _check_count(num_test, 1, 4, "num_test")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    # The source rule has no hidden puzzle-wide color convention.
    unequal_at, dominant_at = (0, 1) if int(rng.integers(2)) else (1, 0)
    test_only_role_exchange = bool(rng.random() < 0.25)
    required_test_role_index = int(rng.integers(num_test))
    used = set()
    puzzle = {"train": [], "test": []}
    for section, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            regime = "source_train" if section == "train" else "target_test"
            witness = ("unequal" if index == unequal_at else
                       "dominance" if index == dominant_at else None) if section == "train" else None
            for attempt in range(30):
                background = 0 if rng.random() < 0.48 else int(rng.integers(1, 10))
                available = [c for c in range(10) if c != background]
                answer = int(rng.choice(available))
                if section == "train" and index == 1 and answer == int(puzzle["train"][0]["output"][0, 0]):
                    continue
                if test_only_role_exchange:
                    role_exchange = (section == "test" and
                                     (index == required_test_role_index or rng.random() < 0.28))
                else:
                    role_exchange = bool(rng.random() < 0.28)
                outline_repeat = bool(rng.random() < 0.31)
                interleaved = bool(rng.random() < 0.22)
                pair_seed = int(rng.integers(0, 2**63))
                try:
                    input_grid, output_grid = _generate_pair(
                        np.random.default_rng(pair_seed), regime, background,
                        answer, witness, role_exchange, outline_repeat, interleaved,
                        test_only_role_exchange and section == "train")
                except (AssertionError, RuntimeError):
                    continue
                key = _key(input_grid)
                if key in used or key in _SOURCE_INPUTS:
                    continue
                used.add(key)
                puzzle[section].append({"input": input_grid, "output": output_grid})
                break
            else:
                raise RuntimeError("Could not generate pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used = set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_grid, output_grid = pair["input"], pair["output"]
            key = _key(input_grid)
            assert key not in used and key not in _SOURCE_INPUTS
            used.add(key)
            answer, objects = _analyze_input(input_grid)
            assert output_grid.shape == (1, 1) and int(output_grid[0, 0]) == answer
            if section == "train":
                assert 3 <= len(objects) <= 5
                assert all(max(obj["height"], obj["width"]) <= 5 for obj in objects)
            else:
                assert 3 <= len(objects) <= 4
                assert all(max(obj["height"], obj["width"]) <= 10 for obj in objects)
            analyses[section].append((answer, objects))
    assert len({answer for answer, _ in analyses["train"]}) >= 2
    assert any(len({obj["accent_area"] for obj in objects if obj["accent"] == answer}) > 1
               for answer, objects in analyses["train"])
    assert any(any(obj["accent_area"] > sum(other["accent_area"] for other in objects
                                                   if other["accent"] == answer)
                   for obj in objects if obj["accent"] != answer)
               for answer, objects in analyses["train"])


def _same_puzzle(a: dict, b: dict) -> bool:
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    first_inputs, structures, backgrounds, kinds, test_compositions = set(), set(), set(), set(), set()
    interleaved = 0
    role_train = role_test = 0
    large_accents = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first_inputs.add(_key(puzzle["train"][0]["input"]))
        for section in ("train", "test"):
            for pair in puzzle[section]:
                answer, objects = _analyze_input(pair["input"])
                assert int(pair["output"][0, 0]) == answer
                backgrounds.add(int(pair["input"][0, 0]))
                kinds.update(obj["kind"] for obj in objects)
                structures.add((pair["input"].shape,
                                tuple(sorted((obj["kind"], obj["height"], obj["width"])
                                             for obj in objects))))
                if section == "test":
                    test_compositions.add(tuple(sorted(obj["kind"] for obj in objects)))
                if any(obj["outline"] == answer and obj["accent"] != answer for obj in objects):
                    if section == "train":
                        role_train += 1
                    else:
                        role_test += 1
                if any(obj["accent_area"] >= 5 for obj in objects):
                    large_accents += 1
                boxes = [(r, r + obj["height"], c, c + obj["width"])
                         for obj in objects for r, c in [obj["origin"]]]
                if any(max(a[0], b[0]) < min(a[1], b[1]) and
                       max(a[2], b[2]) < min(a[3], b[3])
                       for i, a in enumerate(boxes) for b in boxes[i + 1:]):
                    interleaved += 1
        if seed < 4:
            assert _same_puzzle(puzzle, generate_puzzle(seed))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(first_inputs) >= 40 and len(structures) >= 150
    assert len(backgrounds) >= 7 and len(kinds) == 4
    assert len(test_compositions) >= 8 and interleaved >= 5
    assert role_train > 0 and role_test > 0 and large_accents > 0
    for seed, nt, ns in ((100, 2, 1), (101, 10, 1), (102, 10, 4), (103, 5, 2),
                         (104, None, 1), (105, None, 4), (106, 2, None), (107, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ns is not None:
            assert len(puzzle["test"]) == ns
        assert _same_puzzle(puzzle, generate_puzzle(seed, nt, ns))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (4, 4), (3, 4)):
        try:
            generate_puzzle(123, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError((nt, ns))
    invalid = generate_puzzle(200, 3, 1)
    invalid["train"][0]["output"] = invalid["train"][0]["output"].copy()
    invalid["train"][0]["output"][0, 0] = (int(invalid["train"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("Accepted wrong output")
