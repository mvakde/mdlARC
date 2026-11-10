"""Generate three-point puzzles governed by strict collinear betweenness."""

import numpy as np


# Exact signatures of the seven source inputs, used only to prevent copies.
_SOURCE_INPUTS = {
    (10, 10, 0, 2, 4, ((2, 2), (2, 6)), (2, 9)),
    (10, 10, 0, 2, 4, ((2, 1), (8, 1)), (4, 1)),
    (10, 10, 0, 2, 4, ((0, 0), (4, 4)), (2, 2)),
    (10, 10, 0, 2, 4, ((0, 6), (0, 9)), (1, 7)),
    (10, 10, 0, 2, 4, ((3, 2), (9, 2)), (0, 2)),
    (10, 10, 0, 2, 4, ((0, 0), (9, 9)), (8, 8)),
    (10, 10, 0, 2, 4, ((3, 0), (3, 9)), (2, 0)),
}

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
_NEIGHBORS = tuple(
    (r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c
)


def _counts(rng, num_train, num_test):
    for value in (num_train, num_test):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError("pair counts must be integers")
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be 2 through 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be 1 through 4")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 1) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _decode(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 3
    bg_index = int(np.argmax(counts))
    background = int(values[bg_index])
    assert int(counts[bg_index]) == grid.size - 3
    foreground = [(int(v), int(n)) for v, n in zip(values, counts) if int(v) != background]
    assert sorted(n for _, n in foreground) == [1, 2]
    pair_color = next(v for v, n in foreground if n == 2)
    single_color = next(v for v, n in foreground if n == 1)
    a, b = sorted(tuple(map(int, pos)) for pos in np.argwhere(grid == pair_color))
    single = tuple(map(int, np.argwhere(grid == single_color)[0]))
    return background, pair_color, single_color, a, b, single


def _between(a, b, single):
    vr, vc = b[0] - a[0], b[1] - a[1]
    wr, wc = single[0] - a[0], single[1] - a[1]
    dot = vr * wr + vc * wc
    return vr * wc == vc * wr and 0 < dot < vr * vr + vc * vc


def _relation(a, b, single):
    cross = (b[0] - a[0]) * (single[1] - a[1]) - (b[1] - a[1]) * (single[0] - a[0])
    if cross:
        return "off"
    return "between" if _between(a, b, single) else "beyond"


def _orientation(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    if dr == 0:
        return "horizontal"
    if dc == 0:
        return "vertical"
    if abs(dr) == abs(dc):
        return "diag_down" if dr * dc > 0 else "diag_up"
    return "other"


def _expected_output(grid):
    background, _, _, a, b, single = _decode(grid)
    output = grid.copy()
    if _between(a, b, single):
        output[a] = background
        output[b] = background
    else:
        output[single] = background
    return output


def _signature(grid):
    bg, pair_color, single_color, a, b, single = _decode(grid)
    return (grid.shape[0], grid.shape[1], bg, pair_color, single_color, (a, b), single)


def _sample_span(rng, maximum, regime):
    bands = ((2, 4), (5, 7), (8, 12))
    weights = (0.40, 0.40, 0.20) if regime == "source_train" else (0.25, 0.35, 0.40)
    available = [(lo, min(hi, maximum), weight) for (lo, hi), weight in zip(bands, weights) if lo <= maximum]
    probabilities = np.array([item[2] for item in available], dtype=float)
    probabilities /= probabilities.sum()
    lo, hi, _ = available[int(rng.choice(len(available), p=probabilities))]
    return int(rng.integers(lo, hi + 1))


def _relative_points(rng, relation, direction, span, near, classic):
    dr, dc = direction
    a, b = (0, 0), (span * dr, span * dc)
    if relation == "between":
        phase = rng.random()
        if near or phase < 0.22:
            step = int(rng.choice((1, span - 1)))
        elif phase < 0.62:
            step = int(rng.choice((span // 2, (span + 1) // 2)))
        else:
            step = int(rng.integers(1, span))
        single = (step * dr, step * dc)
    elif relation == "beyond":
        extension = 1 if near else int(rng.choice((1, 2, 3), p=(0.45, 0.35, 0.20)))
        step = -extension if rng.random() < 0.5 else span + extension
        single = (step * dr, step * dc)
    else:
        if near or rng.random() < 0.68:
            step = int(rng.choice((0, span))) if near or rng.random() < 0.35 else int(rng.integers(0, span + 1))
            delta = _NEIGHBORS[int(rng.integers(0, len(_NEIGHBORS)))]
            while dr * delta[1] == dc * delta[0]:
                delta = _NEIGHBORS[int(rng.integers(0, len(_NEIGHBORS)))]
            single = (step * dr + delta[0], step * dc + delta[1])
        else:
            step = int(rng.integers(0, span + 1))
            distance = int(rng.integers(2, 4 if classic else 6))
            if rng.random() < 0.5:
                distance = -distance
            single = (step * dr + distance * dc, step * dc - distance * dr)
    return a, b, single


def _make_pair(rng, relation, direction, regime, classic, palette, background, near=False):
    assert relation in ("between", "beyond", "off")
    assert direction in _DIRECTIONS
    assert regime in ("source_train", "target_test")
    limit = 10 if classic else 20
    for _ in range(200):
        span = _sample_span(rng, 8 if classic else 12, regime)
        a, b, single = _relative_points(rng, relation, direction, span, near, classic)
        rows = (a[0], b[0], single[0])
        cols = (a[1], b[1], single[1])
        height_needed = max(rows) - min(rows) + 1
        width_needed = max(cols) - min(cols) + 1
        if height_needed > limit or width_needed > limit:
            continue
        height = 10 if classic else int(rng.integers(max(8, height_needed), 21))
        width = 10 if classic else int(rng.integers(max(8, width_needed), 21))
        offset_r = int(rng.integers(-min(rows), height - max(rows)))
        offset_c = int(rng.integers(-min(cols), width - max(cols)))
        pa = (a[0] + offset_r, a[1] + offset_c)
        pb = (b[0] + offset_r, b[1] + offset_c)
        ps = (single[0] + offset_r, single[1] + offset_c)
        grid = np.full((height, width), background, dtype=np.int8)
        grid[pa] = grid[pb] = palette[0]
        grid[ps] = palette[1]
        return {"input": grid, "output": _expected_output(grid)}
    raise RuntimeError("could not fit three points on a canvas")


def _colors(rng, classic, base, split, index, excluded_background=None):
    if classic:
        return (2, 4), 0
    if split == "train" and index < 2:
        palette = base
    elif split == "train" and index == 2:
        palette = base[::-1]
    elif rng.random() < 0.40:
        palette = base if rng.random() < 0.5 else base[::-1]
    else:
        palette = tuple(map(int, rng.choice(10, size=2, replace=False)))
    options = [color for color in range(10) if color not in palette and color != excluded_background]
    background = int(rng.choice(options))
    return palette, background


def _plans(rng, ntrain, ntest):
    anchor = _DIRECTIONS[int(rng.integers(0, 4))]
    other = [direction for direction in _DIRECTIONS if direction != anchor]
    nonbetween = "off" if rng.random() < 0.5 else "beyond"
    train = [("between", anchor), (nonbetween, anchor)]
    if ntrain >= 3:
        missing = "beyond" if nonbetween == "off" else "off"
        train.append((missing, other[int(rng.integers(0, 3))]))
    while len(train) < ntrain:
        train.append((str(rng.choice(("between", "beyond", "off"))), _DIRECTIONS[int(rng.integers(0, 4))]))
    allowed_test = ("between", nonbetween) if ntrain == 2 else ("between", "beyond", "off")
    test = [(str(rng.choice(allowed_test)), _DIRECTIONS[int(rng.integers(0, 4))]) for _ in range(ntest)]
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, num_train, num_test)
    classic = bool(rng.random() < 0.18)
    base = tuple(map(int, rng.choice(10, size=2, replace=False)))
    train_plan, test_plan = _plans(rng, ntrain, ntest)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_background = None
    first_shape = None
    for split, regime, plan in (("train", "source_train", train_plan), ("test", "target_test", test_plan)):
        for index, (relation, direction) in enumerate(plan):
            near = split == "test" and index == 0
            for _ in range(200):
                excluded = first_background if split == "train" and index == 1 and not classic else None
                palette, background = _colors(rng, classic, base, split, index, excluded)
                pair = _make_pair(rng, relation, direction, regime, classic, palette, background, near)
                signature = _signature(pair["input"])
                if signature in seen or signature in _SOURCE_INPUTS:
                    continue
                if split == "train" and index == 1 and not classic and pair["input"].shape == first_shape:
                    continue
                seen.add(signature)
                puzzle[split].append(pair)
                if split == "train" and index == 0:
                    first_background = background
                    first_shape = pair["input"].shape
                break
            else:
                raise RuntimeError("could not make a distinct pair")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _adjacent_to_endpoint(a, b, single):
    return any(max(abs(single[0] - p[0]), abs(single[1] - p[1])) == 1 for p in (a, b))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert type(puzzle["train"]) is list and type(puzzle["test"]) is list
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen_inputs = set()
    seen_pairs = set()
    facts = {"train": [], "test": []}
    test_has_near = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            background, pair_color, single_color, a, b, single = _decode(inp)
            assert pair_color != single_color and background not in (pair_color, single_color)
            assert _orientation(a, b) != "other"
            assert np.array_equal(out, _expected_output(inp))
            assert _signature(inp) not in _SOURCE_INPUTS
            input_key = (inp.shape, inp.tobytes())
            pair_key = (input_key, out.tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            relation = _relation(a, b, single)
            facts[split].append((_orientation(a, b), relation))
            if split == "test" and _adjacent_to_endpoint(a, b, single):
                test_has_near = True
    train = facts["train"]
    surviving_colors = {
        int(np.unique(pair["output"][pair["output"] != _decode(pair["input"])[0]])[0])
        for pair in puzzle["train"]
    }
    assert len(surviving_colors) >= 2
    assert {"between"} <= {relation for _, relation in train}
    assert any(relation != "between" for _, relation in train)
    assert any(
        direction_a == direction_b and relation_a != relation_b
        for direction_a, relation_a in train
        for direction_b, relation_b in train
    )
    if ntrain >= 3:
        assert {relation for _, relation in train} == {"between", "beyond", "off"}
    else:
        allowed = {relation for _, relation in train}
        assert all(relation in allowed for _, relation in facts["test"])
    assert test_has_near


def _fingerprint(puzzle):
    return tuple(
        (split, pair["input"].shape, pair["input"].tobytes(), pair["output"].tobytes())
        for split in ("train", "test") for pair in puzzle[split]
    )


def self_test() -> None:
    fingerprints = set()
    train_counts, test_counts = set(), set()
    shapes, palettes, backgrounds, layouts = set(), set(), set(), set()
    orientations = {"train": set(), "test": set()}
    relations = {"train": set(), "test": set()}
    spans = {"train": set(), "test": set()}
    near_relations = set()
    direction_relation = {"train": set(), "test": set()}
    close_off_offsets = set()
    beyond_sides = set()
    lone_test_relations = set()
    midpoint_examples = 0
    distant_off_examples = 0
    source_like_puzzles = 0
    tests_smaller_than_train = 0
    mixed_puzzles = 0
    role_reversal_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprint = _fingerprint(puzzle)
        assert fingerprint == _fingerprint(generate_puzzle(seed))
        fingerprints.add(fingerprint)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_shapes, puzzle_backgrounds = set(), set()
        puzzle_palettes = set()
        color_roles = {}
        train_areas = [pair["input"].size for pair in puzzle["train"]]
        if min(pair["input"].size for pair in puzzle["test"]) < max(train_areas):
            tests_smaller_than_train += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                bg, pc, sc, a, b, single = _decode(inp)
                vr, vc = b[0] - a[0], b[1] - a[1]
                wr, wc = single[0] - a[0], single[1] - a[1]
                inside = vr * wc == vc * wr and 0 < vr * wr + vc * wc < vr * vr + vc * vc
                independently_expected = np.where(inp == (sc if inside else pc), inp, bg)
                assert np.array_equal(out, independently_expected)
                shapes.add(inp.shape)
                palettes.add((pc, sc))
                backgrounds.add(bg)
                layouts.add((inp.shape, a, b, single))
                puzzle_shapes.add(inp.shape)
                puzzle_backgrounds.add(bg)
                puzzle_palettes.add((pc, sc))
                color_roles.setdefault(pc, set()).add("pair")
                color_roles.setdefault(sc, set()).add("single")
                orientation = _orientation(a, b)
                relation = _relation(a, b, single)
                orientations[split].add(orientation)
                relations[split].add(relation)
                direction_relation[split].add((orientation, relation))
                span = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                spans[split].add(span)
                if _adjacent_to_endpoint(a, b, single):
                    near_relations.add(relation)
                if relation == "between" and min(
                    max(abs(single[0] - p[0]), abs(single[1] - p[1])) for p in (a, b)
                ) >= 3:
                    midpoint_examples += 1
                elif relation == "beyond":
                    beyond_sides.add("before" if vr * wr + vc * wc < 0 else "after")
                elif relation == "off":
                    dr, dc = vr // span, vc // span
                    neighboring_steps = [
                        (single[0] - (a[0] + step * dr), single[1] - (a[1] + step * dc))
                        for step in range(span + 1)
                    ]
                    close = [(r, c) for r, c in neighboring_steps if max(abs(r), abs(c)) == 1]
                    if not close:
                        distant_off_examples += 1
                    for r, c in close:
                        if abs(r) == 1 and c == 0:
                            close_off_offsets.add("row")
                        elif r == 0 and abs(c) == 1:
                            close_off_offsets.add("column")
                        elif abs(r) == abs(c) == 1:
                            close_off_offsets.add("diagonal")
                if split == "test" and len(puzzle["test"]) == 1:
                    lone_test_relations.add(relation)
        if len(puzzle_shapes) > 1 and len(puzzle_backgrounds) > 1:
            mixed_puzzles += 1
        if puzzle_shapes == {(10, 10)} and puzzle_backgrounds == {0} and puzzle_palettes == {(2, 4)}:
            source_like_puzzles += 1
        if any(roles == {"pair", "single"} for roles in color_roles.values()):
            role_reversal_puzzles += 1
    assert len(fingerprints) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 30 and len(palettes) >= 20 and len(backgrounds) >= 8
    assert len(layouts) >= 150
    assert all(orientations[split] == {"horizontal", "vertical", "diag_down", "diag_up"} for split in orientations)
    assert all(relations[split] == {"between", "beyond", "off"} for split in relations)
    assert all(len(direction_relation[split]) == 12 for split in direction_relation)
    assert all(min(spans[split]) <= 3 and max(spans[split]) >= 9 for split in spans)
    assert near_relations == {"between", "beyond", "off"}
    assert close_off_offsets == {"row", "column", "diagonal"}
    assert beyond_sides == {"before", "after"}
    assert lone_test_relations == {"between", "beyond", "off"}
    assert midpoint_examples >= 5 and distant_off_examples >= 5
    assert source_like_puzzles >= 1 and tests_smaller_than_train >= 10
    assert mixed_puzzles >= 20 and role_reversal_puzzles >= 20
    for seed, train, test in ((100, 2, 1), (101, 10, 1), (102, 10, 4), (103, None, 1), (104, None, 4), (105, 2, None), (106, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed, train, test))
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (True, 1), (2.0, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {train}, {test}")
