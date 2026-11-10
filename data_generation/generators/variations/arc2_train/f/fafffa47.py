"""Generate aligned two-panel common-background ARC puzzles."""

import numpy as np


_SOURCE_ROWS = (
    (("009", "999", "090", "100", "011", "001"), ("020", "000", "200")),
    (("909", "099", "009", "101", "100", "100"), ("020", "000", "020")),
    (("090", "909", "900", "000", "001", "100"), ("202", "020", "022")),
    (("090", "099", "099", "000", "111", "101"), ("202", "000", "000")),
    (("099", "099", "999", "010", "001", "111"), ("200", "200", "000")),
    (("909", "009", "909", "011", "010", "100"), ("000", "200", "020")),
)
_SOURCE_PAIRS = frozenset(
    (
        (tuple(map(int, "".join(inp))), (len(inp), len(inp[0]))),
        (tuple(map(int, "".join(out))), (len(out), len(out[0]))),
    )
    for inp, out in _SOURCE_ROWS
)
_LEGAL_COUNTS = tuple(
    (a, b) for a in range(2, 11) for b in range(1, min(a, 5))
)
_SHAPES = tuple(
    (h, w)
    for h in range(2, 8)
    for w in range(2, 8)
    if max(h, w) < 2 * min(h, w)
)
_RELATIONS = ("equal", "top_subset", "bottom_subset", "disjoint", "partial")


def _ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid_key(grid):
    return (tuple(int(x) for x in grid.ravel()), tuple(int(x) for x in grid.shape))


def _pair_key(pair):
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _resolve_counts(seed, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer or None")
            if not low <= int(value) <= high:
                raise ValueError(f"{name} is outside {low}..{high}")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return int(num_train), int(num_test)
    if num_train is None and num_test is None:
        return _LEGAL_COUNTS[seed % len(_LEGAL_COUNTS)]
    if num_train is None:
        possible = tuple(range(max(2, int(num_test) + 1), 11))
        return possible[seed % len(possible)], int(num_test)
    possible = tuple(range(1, min(4, int(num_train) - 1) + 1))
    return int(num_train), possible[seed % len(possible)]


def _pair_rng(rng, ordinal):
    entropy = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
    return np.random.default_rng(np.random.SeedSequence([entropy, ordinal, 0xA5A5]))


def _roles(rng, highlight, previous=None, stable=None):
    if stable is not None:
        return stable
    if previous is not None and rng.random() < 0.38:
        order = rng.permutation(3)
        if tuple(int(x) for x in order) == (0, 1, 2):
            order = np.array((1, 2, 0))
        return tuple(previous[int(x)] for x in order)
    choices = [x for x in range(10) if x != highlight]
    return tuple(int(x) for x in rng.choice(choices, size=3, replace=False))


def _sample_shape(rng, base, classic):
    if classic or rng.random() < 0.43:
        return base
    return _SHAPES[int(rng.integers(len(_SHAPES)))]


def _max_empty(n, relation):
    if relation == "equal":
        return n - 1
    if relation == "partial":
        return n - 3
    return n - 2


def _choose_empty_count(rng, h, w, relation, anchor, seen=None):
    n = h * w
    maximum = _max_empty(n, relation)
    minimum = 1 if anchor or relation in ("equal", "top_subset", "bottom_subset") else 0
    eligible = list(range(minimum, maximum + 1))
    if seen is not None:
        familiar = [x for x in eligible if x in seen]
        novel = [x for x in eligible if x not in seen]
        if rng.random() < 0.52 and familiar:
            return int(rng.choice(familiar))
        if novel:
            return int(rng.choice(novel))
    if 0 in eligible and rng.random() < 0.10:
        return 0
    if 1 in eligible and rng.random() < 0.16:
        return 1
    # A shape category determines a useful count range before any mask is drawn.
    category = int(rng.integers(4))
    if category == 0:
        upper = min(maximum, w)
    elif category == 1:
        upper = min(maximum, h)
    elif category == 2:
        upper = min(maximum, max(3, n // 2))
    else:
        upper = maximum
    return int(rng.integers(max(1, minimum), upper + 1))


def _empty_mask(rng, h, w, count):
    mask = np.zeros((h, w), dtype=bool)
    if count == 0:
        return mask
    modes = ["scattered"]
    if count <= w:
        modes.append("row")
    if count <= h:
        modes.append("column")
    if count >= 2:
        modes.append("cluster")
    mode = modes[int(rng.integers(len(modes)))]
    if mode == "row":
        row = int(rng.integers(h))
        start = int(rng.integers(w - count + 1))
        mask[row, start:start + count] = True
    elif mode == "column":
        col = int(rng.integers(w))
        start = int(rng.integers(h - count + 1))
        mask[start:start + count, col] = True
    elif mode == "cluster":
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        while len(cells) < count:
            fringe = {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= r + dr < h and 0 <= c + dc < w
                and (r + dr, c + dc) not in cells
            }
            frontier = sorted(fringe)
            cells.add(frontier[int(rng.integers(len(frontier)))])
        for r, c in cells:
            mask[r, c] = True
    else:
        mask.flat[rng.choice(h * w, size=count, replace=False)] = True
    return mask


def _cases_from_mask(rng, empty_mask, relation):
    cases = np.zeros(empty_mask.shape, dtype=np.int8)
    open_positions = np.flatnonzero(~empty_mask)
    required = {
        "equal": (3,),
        "top_subset": (1, 3),
        "bottom_subset": (2, 3),
        "disjoint": (1, 2),
        "partial": (1, 2, 3),
    }[relation]
    if len(required) == 1:
        cases.flat[open_positions] = required[0]
        return cases
    h, w = empty_mask.shape
    mode = int(rng.integers(4))
    if mode == 0:
        rng.shuffle(open_positions)
    elif mode == 1:
        open_positions = np.asarray(
            sorted(open_positions, key=lambda x: (int(x) // w, int(x) % w))
        )
    elif mode == 2:
        open_positions = np.asarray(
            sorted(open_positions, key=lambda x: (int(x) % w, int(x) // w))
        )
    else:
        center_r, center_c = int(rng.integers(h)), int(rng.integers(w))
        open_positions = np.asarray(sorted(
            open_positions,
            key=lambda x: (
                abs(int(x) // w - center_r) + abs(int(x) % w - center_c),
                int(x),
            ),
        ))
    cuts = sorted(
        int(x) for x in rng.choice(
            np.arange(1, len(open_positions)), size=len(required) - 1, replace=False
        )
    )
    values = list(required)
    rng.shuffle(values)
    for value, positions in zip(values, np.split(open_positions, cuts)):
        cases.flat[positions] = value
    return cases


def _make_pair(rng, shape, orientation, roles, highlight, relation, empty_mask):
    h, w = shape
    background, first_color, second_color = roles
    cases = _cases_from_mask(rng, empty_mask, relation)
    first = np.full((h, w), background, dtype=np.int8)
    second = np.full((h, w), background, dtype=np.int8)
    first[(cases & 2) != 0] = first_color
    second[(cases & 1) != 0] = second_color
    if orientation == "vertical":
        inp = np.concatenate((first, second), axis=0)
    else:
        inp = np.concatenate((first, second), axis=1)
    out = np.full((h, w), background, dtype=np.int8)
    out[empty_mask] = highlight
    return {"input": inp, "output": out}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    highlight = int(rng.integers(10))
    classic = bool(rng.random() < 0.16)
    base = (3, 3) if classic else _SHAPES[int(rng.integers(len(_SHAPES)))]
    orientation_mode = "vertical" if classic else ("vertical", "horizontal", "mixed")[
        int(rng.choice(3, p=(0.53, 0.15, 0.32)))
    ]
    stable_palette = bool(classic or rng.random() < 0.25)
    fixed_roles = _roles(rng, highlight) if stable_palette else None
    anchor_index = int(rng.integers(train_count))
    repeat_candidates = [i for i in range(2, train_count) if i != anchor_index]
    repeated_index = (int(rng.choice(repeat_candidates))
                      if repeat_candidates and rng.random() < 0.38 else None)
    repeat_source = int(rng.integers(repeated_index)) if repeated_index is not None else None
    forced_orientations = None
    if orientation_mode == "mixed":
        forced_orientations = ["vertical", "horizontal"]
        rng.shuffle(forced_orientations)

    puzzle = {"train": [], "test": []}
    specs = []
    used_inputs = set()
    previous_roles = None
    train_empty_counts = set()
    for ordinal in range(train_count + test_count):
        section = "train" if ordinal < train_count else "test"
        index = ordinal if section == "train" else ordinal - train_count
        pair_rng = _pair_rng(rng, ordinal)
        repeat = section == "train" and index == repeated_index
        if repeat:
            source = specs[repeat_source]
            shape = source["shape"]
            orientation = source["orientation"]
            roles = source["roles"]
            empty_mask = source["empty_mask"].copy()
            choices = [name for name in _RELATIONS if name != source["relation"]
                       and _max_empty(shape[0] * shape[1], name) >= int(empty_mask.sum())
                       and (int(empty_mask.sum()) > 0 or name not in ("equal", "top_subset", "bottom_subset"))]
            relation = choices[int(pair_rng.integers(len(choices)))]
        else:
            shape = _sample_shape(pair_rng, base, classic)
            if orientation_mode == "mixed":
                orientation = (forced_orientations[index] if section == "train" and index < 2
                               else ("vertical" if pair_rng.random() < 0.5 else "horizontal"))
            else:
                orientation = orientation_mode
            roles = _roles(pair_rng, highlight, previous_roles, fixed_roles)
            if section == "train" and index == 1 and not stable_palette and roles == specs[0]["roles"]:
                roles = (roles[1], roles[2], roles[0])
            anchor = section == "train" and index == anchor_index
            relation = "partial" if anchor else _RELATIONS[int(pair_rng.choice(
                len(_RELATIONS), p=(0.15, 0.19, 0.19, 0.18, 0.29)
            ))]
            count = _choose_empty_count(
                pair_rng, *shape, relation, anchor,
                train_empty_counts if section == "test" else None,
            )
            empty_mask = _empty_mask(pair_rng, *shape, count)
        for attempt in range(120):
            pair = _make_pair(pair_rng, shape, orientation, roles, highlight, relation, empty_mask)
            if _grid_key(pair["input"]) not in used_inputs and _pair_key(pair) not in _SOURCE_PAIRS:
                break
            if repeat:
                # A new allocation preserves the repeated output.
                continue
            if not anchor:
                relation = _RELATIONS[int(pair_rng.integers(len(_RELATIONS)))]
            count = _choose_empty_count(
                pair_rng, *shape, relation, anchor,
                train_empty_counts if section == "test" else None,
            )
            empty_mask = _empty_mask(pair_rng, *shape, count)
        else:
            raise RuntimeError("could not construct a unique pair")
        used_inputs.add(_grid_key(pair["input"]))
        puzzle[section].append(pair)
        previous_roles = roles
        if section == "train":
            train_empty_counts.add(int(empty_mask.sum()))
            specs.append({
                "shape": shape, "orientation": orientation, "roles": roles,
                "relation": relation, "empty_mask": empty_mask,
            })
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair, known_highlight=None):
    inp, out = pair["input"], pair["output"]
    if inp.shape[0] > inp.shape[1]:
        _ensure(inp.shape[0] % 2 == 0, "vertical input height must be even")
        h = inp.shape[0] // 2
        first, second = inp[:h, :], inp[h:, :]
        orientation = "vertical"
    else:
        _ensure(inp.shape[1] > inp.shape[0] and inp.shape[1] % 2 == 0,
                "horizontal input width must be even")
        w = inp.shape[1] // 2
        first, second = inp[:, :w], inp[:, w:]
        orientation = "horizontal"
    _ensure(out.shape == first.shape, "output must have one-panel dimensions")
    first_colors = set(int(x) for x in np.unique(first))
    second_colors = set(int(x) for x in np.unique(second))
    _ensure(len(first_colors) == len(second_colors) == 2, "each panel needs background and foreground")
    common = first_colors & second_colors
    _ensure(len(common) == 1, "panels must share only their background color")
    background = next(iter(common))
    first_color = next(iter(first_colors - common))
    second_color = next(iter(second_colors - common))
    _ensure(first_color != second_color, "panel foreground colors must differ")
    _ensure(background in set(int(x) for x in np.unique(out)), "output background must match input")
    other_output_colors = set(int(x) for x in np.unique(out)) - {background}
    _ensure(len(other_output_colors) <= 1, "output has too many colors")
    if other_output_colors:
        highlight = next(iter(other_output_colors))
        _ensure(highlight not in first_colors | second_colors, "highlight occurs in input")
    else:
        highlight = known_highlight
    if known_highlight is not None:
        _ensure(not other_output_colors or highlight == known_highlight, "highlight changes across pairs")
        _ensure(known_highlight not in first_colors | second_colors, "learned highlight occurs in input")
    cases = 2 * (first != background).astype(np.int8) + (second != background).astype(np.int8)
    if highlight is not None:
        expected = np.full(first.shape, background, dtype=out.dtype)
        expected[cases == 0] = highlight
        _ensure(np.array_equal(out, expected), "output is not the common-background NOR")
    else:
        _ensure(not np.any(cases == 0), "missing highlight in a pair with common background")
        _ensure(np.all(out == background), "zero-highlight output must be background")
    return {
        "background": background, "highlight": highlight, "orientation": orientation,
        "cases": frozenset(int(x) for x in cases.flat),
        "empty_count": int(np.count_nonzero(cases == 0)),
        "empty_mask": (first.shape, tuple(int(x) for x in (cases == 0).flat)),
        "roles": (background, first_color, second_color),
        "shape": first.shape,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _ensure(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _ensure(all(isinstance(puzzle[s], list) for s in ("train", "test")), "sections must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _ensure(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
            "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _ensure(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            for grid in pair.values():
                _ensure(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D ndarray")
                _ensure(all(1 <= x <= 30 for x in grid.shape), "grid dimension outside 1..30")
                _ensure(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                _ensure(bool(np.all((0 <= grid) & (grid <= 9))), "grid color outside 0..9")
            key = _pair_key(pair)
            _ensure(key[0] not in seen_inputs and key not in seen_pairs, "duplicate input or pair")
            _ensure(key not in _SOURCE_PAIRS, "exact source pair copy")
            seen_inputs.add(key[0])
            seen_pairs.add(key)
            analyses[section].append(_analyze_pair(pair))
    train_highlights = {a["highlight"] for a in analyses["train"] if a["highlight"] is not None}
    _ensure(len(train_highlights) == 1, "training must reveal exactly one highlight")
    highlight = next(iter(train_highlights))
    for section in ("train", "test"):
        for pair, analysis in zip(puzzle[section], analyses[section]):
            _analyze_pair(pair, highlight)
    truth_cases = set().union(*(a["cases"] for a in analyses["train"]))
    _ensure(truth_cases == {0, 1, 2, 3}, "training does not demonstrate the NOR truth table")
    train_orientations = {a["orientation"] for a in analyses["train"]}
    test_orientations = {a["orientation"] for a in analyses["test"]}
    _ensure(test_orientations <= train_orientations, "test orientation is absent from training")


def _predict_from_training(train_pairs, test_input):
    highlights = set()
    for pair in train_pairs:
        analysis = _analyze_pair(pair)
        if analysis["highlight"] is not None:
            highlights.add(analysis["highlight"])
    _ensure(len(highlights) == 1, "cannot learn highlight from training")
    if test_input.shape[0] > test_input.shape[1]:
        h = test_input.shape[0] // 2
        first, second = test_input[:h], test_input[h:]
    else:
        w = test_input.shape[1] // 2
        first, second = test_input[:, :w], test_input[:, w:]
    shared = set(int(x) for x in np.unique(first)) & set(int(x) for x in np.unique(second))
    _ensure(len(shared) == 1, "test background is not identifiable")
    background = next(iter(shared))
    result = np.full(first.shape, background, dtype=np.int8)
    result[(first == background) & (second == background)] = next(iter(highlights))
    return result


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(pair) for pair in puzzle[s]) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _ensure(len({_puzzle_key(p) for p in puzzles}) >= 40, "puzzles lack seed diversity")
    _ensure(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack diversity")
    _ensure(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack diversity")
    shapes, orientations, relations, counts, palettes = set(), set(), set(), set(), set()
    nested = disjoint = equal = partial = zero = one = repeated = palette_changes = size_changes = 0
    orientation_changes = role_exchanges = combined_changes = 0
    seen_test_counts = new_test_counts = absent_truth_cases = 0
    geometry = {"row": 0, "column": 0, "connected": 0, "scattered": 0}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_analyses = [_analyze_pair(pair) for pair in puzzle["train"]]
        if len({a["roles"] for a in train_analyses}) > 1:
            palette_changes += 1
        if len({a["shape"] for a in train_analyses}) > 1:
            size_changes += 1
        if len({a["orientation"] for a in train_analyses}) > 1:
            orientation_changes += 1
        role_exchanges += any(
            color in later and later[index] != color
            for index, color in enumerate(train_analyses[0]["roles"])
            for later in (a["roles"] for a in train_analyses[1:])
        )
        combined_changes += (
            len({a["shape"] for a in train_analyses}) > 1
            and len({a["roles"] for a in train_analyses}) > 1
            and len({a["orientation"] for a in train_analyses}) > 1
        )
        masks = [a["empty_mask"] for a in train_analyses]
        repeated += len(masks) > len(set(masks))
        demonstrated_counts = {a["empty_count"] for a in train_analyses}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                analysis = _analyze_pair(pair)
                shapes.add(analysis["shape"])
                orientations.add(analysis["orientation"])
                counts.add(analysis["empty_count"])
                palettes.add(analysis["roles"])
                if analysis["empty_count"] == 0:
                    zero += 1
                if analysis["empty_count"] == 1:
                    one += 1
                absent_truth_cases += analysis["cases"] != frozenset((0, 1, 2, 3))
                if section == "test":
                    seen_test_counts += analysis["empty_count"] in demonstrated_counts
                    new_test_counts += analysis["empty_count"] not in demonstrated_counts
                mask = np.asarray(analysis["empty_mask"][1], dtype=bool).reshape(analysis["shape"])
                points = [tuple(int(v) for v in point) for point in np.argwhere(mask)]
                if len(points) >= 2:
                    geometry["row"] += len({r for r, _ in points}) == 1
                    geometry["column"] += len({c for _, c in points}) == 1
                    reached = {points[0]}
                    remaining = set(points[1:])
                    while remaining:
                        growth = {
                            point for point in remaining
                            if any(abs(point[0] - r) + abs(point[1] - c) == 1
                                   for r, c in reached)
                        }
                        if not growth:
                            break
                        reached.update(growth)
                        remaining.difference_update(growth)
                    geometry["connected" if not remaining else "scattered"] += 1
                inp = pair["input"]
                if analysis["orientation"] == "vertical":
                    first, second = inp[:analysis["shape"][0]], inp[analysis["shape"][0]:]
                else:
                    first, second = inp[:, :analysis["shape"][1]], inp[:, analysis["shape"][1]:]
                a = first != analysis["background"]
                b = second != analysis["background"]
                if np.array_equal(a, b):
                    equal += 1
                    relations.add("equal")
                elif np.all(~a | b):
                    nested += 1
                    relations.add("top_subset")
                elif np.all(~b | a):
                    nested += 1
                    relations.add("bottom_subset")
                elif not np.any(a & b):
                    disjoint += 1
                    relations.add("disjoint")
                else:
                    partial += 1
                    relations.add("partial")
        for pair in puzzle["test"]:
            predicted = _predict_from_training(
                [{"input": q["input"].copy(), "output": q["output"].copy()} for q in puzzle["train"]],
                pair["input"].copy(),
            )
            _ensure(np.array_equal(predicted, pair["output"]), "training-only prediction differs")
    _ensure(len(shapes) >= 12 and len(orientations) == 2, "shape/orientation diversity too low")
    _ensure(relations == set(_RELATIONS), "mask relationship diversity too low")
    _ensure(len(counts) >= 10 and zero > 0 and one > 0, "highlight counts lack diversity")
    _ensure(len(palettes) >= 40, "local palette diversity too low")
    _ensure(repeated >= 3 and palette_changes >= 5 and size_changes >= 5
            and orientation_changes >= 2 and role_exchanges >= 3
            and combined_changes >= 2, "within-puzzle variation is too narrow")
    _ensure(seen_test_counts >= 5 and new_test_counts >= 5 and absent_truth_cases >= 20,
            "test count regimes or unconstrained pair relationships are too narrow")
    _ensure(all(value >= 5 for value in geometry.values()), "spatial arrangements lack diversity")
    _ensure(min(equal, nested, disjoint, partial) >= 5, "mask relationships lack diversity")
    for seed, a, b in ((5001, 2, 1), (5002, 10, 1), (5003, 10, 4),
                       (6001, None, 1), (6002, None, 4), (6003, 2, None), (6004, 10, None)):
        puzzle = generate_puzzle(seed, a, b)
        validate_generated_puzzle(puzzle)
        _ensure(a is None or len(puzzle["train"]) == a, "train request not respected")
        _ensure(b is None or len(puzzle["test"]) == b, "test request not respected")
    for request in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(7000, *request)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
    _ensure(_puzzle_key(generate_puzzle(12345)) == _puzzle_key(generate_puzzle(12345)),
            "automatic generation is not deterministic")
    _ensure(_puzzle_key(generate_puzzle(12345, 7, 3)) == _puzzle_key(generate_puzzle(12345, 7, 3)),
            "explicit generation is not deterministic")
